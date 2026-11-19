from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats

from anvala.diagnostics import _duplication_matrix, _vech_indices
from anvala.polychoric import PolychoricResult, polychoric


@dataclass
class _MGCFASpec:
    lv_names: list[str]
    ov_names: list[str]
    loadings: dict[str, list[str]]
    residual_covs: list[tuple[str, str]]


def _parse_mg_cfa_syntax(syntax: str) -> _MGCFASpec:
    lv_names: list[str] = []
    raw_ovs: list[str] = []
    loadings: dict[str, list[str]] = {}
    cov_pairs: list[tuple[str, str]] = []

    for raw_line in syntax.splitlines():
        clean = raw_line.split("#", 1)[0].strip()
        if not clean:
            continue
        for stmt in clean.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            if "=~" in stmt:
                lhs, rhs = stmt.split("=~", 1)
                lv = lhs.strip()
                if lv not in lv_names:
                    lv_names.append(lv)
                    loadings[lv] = []
                for term in rhs.split("+"):
                    v = term.split("*")[-1].strip()
                    if v and v not in loadings[lv]:
                        loadings[lv].append(v)
                    if v and v not in raw_ovs:
                        raw_ovs.append(v)
            elif "~~" in stmt:
                lhs, rhs = stmt.split("~~", 1)
                v1 = lhs.strip()
                for term in rhs.split("+"):
                    v2 = term.split("*")[-1].strip()
                    if v1 and v2 and v1 != v2:
                        cov_pairs.append((v1, v2))

    lv_set = set(lv_names)
    ov_names = [v for v in raw_ovs if v not in lv_set]
    residual_covs = [(a, b) for a, b in cov_pairs if a not in lv_set and b not in lv_set]
    return _MGCFASpec(lv_names=lv_names, ov_names=ov_names, loadings=loadings, residual_covs=residual_covs)


def _get_fit_summary_stats(fit) -> dict:
    """Extrai estatísticas padronizadas de AnvalaFit, SEMFit ou MultiGroupFit para compare_models."""
    ft = fit.fit_measures()
    est = getattr(fit, "estimator", "ML").upper()
    is_scaled = est in ("MLR", "WLSMV") or bool(getattr(fit, "ordered", False))
    col_idx = "Robust" if (is_scaled and "Robust" in ft.columns and pd.notna(ft.loc["cfi", "Robust"])) else ("Scaled" if is_scaled else "Unscaled")

    chisq_un = float(ft.loc["chisq", "Unscaled"])
    chisq_sc = float(ft.loc["chisq", "Scaled"]) if ("Scaled" in ft.columns and pd.notna(ft.loc["chisq", "Scaled"])) else chisq_un
    df_val = int(round(float(ft.loc["df", "Unscaled"])))
    c_mod = float(ft.loc["scaling_factor", "Scaled"]) if ("scaling_factor" in ft.index and "Scaled" in ft.columns and pd.notna(ft.loc["scaling_factor", "Scaled"])) else 1.0

    # AIC / BIC para modelos contínuos ML/MLR
    pe = fit.parameter_estimates()
    n_free = int(np.sum((pe["se"].fillna(0.0).to_numpy() > 1e-12) & (pe["op"] != ":=")))
    n_obs = int(getattr(fit, "n_obs", 0))
    aic_val = float(ft.loc["aic", "Unscaled"]) if "aic" in ft.index else np.nan
    bic_val = float(ft.loc["bic", "Unscaled"]) if "bic" in ft.index else np.nan
    if np.isnan(aic_val) and not getattr(fit, "ordered", False) and n_obs > 0:
        aic_val = chisq_un + 2.0 * n_free
        bic_val = chisq_un + np.log(float(n_obs)) * n_free

    return {
        "chisq_unscaled": chisq_un,
        "chisq_scaled": chisq_sc,
        "df": df_val,
        "scaling_factor": c_mod,
        "is_scaled": is_scaled,
        "aic": aic_val,
        "bic": bic_val,
        "cfi": float(ft.loc["cfi", col_idx]),
        "tli": float(ft.loc["tli", col_idx]),
        "rmsea": float(ft.loc["rmsea", col_idx]),
        "srmr": float(ft.loc["srmr", "Unscaled"]),
    }


def compare_models(*fits, names: list[str] | None = None) -> pd.DataFrame:
    """
    Compara modelos aninhados em paridade com lavaan::lavTestLRT (Satorra-Bentler 2001 para MLR e WLSMV),
    incluindo variações nos índices de ajuste (delta_cfi, delta_tli, delta_rmsea, delta_srmr).
    """
    if len(fits) < 2:
        raise ValueError("Forneça pelo menos dois modelos ajustados para comparação.")

    if names is None:
        names = [f"Model_{i + 1}" for i in range(len(fits))]
    elif len(names) != len(fits):
        raise ValueError("O comprimento de 'names' deve ser igual ao número de modelos.")

    items = [(nm, ft, _get_fit_summary_stats(ft)) for nm, ft in zip(names, fits)]
    # Ordena do menos restrito (menor df) para o mais restrito (maior df), padrão lavaan::lavTestLRT
    items.sort(key=lambda x: x[2]["df"])

    rows = []
    for idx, (nm, _, st) in enumerate(items):
        if idx == 0:
            rows.append(
                {
                    "model": nm,
                    "df": st["df"],
                    "aic": st["aic"],
                    "bic": st["bic"],
                    "chisq": st["chisq_unscaled"],
                    "chisq_scaled": st["chisq_scaled"],
                    "chisq_diff": np.nan,
                    "df_diff": np.nan,
                    "pvalue": np.nan,
                    "cfi": st["cfi"],
                    "delta_cfi": np.nan,
                    "tli": st["tli"],
                    "delta_tli": np.nan,
                    "rmsea": st["rmsea"],
                    "delta_rmsea": np.nan,
                    "srmr": st["srmr"],
                    "delta_srmr": np.nan,
                }
            )
            continue

        prev_st = items[idx - 1][2]
        df_1, df_0 = prev_st["df"], st["df"]
        df_diff = df_0 - df_1
        t_1, t_0 = prev_st["chisq_unscaled"], st["chisq_unscaled"]

        if df_diff <= 0:
            chisq_diff = np.nan
            pval = np.nan
        elif st["is_scaled"] or prev_st["is_scaled"]:
            # Correção escalonada de Satorra-Bentler (2001), padrão lavaan::lavTestLRT
            c_1, c_0 = prev_st["scaling_factor"], st["scaling_factor"]
            cd = (df_0 * c_0 - df_1 * c_1) / float(df_diff)
            if cd > 1e-12:
                chisq_diff = max(0.0, (t_0 - t_1) / cd)
                pval = float(1.0 - stats.chi2.cdf(chisq_diff, df_diff))
            else:
                chisq_diff = np.nan
                pval = np.nan
        else:
            chisq_diff = max(0.0, t_0 - t_1)
            pval = float(1.0 - stats.chi2.cdf(chisq_diff, df_diff))

        rows.append(
            {
                "model": nm,
                "df": df_0,
                "aic": st["aic"],
                "bic": st["bic"],
                "chisq": t_0,
                "chisq_scaled": st["chisq_scaled"],
                "chisq_diff": chisq_diff,
                "df_diff": float(df_diff) if df_diff > 0 else np.nan,
                "pvalue": pval,
                "cfi": st["cfi"],
                "delta_cfi": st["cfi"] - prev_st["cfi"],
                "tli": st["tli"],
                "delta_tli": st["tli"] - prev_st["tli"],
                "rmsea": st["rmsea"],
                "delta_rmsea": st["rmsea"] - prev_st["rmsea"],
                "srmr": st["srmr"],
                "delta_srmr": st["srmr"] - prev_st["srmr"],
            }
        )

    return pd.DataFrame(rows)


def _normalize_group_equal(group_equal: list[str] | tuple[str, ...] | str | None) -> set[str]:
    if group_equal is None:
        return set()
    if isinstance(group_equal, str):
        items = [group_equal]
    else:
        items = list(group_equal)
    aliases = {
        "loadings": "loadings",
        "lambda": "loadings",
        "intercepts": "intercepts",
        "nu": "intercepts",
        "thresholds": "thresholds",
        "tau": "thresholds",
        "residuals": "residuals",
        "theta": "residuals",
        "lv.variances": "lv.variances",
        "lv.covariances": "lv.covariances",
        "means": "means",
        "alpha": "means",
    }
    out = set()
    for it in items:
        k = it.strip().lower()
        if k in aliases:
            out.add(aliases[k])
        elif k:
            raise ValueError(f"Restrição '{it}' não reconhecida em group_equal.")
    return out


def _normalize_group_partial(group_partial: list[str] | tuple[str, ...] | str | None) -> set[tuple[str, str, str]]:
    """Converte strings como ['visual =~ x2', 'x4 ~ 1', 'x3 ~~ x3'] em tuplas canônicas (lhs, op, rhs)."""
    if group_partial is None:
        return set()
    items = [group_partial] if isinstance(group_partial, str) else list(group_partial)
    out = set()
    for raw in items:
        s = raw.strip()
        if not s:
            continue
        if "=~" in s:
            l_p, r_p = s.split("=~", 1)
            for term in r_p.split("+"):
                out.add((l_p.strip(), "=~", term.strip()))
        elif "~~" in s:
            l_p, r_p = s.split("~~", 1)
            for term in r_p.split("+"):
                out.add((l_p.strip(), "~~", term.strip()))
        elif "~" in s:
            l_p, r_p = s.split("~", 1)
            r_clean = r_p.strip()
            out.add((l_p.strip(), "~1" if r_clean == "1" else "~", r_clean))
        elif "|" in s:
            l_p, r_p = s.split("|", 1)
            out.add((l_p.strip(), "|", r_p.strip()))
    return out


@dataclass
class MultiGroupFit:
    group_col: str
    group_labels: list[str]
    n_obs_per_group: list[int]
    n_obs: int
    lv_names: list[str]
    ov_names: list[str]
    estimator: str
    ordered: bool
    group_equal: list[str]
    group_partial: list[str]
    pe_df: pd.DataFrame
    fit_table: pd.DataFrame
    group_chisq: dict[str, float]
    sigma_per_group: list[np.ndarray]
    s_per_group: list[np.ndarray]
    vcov: np.ndarray
    _score_context: dict | None = None
    converged: bool = True

    def parameter_estimates(self) -> pd.DataFrame:
        return self.pe_df.copy()

    def fit_measures(self) -> pd.DataFrame:
        return self.fit_table.copy()

    def score_test(self, sort: bool = True, return_total: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula testes escore (multiplicadores de Lagrange / lavTestScore) para as restrições
        de igualdade entre grupos (total e univariado, com versões escalonada/robusta).

        - return_total=False (padrão): retorna o DataFrame univariado por restrição.
        - return_total=True: retorna tupla (total_df, uni_df) em paridade com lavTestScore().
        """
        ctx = self._score_context
        uni_cols = ["lhs", "op", "rhs", "x2", "df", "pvalue", "x2_scaled", "pvalue_scaled"]
        tot_cols = ["test", "x2", "df", "pvalue"]

        if ctx is None or len(ctx.get("constraints", [])) == 0:
            empty_uni = pd.DataFrame(columns=uni_cols)
            if return_total:
                return pd.DataFrame(columns=tot_cols), empty_uni
            return empty_uni

        score_vec = ctx["score"]
        info_mat = ctx["info"]
        meat_mat = ctx.get("meat")
        r_mat = ctx["r_mat"]
        n_eff = float(ctx["n_eff"])
        constraints = ctx["constraints"]
        n_con = r_mat.shape[0]

        def _bordered_iinv(info_m: np.ndarray, r_sub: np.ndarray | None) -> np.ndarray:
            q = info_m.shape[0]
            if r_sub is None or r_sub.shape[0] == 0:
                return np.linalg.pinv(info_m)
            k_r = r_sub.shape[0]
            aug = np.block([
                [info_m, r_sub.T],
                [r_sub, np.zeros((k_r, k_r), dtype=np.float64)],
            ])
            aug_inv = np.linalg.pinv(aug)
            return aug_inv[:q, :q]

        # 1. Teste escore global (todas as restrições simultaneamente)
        j_inv = _bordered_iinv(info_mat, None)
        stat_tot = float(n_eff * (score_vec @ j_inv @ score_vec))
        pval_tot = float(1.0 - stats.chi2.cdf(max(0.0, stat_tot), n_con))
        tot_rows = [{"test": "score", "x2": stat_tot, "df": float(n_con), "pvalue": pval_tot}]

        if meat_mat is not None:
            aj = r_mat @ j_inv
            m1 = aj @ r_mat.T
            m2 = aj @ meat_mat @ aj.T
            v_vec = aj @ score_vec
            try:
                u_mat = np.linalg.solve(m1, m2)
            except np.linalg.LinAlgError:
                u_mat = np.linalg.pinv(m1) @ m2
            tr_u = float(np.trace(u_mat))
            tr_u2 = float(np.trace(u_mat @ u_mat))
            c_sc = tr_u / max(1e-12, float(n_con))
            x2_sc = stat_tot / max(1e-12, c_sc)
            df_adj = (tr_u * tr_u) / max(1e-12, tr_u2)
            x2_adj = stat_tot * (df_adj / max(1e-12, tr_u))
            x2_rob = float(n_eff * (v_vec @ np.linalg.pinv(m2) @ v_vec))

            tot_rows.extend([
                {"test": "score.scaled", "x2": x2_sc, "df": float(n_con), "pvalue": float(1.0 - stats.chi2.cdf(max(0.0, x2_sc), n_con))},
                {"test": "score.adjusted", "x2": x2_adj, "df": df_adj, "pvalue": float(1.0 - stats.chi2.cdf(max(0.0, x2_adj), max(1e-6, df_adj)))},
                {"test": "score.robust", "x2": x2_rob, "df": float(n_con), "pvalue": float(1.0 - stats.chi2.cdf(max(0.0, x2_rob), n_con))},
            ])

        total_df = pd.DataFrame(tot_rows, columns=tot_cols)

        # 2. Testes escore univariados (liberando uma restrição por vez: R[-r, :])
        uni_rows = []
        row_cursor = 0
        for c_info in constraints:
            k_rows = int(c_info["n_rows"])
            r_indices = list(range(row_cursor, row_cursor + k_rows))
            row_cursor += k_rows

            keep_mask = np.ones(n_con, dtype=bool)
            keep_mask[r_indices] = False
            r_minus = r_mat[keep_mask]
            z1_p1 = _bordered_iinv(info_mat, r_minus)

            x2_u = float(n_eff * (score_vec @ z1_p1 @ score_vec))
            pv_u = float(1.0 - stats.chi2.cdf(max(0.0, x2_u), k_rows))

            x2_u_sc = np.nan
            pv_u_sc = np.nan
            if meat_mat is not None:
                a_r = r_mat[r_indices]
                aj_r = a_r @ z1_p1
                m1_r = aj_r @ a_r.T
                m2_r = aj_r @ meat_mat @ aj_r.T
                try:
                    u_r = np.linalg.solve(m1_r, m2_r)
                except np.linalg.LinAlgError:
                    u_r = np.linalg.pinv(m1_r) @ m2_r
                c_r = float(np.trace(u_r)) / max(1e-12, float(k_rows))
                if np.isfinite(c_r) and c_r > 0:
                    x2_u_sc = x2_u / c_r
                    pv_u_sc = float(1.0 - stats.chi2.cdf(max(0.0, x2_u_sc), k_rows))

            uni_rows.append({
                "lhs": c_info["lhs"],
                "op": c_info["op"],
                "rhs": c_info["rhs"],
                "x2": x2_u,
                "df": k_rows,
                "pvalue": pv_u,
                "x2_scaled": x2_u_sc,
                "pvalue_scaled": pv_u_sc,
            })

        uni_df = pd.DataFrame(uni_rows, columns=uni_cols)
        if sort and not uni_df.empty:
            uni_df = uni_df.sort_values(by="x2", ascending=False).reset_index(drop=True)

        if return_total:
            return total_df, uni_df
        return uni_df

    def summary(self) -> str:
        lines = [
            f"anvala Multi-Group CFA ({self.estimator}, groups={self.group_labels}, N={self.n_obs}, converged={self.converged})",
            f"group.equal = {self.group_equal}",
            "--- Fit Measures ---",
            self.fit_table.round(4).to_string(),
        ]
        return "\n".join(lines)


def cfa_multigroup(
    model: str,
    data: pd.DataFrame,
    group: str,
    estimator: str = "ML",
    ordered: bool = False,
    group_equal: list[str] | tuple[str, ...] | str | None = None,
    group_partial: list[str] | tuple[str, ...] | str | None = None,
    std_lv: bool = False,
) -> MultiGroupFit:
    """
    Estima Análise Fatorial Confirmatória Multigrupo (MG-CFA) em paridade com lavaan::cfa(..., group=..., group.equal=...).
    Suporta estimadores ML, MLR (contínuos com estrutura de médias) e WLSMV (ordinais na parametrização delta).
    """
    if group not in data.columns:
        raise ValueError(f"Coluna de agrupamento '{group}' não encontrada nos dados.")

    est_upper = estimator.upper()
    if est_upper == "WLSMV":
        ordered = True

    eq_set = _normalize_group_equal(group_equal)
    partial_set = _normalize_group_partial(group_partial)

    spec = _parse_mg_cfa_syntax(model)
    lv_names = list(spec.lv_names)
    ov_names = list(spec.ov_names)
    m = len(lv_names)
    p = len(ov_names)
    ov_to_idx = {v: i for i, v in enumerate(ov_names)}
    lv_to_idx = {v: i for i, v in enumerate(lv_names)}

    req_cols = ov_names + [group]
    df_clean = data[req_cols].dropna().copy()
    group_labels = [str(g) for g in pd.unique(df_clean[group])]
    n_groups = len(group_labels)
    if n_groups < 2:
        raise ValueError("São necessários pelo menos 2 grupos distintos para análise multigrupo.")

    n_obs_list: list[int] = []
    s_list: list[np.ndarray] = []
    mean_list: list[np.ndarray] = []
    y_centered_list: list[np.ndarray] = []
    poly_list: list[PolychoricResult] = []

    for g_lbl in group_labels:
        sub = df_clean[df_clean[group].astype(str) == g_lbl][ov_names]
        n_g = int(sub.shape[0])
        n_obs_list.append(n_g)
        if ordered:
            pr = polychoric(sub)
            poly_list.append(pr)
            s_list.append(pr.corr.to_numpy(dtype=np.float64))
            mean_list.append(np.zeros(p, dtype=np.float64))
        else:
            y_g = sub.to_numpy(dtype=np.float64)
            m_g = np.mean(y_g, axis=0)
            s_g = np.cov(y_g, rowvar=False, bias=True)
            s_list.append(s_g)
            mean_list.append(m_g)
            y_centered_list.append(y_g - m_g[None, :])

    n_total = int(sum(n_obs_list))

    # Mapeamento de parâmetros por grupo (slots livres e restrições de igualdade)
    init_vals: list[float] = []
    is_log_var: list[bool] = []
    shared_slot_map: dict[tuple, int] = {}

    def _alloc_slot(key: tuple, default_val: float, log_scale: bool, share: bool) -> int:
        if share and key in shared_slot_map:
            return shared_slot_map[key]
        idx = len(init_vals)
        is_log_var.append(log_scale)
        init_vals.append(np.log(max(1e-3, default_val)) if log_scale else default_val)
        if share:
            shared_slot_map[key] = idx
        return idx

    group_param_maps: list[dict] = []
    for g in range(n_groups):
        lam_fixed = np.zeros((p, m), dtype=np.float64)
        lam_slots = np.full((p, m), -1, dtype=np.int64)
        for lv_i, lv in enumerate(lv_names):
            inds = spec.loadings[lv]
            for j_ind, ov in enumerate(inds):
                ov_i = ov_to_idx[ov]
                is_marker = (j_ind == 0) and (not std_lv)
                if is_marker:
                    lam_fixed[ov_i, lv_i] = 1.0
                else:
                    share_lam = ("loadings" in eq_set) and ((lv, "=~", ov) not in partial_set)
                    lam_slots[ov_i, lv_i] = _alloc_slot(("=~", lv, ov), 0.65, False, share_lam)

        # Variâncias e covariâncias latentes (Psi)
        psi_fixed = np.zeros((m, m), dtype=np.float64)
        psi_slots = np.full((m, m), -1, dtype=np.int64)
        for lv_i, lv in enumerate(lv_names):
            # Se std_lv=True: fixo em 1 no grupo 0; livre nos grupos >=1 se loadings forem iguais
            if std_lv and (g == 0 or "loadings" not in eq_set):
                psi_fixed[lv_i, lv_i] = 1.0
            else:
                share_lv_var = ("lv.variances" in eq_set) and ((lv, "~~", lv) not in partial_set)
                psi_slots[lv_i, lv_i] = _alloc_slot(("~~", lv, lv), 0.5, True, share_lv_var)

        for i_lv in range(m):
            for j_lv in range(i_lv + 1, m):
                l1, l2 = lv_names[i_lv], lv_names[j_lv]
                share_lv_cov = ("lv.covariances" in eq_set) and ((l1, "~~", l2) not in partial_set)
                sl = _alloc_slot(("~~", l1, l2), 0.1, False, share_lv_cov)
                psi_slots[i_lv, j_lv] = sl
                psi_slots[j_lv, i_lv] = sl

        # Variâncias e covariâncias residuais (Theta)
        theta_fixed = np.zeros((p, p), dtype=np.float64)
        theta_slots = np.full((p, p), -1, dtype=np.int64)
        if not ordered:
            for ov_i, ov in enumerate(ov_names):
                share_res = ("residuals" in eq_set) and ((ov, "~~", ov) not in partial_set)
                theta_slots[ov_i, ov_i] = _alloc_slot(("~~", ov, ov), float(s_list[g][ov_i, ov_i]) * 0.5, True, share_res)

        for ov1, ov2 in spec.residual_covs:
            i1, i2 = ov_to_idx[ov1], ov_to_idx[ov2]
            share_rc = ("residuals" in eq_set) and ((ov1, "~~", ov2) not in partial_set) and ((ov2, "~~", ov1) not in partial_set)
            sl = _alloc_slot(("~~", ov1, ov2), 0.05, False, share_rc)
            theta_slots[i1, i2] = sl
            theta_slots[i2, i1] = sl

        # Interceptos (contínuo) ou Limiares + Escalas (ordinal)
        nu_slots = np.full(p, -1, dtype=np.int64)
        alpha_slots = np.full(m, -1, dtype=np.int64)
        tau_slots_per_item: list[list[int]] = []
        scale_slots = np.full(p, -1, dtype=np.int64)

        if ordered:
            for ov_i, ov in enumerate(ov_names):
                taus_g = poly_list[g].thresholds_list[ov_i]
                item_t_slots = []
                for k_t, t_val in enumerate(taus_g):
                    share_tau = ("thresholds" in eq_set) and ((ov, "|", f"t{k_t + 1}") not in partial_set)
                    item_t_slots.append(_alloc_slot(("|", ov, k_t), float(t_val), False, share_tau))
                tau_slots_per_item.append(item_t_slots)

                # Na parametrização delta: escalas e interceptos latentes dos itens livres em g > 0 quando thresholds são igualados
                if g > 0 and ("thresholds" in eq_set):
                    if "residuals" not in eq_set:
                        scale_slots[ov_i] = _alloc_slot(("scale", ov, g), 1.0, True, False)
                    if ("intercepts" not in eq_set) and ((ov, "~1", "1") not in partial_set):
                        nu_slots[ov_i] = _alloc_slot(("nu_ord", ov, g), 0.0, False, False)

            if g > 0 and ("thresholds" in eq_set) and ("intercepts" in eq_set):
                for lv_i, lv in enumerate(lv_names):
                    if ("means" not in eq_set) and ((lv, "~1", "1") not in partial_set):
                        alpha_slots[lv_i] = _alloc_slot(("alpha", lv, g), 0.0, False, False)
        else:
            for ov_i, ov in enumerate(ov_names):
                share_nu = ("intercepts" in eq_set) and ((ov, "~1", "1") not in partial_set)
                nu_slots[ov_i] = _alloc_slot(("~1", ov), float(mean_list[g][ov_i]), False, share_nu)

            if g > 0 and ("intercepts" in eq_set):
                for lv_i, lv in enumerate(lv_names):
                    if ("means" not in eq_set) and ((lv, "~1", "1") not in partial_set):
                        alpha_slots[lv_i] = _alloc_slot(("alpha", lv, g), 0.0, False, False)

        group_param_maps.append(
            {
                "lam_fixed": lam_fixed,
                "lam_slots": lam_slots,
                "psi_fixed": psi_fixed,
                "psi_slots": psi_slots,
                "theta_fixed": theta_fixed,
                "theta_slots": theta_slots,
                "nu_slots": nu_slots,
                "alpha_slots": alpha_slots,
                "tau_slots": tau_slots_per_item,
                "scale_slots": scale_slots,
            }
        )

    n_free = len(init_vals)
    theta_init = np.array(init_vals, dtype=np.float64)
    is_log_arr = np.array(is_log_var, dtype=bool)

    s_torch_list = [torch.tensor(s_g, dtype=torch.float64) for s_g in s_list]
    logdet_s_list = [float(torch.linalg.slogdet(st)[1].numpy()) for st in s_torch_list]
    mean_torch_list = [torch.tensor(m_g, dtype=torch.float64) for m_g in mean_list]

    if ordered:
        vech_r, vech_c = _vech_indices(p, strict=True)
        p_star = len(vech_r)
        obs_vec_ord_list = []
        w_diag_ord_list = []
        for g in range(n_groups):
            pr = poly_list[g]
            tau_obs = np.concatenate(pr.thresholds_list)
            r_obs = s_list[g][vech_r, vech_c]
            obs_vec_ord_list.append(torch.tensor(np.concatenate([tau_obs, r_obs]), dtype=torch.float64))
            w_d = 1.0 / np.maximum(1e-12, np.diag(pr.gamma))
            w_diag_ord_list.append(torch.tensor(w_d, dtype=torch.float64))

    def _raw_to_natural(raw_t: torch.Tensor) -> torch.Tensor:
        nat = [torch.exp(raw_t[i]) if is_log_var[i] else raw_t[i] for i in range(n_free)]
        return torch.stack(nat) if n_free > 0 else torch.zeros(0, dtype=torch.float64)

    def _unpack_group(nat_t: torch.Tensor, g: int):
        gm = group_param_maps[g]
        lam = torch.tensor(gm["lam_fixed"], dtype=torch.float64)
        for i in range(p):
            for j in range(m):
                sl = gm["lam_slots"][i, j]
                if sl >= 0:
                    lam[i, j] = nat_t[sl]

        psi = torch.tensor(gm["psi_fixed"], dtype=torch.float64)
        for i in range(m):
            for j in range(m):
                sl = gm["psi_slots"][i, j]
                if sl >= 0:
                    psi[i, j] = nat_t[sl]

        alpha = torch.zeros(m, dtype=torch.float64)
        for j in range(m):
            sl = gm["alpha_slots"][j]
            if sl >= 0:
                alpha[j] = nat_t[sl]

        if ordered:
            scales = torch.ones(p, dtype=torch.float64)
            for i in range(p):
                sl = gm["scale_slots"][i]
                if sl >= 0:
                    scales[i] = nat_t[sl]

            # Se residuals forem iguais entre grupos em WLSMV, escala dos grupos >=1 vem de Theta_0
            if ("residuals" in eq_set) and g > 0:
                lam0, psi0, _, _, _ = _unpack_group(nat_t, 0)
                theta0_diag = 1.0 - torch.sum((lam0 @ psi0) * lam0, dim=1)
                comm_g = torch.sum((lam @ psi) * lam, dim=1)
                tot_var_g = torch.clamp(comm_g + theta0_diag, min=1e-6)
                scales = 1.0 / torch.sqrt(tot_var_g)

            comm = torch.sum((lam @ psi) * lam, dim=1)
            theta_diag = (1.0 / (scales * scales)) - comm
            theta = torch.diag(theta_diag)
            for i in range(p):
                for j in range(p):
                    if i != j and gm["theta_slots"][i, j] >= 0:
                        theta[i, j] = nat_t[gm["theta_slots"][i, j]]

            sigma_star = lam @ psi @ lam.T + theta
            d_mat_sc = torch.diag(scales)
            p_corr = d_mat_sc @ sigma_star @ d_mat_sc

            # Limiares implicados: d_i * (tau_ik - (Lambda * alpha)_i)
            nu = torch.zeros(p, dtype=torch.float64)
            for i in range(p):
                sl = gm["nu_slots"][i]
                if sl >= 0:
                    nu[i] = nat_t[sl]
            shift_ov = nu + (lam @ alpha)
            tau_imp_list = []
            for i in range(p):
                t_raw = torch.stack([nat_t[sl] for sl in gm["tau_slots"][i]])
                tau_imp_list.append(scales[i] * (t_raw - shift_ov[i]))
            tau_imp = torch.cat(tau_imp_list)
            return lam, psi, theta, p_corr, tau_imp

        theta = torch.tensor(gm["theta_fixed"], dtype=torch.float64)
        for i in range(p):
            for j in range(p):
                sl = gm["theta_slots"][i, j]
                if sl >= 0:
                    theta[i, j] = nat_t[sl]

        nu = torch.zeros(p, dtype=torch.float64)
        for i in range(p):
            sl = gm["nu_slots"][i]
            if sl >= 0:
                nu[i] = nat_t[sl]

        sigma = lam @ psi @ lam.T + theta
        mu = nu + lam @ alpha
        return lam, psi, theta, sigma, mu

    def objective_mg(raw_t: torch.Tensor) -> torch.Tensor:
        nat_t = _raw_to_natural(raw_t)
        total_loss = torch.tensor(0.0, dtype=torch.float64)
        for g in range(n_groups):
            if ordered:
                _, _, _, p_corr, tau_imp = _unpack_group(nat_t, g)
                imp_vec = torch.cat([tau_imp, p_corr[vech_r, vech_c]])
                diff_v = obs_vec_ord_list[g] - imp_vec
                w_g = float(n_obs_list[g] - 1) / float(n_total - n_groups)
                total_loss = total_loss + w_g * 0.5 * torch.sum(w_diag_ord_list[g] * diff_v * diff_v)
            else:
                _, _, _, sigma, mu = _unpack_group(nat_t, g)
                sign, logdet = torch.linalg.slogdet(sigma)
                if sign <= 0:
                    return torch.tensor(1e10, dtype=torch.float64)
                tr_term = torch.trace(torch.linalg.solve(sigma, s_torch_list[g]))
                mdiff = mean_torch_list[g] - mu
                mean_term = torch.dot(mdiff, torch.linalg.solve(sigma, mdiff))
                f_g = 0.5 * (logdet + tr_term - logdet_s_list[g] - float(p) + mean_term)
                total_loss = total_loss + (float(n_obs_list[g]) / float(n_total)) * f_g
        return total_loss

    def obj_and_grad(th_np: np.ndarray):
        tv = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        loss = objective_mg(tv)
        val = float(loss.detach().numpy())
        grad = torch.autograd.grad(loss, tv)[0]
        return val, grad.detach().numpy().copy()

    def obj_grad_hess(th_np: np.ndarray):
        tv = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        loss = objective_mg(tv)
        val = float(loss.detach().numpy())
        grad = torch.autograd.grad(loss, tv, create_graph=True)[0]
        g_np = grad.detach().numpy().copy()
        h_rows = [torch.autograd.grad(grad[i], tv, retain_graph=True)[0].detach().numpy() for i in range(len(th_np))]
        return val, g_np, np.vstack(h_rows)

    res_bfgs = optimize.minimize(
        obj_and_grad,
        theta_init,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-6, "maxiter": 600},
    )
    raw_hat = res_bfgs.x.copy()
    f_curr, g_curr, h_curr = obj_grad_hess(raw_hat)
    converged = False
    for _ in range(50):
        if np.max(np.abs(g_curr)) < 1e-12:
            converged = True
            break
        try:
            step = -np.linalg.solve(h_curr, g_curr)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(h_curr, g_curr, rcond=None)[0]
        alpha_step = 1.0
        accepted = False
        while alpha_step >= 1e-10:
            cand = raw_hat + alpha_step * step
            fc, gc, hc = obj_grad_hess(cand)
            if np.isfinite(fc) and fc <= f_curr + 1e-12:
                raw_hat, f_curr, g_curr, h_curr = cand, fc, gc, hc
                accepted = True
                break
            alpha_step *= 0.5
        if not accepted:
            break
    if np.max(np.abs(g_curr)) < 1e-8:
        converged = True

    with torch.no_grad():
        nat_t_hat = _raw_to_natural(torch.tensor(raw_hat, dtype=torch.float64))
        nat_hat = nat_t_hat.numpy()
        sigma_per_group = []
        mu_per_group = []
        for g in range(n_groups):
            _, _, _, sig_g, fifth_g = _unpack_group(nat_t_hat, g)
            sigma_per_group.append(sig_g.numpy())
            mu_per_group.append(np.zeros(p, dtype=np.float64) if ordered else fifth_g.numpy())

    # VCOV e Índices de Ajuste Multigrupo
    group_chisq: dict[str, float] = {}
    if ordered:
        n_wls_tot = float(n_total - n_groups)
        jac_list, w_list, gamma_list = [], [], []
        delta_r_list, w_rr_block_list, v_ml_block_list, gamma_rr_block_list = [], [], [], []
        t_model_total = 0.0
        t_base_total = 0.0
        t_b_catml_tot = 0.0

        full_r, full_c = _vech_indices(p, strict=False)
        d_full = _duplication_matrix(p, full_r, full_c)
        off_idx = np.array([idx_i for idx_i, (r, c) in enumerate(zip(full_r, full_c)) if r != c], dtype=np.int64)

        for g in range(n_groups):
            def imp_g_nat(nt: torch.Tensor, g_idx=g) -> torch.Tensor:
                _, _, _, pc, ti = _unpack_group(nt, g_idx)
                return torch.cat([ti, pc[vech_r, vech_c]])

            w_group_share = float(n_obs_list[g] - 1) / n_wls_tot
            fg = float(n_obs_list[g]) / float(n_total)
            jg = torch.autograd.functional.jacobian(imp_g_nat, torch.tensor(nat_hat, dtype=torch.float64)).numpy()
            jac_list.append(jg)
            w_g_mat = np.diag(w_diag_ord_list[g].numpy())
            w_list.append(w_group_share * w_g_mat)
            gam_g = poly_list[g].gamma
            gamma_list.append((1.0 / w_group_share) * gam_g)

            imp_np = imp_g_nat(nat_t_hat).numpy()
            obs_np = obs_vec_ord_list[g].numpy()
            t_g = float(n_obs_list[g] - 1) * float(np.sum(np.diag(w_g_mat) * ((obs_np - imp_np) ** 2)))
            group_chisq[group_labels[g]] = t_g
            t_model_total += t_g

            # Modelo nulo (thresholds livres, correlações = 0)
            n_tau_g = sum(len(t) for t in poly_list[g].thresholds_list)
            r_obs_g = s_list[g][vech_r, vech_c]
            w_rr_g = np.diag(w_g_mat)[n_tau_g:]
            t_base_total += float(n_obs_list[g] - 1) * float(np.sum(w_rr_g * (r_obs_g ** 2)))

            logdet_r = logdet_s_list[g]
            t_b_catml_tot += float(n_obs_list[g]) * (-logdet_r)

            sig_inv = np.linalg.inv(sigma_per_group[g])
            v_ml_g = (0.5 * (d_full.T @ np.kron(sig_inv, sig_inv) @ d_full))[np.ix_(off_idx, off_idx)]
            delta_r_list.append(jg[n_tau_g:, :])
            w_rr_block_list.append(fg * np.diag(w_rr_g))
            v_ml_block_list.append(fg * v_ml_g)
            gamma_rr_block_list.append((1.0 / fg) * gam_g[n_tau_g:, n_tau_g:])

        jac_all = np.vstack(jac_list)
        w_all = np.block([[w_list[i] if i == j else np.zeros((w_list[i].shape[0], w_list[j].shape[1])) for j in range(n_groups)] for i in range(n_groups)])
        gamma_all = np.block([[gamma_list[i] if i == j else np.zeros((gamma_list[i].shape[0], gamma_list[j].shape[1])) for j in range(n_groups)] for i in range(n_groups)])

        dtwd = jac_all.T @ w_all @ jac_all
        dtwd_inv = np.linalg.pinv(dtwd)
        vcov_nat = (dtwd_inv @ (jac_all.T @ w_all @ gamma_all @ w_all @ jac_all) @ dtwd_inv) / n_wls_tot

        tot_tau_obs = sum(sum(len(t) for t in poly_list[g].thresholds_list) for g in range(n_groups))
        df_model = (tot_tau_obs + n_groups * p_star) - n_free
        df_base = n_groups * p_star

        u_mat = w_all - w_all @ jac_all @ dtwd_inv @ jac_all.T @ w_all
        m_mat = u_mat @ gamma_all
        a_scale = np.sqrt(float(df_model) / max(1e-12, float(np.trace(m_mat @ m_mat))))
        c_model = 1.0 / a_scale
        b_model = float(df_model) - a_scale * float(np.trace(m_mat))

        r_rows_idx = []
        offset_r = 0
        for g in range(n_groups):
            nt_g = sum(len(t) for t in poly_list[g].thresholds_list)
            r_rows_idx.extend(range(offset_r + nt_g, offset_r + nt_g + p_star))
            offset_r += nt_g + p_star
        r_idx_arr = np.array(r_rows_idx, dtype=np.int64)
        m_base = w_all[np.ix_(r_idx_arr, r_idx_arr)] @ gamma_all[np.ix_(r_idx_arr, r_idx_arr)]
        a_base = np.sqrt(float(df_base) / max(1e-12, float(np.trace(m_base @ m_base))))
        c_base = 1.0 / a_base
        b_base = float(df_base) - a_base * float(np.trace(m_base))

        # Exact lav_fit_catml_dwls (lavaan 0.7-2)
        corr_slots = sorted({
            int(i)
            for g_m in group_param_maps
            for key_k in ("lam_slots", "psi_slots", "theta_slots", "scale_slots")
            for i in g_m[key_k].flatten()
            if i >= 0
        })
        df3_catml = n_groups * p_star - len(corr_slots)

        def _catml_loss(x_sub: np.ndarray) -> float:
            raw_c = raw_hat.copy()
            for idx_s, sl_i in enumerate(corr_slots):
                raw_c[sl_i] = x_sub[idx_s]
            with torch.no_grad():
                nt_c = _raw_to_natural(torch.tensor(raw_c, dtype=torch.float64))
                tot_f = 0.0
                for g in range(n_groups):
                    _, _, _, pc_g, _ = _unpack_group(nt_c, g)
                    pc_np = pc_g.numpy()
                    sgn, ldet_p = np.linalg.slogdet(pc_np)
                    if sgn <= 0:
                        return 1e10
                    tot_f += float(n_obs_list[g]) * float(ldet_p + np.trace(np.linalg.solve(pc_np, s_list[g])) - logdet_s_list[g] - float(p))
            return tot_f

        t_m_catml_tot = _catml_loss(raw_hat[corr_slots])
        delta_all_r = np.vstack(delta_r_list)
        w_dwls_all_r = np.block([[w_rr_block_list[i] if i == j else np.zeros((p_star, p_star)) for j in range(n_groups)] for i in range(n_groups)])
        v_all_r = np.block([[v_ml_block_list[i] if i == j else np.zeros((p_star, p_star)) for j in range(n_groups)] for i in range(n_groups)])
        gamma_all_r = np.block([[gamma_rr_block_list[i] if i == j else np.zeros((p_star, p_star)) for j in range(n_groups)] for i in range(n_groups)])

        wi_u_all = np.eye(n_groups * p_star) - delta_all_r @ dtwd_inv @ delta_all_r.T @ w_dwls_all_r
        ks_catml = float(np.trace(wi_u_all.T @ v_all_r @ wi_u_all @ gamma_all_r))
        c_hat3_m = ks_catml / float(df3_catml)
        c_hat3_b = float(np.trace(gamma_all_r)) / float(df_base)
    else:
        full_r, full_c = _vech_indices(p, strict=False)
        d_mat = _duplication_matrix(p, full_r, full_c)
        p_cov_moments = len(full_r)
        p_tot_moments = p + p_cov_moments

        t_model_total = 0.0
        t_base_total = 0.0
        info_total = np.zeros((n_free, n_free), dtype=np.float64)
        meat_total = np.zeros((n_free, n_free), dtype=np.float64)
        tr_h1_total = 0.0
        tr_b0_total = 0.0

        for g in range(n_groups):
            def mom_g_nat(nt: torch.Tensor, g_idx=g) -> torch.Tensor:
                _, _, _, sig_g, mu_g = _unpack_group(nt, g_idx)
                return torch.cat([mu_g, sig_g[full_r, full_c]])

            jg = torch.autograd.functional.jacobian(mom_g_nat, torch.tensor(nat_hat, dtype=torch.float64)).numpy()
            sig_g_np = sigma_per_group[g]
            sig_inv = np.linalg.inv(sig_g_np)
            w_cov = 0.5 * (d_mat.T @ np.kron(sig_inv, sig_inv) @ d_mat)
            w_g = np.block([[sig_inv, np.zeros((p, p_cov_moments))], [np.zeros((p_cov_moments, p)), w_cov]])

            info_total += float(n_obs_list[g]) * (jg.T @ w_g @ jg)

            with torch.no_grad():
                _, _, _, _, mu_g_t = _unpack_group(nat_t_hat, g)
                mu_g_np = mu_g_t.numpy()
            mdiff = mean_list[g] - mu_g_np
            t_g = float(n_obs_list[g]) * float(
                np.linalg.slogdet(sig_g_np)[1] + np.trace(np.linalg.solve(sig_g_np, s_list[g])) - logdet_s_list[g] - float(p) + mdiff @ np.linalg.solve(sig_g_np, mdiff)
            )
            group_chisq[group_labels[g]] = t_g
            t_model_total += t_g

            s_diag = np.diag(np.diag(s_list[g]))
            _, logdet_d = np.linalg.slogdet(s_diag)
            t_base_total += float(n_obs_list[g]) * float(logdet_d - logdet_s_list[g])

            if est_upper == "MLR":
                yc = y_centered_list[g]
                s_vech = s_list[g][full_r, full_c]
                zi_cov = np.einsum("ni,nj->nij", yc, yc)[:, full_r, full_c] - s_vech[None, :]
                zi = np.hstack([yc, zi_cov])
                gamma_g = (zi.T @ zi) / float(n_obs_list[g])

                s_inv = np.linalg.inv(s_list[g])
                w_h1_cov = 0.5 * (d_mat.T @ np.kron(s_inv, s_inv) @ d_mat)
                w_h1_g = np.block([[s_inv, np.zeros((p, p_cov_moments))], [np.zeros((p_cov_moments, p)), w_h1_cov]])
                tr_h1_total += float(np.trace(w_h1_g @ gamma_g))

                # Escoros individuais avaliados em (mu_hat, Sigma_hat)
                sig_vech = sig_g_np[full_r, full_c]
                y_raw_c = yc + (mean_list[g] - mu_g_np)[None, :]
                zi_sig_cov = np.einsum("ni,nj->nij", y_raw_c, y_raw_c)[:, full_r, full_c] - sig_vech[None, :]
                zi_sig = np.hstack([y_raw_c, zi_sig_cov])
                sc_i = zi_sig @ (w_g @ jg)
                meat_total += (sc_i.T @ sc_i)

                # Modelo nulo MLR por grupo (médias e variâncias livres)
                s_d_inv = np.diag(1.0 / np.diag(s_list[g]))
                w_b_cov = 0.5 * (d_mat.T @ np.kron(s_d_inv, s_d_inv) @ d_mat)
                w_b_g = np.block([[s_d_inv, np.zeros((p, p_cov_moments))], [np.zeros((p_cov_moments, p)), w_b_cov]])
                base_idx = np.array(
                    list(range(p)) + [p + idx_i for idx_i, (r, c) in enumerate(zip(full_r, full_c)) if r == c],
                    dtype=np.int64,
                )
                j_b = np.eye(p_tot_moments)[:, base_idx]
                tr_b0_total += float(np.trace(w_b_g @ j_b @ np.linalg.inv(j_b.T @ w_b_g @ j_b) @ j_b.T @ w_b_g @ gamma_g))

        df_model = n_groups * p_tot_moments - n_free
        df_base = n_groups * (p * (p - 1) // 2)

        if est_upper == "MLR":
            d_raw_d_nat = np.diag(np.where(is_log_arr, 1.0 / np.maximum(1e-12, nat_hat), 1.0))
            h_obs_tot = float(n_total) * (d_raw_d_nat @ h_curr @ d_raw_d_nat)
            h_obs_inv = np.linalg.pinv(h_obs_tot)
            vcov_nat = h_obs_inv @ meat_total @ h_obs_inv
            tr_h0_total = float(np.trace(h_obs_inv @ meat_total))
            c_model = (tr_h1_total - tr_h0_total) / float(df_model)
            c_base = (tr_h1_total - tr_b0_total) / float(df_base)
        else:
            vcov_nat = np.linalg.pinv(info_total)
            c_model, c_base = 1.0, 1.0

    se_nat = np.sqrt(np.maximum(0.0, np.diag(vcov_nat)))

    # Monta tabela completa de parâmetros por grupo
    pe_rows = []
    for g in range(n_groups):
        gm = group_param_maps[g]
        g_lbl = group_labels[g]
        with torch.no_grad():
            lam_g, psi_g, theta_g, sig_g, _ = _unpack_group(nat_t_hat, g)
            lam_np = lam_g.numpy()
            psi_np = psi_g.numpy()
            theta_np = theta_g.numpy()
            sig_np = sig_g.numpy()

        sd_lv = np.sqrt(np.maximum(1e-12, np.diag(psi_np)))
        sd_ov = np.sqrt(np.maximum(1e-12, np.diag(sig_np)))

        def _add_pe(lhs_s, op_s, rhs_s, est_v, sl_idx, std_lv_v, std_all_v):
            se_v = float(se_nat[sl_idx]) if sl_idx >= 0 else 0.0
            z_v = est_v / se_v if se_v > 1e-12 else np.nan
            p_v = float(2.0 * (1.0 - stats.norm.cdf(abs(z_v)))) if np.isfinite(z_v) else np.nan
            pe_rows.append(
                {
                    "group": g_lbl,
                    "lhs": lhs_s,
                    "op": op_s,
                    "rhs": rhs_s,
                    "est": float(est_v),
                    "se": se_v,
                    "z": z_v,
                    "pvalue": p_v,
                    "std_lv": float(std_lv_v),
                    "std_all": float(std_all_v),
                }
            )

        for lv_i, lv in enumerate(lv_names):
            for ov in spec.loadings[lv]:
                ov_i = ov_to_idx[ov]
                val = lam_np[ov_i, lv_i]
                _add_pe(lv, "=~", ov, val, int(gm["lam_slots"][ov_i, lv_i]), val * sd_lv[lv_i], val * sd_lv[lv_i] / sd_ov[ov_i])

        for lv_i, lv in enumerate(lv_names):
            val = psi_np[lv_i, lv_i]
            _add_pe(lv, "~~", lv, val, int(gm["psi_slots"][lv_i, lv_i]), 1.0, 1.0)

        for i_lv in range(m):
            for j_lv in range(i_lv + 1, m):
                val = psi_np[i_lv, j_lv]
                cor_v = val / (sd_lv[i_lv] * sd_lv[j_lv])
                _add_pe(lv_names[i_lv], "~~", lv_names[j_lv], val, int(gm["psi_slots"][i_lv, j_lv]), cor_v, cor_v)

        for ov_i, ov in enumerate(ov_names):
            val = theta_np[ov_i, ov_i]
            _add_pe(ov, "~~", ov, val, int(gm["theta_slots"][ov_i, ov_i]), val, val / (sd_ov[ov_i] ** 2))

        if ordered:
            for ov_i, ov in enumerate(ov_names):
                for k_t, sl in enumerate(gm["tau_slots"][ov_i]):
                    val = float(nat_hat[sl])
                    _add_pe(ov, "|", f"t{k_t + 1}", val, int(sl), val, val)
            for ov_i, ov in enumerate(ov_names):
                sl = int(gm["nu_slots"][ov_i])
                val = float(nat_hat[sl]) if sl >= 0 else 0.0
                _add_pe(ov, "~1", "1", val, sl, val, val)
        
        else:
            for ov_i, ov in enumerate(ov_names):
                sl = int(gm["nu_slots"][ov_i])
                val = float(nat_hat[sl]) if sl >= 0 else 0.0
                _add_pe(ov, "~1", "1", val, sl, val, val / sd_ov[ov_i])

        for lv_i, lv in enumerate(lv_names):
            sl = int(gm["alpha_slots"][lv_i])
            val = float(nat_hat[sl]) if sl >= 0 else 0.0
            _add_pe(lv, "~1", "1", val, sl, val / sd_lv[lv_i], val / sd_lv[lv_i])

    pe_df = pd.DataFrame(pe_rows)

    # Tabela de ajuste global multigrupo
    def _build_mg_fit_table() -> pd.DataFrame:
        n_rm = float(n_total - n_groups) if ordered else float(n_total)
        t_m_sc = (t_model_total / c_model + b_model) if ordered else (t_model_total / c_model)
        t_b_sc = (t_base_total / c_base + b_base) if ordered else (t_base_total / c_base)

        def _cfi(tm, dfm, tb, dfb):
            num = max(0.0, tm - dfm)
            den = max(num, tb - dfb)
            return 1.0 - (num / den) if den > 1e-12 else 1.0

        def _tli(tm, dfm, tb, dfb):
            rb, rm = tb / dfb, tm / dfm
            return (rb - rm) / (rb - 1.0) if abs(rb - 1.0) > 1e-12 else 1.0

        def _rmsea(tm, dfm):
            # Em multigrupo, RMSEA padrão do lavaan multiplica por sqrt(G)
            return float(np.sqrt(float(n_groups) * max(0.0, (tm - dfm) / (dfm * n_rm))))

        # SRMR multigrupo (média ponderada por N_g dos SRMRs de cada grupo, incluindo resíduos de média se contínuo)        
        srmr_sum = 0.0
        tril_r, tril_c = np.tril_indices(p, k=0)
        for g in range(n_groups):
            sd_s = np.sqrt(np.maximum(1e-12, np.diag(s_list[g])))
            r_o = s_list[g] / np.outer(sd_s, sd_s)
            r_i = sigma_per_group[g] / np.outer(sd_s, sd_s)
            cov_res_sq = (r_o[tril_r, tril_c] - r_i[tril_r, tril_c]) ** 2
            if not ordered:
                mean_res_sq = ((mean_list[g] - mu_per_group[g]) / sd_s) ** 2
                all_res_sq = np.concatenate([cov_res_sq, mean_res_sq])
            else:
                all_res_sq = cov_res_sq
            srmr_g = float(np.sqrt(np.mean(all_res_sq)))
            srmr_sum += (float(n_obs_list[g]) / float(n_total)) * srmr_g

        cfi_un, tli_un, rmsea_un = _cfi(t_model_total, df_model, t_base_total, df_base), _tli(t_model_total, df_model, t_base_total, df_base), _rmsea(t_model_total, df_model)
        cfi_sc, tli_sc, rmsea_sc = _cfi(t_m_sc, df_model, t_b_sc, df_base), _tli(t_m_sc, df_model, t_b_sc, df_base), _rmsea(t_m_sc, df_model)

        if ordered:
            num_r = max(0.0, t_m_catml_tot - c_hat3_m * df_model)
            den_r = max(num_r, t_b_catml_tot - c_hat3_b * df_base)
            cfi_rob = 1.0 - (num_r / den_r) if den_r > 1e-12 else 1.0
            rb_r = t_b_catml_tot / (c_hat3_b * df_base)
            rm_r = t_m_catml_tot / (c_hat3_m * df_model)
            tli_rob = (rb_r - rm_r) / (rb_r - 1.0) if abs(rb_r - 1.0) > 1e-12 else 1.0
            rmsea_rob = float(np.sqrt(float(n_groups) * max(0.0, (t_m_catml_tot - c_hat3_m * df3_catml) / (df3_catml * float(n_total)))))
        else:
            num_r = max(0.0, c_model * (t_m_sc - df_model))
            den_r = max(num_r, c_base * (t_b_sc - df_base))
            cfi_rob = 1.0 - (num_r / den_r) if den_r > 1e-12 else 1.0
            tli_rob = tli_sc
            rmsea_rob = float(np.sqrt(float(n_groups) * max(0.0, c_model * (t_m_sc - df_model) / (df_model * n_rm))))

        pval_un = float(1.0 - stats.chi2.cdf(t_model_total, df_model))
        pval_sc = float(1.0 - stats.chi2.cdf(t_m_sc, df_model))
        rows_ft = {
            "chisq": [t_model_total, t_m_sc, np.nan],
            "df": [float(df_model), float(df_model), np.nan],
            "pvalue": [pval_un, pval_sc, np.nan],
            "scaling_factor": [1.0, c_model, np.nan],
            "cfi": [cfi_un, cfi_sc, cfi_rob],
            "tli": [tli_un, tli_sc, tli_rob],
            "rmsea": [rmsea_un, rmsea_sc, rmsea_rob],
            "srmr": [srmr_sum, srmr_sum, srmr_sum],
        }
        return pd.DataFrame.from_dict(rows_ft, orient="index", columns=["Unscaled", "Scaled", "Robust"])

    fit_table = _build_mg_fit_table()

    # Contexto exato para score_test() (paridade com lavaan::lavTestScore)
    n_raw = len(raw_hat)
    slot_active_groups: dict[int, list[int]] = {s: [] for s in range(n_raw)}
    for g in range(n_groups):
        gm = group_param_maps[g]
        active_s: set[int] = set()
        for k_mat in ("lam_slots", "psi_slots", "theta_slots", "nu_slots", "alpha_slots", "tau_slots", "scale_slots"):
            if k_mat in gm:
                vals_iter = [x for sub in gm[k_mat] for x in sub] if isinstance(gm[k_mat], list) else gm[k_mat].flatten()
                for val_s in vals_iter:
                    if int(val_s) >= 0:
                        active_s.add(int(val_s))
        for s_idx in sorted(active_s):
            slot_active_groups[s_idx].append(g)

    exp_col_map: dict[tuple[int, int], int] = {}
    q_exp = 0
    for g in range(n_groups):
        for s_idx in range(n_raw):
            if g in slot_active_groups[s_idx]:
                exp_col_map[(g, s_idx)] = q_exp
                q_exp += 1

    score_exp = np.zeros(q_exp, dtype=np.float64)
    info_exp = np.zeros((q_exp, q_exp), dtype=np.float64)
    meat_exp = np.zeros((q_exp, q_exp), dtype=np.float64)

    if ordered:
        n_eff_sc = float(n_total - n_groups)
        for g in range(n_groups):
            fg = float(n_obs_list[g]) / max(1.0, n_eff_sc)
            tv_g = torch.tensor(raw_hat, dtype=torch.float64, requires_grad=True)
            nat_g = _raw_to_natural(tv_g)
            _, _, _, p_corr, tau_imp = _unpack_group(nat_g, g)
            imp_v = torch.cat([tau_imp, p_corr[vech_r, vech_c]])
            diff_v = obs_vec_ord_list[g] - imp_v
            loss_g = 0.5 * fg * torch.sum(w_diag_ord_list[g] * diff_v * diff_v)
            grad_g = torch.autograd.grad(loss_g, tv_g)[0].numpy()

            def _imp_g_raw(rt: torch.Tensor, g_i=g) -> torch.Tensor:
                nt = _raw_to_natural(rt)
                _, _, _, pc, ti = _unpack_group(nt, g_i)
                return torch.cat([ti, pc[vech_r, vech_c]])

            jg_raw = torch.autograd.functional.jacobian(_imp_g_raw, torch.tensor(raw_hat, dtype=torch.float64)).numpy()
            w_g_np = w_list[g]
            gam_g_np = gamma_list[g]
            inf_g = fg * (jg_raw.T @ w_g_np @ jg_raw)
            mt_g = fg * (jg_raw.T @ w_g_np @ gam_g_np @ w_g_np @ jg_raw)

            cols_g = [(s, exp_col_map[(g, s)]) for s in range(n_raw) if (g, s) in exp_col_map]
            s_idx_g = [c[0] for c in cols_g]
            e_idx_g = [c[1] for c in cols_g]
            score_exp[e_idx_g] = -grad_g[s_idx_g]
            info_exp[np.ix_(e_idx_g, e_idx_g)] = inf_g[np.ix_(s_idx_g, s_idx_g)]
            meat_exp[np.ix_(e_idx_g, e_idx_g)] = mt_g[np.ix_(s_idx_g, s_idx_g)]
    else:
        n_eff_sc = float(n_total)
        full_r_sc, full_c_sc = _vech_indices(p, strict=False)
        d_mat_sc = _duplication_matrix(p, full_r_sc, full_c_sc)
        p_cov_sc = len(full_r_sc)

        for g in range(n_groups):
            fg = float(n_obs_list[g]) / max(1.0, n_eff_sc)
            tv_g = torch.tensor(raw_hat, dtype=torch.float64, requires_grad=True)
            nat_g = _raw_to_natural(tv_g)
            _, _, _, sig_g, mu_g = _unpack_group(nat_g, g)
            tr_t = torch.trace(torch.linalg.solve(sig_g, s_torch_list[g]))
            md = mean_torch_list[g] - mu_g
            loss_g = 0.5 * fg * (torch.linalg.slogdet(sig_g)[1] + tr_t - logdet_s_list[g] - float(p) + torch.dot(md, torch.linalg.solve(sig_g, md)))
            grad_g = torch.autograd.grad(loss_g, tv_g)[0].numpy()

            def _mom_g_raw(rt: torch.Tensor, g_i=g) -> torch.Tensor:
                nt = _raw_to_natural(rt)
                _, _, _, sg, mg = _unpack_group(nt, g_i)
                return torch.cat([mg, sg[full_r_sc, full_c_sc]])

            jg_raw = torch.autograd.functional.jacobian(_mom_g_raw, torch.tensor(raw_hat, dtype=torch.float64)).numpy()
            sig_g_np = sigma_per_group[g]
            mu_g_np = mu_per_group[g]
            sig_inv_np = np.linalg.inv(sig_g_np)
            w_cov_np = 0.5 * (d_mat_sc.T @ np.kron(sig_inv_np, sig_inv_np) @ d_mat_sc)
            w_g_np = np.block([
                [sig_inv_np, np.zeros((p, p_cov_sc))],
                [np.zeros((p_cov_sc, p)), w_cov_np],
            ])
            inf_g = fg * (jg_raw.T @ w_g_np @ jg_raw)

            y_g_raw = y_centered_list[g] + mean_list[g][None, :]
            diff_y = y_g_raw - mu_g_np[None, :]
            outer_y = diff_y[:, :, None] * diff_y[:, None, :] - sig_g_np[None, :, :]
            vech_y = outer_y[:, full_r_sc, full_c_sc]
            d_case = np.hstack([diff_y, vech_y])
            g_case = (d_case @ w_g_np) @ jg_raw
            mt_g = (g_case.T @ g_case) / max(1.0, n_eff_sc)

            cols_g = [(s, exp_col_map[(g, s)]) for s in range(n_raw) if (g, s) in exp_col_map]
            s_idx_g = [c[0] for c in cols_g]
            e_idx_g = [c[1] for c in cols_g]
            score_exp[e_idx_g] = -grad_g[s_idx_g]
            info_exp[np.ix_(e_idx_g, e_idx_g)] = inf_g[np.ix_(s_idx_g, s_idx_g)]
            meat_exp[np.ix_(e_idx_g, e_idx_g)] = mt_g[np.ix_(s_idx_g, s_idx_g)]

    r_rows: list[np.ndarray] = []
    constraints_list = []
    for key, shared_idx in shared_slot_map.items():
        act_gs = slot_active_groups.get(shared_idx, [])
        if len(act_gs) < 2:
            continue
        if key[0] == "=~":
            lhs_s, op_s, rhs_s = key[1], "=~", key[2]
        elif key[0] == "~~":
            lhs_s, op_s, rhs_s = key[1], "~~", key[2]
        elif key[0] == "|":
            lhs_s, op_s, rhs_s = key[1], "|", f"t{key[2] + 1}"
        elif key[0] == "~1":
            lhs_s, op_s, rhs_s = key[1], "~1", "1"
        else:
            continue

        c0 = exp_col_map[(act_gs[0], shared_idx)]
        for g_other in act_gs[1:]:
            row_r = np.zeros(q_exp, dtype=np.float64)
            row_r[c0] = 1.0
            row_r[exp_col_map[(g_other, shared_idx)]] = -1.0
            r_rows.append(row_r)

        constraints_list.append({
            "lhs": lhs_s,
            "op": op_s,
            "rhs": rhs_s,
            "n_rows": len(act_gs) - 1,
        })

    r_mat_all = np.vstack(r_rows) if len(r_rows) > 0 else np.zeros((0, q_exp), dtype=np.float64)
    score_ctx = {
        "score": score_exp,
        "info": info_exp,
        "meat": meat_exp,
        "r_mat": r_mat_all,
        "n_eff": n_eff_sc,
        "constraints": constraints_list,
    }

    return MultiGroupFit(
        group_col=group,
        group_labels=group_labels,
        n_obs_per_group=n_obs_list,
        n_obs=n_total,
        lv_names=lv_names,
        ov_names=ov_names,
        estimator=est_upper,
        ordered=ordered,
        group_equal=sorted(eq_set),
        group_partial=sorted(f"{l} {o} {r}" for l, o, r in partial_set),
        pe_df=pe_df,
        fit_table=fit_table,
        group_chisq=group_chisq,
        sigma_per_group=sigma_per_group,
        s_per_group=s_list,
        vcov=vcov_nat,
        _score_context=score_ctx,
        converged=converged,
    )


@dataclass
class InvarianceResult:
    table: pd.DataFrame
    fits: dict[str, MultiGroupFit]
    group_col: str
    group_labels: list[str]
    estimator: str
    ordered: bool

    def summary(self) -> str:
        lines = [
            f"anvala Measurement Invariance ({self.estimator}, groups={self.group_labels})",
            self.table.round(4).to_string(index=False),
        ]
        return "\n".join(lines)


def measurement_invariance(
    model: str,
    data: pd.DataFrame,
    group: str,
    estimator: str = "ML",
    ordered: bool = False,
    levels: tuple[str, ...] = ("configural", "metric", "scalar", "strict"),
    group_partial: list[str] | tuple[str, ...] | str | None = None,
    std_lv: bool = False,
) -> InvarianceResult:
    """
    Executa a sequência hierárquica de invariância de medida multigrupo:
    configural -> metric -> scalar -> strict, com testes LRT escalonados (Satorra-Bentler)
    e variações nos índices de ajuste (delta_cfi, delta_rmsea, delta_srmr).
    """
    is_ord = ordered or (estimator.upper() == "WLSMV")
    loc_key = "thresholds" if is_ord else "intercepts"

    level_map = {
        "configural": [],
        "metric": ["loadings"],
        "scalar": ["loadings", loc_key],
        "strict": ["loadings", loc_key, "residuals"],
    }

    fits_dict: dict[str, MultiGroupFit] = {}
    ordered_names: list[str] = []
    ordered_fits: list[MultiGroupFit] = []

    for lvl in levels:
        lvl_k = lvl.strip().lower()
        if lvl_k not in level_map:
            raise ValueError(f"Nível de invariância '{lvl}' inválido. Use: {list(level_map.keys())}")
        fit_lvl = cfa_multigroup(
            model=model,
            data=data,
            group=group,
            estimator=estimator,
            ordered=is_ord,
            group_equal=level_map[lvl_k],
            group_partial=group_partial,
            std_lv=std_lv,
        )
        fits_dict[lvl_k] = fit_lvl
        ordered_names.append(lvl_k)
        ordered_fits.append(fit_lvl)

    comp_df = compare_models(*ordered_fits, names=ordered_names)
    first_fit = ordered_fits[0]
    return InvarianceResult(
        table=comp_df,
        fits=fits_dict,
        group_col=group,
        group_labels=first_fit.group_labels,
        estimator=first_fit.estimator,
        ordered=first_fit.ordered,
    )