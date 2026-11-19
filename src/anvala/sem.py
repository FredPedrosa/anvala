from __future__ import annotations

import ast
import operator as py_op
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats

from anvala.diagnostics import _duplication_matrix, _vech_indices, modindices, residuals
from anvala.polychoric import PolychoricResult, polychoric


_SAFE_BIN_OPS = {
    ast.Add: py_op.add,
    ast.Sub: py_op.sub,
    ast.Mult: py_op.mul,
    ast.Div: py_op.truediv,
    ast.Pow: py_op.pow,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: py_op.pos,
    ast.USub: py_op.neg,
}


def _eval_defined_ast(node: ast.AST, env: dict[str, torch.Tensor | float]):
    if isinstance(node, ast.Expression):
        return _eval_defined_ast(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Constante não numérica em expressão ':=': {node.value}")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Rótulo '{node.id}' usado em ':=' não foi definido no modelo.")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](_eval_defined_ast(node.left, env), _eval_defined_ast(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_defined_ast(node.operand, env))
    raise ValueError(f"Expressão não suportada em ':=': {ast.dump(node)}")


def _parse_rhs_term(term_str: str) -> tuple[str | None, float | None, bool, str]:
    """
    Analisa um termo do lado direito como 'a*x1', '0.5*x1', 'NA*x1' ou 'x1'.
    Retorna (label, fixed_val, force_free, var_name).
    """
    t = term_str.strip()
    if "*" not in t:
        return None, None, False, t
    mod_part, var_name = t.split("*", 1)
    mod_part = mod_part.strip()
    var_name = var_name.strip()
    if mod_part.upper() == "NA":
        return None, None, True, var_name
    try:
        fixed_v = float(mod_part)
        return None, fixed_v, False, var_name
    except ValueError:
        return mod_part, None, False, var_name


def _parse_sem_syntax(
    syntax: str,
    data_cols: list[str],
    std_lv: bool = False,
    orthogonal: bool = False,
    fixed_x: bool = True,
    s_mat: np.ndarray | None = None,
) -> dict:
    """Compila sintaxe lavaan (=~, ~, ~~, :=) para matrizes RAM (B, Psi)."""
    raw_lines: list[str] = []
    for line in syntax.splitlines():
        line_clean = line.split("#", 1)[0].strip()
        if not line_clean:
            continue
        for sub in line_clean.split(";"):
            sub_c = sub.strip()
            if sub_c:
                raw_lines.append(sub_c)

    explicit_entries: list[dict] = []
    defined_params: list[tuple[str, str]] = []
    lv_order: list[str] = []
    mentioned_vars: list[str] = []

    def _reg_var(v: str):
        if v not in mentioned_vars:
            mentioned_vars.append(v)

    for line in raw_lines:
        if ":=" in line:
            lhs, expr = line.split(":=", 1)
            defined_params.append((lhs.strip(), expr.strip()))
            continue

        if "=~" in line:
            op = "=~"
        elif "~~" in line:
            op = "~~"
        elif "~" in line:
            op = "~"
        else:
            raise ValueError(f"Linha de sintaxe inválida: '{line}'")

        lhs_raw, rhs_raw = line.split(op, 1)
        lhs = lhs_raw.strip()
        if op == "=~" and lhs not in lv_order:
            lv_order.append(lhs)
        _reg_var(lhs)

        terms = [t.strip() for t in rhs_raw.split("+") if t.strip()]
        for term in terms:
            lbl, fixed_v, force_free, rhs_var = _parse_rhs_term(term)
            _reg_var(rhs_var)
            explicit_entries.append(
                {
                    "lhs": lhs,
                    "op": op,
                    "rhs": rhs_var,
                    "label": lbl,
                    "fixed_val": fixed_v,
                    "force_free": force_free,
                }
            )

    ov_names = [c for c in data_cols if c in mentioned_vars and c not in lv_order]
    if not ov_names:
        raise ValueError("Nenhuma variável observada correspondente às colunas dos dados foi encontrada na sintaxe.")

    # Regra de identificação do primeiro indicador para cada LV
    first_ind_done: set[str] = set()
    for ent in explicit_entries:
        if ent["op"] == "=~":
            lv = ent["lhs"]
            if lv not in first_ind_done:
                first_ind_done.add(lv)
                if not std_lv and ent["fixed_val"] is None and not ent["force_free"]:
                    ent["fixed_val"] = 1.0

    # Identifica variáveis com setas entrantes (=~ ou ~) -> endógenas
    has_incoming: set[str] = set()
    for ent in explicit_entries:
        if ent["op"] == "=~":
            has_incoming.add(ent["rhs"])
        elif ent["op"] == "~":
            has_incoming.add(ent["lhs"])

    exo_lvs = [lv for lv in lv_order if lv not in has_incoming]
    exo_ovs = [ov for ov in ov_names if ov not in has_incoming]

    # Adiciona variâncias automáticas para todas as LVs e OVs caso não explícitas
    existing_pairs = set()
    for ent in explicit_entries:
        if ent["op"] == "~~":
            existing_pairs.add((ent["lhs"], ent["rhs"]))
            existing_pairs.add((ent["rhs"], ent["lhs"]))

    ov_idx_map = {ov: i for i, ov in enumerate(ov_names)}

    for lv in lv_order:
        if (lv, lv) not in existing_pairs:
            fixed_var = 1.0 if (std_lv and lv in exo_lvs) else None
            explicit_entries.append(
                {"lhs": lv, "op": "~~", "rhs": lv, "label": None, "fixed_val": fixed_var, "force_free": False}
            )
            existing_pairs.add((lv, lv))

    for ov in ov_names:
        if (ov, ov) not in existing_pairs:
            if fixed_x and ov in exo_ovs and s_mat is not None:
                i_ov = ov_idx_map[ov]
                fixed_v = float(s_mat[i_ov, i_ov])
            else:
                fixed_v = None
            explicit_entries.append(
                {"lhs": ov, "op": "~~", "rhs": ov, "label": None, "fixed_val": fixed_v, "force_free": False}
            )
            existing_pairs.add((ov, ov))

    # Covariâncias automáticas entre LVs exógenas (se orthogonal=False)
    if not orthogonal and len(exo_lvs) > 1:
        for i in range(len(exo_lvs)):
            for j in range(i + 1, len(exo_lvs)):
                l1, l2 = exo_lvs[i], exo_lvs[j]
                if (l1, l2) not in existing_pairs:
                    explicit_entries.append(
                        {"lhs": l1, "op": "~~", "rhs": l2, "label": None, "fixed_val": None, "force_free": False}
                    )
                    existing_pairs.add((l1, l2))
                    existing_pairs.add((l2, l1))

    # Covariâncias automáticas entre OVs exógenas (fixed.x=True fixa na covariância amostral)
    if len(exo_ovs) > 1:
        for i in range(len(exo_ovs)):
            for j in range(i + 1, len(exo_ovs)):
                o1, o2 = exo_ovs[i], exo_ovs[j]
                if (o1, o2) not in existing_pairs:
                    if fixed_x and s_mat is not None:
                        fixed_c = float(s_mat[ov_idx_map[o1], ov_idx_map[o2]])
                    else:
                        fixed_c = None
                    explicit_entries.append(
                        {"lhs": o1, "op": "~~", "rhs": o2, "label": None, "fixed_val": fixed_c, "force_free": False}
                    )
                    existing_pairs.add((o1, o2))
                    existing_pairs.add((o2, o1))

    return {
        "entries": explicit_entries,
        "defined": defined_params,
        "lv_names": lv_order,
        "ov_names": ov_names,
        "exo_ovs": exo_ovs,
    }


@dataclass
class SEMFit:
    lv_names: list[str]
    ov_names: list[str]
    estimator: str
    ordered: bool
    n_obs: int
    s_mat: np.ndarray
    sigma_hat: np.ndarray
    sigma_all: np.ndarray
    vcov: np.ndarray
    pe_df: pd.DataFrame
    fit_table: pd.DataFrame
    data: pd.DataFrame | None = None
    poly_result: PolychoricResult | None = None
    converged: bool = True

    def parameter_estimates(self) -> pd.DataFrame:
        return self.pe_df.copy()

    def fit_measures(self) -> pd.DataFrame:
        return self.fit_table.copy()

    def implied_cov(self) -> pd.DataFrame:
        return pd.DataFrame(self.sigma_hat, index=self.ov_names, columns=self.ov_names)

    def residuals(self, type: str = "raw") -> pd.DataFrame:
        return residuals(self, type=type)

    def modindices(self, sort: bool = True, minimum_value: float = 0.0, op: str | None = None) -> pd.DataFrame:
        return modindices(self, sort=sort, minimum_value=minimum_value, op=op)

    def summary(self) -> str:
        lines = [
            f"anvala SEM ({self.estimator}, N={self.n_obs}, converged={self.converged})",
            "--- Fit Measures ---",
            self.fit_table.round(4).to_string(),
            "\n--- Parameter Estimates ---",
            self.pe_df.round(4).to_string(index=False),
        ]
        return "\n".join(lines)


def _build_sem_fit_table(
    t_model: float,
    df_model: int,
    t_base: float,
    df_base: int,
    n_obs: int,
    s_mat: np.ndarray,
    sigma_hat: np.ndarray,
    c_model: float = 1.0,
    c_base: float = 1.0,
    b_model: float = 0.0,
    b_base: float = 0.0,
    scaled_shifted: bool = False,
    n_rmsea: float | None = None,
    t_m_catml: float | None = None,
    t_b_catml: float | None = None,
    c_hat3_m: float = 1.0,
    c_hat3_b: float = 1.0,
) -> pd.DataFrame:
    n_rm = float(n_obs) if n_rmsea is None else float(n_rmsea)
    p = s_mat.shape[0]

    def _cfi(tm, dfm, tb, dfb):
        if dfm <= 0:
            return 1.0
        num = max(0.0, tm - dfm)
        den = max(num, tb - dfb)
        return 1.0 - (num / den) if den > 1e-12 else 1.0

    def _tli(tm, dfm, tb, dfb):
        if dfm <= 0 or dfb <= 0:
            return 1.0
        r_b = tb / dfb
        r_m = tm / dfm
        return (r_b - r_m) / (r_b - 1.0) if abs(r_b - 1.0) > 1e-12 else 1.0

    def _rmsea(tm, dfm):
        if dfm <= 0:
            return 0.0
        return float(np.sqrt(max(0.0, (tm - dfm) / (dfm * n_rm))))

    # SRMR
    sd_s = np.sqrt(np.maximum(1e-12, np.diag(s_mat)))
    sd_sig = np.sqrt(np.maximum(1e-12, np.diag(sigma_hat)))
    r_obs = s_mat / np.outer(sd_s, sd_s)
    r_imp = sigma_hat / np.outer(sd_sig, sd_sig)
    tril_r, tril_c = np.tril_indices(p, k=0)
    srmr_val = float(np.sqrt(np.mean((r_obs[tril_r, tril_c] - r_imp[tril_r, tril_c]) ** 2)))

    t_m_sc = (t_model / c_model + b_model) if scaled_shifted else (t_model / c_model)
    t_b_sc = (t_base / c_base + b_base) if scaled_shifted else (t_base / c_base)

    cfi_un = _cfi(t_model, df_model, t_base, df_base)
    tli_un = _tli(t_model, df_model, t_base, df_base)
    rmsea_un = _rmsea(t_model, df_model)

    cfi_sc = _cfi(t_m_sc, df_model, t_b_sc, df_base)
    tli_sc = _tli(t_m_sc, df_model, t_b_sc, df_base)
    rmsea_sc = _rmsea(t_m_sc, df_model)

    if scaled_shifted and t_m_catml is not None and t_b_catml is not None:
        # Fórmulas robustas categóricas (Savalei, 2021)
        if df_model <= 0:
            cfi_rob, tli_rob, rmsea_rob = 1.0, 1.0, 0.0
        else:
            num_r = max(0.0, t_m_catml - c_hat3_m * df_model)
            den_r = max(num_r, t_b_catml - c_hat3_b * df_base)
            cfi_rob = 1.0 - (num_r / den_r) if den_r > 1e-12 else 1.0
            r_b_rob = t_b_catml / (c_hat3_b * df_base)
            r_m_rob = t_m_catml / (c_hat3_m * df_model)
            tli_rob = (r_b_rob - r_m_rob) / (r_b_rob - 1.0) if abs(r_b_rob - 1.0) > 1e-12 else 1.0
            rmsea_rob = float(np.sqrt(max(0.0, (t_m_catml - c_hat3_m * df_model) / (df_model * n_rm))))
    elif df_model > 0 and c_model > 1e-12:
        # Fórmula robusta contínua (Brosseau-Liard & Savalei, 2014)
        num_r = max(0.0, c_model * (t_m_sc - df_model))
        den_r = max(num_r, c_base * (t_b_sc - df_base))
        cfi_rob = 1.0 - (num_r / den_r) if den_r > 1e-12 else 1.0
        tli_rob = tli_sc
        rmsea_rob = float(np.sqrt(max(0.0, c_model * (t_m_sc - df_model) / (df_model * n_rm))))
    else:
        cfi_rob, tli_rob, rmsea_rob = cfi_sc, tli_sc, rmsea_sc

    pval_un = float(1.0 - stats.chi2.cdf(t_model, df_model)) if df_model > 0 else np.nan
    pval_sc = float(1.0 - stats.chi2.cdf(t_m_sc, df_model)) if df_model > 0 else np.nan

    data_rows = {
        "chisq": [t_model, t_m_sc, np.nan],
        "df": [float(df_model), float(df_model), np.nan],
        "pvalue": [pval_un, pval_sc, np.nan],
        "scaling_factor": [1.0, c_model, np.nan],
        "cfi": [cfi_un, cfi_sc, cfi_rob],
        "tli": [tli_un, tli_sc, tli_rob],
        "rmsea": [rmsea_un, rmsea_sc, rmsea_rob],
        "srmr": [srmr_val, srmr_val, srmr_val],
    }
    return pd.DataFrame.from_dict(data_rows, orient="index", columns=["Unscaled", "Scaled", "Robust"])


def sem(
    model: str,
    data: pd.DataFrame,
    estimator: str = "ML",
    ordered: bool = False,
    std_lv: bool = False,
    orthogonal: bool = False,
    fixed_x: bool = True,
) -> SEMFit:
    """
    Estima Modelos de Equações Estruturais (SEM / Path Analysis / CFA) com paridade numérica com lavaan::sem.
    Suporta operadores '=~', '~', '~~', rótulos ('a*x'), restrições de igualdade e parâmetros definidos ':='.
    """
    est_upper = estimator.upper()
    if est_upper == "WLSMV":
        ordered = True

    df_clean = data.dropna().copy()
    n_obs = int(df_clean.shape[0])
    data_cols = list(df_clean.columns)

    # Primeiro parse para identificar OVs relevantes
    pre_spec = _parse_sem_syntax(model, data_cols, std_lv=std_lv, orthogonal=orthogonal, fixed_x=False)
    ov_names = pre_spec["ov_names"]
    lv_names = pre_spec["lv_names"]
    p = len(ov_names)
    n_lv = len(lv_names)

    if ordered:
        poly_res = polychoric(df_clean[ov_names])
        s_mat = poly_res.corr.to_numpy(dtype=np.float64)
    else:
        poly_res = None
        s_mat = np.cov(df_clean[ov_names].to_numpy(dtype=np.float64), rowvar=False, bias=True)

    spec = _parse_sem_syntax(model, data_cols, std_lv=std_lv, orthogonal=orthogonal, fixed_x=fixed_x, s_mat=s_mat)
    entries = spec["entries"]
    defined_params = spec["defined"]
    exo_ovs = spec["exo_ovs"]

    all_vars = lv_names + ov_names
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    k_tot = len(all_vars)

    b_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)
    psi_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)

    free_slots: list[dict] = []
    label_to_slot_idx: dict[str, int] = {}
    init_vals: list[float] = []
    is_log_var: list[bool] = []

    ov_idx_local = {ov: i for i, ov in enumerate(ov_names)}

    for ent in entries:
        lhs, op, rhs = ent["lhs"], ent["op"], ent["rhs"]
        fixed_v = ent["fixed_val"]
        lbl = ent["label"]

        # Em modelos ordinais, variâncias residuais das OVs são determinadas pela parametrização delta
        if ordered and op == "~~" and lhs == rhs and lhs in ov_idx_local:
            ent["is_ord_res"] = True
            ent["slot_idx"] = None
            continue
        ent["is_ord_res"] = False

        if op == "=~":
            r_i, c_i = var_to_idx[rhs], var_to_idx[lhs]
            mat_tag = "B"
            default_init = 0.6
        elif op == "~":
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            mat_tag = "B"
            default_init = 0.2
        else:
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            mat_tag = "Psi"
            if lhs == rhs:
                default_init = float(s_mat[ov_idx_local[lhs], ov_idx_local[lhs]]) * 0.5 if lhs in ov_idx_local else 0.5
            else:
                default_init = 0.05

        if fixed_v is not None:
            if mat_tag == "B":
                b_fixed[r_i, c_i] = float(fixed_v)
            else:
                psi_fixed[r_i, c_i] = float(fixed_v)
                psi_fixed[c_i, r_i] = float(fixed_v)
            ent["slot_idx"] = None
        else:
            if lbl and lbl in label_to_slot_idx:
                s_idx = label_to_slot_idx[lbl]
            else:
                s_idx = len(init_vals)
                if lbl:
                    label_to_slot_idx[lbl] = s_idx
                is_diag_var = (mat_tag == "Psi") and (r_i == c_i)
                is_log_var.append(is_diag_var)
                init_vals.append(np.log(max(1e-3, default_init)) if is_diag_var else default_init)
            ent["slot_idx"] = s_idx
            free_slots.append({"mat": mat_tag, "r": r_i, "c": c_i, "slot": s_idx, "is_var": is_log_var[s_idx]})

    n_free = len(init_vals)
    theta_init = np.array(init_vals, dtype=np.float64)
    is_log_arr = np.array(is_log_var, dtype=bool)

    s_torch = torch.tensor(s_mat, dtype=torch.float64)
    _, logdet_s = torch.linalg.slogdet(s_torch)

    if ordered:
        vech_r, vech_c = _vech_indices(p, strict=True)
        p_star = len(vech_r)
        r_obs_t = s_torch[vech_r, vech_c]
        n_tau = sum(len(t) for t in poly_res.thresholds_list)
        gamma_rr = poly_res.gamma[n_tau:, n_tau:]
        w_rr_diag = 1.0 / np.maximum(1e-12, np.diag(gamma_rr))
        w_rr_t = torch.tensor(w_rr_diag, dtype=torch.float64)

    def _raw_to_natural(raw_t: torch.Tensor) -> torch.Tensor:
        nat_list = []
        for idx_p in range(n_free):
            if is_log_var[idx_p]:
                nat_list.append(torch.exp(raw_t[idx_p]))
            else:
                nat_list.append(raw_t[idx_p])
        return torch.stack(nat_list) if n_free > 0 else torch.zeros(0, dtype=torch.float64)

    def _unpack_natural(nat_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b_t = torch.tensor(b_fixed, dtype=torch.float64)
        psi_t = torch.tensor(psi_fixed, dtype=torch.float64)
        for fs in free_slots:
            val_t = nat_t[fs["slot"]]
            if fs["mat"] == "B":
                b_t[fs["r"], fs["c"]] = val_t
            else:
                psi_t[fs["r"], fs["c"]] = val_t
                psi_t[fs["c"], fs["r"]] = val_t

        eye_k = torch.eye(k_tot, dtype=torch.float64)
        inv_ib = torch.linalg.solve(eye_k - b_t, eye_k)
        if ordered:
            psi_lv = psi_t[:n_lv, :n_lv]
            b_lv = b_t[:n_lv, :n_lv]
            inv_ib_lv = torch.linalg.solve(torch.eye(n_lv, dtype=torch.float64) - b_lv, torch.eye(n_lv, dtype=torch.float64))
            cov_lv = inv_ib_lv @ psi_lv @ inv_ib_lv.T
            lam_ov = b_t[n_lv:, :n_lv]
            comm_ov = torch.sum((lam_ov @ cov_lv) * lam_ov, dim=1)
            ov_res = 1.0 - comm_ov
            for i_ov in range(p):
                psi_t[n_lv + i_ov, n_lv + i_ov] = ov_res[i_ov]
        sigma_all_t = inv_ib @ psi_t @ inv_ib.T
        sigma_ov_t = sigma_all_t[n_lv:, n_lv:]
        return sigma_ov_t, sigma_all_t

    def objective_half_f(raw_t: torch.Tensor) -> torch.Tensor:
        nat_t = _raw_to_natural(raw_t)
        sigma, _ = _unpack_natural(nat_t)
        if ordered:
            diff_r = r_obs_t - sigma[vech_r, vech_c]
            return 0.5 * torch.sum(w_rr_t * diff_r * diff_r)
        sign, logdet = torch.linalg.slogdet(sigma)
        if sign <= 0:
            return torch.tensor(1e10, dtype=torch.float64)
        tr_term = torch.trace(torch.linalg.solve(sigma, s_torch))
        return 0.5 * (logdet + tr_term - logdet_s - float(p))

    def obj_grad_hess(th_np: np.ndarray):
        tv = torch.tensor(th_np, dtype=torch.float64, requires_grad=True)
        loss = objective_half_f(tv)
        val = float(loss.detach().numpy())
        grad = torch.autograd.grad(loss, tv, create_graph=True)[0]
        g_np = grad.detach().numpy().copy()
        h_rows = [torch.autograd.grad(grad[i], tv, retain_graph=True)[0].detach().numpy() for i in range(len(th_np))]
        return val, g_np, np.vstack(h_rows)

    res_bfgs = optimize.minimize(
        lambda x: obj_grad_hess(x)[:2],
        theta_init,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-6, "maxiter": 500},
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
        alpha = 1.0
        accepted = False
        while alpha >= 1e-10:
            cand = raw_hat + alpha * step
            fc, gc, hc = obj_grad_hess(cand)
            if np.isfinite(fc) and fc <= f_curr + 1e-12:
                raw_hat, f_curr, g_curr, h_curr = cand, fc, gc, hc
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    if np.max(np.abs(g_curr)) < 1e-8:
        converged = True

    with torch.no_grad():
        nat_t_hat = _raw_to_natural(torch.tensor(raw_hat, dtype=torch.float64))
        nat_hat = nat_t_hat.numpy()
        sig_ov_t, sig_all_t = _unpack_natural(nat_t_hat)
        sigma_hat = sig_ov_t.numpy()
        sigma_all = sig_all_t.numpy()

    # Matriz de covariância assintótica (VCOV) na escala natural dos parâmetros
    if ordered:
        n_wls = float(n_obs - 1)
        def vech_r_nat(nt: torch.Tensor) -> torch.Tensor:
            sig, _ = _unpack_natural(nt)
            return sig[vech_r, vech_c]
        jac_r = torch.autograd.functional.jacobian(vech_r_nat, torch.tensor(nat_hat, dtype=torch.float64)).numpy()
        w_rr = np.diag(w_rr_diag)
        dtwd = jac_r.T @ w_rr @ jac_r
        dtwd_inv = np.linalg.pinv(dtwd)
        meat = jac_r.T @ w_rr @ gamma_rr @ w_rr @ jac_r
        vcov_nat = (dtwd_inv @ meat @ dtwd_inv) / n_wls
    else:
        full_r, full_c = _vech_indices(p, strict=False)
        d_mat = _duplication_matrix(p, full_r, full_c)
        def vech_s_nat(nt: torch.Tensor) -> torch.Tensor:
            sig, _ = _unpack_natural(nt)
            return sig[full_r, full_c]
        jac_s = torch.autograd.functional.jacobian(vech_s_nat, torch.tensor(nat_hat, dtype=torch.float64)).numpy()
        sigma_inv = np.linalg.inv(sigma_hat)
        w_ml = 0.5 * (d_mat.T @ np.kron(sigma_inv, sigma_inv) @ d_mat)
        a_inv = np.linalg.pinv(jac_s.T @ w_ml @ jac_s)
        if est_upper == "MLR":
            y_c = df_clean[ov_names].to_numpy(dtype=np.float64) - np.mean(df_clean[ov_names].to_numpy(dtype=np.float64), axis=0)
            s_vech = s_mat[full_r, full_c]
            sig_vech = sigma_hat[full_r, full_c]
            yi_s = np.einsum("ni,nj->nij", y_c, y_c)[:, full_r, full_c] - s_vech[None, :]
            gamma_mlr = (yi_s.T @ yi_s) / float(n_obs)
            yi_sig = np.einsum("ni,nj->nij", y_c, y_c)[:, full_r, full_c] - sig_vech[None, :]
            scores_i = yi_sig @ (w_ml @ jac_s)
            b_meat = (scores_i.T @ scores_i) / float(n_obs)
            # Converte Hessiana da escala log para a escala natural via regra da cadeia
            d_raw_d_nat = np.diag(np.where(is_log_arr, 1.0 / np.maximum(1e-12, nat_hat), 1.0))
            h_nat = d_raw_d_nat @ h_curr @ d_raw_d_nat
            h_obs_inv = np.linalg.pinv(h_nat)
            vcov_nat = (h_obs_inv @ b_meat @ h_obs_inv) / float(n_obs)
        else:
            vcov_nat = a_inv / float(n_obs)

    se_nat = np.sqrt(np.maximum(0.0, np.diag(vcov_nat)))

    # Graus de liberdade (descontando covariâncias fixas de exo_ovs se fixed_x=True)
    n_exo_moments = (len(exo_ovs) * (len(exo_ovs) + 1) // 2) if (fixed_x and not ordered) else 0
    if ordered:
        df_base = p_star
        df_model = p_star - n_free
    else:
        df_base = (p * (p + 1) // 2) - n_exo_moments - (p - (len(exo_ovs) if fixed_x else 0))
        df_model = (p * (p + 1) // 2) - n_exo_moments - n_free

    # Tabela de estimativas de parâmetros (incluindo := via método Delta)
    pe_rows = []
    label_est_map: dict[str, torch.Tensor] = {}
    label_std_lv_map: dict[str, float] = {}
    label_std_all_map: dict[str, float] = {}

    nat_t_grad = torch.tensor(nat_hat, dtype=torch.float64, requires_grad=True)

    for ent in entries:
        lhs, op, rhs = ent["lhs"], ent["op"], ent["rhs"]
        lbl = ent["label"] or ""
        s_idx = ent["slot_idx"]

        if ent.get("is_ord_res", False):
            ov_i = var_to_idx[lhs]
            est_val = float(sigma_all[ov_i, ov_i] - np.sum((sigma_all[ov_i, :n_lv] @ np.linalg.pinv(sigma_all[:n_lv, :n_lv])) * sigma_all[ov_i, :n_lv]))
            se_val = 0.0
        elif s_idx is None:
            est_val = float(ent["fixed_val"])
            se_val = 0.0
            if lbl:
                label_est_map[lbl] = torch.tensor(est_val, dtype=torch.float64)
        else:
            est_val = float(nat_hat[s_idx])
            se_val = float(se_nat[s_idx])
            if lbl:
                label_est_map[lbl] = nat_t_grad[s_idx]

        idx_l = var_to_idx[lhs]
        idx_r = var_to_idx[rhs]
        sd_lv_l = np.sqrt(max(1e-12, sigma_all[idx_l, idx_l])) if lhs in lv_names else 1.0
        sd_lv_r = np.sqrt(max(1e-12, sigma_all[idx_r, idx_r])) if rhs in lv_names else 1.0
        sd_all_l = np.sqrt(max(1e-12, sigma_all[idx_l, idx_l]))
        sd_all_r = np.sqrt(max(1e-12, sigma_all[idx_r, idx_r]))

        if op == "=~":
            std_lv_v = est_val * sd_lv_l
            std_all_v = est_val * (sd_lv_l / sd_all_r)
        elif op == "~":
            std_lv_v = est_val * (sd_lv_r / sd_lv_l)
            std_all_v = est_val * (sd_all_r / sd_all_l)
        else:
            std_lv_v = est_val / (sd_lv_l * sd_lv_r)
            std_all_v = est_val / (sd_all_l * sd_all_r)

        if lbl:
            label_std_lv_map[lbl] = std_lv_v
            label_std_all_map[lbl] = std_all_v

        z_v = est_val / se_val if se_val > 1e-12 else np.nan
        p_v = float(2.0 * (1.0 - stats.norm.cdf(abs(z_v)))) if np.isfinite(z_v) else np.nan
        pe_rows.append(
            {
                "lhs": lhs,
                "op": op,
                "rhs": rhs,
                "label": lbl,
                "est": est_val,
                "se": se_val,
                "z": z_v,
                "pvalue": p_v,
                "std_lv": std_lv_v,
                "std_all": std_all_v,
            }
        )

    for def_name, expr_str in defined_params:
        parsed_expr = ast.parse(expr_str, mode="eval")
        def_tensor = _eval_defined_ast(parsed_expr, label_est_map)
        if isinstance(def_tensor, torch.Tensor) and def_tensor.requires_grad:
            grad_def = torch.autograd.grad(def_tensor, nat_t_grad, retain_graph=True)[0].detach().numpy()
            def_est = float(def_tensor.detach().numpy())
            def_se = float(np.sqrt(max(0.0, float(grad_def @ vcov_nat @ grad_def))))
        else:
            def_est = float(def_tensor)
            def_se = 0.0

        def_std_lv = float(_eval_defined_ast(parsed_expr, label_std_lv_map))
        def_std_all = float(_eval_defined_ast(parsed_expr, label_std_all_map))
        z_v = def_est / def_se if def_se > 1e-12 else np.nan
        p_v = float(2.0 * (1.0 - stats.norm.cdf(abs(z_v)))) if np.isfinite(z_v) else np.nan
        pe_rows.append(
            {
                "lhs": def_name,
                "op": ":=",
                "rhs": expr_str,
                "label": def_name,
                "est": def_est,
                "se": def_se,
                "z": z_v,
                "pvalue": p_v,
                "std_lv": def_std_lv,
                "std_all": def_std_all,
            }
        )

    pe_df = pd.DataFrame(pe_rows)

    # Índices de ajuste (Unscaled / Scaled / Robust)
    if ordered:
        n_wls = float(n_obs - 1)
        r_obs_v = s_mat[vech_r, vech_c]
        r_hat_v = sigma_hat[vech_r, vech_c]
        t_model = n_wls * float(np.sum(w_rr_diag * ((r_obs_v - r_hat_v) ** 2)))
        t_base = n_wls * float(np.sum(w_rr_diag * (r_obs_v ** 2)))

        if df_model > 0:
            u_mat = w_rr - w_rr @ jac_r @ dtwd_inv @ jac_r.T @ w_rr
            m_mat = u_mat @ gamma_rr
            a_scale = np.sqrt(float(df_model) / max(1e-12, float(np.trace(m_mat @ m_mat))))
            c_model = 1.0 / a_scale
            b_model = float(df_model) - a_scale * float(np.trace(m_mat))
        else:
            c_model, b_model = 1.0, 0.0

        m_base = w_rr @ gamma_rr
        a_base = np.sqrt(float(df_base) / max(1e-12, float(np.trace(m_base @ m_base))))
        c_base = 1.0 / a_base
        b_base = float(df_base) - a_base * float(np.trace(m_base))

        full_r, full_c = _vech_indices(p, strict=False)
        d_full = _duplication_matrix(p, full_r, full_c)
        off_idx = np.array([idx_i for idx_i, (r, c) in enumerate(zip(full_r, full_c)) if r != c], dtype=np.int64)
        sigma_inv = np.linalg.inv(sigma_hat)
        v_ml = (0.5 * (d_full.T @ np.kron(sigma_inv, sigma_inv) @ d_full))[np.ix_(off_idx, off_idx)]
        wi_u = np.eye(p_star) - jac_r @ dtwd_inv @ jac_r.T @ w_rr
        c_hat3_m = (float(np.trace(wi_u.T @ v_ml @ wi_u @ gamma_rr)) / float(df_model)) if df_model > 0 else 1.0
        c_hat3_b = float(np.trace(gamma_rr)) / float(df_base)

        _, logdet_sig = np.linalg.slogdet(sigma_hat)
        logdet_r = float(logdet_s.numpy())
        t_m_catml = float(n_obs) * float(logdet_sig + np.trace(np.linalg.solve(sigma_hat, s_mat)) - logdet_r - float(p))
        t_b_catml = float(n_obs) * (-logdet_r)

        fit_table = _build_sem_fit_table(
            t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat,
            c_model=c_model, c_base=c_base, b_model=b_model, b_base=b_base,
            scaled_shifted=True, n_rmsea=n_wls,
            t_m_catml=t_m_catml, t_b_catml=t_b_catml, c_hat3_m=c_hat3_m, c_hat3_b=c_hat3_b,
        )
    else:
        t_model = float(n_obs) * (2.0 * f_curr)
        # Modelo nulo respeitando fixed.x para variáveis exógenas observadas
        sigma_base = np.diag(np.diag(s_mat))
        if fixed_x and len(exo_ovs) > 0:
            exo_idx = [ov_idx_local[ov] for ov in exo_ovs]
            sigma_base[np.ix_(exo_idx, exo_idx)] = s_mat[np.ix_(exo_idx, exo_idx)]
        _, logdet_b = np.linalg.slogdet(sigma_base)
        t_base = float(n_obs) * float(logdet_b + np.trace(np.linalg.solve(sigma_base, s_mat)) - float(logdet_s.numpy()) - float(p))

        if est_upper == "MLR":
            s_inv = np.linalg.inv(s_mat)
            w_h1 = 0.5 * (d_mat.T @ np.kron(s_inv, s_inv) @ d_mat)
            tr_h1 = float(np.trace(w_h1 @ gamma_mlr))
            tr_h0 = float(np.trace(h_obs_inv @ b_meat))
            c_model = ((tr_h1 - tr_h0) / float(df_model)) if df_model > 0 else 1.0

            sb_inv = np.linalg.inv(sigma_base)
            w_base = 0.5 * (d_mat.T @ np.kron(sb_inv, sb_inv) @ d_mat)
            base_free_mask = np.array(
                [(r == c) or (fixed_x and ov_names[r] in exo_ovs and ov_names[c] in exo_ovs) for r, c in zip(full_r, full_c)],
                dtype=bool,
            )
            j_base = np.eye(len(full_r))[:, base_free_mask]
            tr_b0 = float(np.trace(w_base @ j_base @ np.linalg.pinv(j_base.T @ w_base @ j_base) @ j_base.T @ w_base @ gamma_mlr))
            c_base = ((tr_h1 - tr_b0) / float(df_base)) if df_base > 0 else 1.0
            fit_table = _build_sem_fit_table(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat, c_model=c_model, c_base=c_base)
        else:
            fit_table = _build_sem_fit_table(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat)

    return SEMFit(
        lv_names=lv_names,
        ov_names=ov_names,
        estimator=est_upper,
        ordered=ordered,
        n_obs=n_obs,
        s_mat=s_mat,
        sigma_hat=sigma_hat,
        sigma_all=sigma_all,
        vcov=vcov_nat,
        pe_df=pe_df,
        fit_table=fit_table,
        data=df_clean,
        poly_result=poly_res,
        converged=converged,
    )