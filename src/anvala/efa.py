from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize, stats
import torch

from anvala.cfa import _duplication_matrix, _vech_col_major_indices
from anvala.ega import _near_pd_corr, _resolve_corr_matrix
from anvala.fit_measures import compute_fit_triad
from anvala.polychoric import polychoric


def kmo_bartlett(data: pd.DataFrame, corkind: str = "auto") -> dict:
    """Calcula KMO global, MSA por item e Teste de Esfericidade de Bartlett."""
    r_mat, var_names, n_obs, _ = _resolve_corr_matrix(data, corr=corkind)
    p = r_mat.shape[0]
    r_inv = np.linalg.pinv(r_mat)
    d_inv = np.sqrt(np.maximum(1e-12, np.diag(r_inv)))
    pcor = -r_inv / np.outer(d_inv, d_inv)
    np.fill_diagonal(pcor, 0.0)

    r_off = r_mat.copy()
    np.fill_diagonal(r_off, 0.0)
    r2_sum = float(np.sum(r_off ** 2))
    p2_sum = float(np.sum(pcor ** 2))
    kmo_overall = r2_sum / max(1e-12, r2_sum + p2_sum)

    msa_items = np.sum(r_off ** 2, axis=1) / np.maximum(
        1e-12, np.sum(r_off ** 2, axis=1) + np.sum(pcor ** 2, axis=1)
    )
    msa_series = pd.Series(msa_items, index=var_names, name="MSA")

    sign, logdet = np.linalg.slogdet(r_mat)
    chisq_bartlett = -(float(n_obs) - 1.0 - (2.0 * float(p) + 5.0) / 6.0) * float(logdet)
    df_bartlett = p * (p - 1) // 2
    p_bartlett = float(1.0 - stats.chi2.cdf(max(0.0, chisq_bartlett), df_bartlett))

    return {
        "kmo": float(kmo_overall),
        "msa": msa_series,
        "bartlett_chisq": float(chisq_bartlett),
        "bartlett_df": int(df_bartlett),
        "bartlett_pvalue": p_bartlett,
    }


def _smc_reduced_eigvals(r_mat: np.ndarray) -> np.ndarray:
    """Autovalores da matriz de correlação reduzida com SMC na diagonal (padrão psych / EFA.dimensions)."""
    r_inv = np.linalg.pinv(r_mat)
    smc = 1.0 - 1.0 / np.maximum(1e-10, np.diag(r_inv))
    smc = np.clip(smc, 0.0, 0.9999)
    r_red = r_mat.copy()
    np.fill_diagonal(r_red, smc)
    vals = np.linalg.eigvalsh(r_red)[::-1]
    return vals


def _paf_fit(r_mat: np.ndarray, n_factors: int, max_iter: int = 200, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Principal Axis Factoring (PAF) rápido para Hull e Comparison Data."""
    p = r_mat.shape[0]
    if n_factors == 0:
        return np.zeros((p, 0)), np.zeros(p)
    r_inv = np.linalg.pinv(r_mat)
    h2 = np.clip(1.0 - 1.0 / np.maximum(1e-10, np.diag(r_inv)), 0.0, 0.999)
    loadings = np.zeros((p, n_factors), dtype=np.float64)
    for _ in range(max_iter):
        h2_old = h2.copy()
        r_red = r_mat.copy()
        np.fill_diagonal(r_red, h2)
        vals, vecs = np.linalg.eigh(r_red)
        idx = np.argsort(vals)[::-1][:n_factors]
        vals_pos = np.maximum(vals[idx], 0.0)
        loadings = vecs[:, idx] * np.sqrt(vals_pos)[None, :]
        h2 = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 0.9999)
        if np.max(np.abs(h2 - h2_old)) < tol:
            break
    return loadings, h2


@dataclass
class DimTestsResult:
    results: pd.DataFrame
    n_factors: int
    primary_method: str
    eigenvalues: pd.DataFrame
    adequacy: dict

    def __repr__(self) -> str:
        return (
            f"DimTestsResult(recommended_n_factors={self.n_factors} [padrão ouro: {self.primary_method}], "
            f"KMO={self.adequacy['kmo']:.3f})\n"
            f"{self.results.to_string(index=False)}"
        )


def dimtests(
    data: pd.DataFrame,
    tests: list[str] | None = None,
    corkind: str = "auto",
    n_cases: int | None = None,
    hull_method: str = "PAF",
    hull_gof: str = "CAF",
    n_sim: int = 100,
    percentile: float = 95.0,
    seed: int = 42,
) -> DimTestsResult:
    """
    Bateria de testes de retenção dimensional (inspirada em EFA.dimensions::DIMTESTS e Auerswald & Moshagen, 2019).
    Utiliza a Análise Paralela de Horn (RAWPAR) como primeiro critério e padrão ouro.
    """
    if tests is None:
        tests = ["RAWPAR", "EMPKC", "HULL", "SMT"]
    tests_upper = [t.upper() for t in tests]

    is_corr_input = (
        data.shape[0] == data.shape[1]
        and np.allclose(np.diag(data.to_numpy()), 1.0, atol=1e-4)
        and np.allclose(data.to_numpy(), data.to_numpy().T, atol=1e-4)
    )
    if is_corr_input:
        if n_cases is None:
            raise ValueError("Quando 'data' é uma matriz de correlação, informe 'n_cases'.")
        r_mat = _near_pd_corr(data.to_numpy(dtype=np.float64))
        n_obs = int(n_cases)
        df_clean = None
        adequacy = {"kmo": np.nan, "msa": pd.Series(dtype=float), "bartlett_chisq": np.nan, "bartlett_df": 0, "bartlett_pvalue": np.nan}
    else:
        df_clean = data.dropna()
        r_mat, _, n_obs, _ = _resolve_corr_matrix(df_clean, corr=corkind)
        adequacy = kmo_bartlett(df_clean, corkind=corkind)

    p = r_mat.shape[0]
    rng = np.random.default_rng(seed)

    eig_pca = np.linalg.eigvalsh(r_mat)[::-1]
    eig_fa = _smc_reduced_eigvals(r_mat)

    # Simulação de Horn (RAWPAR)
    sim_pca = np.zeros((n_sim, p), dtype=np.float64)
    sim_fa = np.zeros((n_sim, p), dtype=np.float64)
    for b in range(n_sim):
        z = rng.standard_normal((n_obs, p))
        r_sim = np.corrcoef(z, rowvar=False)
        sim_pca[b] = np.linalg.eigvalsh(r_sim)[::-1]
        sim_fa[b] = _smc_reduced_eigvals(r_sim)

    pa_mean_pca = np.mean(sim_pca, axis=0)
    pa_perc_pca = np.percentile(sim_pca, percentile, axis=0)
    pa_mean_fa = np.mean(sim_fa, axis=0)
    pa_perc_fa = np.percentile(sim_fa, percentile, axis=0)

    eig_df = pd.DataFrame({
        "factor": np.arange(1, p + 1),
        "eig_pca": eig_pca,
        "pa_mean_pca": pa_mean_pca,
        "pa_p95_pca": pa_perc_pca,
        "eig_fa": eig_fa,
        "pa_mean_fa": pa_mean_fa,
        "pa_p95_fa": pa_perc_fa,
    })

    # Número máximo identificável de fatores: df >= 0
    m_max = 0
    for m_cand in range(1, p):
        if ((p - m_cand) ** 2 - (p + m_cand)) >= 0:
            m_max = m_cand
        else:
            break

    rows = []
    for t_name in tests_upper:
        if t_name == "RAWPAR":
            # Padrão ouro: Análise Paralela de Horn (fatores comuns vs. percentil 95 / média)
            k_fa_p95 = int(np.sum(eig_fa > pa_perc_fa))
            k_pca_p95 = int(np.sum(eig_pca > pa_perc_pca))
            rows.append({
                "method": "RAWPAR (Horn Parallel Analysis)",
                "n_factors": max(1, k_fa_p95),
                "detail": f"FA p95={k_fa_p95} | PCA p95={k_pca_p95}",
            })
        elif t_name == "EMPKC":
            # Empirical Kaiser Criterion (Braeken & van Assen, 2017)
            gamma_p_n = float(p) / float(n_obs)
            k_ekc = 0
            cum_eig = 0.0
            for j in range(p):
                l_ref = max(1.0, ((float(p) - cum_eig) / float(p - j)) * ((1.0 + np.sqrt(gamma_p_n)) ** 2))
                if eig_pca[j] > l_ref:
                    k_ekc += 1
                    cum_eig += eig_pca[j]
                else:
                    break
            rows.append({
                "method": "EMPKC (Empirical Kaiser)",
                "n_factors": max(1, k_ekc),
                "detail": f"Primeiro autovalor={eig_pca[0]:.3f}",
            })
        elif t_name == "HULL":
            # Hull Method (Lorenzo-Seva et al., 2011)
            f_vals = []
            df_vals = []
            for m in range(0, m_max + 1):
                df_m = ((p - m) ** 2 - (p + m)) // 2
                if m == 0:
                    r_res = r_mat - np.eye(p)
                else:
                    lam_m, h2_m = _paf_fit(r_mat, m)
                    r_hat = lam_m @ lam_m.T
                    np.fill_diagonal(r_hat, 1.0)
                    r_res = r_mat - r_hat
                off = r_res[np.tril_indices(p, k=-1)]
                if hull_gof.upper() == "CAF":
                    # Common part Accounted For (1 - KMO dos resíduos)
                    r_res_full = r_res.copy()
                    np.fill_diagonal(r_res_full, 1.0)
                    r_res_pd = _near_pd_corr(r_res_full)
                    inv_res = np.linalg.pinv(r_res_pd)
                    d_i = np.sqrt(np.maximum(1e-10, np.diag(inv_res)))
                    pc_res = -inv_res / np.outer(d_i, d_i)
                    np.fill_diagonal(pc_res, 0.0)
                    r2_s = float(np.sum(off ** 2)) * 2.0
                    p2_s = float(np.sum(pc_res ** 2))
                    kmo_res = r2_s / max(1e-12, r2_s + p2_s)
                    gof_m = 1.0 - kmo_res
                else:
                    srmr_m = float(np.sqrt(np.mean(off ** 2)))
                    gof_m = 1.0 - srmr_m
                f_vals.append(gof_m)
                df_vals.append(float(df_m))

            # Envoltória convexa superior e razão de inclinação st_j
            hull_idx = [0]
            for j in range(1, len(f_vals)):
                while len(hull_idx) >= 2:
                    i1, i2 = hull_idx[-2], hull_idx[-1]
                    slope1 = (f_vals[i2] - f_vals[i1]) / max(1e-9, df_vals[i2] - df_vals[i1])
                    slope2 = (f_vals[j] - f_vals[i2]) / max(1e-9, df_vals[j] - df_vals[i2])
                    if slope2 >= slope1:
                        hull_idx.pop()
                    else:
                        break
                hull_idx.append(j)

            best_m = 1
            best_st = -np.inf
            for pos in range(1, len(hull_idx) - 1):
                prev_i, curr_i, next_i = hull_idx[pos - 1], hull_idx[pos], hull_idx[pos + 1]
                s_prev = (f_vals[curr_i] - f_vals[prev_i]) / max(1e-9, df_vals[curr_i] - df_vals[prev_i])
                s_next = (f_vals[next_i] - f_vals[curr_i]) / max(1e-9, df_vals[next_i] - df_vals[curr_i])
                st = s_prev / max(1e-9, s_next)
                if st > best_st:
                    best_st = st
                    best_m = curr_i
            rows.append({
                "method": f"HULL ({hull_method.upper()}/{hull_gof.upper()})",
                "n_factors": max(1, best_m),
                "detail": f"st_max={best_st:.3f}" if np.isfinite(best_st) else "boundary",
            })
        elif t_name == "SMT":
            # Sequential Model Testing (chi-square ML sequencial até p > 0.05)
            k_smt = m_max
            p_last = 0.0
            for m in range(1, m_max + 1):
                fit_m = efa(
                    pd.DataFrame(r_mat) if is_corr_input else df_clean,
                    n_factors=m,
                    estimator="ML",
                    rotation="none",
                    sample_cov=r_mat if is_corr_input else None,
                    sample_nobs=n_obs if is_corr_input else None,
                )
                p_val = float(fit_m.fit_measures().loc["pvalue", "Unscaled"])
                p_last = p_val
                if p_val > 0.05:
                    k_smt = m
                    break
            rows.append({
                "method": "SMT (Sequential Model Testing)",
                "n_factors": max(1, k_smt),
                "detail": f"p={p_last:.4f}",
            })
        elif t_name == "MAP":
            # Velicer's Minimum Average Partial (MAP, 1976)
            vals, vecs = np.linalg.eigh(r_mat)
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
            load_pca = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]
            map_vals = []
            for m in range(0, p - 1):
                if m == 0:
                    r_part = r_mat.copy()
                else:
                    a_m = load_pca[:, :m]
                    r_part = r_mat - a_m @ a_m.T
                d_p = np.sqrt(np.maximum(1e-10, np.diag(r_part)))
                r_part_cor = r_part / np.outer(d_p, d_p)
                off_p = r_part_cor[np.tril_indices(p, k=-1)]
                map_vals.append(float(np.mean(off_p ** 2)))
            k_map = int(np.argmin(map_vals))
            rows.append({
                "method": "MAP (Velicer)",
                "n_factors": max(1, k_map),
                "detail": f"min_avg_sq_pcor={min(map_vals):.5f}",
            })
        elif t_name == "CD":
            # Comparison Data (Ruscio & Roche, 2012)
            if is_corr_input or df_clean is None:
                rows.append({"method": "CD (Comparison Data)", "n_factors": np.nan, "detail": "Requer dados brutos"})
                continue
            k_cd = 1
            for j in range(1, min(m_max, 6)):
                lam_j, _ = _paf_fit(r_mat, j)
                lam_jp1, _ = _paf_fit(r_mat, j + 1)
                r_j = lam_j @ lam_j.T
                np.fill_diagonal(r_j, 1.0)
                r_jp1 = lam_jp1 @ lam_jp1.T
                np.fill_diagonal(r_jp1, 1.0)
                r_j, r_jp1 = _near_pd_corr(r_j), _near_pd_corr(r_jp1)

                rmse_j = []
                rmse_jp1 = []
                for _ in range(30):
                    ev1 = np.linalg.eigvalsh(np.corrcoef(rng.multivariate_normal(np.zeros(p), r_j, n_obs), rowvar=False))[::-1]
                    ev2 = np.linalg.eigvalsh(np.corrcoef(rng.multivariate_normal(np.zeros(p), r_jp1, n_obs), rowvar=False))[::-1]
                    rmse_j.append(float(np.sqrt(np.mean((ev1 - eig_pca) ** 2))))
                    rmse_jp1.append(float(np.sqrt(np.mean((ev2 - eig_pca) ** 2))))
                _, p_mann = stats.mannwhitneyu(rmse_jp1, rmse_j, alternative="less")
                if p_mann < 0.05:
                    k_cd = j + 1
                else:
                    break
            rows.append({
                "method": "CD (Comparison Data)",
                "n_factors": k_cd,
                "detail": f"Ruscio & Roche (2012)",
            })
        elif t_name == "NEVALSGT1":
            k_k1 = int(np.sum(eig_pca > 1.0))
            rows.append({"method": "NEVALSGT1 (Kaiser > 1)", "n_factors": max(1, k_k1), "detail": f"ev > 1.0"})
        elif t_name == "SESCREE":
            # Standard Error Scree (Zoski & Jurs, 1996)
            k_se = 1
            for m in range(1, p - 2):
                rem = eig_pca[m:]
                x_idx = np.arange(len(rem), dtype=np.float64)
                slope, intercept = np.polyfit(x_idx, rem, 1)
                se_reg = float(np.sqrt(np.sum((rem - (slope * x_idx + intercept)) ** 2) / max(1, len(rem) - 2)))
                if se_reg <= 1.0 / np.sqrt(float(p)):
                    k_se = m
                    break
            rows.append({"method": "SESCREE (Zoski & Jurs)", "n_factors": k_se, "detail": "SE scree"})
        else:
            raise ValueError(f"Teste dimensional '{t_name}' desconhecido.")

    res_df = pd.DataFrame(rows)
    first_n = int(res_df.iloc[0]["n_factors"])
    return DimTestsResult(
        results=res_df,
        n_factors=first_n,
        primary_method=tests_upper[0],
        eigenvalues=eig_df,
        adequacy=adequacy,
    )


def _rotation_criterion_and_grad(
    l_mat: np.ndarray,
    method: str,
    geomin_epsilon: float = 0.001,
    oblimin_gamma: float = 0.0,
    orthomax_gamma: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Calcula critério Q(L) e gradiente dQ/dL (padrão lavaan / GPArotation)."""
    p, m = l_mat.shape
    l2 = l_mat ** 2

    if method == "geomin":
        l2_eps = l2 + geomin_epsilon
        g_row = np.prod(l2_eps, axis=1) ** (1.0 / float(m))
        q_val = float(np.sum(g_row))
        grad = (2.0 / float(m)) * g_row[:, None] * (l_mat / l2_eps)
        return q_val, grad

    if method == "oblimin":
        n_mat = np.ones((m, m), dtype=np.float64) - np.eye(m, dtype=np.float64)
        c_l2 = l2 - (oblimin_gamma / float(p)) * np.sum(l2, axis=0, keepdims=True)
        q_val = 0.25 * float(np.sum(l2 * (c_l2 @ n_mat)))
        grad = l_mat * (c_l2 @ n_mat)
        return q_val, grad

    # Orthomax family (varimax, quartimax, equamax, parsimax)
    c_l2 = l2 - (orthomax_gamma / float(p)) * np.sum(l2, axis=0, keepdims=True)
    q_val = -0.25 * float(np.sum(l2 * c_l2))
    grad = -l_mat * c_l2
    return q_val, grad


def _gpa_rotate(
    a_unrot: np.ndarray,
    method: str = "geomin",
    orthogonal: bool = False,
    geomin_epsilon: float = 0.001,
    oblimin_gamma: float = 0.0,
    promax_kappa: float = 4.0,
    rstarts: int = 30,
    init_t: np.ndarray | None = None,
    gpa_tol: float = 1e-6,
    max_iter: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradient Projection Algorithm (Bernaards & Jennrich, 2005) com multi-start (padrão lavaan).
    Retorna (L_rot, Phi_rot, T_opt) onde L_rot = A @ inv(T_opt).T e Phi_rot = T_opt.T @ T_opt.
    """
    p, m = a_unrot.shape
    method_l = method.lower()

    if m == 1 or method_l == "none":
        return a_unrot.copy(), np.eye(m, dtype=np.float64), np.eye(m, dtype=np.float64)

    if method_l == "promax":
        l_var, _, t_var = _gpa_rotate(
            a_unrot,
            method="varimax",
            orthogonal=True,
            rstarts=rstarts,
            gpa_tol=gpa_tol,
            max_iter=max_iter,
            seed=seed,
        )
        target = np.sign(l_var) * (np.abs(l_var) ** promax_kappa)
        u_mat = np.linalg.lstsq(l_var, target, rcond=None)[0]
        d_u = np.sqrt(np.maximum(1e-12, np.diag(u_mat.T @ u_mat)))
        u_norm = u_mat / d_u[None, :]
        t_tot = t_var @ np.linalg.inv(u_norm).T
        l_rot = a_unrot @ np.linalg.inv(t_tot).T
        phi_rot = t_tot.T @ t_tot
        return l_rot, phi_rot, t_tot

    orthomax_map = {
        "varimax": 1.0,
        "quartimax": 0.0,
        "equamax": float(m) / 2.0,
        "parsimax": float(p * (m - 1)) / max(1.0, float(p + m - 2)),
    }
    if method_l in orthomax_map:
        orthogonal = True
        orthomax_gamma = orthomax_map[method_l]
        crit_method = "orthomax"
    else:
        orthomax_gamma = 1.0
        crit_method = method_l

    rng = np.random.default_rng(seed)
    starts = [init_t.copy()] if init_t is not None else [np.eye(m, dtype=np.float64)]
    if init_t is None:
        for _ in range(max(0, rstarts - 1)):
            q_rand, _ = np.linalg.qr(rng.standard_normal((m, m)))
            starts.append(q_rand)

    best_q = np.inf
    best_t = np.eye(m, dtype=np.float64)

    for t_start in starts:
        t_curr = t_start.copy()
        alpha = 1.0
        if orthogonal:
            l_curr = a_unrot @ t_curr
        else:
            t_curr = t_curr / np.linalg.norm(t_curr, axis=0, keepdims=True)
            l_curr = a_unrot @ np.linalg.inv(t_curr).T

        q_curr, gq = _rotation_criterion_and_grad(
            l_curr, crit_method, geomin_epsilon=geomin_epsilon, oblimin_gamma=oblimin_gamma, orthomax_gamma=orthomax_gamma
        )

        for _ in range(max_iter):
            if orthogonal:
                gt = a_unrot.T @ gq
                m_sym = t_curr.T @ gt
                gp = t_curr @ (0.5 * (m_sym - m_sym.T))
            else:
                t_inv = np.linalg.inv(t_curr)
                gt = -(t_inv.T @ gq.T @ l_curr)
                col_dots = np.sum(t_curr * gt, axis=0, keepdims=True)
                gp = gt - t_curr * col_dots

            gnorm = float(np.linalg.norm(gp, ord="fro"))
            if gnorm < gpa_tol:
                break

            accepted = False
            for _ in range(30):
                t_cand = t_curr - alpha * gp
                if orthogonal:
                    u_s, _, vt_s = np.linalg.svd(t_cand)
                    t_cand = u_s @ vt_s
                    l_cand = a_unrot @ t_cand
                else:
                    norms = np.linalg.norm(t_cand, axis=0, keepdims=True)
                    if np.any(norms < 1e-10):
                        alpha *= 0.5
                        continue
                    t_cand = t_cand / norms
                    try:
                        l_cand = a_unrot @ np.linalg.inv(t_cand).T
                    except np.linalg.LinAlgError:
                        alpha *= 0.5
                        continue

                q_cand, gq_cand = _rotation_criterion_and_grad(
                    l_cand, crit_method, geomin_epsilon=geomin_epsilon, oblimin_gamma=oblimin_gamma, orthomax_gamma=orthomax_gamma
                )
                if q_cand <= q_curr - 0.5 * alpha * (gnorm ** 2) + 1e-12:
                    t_curr, l_curr, q_curr, gq = t_cand, l_cand, q_cand, gq_cand
                    alpha *= 2.0
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

        if q_curr < best_q:
            best_q = q_curr
            best_t = t_curr.copy()

    if orthogonal:
        l_final = a_unrot @ best_t
        phi_final = np.eye(m, dtype=np.float64)
    else:
        l_final = a_unrot @ np.linalg.inv(best_t).T
        phi_final = best_t.T @ best_t

    return l_final, phi_final, best_t


@dataclass
class EFAFit:
    n_factors: int
    estimator: str
    rotation: str
    ordered: bool
    n_obs: int
    var_names: list[str]
    factor_names: list[str]
    loadings_unstd: pd.DataFrame
    loadings_se: pd.DataFrame
    loadings_std: pd.DataFrame
    factor_corr: pd.DataFrame
    factor_corr_se: pd.DataFrame
    communalities: pd.Series
    uniquenesses: pd.Series
    variance_explained: pd.DataFrame
    eigenvalues: np.ndarray
    fit_table: pd.DataFrame
    thresholds: pd.DataFrame | None
    polychoric_corr: pd.DataFrame | None
    converged: bool

    def loadings(self, standardized: bool = True) -> pd.DataFrame:
        return self.loadings_std if standardized else self.loadings_unstd

    def fit_measures(self) -> pd.DataFrame:
        return self.fit_table

    def parameter_estimates(self) -> pd.DataFrame:
        rows = []
        for f_idx, f_name in enumerate(self.factor_names):
            for v_idx, v_name in enumerate(self.var_names):
                est_v = float(self.loadings_unstd.iloc[v_idx, f_idx])
                se_v = float(self.loadings_se.iloc[v_idx, f_idx])
                std_v = float(self.loadings_std.iloc[v_idx, f_idx])
                z_v = est_v / se_v if se_v > 1e-12 else np.nan
                p_v = float(2.0 * (1.0 - stats.norm.cdf(abs(z_v)))) if np.isfinite(z_v) else np.nan
                rows.append({
                    "lhs": f_name,
                    "op": "=~",
                    "rhs": v_name,
                    "est": est_v,
                    "se": se_v,
                    "z": z_v,
                    "pvalue": p_v,
                    "std_all": std_v,
                })
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                est_c = float(self.factor_corr.iloc[i, j])
                se_c = float(self.factor_corr_se.iloc[i, j])
                z_c = est_c / se_c if se_c > 1e-12 else np.nan
                p_c = float(2.0 * (1.0 - stats.norm.cdf(abs(z_c)))) if np.isfinite(z_c) else np.nan
                rows.append({
                    "lhs": self.factor_names[i],
                    "op": "~~",
                    "rhs": self.factor_names[j],
                    "est": est_c,
                    "se": se_c,
                    "z": z_c,
                    "pvalue": p_c,
                    "std_all": est_c,
                })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        chisq_col = "Scaled" if "Scaled" in self.fit_table.columns else "Unscaled"
        chi_v = self.fit_table.loc["chisq", chisq_col]
        df_v = int(self.fit_table.loc["df", chisq_col])
        cfi_col = "Robust" if "Robust" in self.fit_table.columns else chisq_col
        cfi_v = self.fit_table.loc["cfi", cfi_col]
        rmsea_v = self.fit_table.loc["rmsea", cfi_col]
        return (
            f"EFAFit(n_factors={self.n_factors}, estimator='{self.estimator}', rotation='{self.rotation}', "
            f"chisq={chi_v:.3f} (df={df_v}), CFI={cfi_v:.3f}, RMSEA={rmsea_v:.3f})\n"
            f"{self.loadings_std.round(3)}"
        )


def efa(
    data: pd.DataFrame,
    n_factors: int = 1,
    estimator: str = "MLR",
    ordered: bool = False,
    rotation: str = "geomin",
    orthogonal: bool = False,
    geomin_epsilon: float = 0.001,
    oblimin_gamma: float = 0.0,
    promax_kappa: float = 4.0,
    rstarts: int = 30,
    std_ov: bool = True,
    sample_cov: np.ndarray | None = None,
    sample_nobs: int | None = None,
    seed: int = 42,
) -> EFAFit:
    """
    Análise Fatorial Exploratória com paridade numérica com lavaan::efa.
    Suporta ML, MLR, WLSMV (ordinal), ULS e DWLS, rotação GPA multi-start e SEs rotacionados.
    """
    est_upper = estimator.upper()
    if est_upper == "WLSMV":
        ordered = True

    if sample_cov is not None:
        if sample_nobs is None:
            raise ValueError("Informe 'sample_nobs' quando fornecer 'sample_cov'.")
        var_names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"x{i+1}" for i in range(sample_cov.shape[0])]
        n_obs = int(sample_nobs)
        s_mat = np.asarray(sample_cov, dtype=np.float64)
        poly_res = None
        df_clean = None
    else:
        df_clean = data.dropna()
        var_names = list(df_clean.columns)
        n_obs = df_clean.shape[0]
        if ordered:
            poly_res = polychoric(df_clean)
            s_mat = poly_res.corr.to_numpy(dtype=np.float64)
        else:
            poly_res = None
            s_mat = np.cov(df_clean.to_numpy(dtype=np.float64), rowvar=False, bias=True)

    p = len(var_names)
    m = int(n_factors)
    df_model = ((p - m) ** 2 - (p + m)) // 2
    if df_model < 0:
        raise ValueError(f"Modelo subidentificado: n_factors={m} excede o máximo permitido para {p} itens.")

    # Inicialização via PAF sobre a matriz de correlação
    s_sd = np.sqrt(np.diag(s_mat))
    r_obs = s_mat / np.outer(s_sd, s_sd)
    eigvals_corr = np.linalg.eigvalsh(r_obs)[::-1]
    lam_paf, h2_paf = _paf_fit(r_obs, m)
    # Transforma PAF em forma escalonada (triângulo superior direito = 0) via QR
    if m > 1:
        q_ech, _ = np.linalg.qr(lam_paf[:m, :].T)
        lam_init = (lam_paf @ q_ech) * s_sd[:, None]
        for j in range(1, m):
            lam_init[:j, j] = 0.0
    else:
        lam_init = lam_paf * s_sd[:, None]

    # Índices livres da forma escalonada
    lam_free_mask = np.ones((p, m), dtype=bool)
    for j in range(1, m):
        lam_free_mask[:j, j] = False
    lam_rows, lam_cols = np.where(lam_free_mask)
    n_lam_free = len(lam_rows)

    if ordered:
        theta_init = lam_init[lam_rows, lam_cols].copy()
    else:
        psi_res_init = np.maximum(0.05, (1.0 - h2_paf) * (s_sd ** 2))
        theta_init = np.concatenate([lam_init[lam_rows, lam_cols], np.log(psi_res_init)])

    s_torch = torch.tensor(s_mat, dtype=torch.float64)
    sign_s, logdet_s = torch.linalg.slogdet(s_torch)

    if ordered:
        vech_r, vech_c = _vech_col_major_indices(p, strict=True)
        p_star = len(vech_r)
        r_obs_t = s_torch[vech_r, vech_c]
        n_tau_total = sum(len(t) for t in poly_res.thresholds_list)
        gamma_rr = poly_res.gamma[n_tau_total:, n_tau_total:]
        w_rr_diag = 1.0 / np.maximum(1e-12, np.diag(gamma_rr))
        w_rr_t = torch.tensor(w_rr_diag, dtype=torch.float64)

    def unpack_torch(theta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lam_t = torch.zeros((p, m), dtype=torch.float64)
        lam_t[lam_rows, lam_cols] = theta_t[:n_lam_free]
        if ordered:
            res_var = 1.0 - torch.sum(lam_t * lam_t, dim=1)
        else:
            res_var = torch.exp(theta_t[n_lam_free:])
        theta_diag = torch.diag(res_var)
        sigma = lam_t @ lam_t.T + theta_diag
        return sigma, lam_t, res_var

    def objective_half_f(theta_t: torch.Tensor) -> torch.Tensor:
        sigma, _, _ = unpack_torch(theta_t)
        if ordered:
            diff_r = r_obs_t - sigma[vech_r, vech_c]
            return 0.5 * torch.sum(w_rr_t * diff_r * diff_r)
        if est_upper in ("ULS", "DWLS"):
            full_r, full_c = _vech_col_major_indices(p, strict=False)
            diff_v = s_torch[full_r, full_c] - sigma[full_r, full_c]
            return 0.5 * torch.sum(diff_v * diff_v)
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
        options={"gtol": 1e-6, "maxiter": 400},
    )
    th_hat = res_bfgs.x.copy()
    f_curr, g_curr, h_curr = obj_grad_hess(th_hat)
    converged = False
    for _ in range(40):
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
            cand = th_hat + alpha * step
            fc, gc, hc = obj_grad_hess(cand)
            if np.isfinite(fc) and fc <= f_curr + 1e-12:
                th_hat, f_curr, g_curr, h_curr = cand, fc, gc, hc
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    if np.max(np.abs(g_curr)) < 1e-9:
        converged = True

    # Extrai matrizes não-rotacionadas e VCOV não-rotacionado
    with torch.no_grad():
        th_t = torch.tensor(th_hat, dtype=torch.float64)
        sig_t, lam_u_t, res_u_t = unpack_torch(th_t)
        sigma_hat = sig_t.numpy()
        lam_unrot = lam_u_t.numpy()
        res_unrot = res_u_t.numpy()

    if ordered:
        n_wls = float(n_obs - 1)
        def vech_r_fn(tv: torch.Tensor) -> torch.Tensor:
            sig, _, _ = unpack_torch(tv)
            return sig[vech_r, vech_c]
        jac_r = torch.autograd.functional.jacobian(vech_r_fn, torch.tensor(th_hat, dtype=torch.float64)).numpy()
        w_rr = np.diag(w_rr_diag)
        dtwd = jac_r.T @ w_rr @ jac_r
        dtwd_inv = np.linalg.inv(dtwd)
        meat = jac_r.T @ w_rr @ gamma_rr @ w_rr @ jac_r
        vcov_unrot = (dtwd_inv @ meat @ dtwd_inv) / n_wls
    else:
        full_r, full_c = _vech_col_major_indices(p, strict=False)
        d_mat = _duplication_matrix(p, full_r, full_c)
        def vech_s_fn(tv: torch.Tensor) -> torch.Tensor:
            sig, _, _ = unpack_torch(tv)
            return sig[full_r, full_c]
        jac_s = torch.autograd.functional.jacobian(vech_s_fn, torch.tensor(th_hat, dtype=torch.float64)).numpy()
        sigma_inv = np.linalg.inv(sigma_hat)
        w_ml = 0.5 * (d_mat.T @ np.kron(sigma_inv, sigma_inv) @ d_mat)
        a_inv = np.linalg.inv(jac_s.T @ w_ml @ jac_s)
        if est_upper == "MLR" and df_clean is not None:
            y_c = df_clean.to_numpy(dtype=np.float64) - np.mean(df_clean.to_numpy(dtype=np.float64), axis=0)
            s_vech = s_mat[full_r, full_c]
            yi_s = np.einsum("ni,nj->nij", y_c, y_c)[:, full_r, full_c] - s_vech[None, :]
            gamma_mlr = (yi_s.T @ yi_s) / float(n_obs)
            sig_vech = sigma_hat[full_r, full_c]
            yi_sig = np.einsum("ni,nj->nij", y_c, y_c)[:, full_r, full_c] - sig_vech[None, :]
            scores_i = yi_sig @ (w_ml @ jac_s)
            b_meat = (scores_i.T @ scores_i) / float(n_obs)
            h_obs_inv = np.linalg.inv(h_curr)
            vcov_unrot = (h_obs_inv @ b_meat @ h_obs_inv) / float(n_obs)
        else:
            vcov_unrot = a_inv / float(n_obs)

    # Rotação GPA (std_ov e normalização de Kaiser automática para rotações ortogonais, padrão lavaan)
    is_orth_rot = orthogonal or rotation.lower() in ("varimax", "quartimax", "equamax", "parsimax")
    use_kaiser = is_orth_rot

    ov_sd = np.sqrt(np.diag(sigma_hat)) if std_ov else np.ones(p, dtype=np.float64)
    a_std = lam_unrot / ov_sd[:, None]
    row_norms = np.sqrt(np.maximum(1e-12, np.sum(a_std ** 2, axis=1))) if use_kaiser else np.ones(p, dtype=np.float64)
    scale_vec = ov_sd * row_norms
    a_for_rot = lam_unrot / scale_vec[:, None]

    l_std_rot, phi_rot, t_opt = _gpa_rotate(
        a_for_rot,
        method=rotation,
        orthogonal=orthogonal,
        geomin_epsilon=geomin_epsilon,
        oblimin_gamma=oblimin_gamma,
        promax_kappa=promax_kappa,
        rstarts=rstarts,
        seed=seed,
    )
    lam_rot = l_std_rot * scale_vec[:, None]

    # Reflexão (sinais positivos) e ordenação por índice da variável de maior carga (padrão lavaan)
    for j in range(m):
        if np.sum(lam_rot[:, j]) < 0.0:
            lam_rot[:, j] = -lam_rot[:, j]
            phi_rot[j, :] = -phi_rot[j, :]
            phi_rot[:, j] = -phi_rot[:, j]
            t_opt[:, j] = -t_opt[:, j]

    if m > 1:
        lead_idx = [int(np.argmax(np.abs(lam_rot[:, j]))) for j in range(m)]
        order_f = np.argsort(lead_idx)
        lam_rot = lam_rot[:, order_f]
        phi_rot = phi_rot[np.ix_(order_f, order_f)]
        t_opt = t_opt[:, order_f]

    if m > 1 and rotation.lower() not in ("none", "promax"):
        l_std_rot, phi_rot, t_opt = _gpa_rotate(
            a_for_rot,
            method=rotation,
            orthogonal=orthogonal,
            geomin_epsilon=geomin_epsilon,
            oblimin_gamma=oblimin_gamma,
            promax_kappa=promax_kappa,
            rstarts=1,
            init_t=t_opt,
            gpa_tol=1e-12,
            max_iter=5000,
            seed=seed,
        )
        lam_rot = l_std_rot * scale_vec[:, None]

    # Propagação de SEs rotacionados via Teorema da Função Implícita (Jennrich, 1973 / lavaan)
    # Na solução estacionária: L_rot = A @ inv(T).T, onde T satisfaz F(A, T) = 0
    eps_fd = 1e-6

    def _stationarity_residual(a_mat: np.ndarray, t_mat: np.ndarray) -> np.ndarray:
        if orthogonal:
            l_tmp = a_mat @ t_mat
            _, gq = _rotation_criterion_and_grad(
                l_tmp,
                "orthomax" if rotation.lower() in ("varimax", "quartimax", "equamax", "parsimax") else rotation.lower(),
                geomin_epsilon=geomin_epsilon,
                oblimin_gamma=oblimin_gamma,
                orthomax_gamma={"varimax": 1.0, "quartimax": 0.0, "equamax": float(m) / 2.0, "parsimax": float(p * (m - 1)) / max(1.0, float(p + m - 2))}.get(rotation.lower(), 1.0),
            )
            z = l_tmp.T @ gq
            diff_z = z - z.T
            return diff_z[np.tril_indices(m, k=-1)]
        else:
            t_col_norms = np.linalg.norm(t_mat, axis=0)
            l_tmp = a_mat @ np.linalg.inv(t_mat).T
            _, gq = _rotation_criterion_and_grad(
                l_tmp,
                rotation.lower(),
                geomin_epsilon=geomin_epsilon,
                oblimin_gamma=oblimin_gamma,
            )
            z = l_tmp.T @ gq
            phi_tmp = t_mat.T @ t_mat
            tmp = z - phi_tmp * np.diag(z)[None, :]
            off_mask = ~np.eye(m, dtype=bool)
            return np.concatenate([tmp[off_mask], t_col_norms - 1.0])

    def _outputs_from_a_t(a_mat: np.ndarray, t_mat: np.ndarray, ov_sd_vec: np.ndarray) -> np.ndarray:
        if orthogonal:
            l_std_p = a_mat @ t_mat
            phi_p = np.eye(m, dtype=np.float64)
        else:
            l_std_p = a_mat @ np.linalg.inv(t_mat).T
            phi_p = t_mat.T @ t_mat
        lam_r_p = l_std_p * ov_sd_vec[:, None]
        phi_vech = phi_p[np.tril_indices(m, k=-1)]
        return np.concatenate([lam_r_p.ravel(order="F"), phi_vech])

    n_rot_out = p * m + (m * (m - 1) // 2)
    if m == 1 or rotation.lower() in ("none", "promax"):
        def _rotate_from_theta(th_vec: np.ndarray) -> np.ndarray:
            lam_p = np.zeros((p, m), dtype=np.float64)
            lam_p[lam_rows, lam_cols] = th_vec[:n_lam_free]
            res_p = (1.0 - np.sum(lam_p ** 2, axis=1)) if ordered else np.exp(th_vec[n_lam_free:])
            ov_sd_p = np.sqrt(np.sum(lam_p ** 2, axis=1) + res_p) if std_ov else np.ones(p, dtype=np.float64)
            l_std_p, phi_p, _ = _gpa_rotate(
                lam_p / ov_sd_p[:, None], method=rotation, orthogonal=orthogonal,
                geomin_epsilon=geomin_epsilon, oblimin_gamma=oblimin_gamma, promax_kappa=promax_kappa,
                rstarts=1, init_t=t_opt, gpa_tol=1e-11, max_iter=500,
            )
            return np.concatenate([(l_std_p * ov_sd_p[:, None]).ravel(order="F"), phi_p[np.tril_indices(m, k=-1)]])

        jac_rot = np.zeros((n_rot_out, len(th_hat)), dtype=np.float64)
        for k in range(len(th_hat)):
            th_p, th_m = th_hat.copy(), th_hat.copy()
            th_p[k] += eps_fd
            th_m[k] -= eps_fd
            jac_rot[:, k] = (_rotate_from_theta(th_p) - _rotate_from_theta(th_m)) / (2.0 * eps_fd)
    else:
        # Jacobianos em relação a A (p*m) e T (m*m) no ponto ótimo (a_for_rot, t_opt)
        f0 = _stationarity_residual(a_for_rot, t_opt)
        n_cond = len(f0)
        df_da = np.zeros((n_cond, p * m), dtype=np.float64)
        a_flat = a_for_rot.ravel(order="F")
        for idx_a in range(p * m):
            ap, am = a_flat.copy(), a_flat.copy()
            ap[idx_a] += eps_fd
            am[idx_a] -= eps_fd
            df_da[:, idx_a] = (
                _stationarity_residual(ap.reshape((p, m), order="F"), t_opt)
                - _stationarity_residual(am.reshape((p, m), order="F"), t_opt)
            ) / (2.0 * eps_fd)

        df_dt = np.zeros((n_cond, m * m), dtype=np.float64)
        t_flat = t_opt.ravel(order="F")
        for idx_t in range(m * m):
            tp, tm = t_flat.copy(), t_flat.copy()
            tp[idx_t] += eps_fd
            tm[idx_t] -= eps_fd
            df_dt[:, idx_t] = (
                _stationarity_residual(a_for_rot, tp.reshape((m, m), order="F"))
                - _stationarity_residual(a_for_rot, tm.reshape((m, m), order="F"))
            ) / (2.0 * eps_fd)

        # dT / dA = -pinv(dF/dT) @ (dF/dA)
        dt_da = -np.linalg.pinv(df_dt) @ df_da

        # d(out)/dA total = d(out)/dA + d(out)/dT @ dT/dA
        dout_da_part = np.zeros((n_rot_out, p * m), dtype=np.float64)
        for idx_a in range(p * m):
            ap, am = a_flat.copy(), a_flat.copy()
            ap[idx_a] += eps_fd
            am[idx_a] -= eps_fd
            dout_da_part[:, idx_a] = (
                _outputs_from_a_t(ap.reshape((p, m), order="F"), t_opt, ov_sd)
                - _outputs_from_a_t(am.reshape((p, m), order="F"), t_opt, ov_sd)
            ) / (2.0 * eps_fd)

        dout_dt_part = np.zeros((n_rot_out, m * m), dtype=np.float64)
        for idx_t in range(m * m):
            tp, tm = t_flat.copy(), t_flat.copy()
            tp[idx_t] += eps_fd
            tm[idx_t] -= eps_fd
            dout_dt_part[:, idx_t] = (
                _outputs_from_a_t(a_for_rot, tp.reshape((m, m), order="F"), ov_sd)
                - _outputs_from_a_t(a_for_rot, tm.reshape((m, m), order="F"), ov_sd)
            ) / (2.0 * eps_fd)

        dout_da_total = dout_da_part + dout_dt_part @ dt_da

        # Regra da cadeia dA / d(theta) e d(scale_vec) / d(theta)
        def _a_and_sd_from_theta(th_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lam_p = np.zeros((p, m), dtype=np.float64)
            lam_p[lam_rows, lam_cols] = th_vec[:n_lam_free]
            if ordered or not std_ov:
                ov_sd_p = np.ones(p, dtype=np.float64)
            else:
                res_p = np.exp(th_vec[n_lam_free:])
                ov_sd_p = np.sqrt(np.sum(lam_p ** 2, axis=1) + res_p)
            a_std_p = lam_p / ov_sd_p[:, None]
            rn_p = np.sqrt(np.maximum(1e-12, np.sum(a_std_p ** 2, axis=1))) if use_kaiser else np.ones(p, dtype=np.float64)
            sc_p = ov_sd_p * rn_p
            return (lam_p / sc_p[:, None]).ravel(order="F"), sc_p

        da_dtheta = np.zeros((p * m, len(th_hat)), dtype=np.float64)
        dsd_dtheta = np.zeros((p, len(th_hat)), dtype=np.float64)
        for k in range(len(th_hat)):
            th_p, th_m = th_hat.copy(), th_hat.copy()
            th_p[k] += eps_fd
            th_m[k] -= eps_fd
            ap_v, sdp_v = _a_and_sd_from_theta(th_p)
            am_v, sdm_v = _a_and_sd_from_theta(th_m)
            da_dtheta[:, k] = (ap_v - am_v) / (2.0 * eps_fd)
            dsd_dtheta[:, k] = (sdp_v - sdm_v) / (2.0 * eps_fd)

            dout_dsd = np.zeros((n_rot_out, p), dtype=np.float64)
            for i_ov in range(p):
                sdp, sdm = scale_vec.copy(), scale_vec.copy()
                sdp[i_ov] += eps_fd
                sdm[i_ov] -= eps_fd
                dout_dsd[:, i_ov] = (
                    _outputs_from_a_t(a_for_rot, t_opt, sdp)
                    - _outputs_from_a_t(a_for_rot, t_opt, sdm)
                ) / (2.0 * eps_fd)

        dout_dsd = np.zeros((n_rot_out, p), dtype=np.float64)
        for i_ov in range(p):
            sdp, sdm = ov_sd.copy(), ov_sd.copy()
            sdp[i_ov] += eps_fd
            sdm[i_ov] -= eps_fd
            dout_dsd[:, i_ov] = (
                _outputs_from_a_t(a_for_rot, t_opt, sdp)
                - _outputs_from_a_t(a_for_rot, t_opt, sdm)
            ) / (2.0 * eps_fd)

        jac_rot = dout_da_total @ da_dtheta + dout_dsd @ dsd_dtheta

    vcov_rot = jac_rot @ vcov_unrot @ jac_rot.T
    se_rot_vec = np.sqrt(np.maximum(0.0, np.diag(vcov_rot)))

    lam_se_mat = se_rot_vec[: p * m].reshape((p, m), order="F")
    phi_se_mat = np.zeros((m, m), dtype=np.float64)
    if m > 1:
        tril_m_r, tril_m_c = np.tril_indices(m, k=-1)
        phi_se_mat[tril_m_r, tril_m_c] = se_rot_vec[p * m :]
        phi_se_mat[tril_m_c, tril_m_r] = se_rot_vec[p * m :]

    lam_std_mat = lam_rot / np.sqrt(np.diag(sigma_hat))[:, None]
    uniqueness_vec = res_unrot / np.diag(sigma_hat)
    communality_vec = 1.0 - uniqueness_vec

    # Variância explicada por fator (traço de L_std @ Phi @ L_std^T distribuído)
    ss_loadings = np.diag(lam_std_mat.T @ lam_std_mat @ phi_rot)
    prop_var = ss_loadings / float(p)
    cum_var = np.cumsum(prop_var)
    factor_names = [f"f{j + 1}" for j in range(m)]
    var_df = pd.DataFrame(
        {
            "ss_loadings": ss_loadings,
            "proportion_var": prop_var,
            "cumulative_var": cum_var,
            "proportion_explained": ss_loadings / max(1e-12, float(np.sum(ss_loadings))),
        },
        index=factor_names,
    )

    # Índices de ajuste (Unscaled / Scaled / Robust)
    if ordered:
        n_wls = float(n_obs - 1)
        df_base = p_star
        r_obs_v = s_mat[vech_r, vech_c]
        r_hat_v = sigma_hat[vech_r, vech_c]
        t_model = n_wls * float(np.sum(w_rr_diag * ((r_obs_v - r_hat_v) ** 2)))
        t_base = n_wls * float(np.sum(w_rr_diag * (r_obs_v ** 2)))

        u_mat = w_rr - w_rr @ jac_r @ dtwd_inv @ jac_r.T @ w_rr
        m_mat = u_mat @ gamma_rr
        a_scale = np.sqrt(float(df_model) / float(np.trace(m_mat @ m_mat)))
        c_model = 1.0 / a_scale
        b_model = float(df_model) - a_scale * float(np.trace(m_mat))

        m_base = w_rr @ gamma_rr
        a_base = np.sqrt(float(df_base) / float(np.trace(m_base @ m_base)))
        c_base = 1.0 / a_base
        b_base = float(df_base) - a_base * float(np.trace(m_base))

        full_r, full_c = _vech_col_major_indices(p, strict=False)
        d_full = _duplication_matrix(p, full_r, full_c)
        off_idx = np.array([idx_i for idx_i, (r, c) in enumerate(zip(full_r, full_c)) if r != c], dtype=np.int64)
        sigma_inv = np.linalg.inv(sigma_hat)
        v_ml = (0.5 * (d_full.T @ np.kron(sigma_inv, sigma_inv) @ d_full))[np.ix_(off_idx, off_idx)]
        wi_u = np.eye(p_star) - jac_r @ dtwd_inv @ jac_r.T @ w_rr
        c_hat3_m = float(np.trace(wi_u.T @ v_ml @ wi_u @ gamma_rr)) / float(df_model)
        c_hat3_b = float(np.trace(gamma_rr)) / float(df_base)

        _, logdet_sig = np.linalg.slogdet(sigma_hat)
        logdet_r = float(logdet_s.numpy())
        t_m_catml = float(n_obs) * float(logdet_sig + np.trace(np.linalg.solve(sigma_hat, s_mat)) - logdet_r - float(p))
        t_b_catml = float(n_obs) * (-logdet_r)

        fit_table = compute_fit_triad(
            t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat,
            c_model=c_model, c_base=c_base, b_model=b_model, b_base=b_base,
            scaled_shifted=True, n_rmsea=n_wls,
            t_m_catml=t_m_catml, t_b_catml=t_b_catml, c_hat3_m=c_hat3_m, c_hat3_b=c_hat3_b,
        )
    else:
        df_base = p * (p - 1) // 2
        t_model = float(n_obs) * (2.0 * f_curr)
        s_diag = np.diag(np.diag(s_mat))
        _, logdet_d = np.linalg.slogdet(s_diag)
        t_base = float(n_obs) * float(logdet_d - float(logdet_s.numpy()))
        if est_upper == "MLR" and df_clean is not None:
            s_inv = np.linalg.inv(s_mat)
            w_h1 = 0.5 * (d_mat.T @ np.kron(s_inv, s_inv) @ d_mat)
            tr_h1 = float(np.trace(w_h1 @ gamma_mlr))
            tr_h0 = float(np.trace(h_obs_inv @ b_meat))
            c_model = (tr_h1 - tr_h0) / float(df_model)
            s_d_inv = np.diag(1.0 / np.diag(s_mat))
            w_base = 0.5 * (d_mat.T @ np.kron(s_d_inv, s_d_inv) @ d_mat)
            var_idx = np.array([idx_i for idx_i, (r, c) in enumerate(zip(full_r, full_c)) if r == c], dtype=np.int64)
            j_base = np.eye(len(full_r))[:, var_idx]
            tr_b0 = float(np.trace(w_base @ j_base @ np.linalg.inv(j_base.T @ w_base @ j_base) @ j_base.T @ w_base @ gamma_mlr))
            c_base = (tr_h1 - tr_b0) / float(df_base)
            fit_table = compute_fit_triad(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat, c_model=c_model, c_base=c_base)
        else:
            fit_table = compute_fit_triad(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat)

    return EFAFit(
        n_factors=m,
        estimator=est_upper,
        rotation=rotation.lower(),
        ordered=ordered,
        n_obs=n_obs,
        var_names=var_names,
        factor_names=factor_names,
        loadings_unstd=pd.DataFrame(lam_rot, index=var_names, columns=factor_names),
        loadings_se=pd.DataFrame(lam_se_mat, index=var_names, columns=factor_names),
        loadings_std=pd.DataFrame(lam_std_mat, index=var_names, columns=factor_names),
        factor_corr=pd.DataFrame(phi_rot, index=factor_names, columns=factor_names),
        factor_corr_se=pd.DataFrame(phi_se_mat, index=factor_names, columns=factor_names),
        communalities=pd.Series(communality_vec, index=var_names, name="communality"),
        uniquenesses=pd.Series(uniqueness_vec, index=var_names, name="uniqueness"),
        variance_explained=var_df,
        eigenvalues=eigvals_corr,
        fit_table=fit_table,
        thresholds=poly_res.thresholds if poly_res is not None else None,
        polychoric_corr=poly_res.corr if poly_res is not None else None,
        converged=converged,
    )