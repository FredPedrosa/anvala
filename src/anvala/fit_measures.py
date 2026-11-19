import numpy as np
import pandas as pd
from scipy import optimize, stats


def _rmsea_point(t_stat: float, df: float, n_rmsea: float, c_hat: float = 1.0) -> float:
    if df <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, (t_stat / n_rmsea) / df - (c_hat / n_rmsea))))


def _rmsea_ci(t_stat: float, df: float, n_rmsea: float, c_hat: float = 1.0, alpha: float = 0.10) -> tuple[float, float]:
    """Calcula os limites inferior e superior do IC 90% do RMSEA (paridade exata com lav_fit_rmsea_ci)."""
    if df < 1 or not np.isfinite(t_stat) or t_stat <= 0.0:
        return 0.0, 0.0

    p_upper_tail = 1.0 - (alpha / 2.0)
    if stats.chi2.cdf(t_stat, df) <= p_upper_tail:
        ncp_low = 0.0
    else:
        upper_bound = max(t_stat * 2.0, 50.0)
        while stats.ncx2.cdf(t_stat, df, upper_bound) > p_upper_tail:
            upper_bound *= 2.0
        ncp_low = optimize.brentq(
            lambda lmb: stats.ncx2.cdf(t_stat, df, lmb) - p_upper_tail,
            0.0,
            upper_bound,
        )

    p_lower_tail = alpha / 2.0
    if stats.chi2.cdf(t_stat, df) <= p_lower_tail:
        ncp_high = 0.0
    else:
        upper_bound = max(t_stat * 2.0, 50.0)
        while stats.ncx2.cdf(t_stat, df, upper_bound) > p_lower_tail:
            upper_bound *= 2.0
        ncp_high = optimize.brentq(
            lambda lmb: stats.ncx2.cdf(t_stat, df, lmb) - p_lower_tail,
            0.0,
            upper_bound,
        )

    ci_low = float(np.sqrt((c_hat * ncp_low) / (n_rmsea * df)))
    ci_high = float(np.sqrt((c_hat * ncp_high) / (n_rmsea * df)))
    return ci_low, ci_high


def compute_fit_triad(
    t_model: float,
    df_model: int,
    t_base: float,
    df_base: int,
    n_obs: int,
    s_mat: np.ndarray,
    sigma_hat: np.ndarray,
    c_model: float | None = None,
    c_base: float | None = None,
    b_model: float = 0.0,
    b_base: float = 0.0,
    scaled_shifted: bool = False,
    n_rmsea: float | None = None,
    t_m_catml: float | None = None,
    t_b_catml: float | None = None,
    c_hat3_m: float | None = None,
    c_hat3_b: float | None = None,
) -> pd.DataFrame:
    """Gera a tabela Unscaled | Scaled | Robust de índices de ajuste (padrão lavaan 0.7)."""
    n_u = float(n_obs) if n_rmsea is None else float(n_rmsea)

    # 1. Unscaled
    p_unscaled = float(1.0 - stats.chi2.cdf(t_model, df_model)) if df_model > 0 else np.nan
    num_m_u = max(0.0, t_model - df_model)
    num_b_u = max(0.0, t_base - df_base, num_m_u)
    cfi_u = 1.0 - (num_m_u / num_b_u) if num_b_u > 0 else 1.0
    tli_u = ((t_base / df_base) - (t_model / df_model)) / ((t_base / df_base) - 1.0)
    rmsea_u = _rmsea_point(t_model, df_model, n_u, c_hat=1.0)
    rmsea_ci_u = _rmsea_ci(t_model, df_model, n_u, c_hat=1.0)

    # SRMR
    s_sd = np.sqrt(np.diag(s_mat))
    sig_sd = np.sqrt(np.diag(sigma_hat))
    s_cor = s_mat / np.outer(s_sd, s_sd)
    sig_cor = sigma_hat / np.outer(sig_sd, sig_sd)
    p = s_mat.shape[0]
    tril_r, tril_c = np.tril_indices(p)
    srmr = float(np.sqrt(np.mean((s_cor[tril_r, tril_c] - sig_cor[tril_r, tril_c]) ** 2)))

    unscaled_col = {
        "chisq": t_model,
        "df": float(df_model),
        "pvalue": p_unscaled,
        "scaling_factor": 1.0,
        "shift_parameter": 0.0,
        "cfi": cfi_u,
        "tli": tli_u,
        "rmsea": rmsea_u,
        "rmsea_ci_lower": rmsea_ci_u[0],
        "rmsea_ci_upper": rmsea_ci_u[1],
        "srmr": srmr,
    }

    if c_model is None or c_base is None:
        return pd.DataFrame({"Unscaled": unscaled_col})

    # 2. Scaled
    t_m_scaled = (t_model / c_model) + b_model
    t_b_scaled = (t_base / c_base) + b_base
    p_scaled = float(1.0 - stats.chi2.cdf(t_m_scaled, df_model)) if df_model > 0 else np.nan

    num_m_s = max(0.0, t_m_scaled - df_model)
    num_b_s = max(0.0, t_b_scaled - df_base, num_m_s)
    cfi_s = 1.0 - (num_m_s / num_b_s) if num_b_s > 0 else 1.0
    tli_s = ((t_b_scaled / df_base) - (t_m_scaled / df_model)) / ((t_b_scaled / df_base) - 1.0)

    if scaled_shifted:
        rmsea_s = _rmsea_point(t_m_scaled, df_model, n_u, c_hat=1.0)
        rmsea_ci_s_eff = _rmsea_ci(t_m_scaled, df_model, n_u, c_hat=1.0)
    else:
        rmsea_s = _rmsea_point(t_m_scaled, df_model, n_u, c_hat=1.0)
        rmsea_ci_s_eff = _rmsea_ci(t_model, df_model * c_model, n_u, c_hat=1.0)

    scaled_col = {
        "chisq": t_m_scaled,
        "df": float(df_model),
        "pvalue": p_scaled,
        "scaling_factor": c_model,
        "shift_parameter": b_model,
        "cfi": cfi_s,
        "tli": tli_s,
        "rmsea": rmsea_s,
        "rmsea_ci_lower": rmsea_ci_s_eff[0],
        "rmsea_ci_upper": rmsea_ci_s_eff[1],
        "srmr": srmr,
    }

    # 3. Robust (Savalei, 2018 para contínuos; Savalei, 2021 catML para ordinais)
    if t_m_catml is not None and t_b_catml is not None and c_hat3_m is not None and c_hat3_b is not None:
        t_m_r_base = t_m_catml
        t_m_r_scaled = t_m_catml / c_hat3_m
        t_b_r_scaled = t_b_catml / c_hat3_b
        c3_m = c_hat3_m
        c3_b = c_hat3_b
        n_r = float(n_obs)
    else:
        t_m_r_base = t_model
        t_m_r_scaled = t_m_scaled
        t_b_r_scaled = t_b_scaled
        c3_m = c_model
        c3_b = c_base
        n_r = n_u

    num_m_r = c3_m * max(0.0, t_m_r_scaled - df_model)
    num_b_r = max(0.0, c3_b * (t_b_r_scaled - df_base), num_m_r)
    cfi_r = 1.0 - (num_m_r / num_b_r) if num_b_r > 0 else 1.0

    tli_r = 1.0 - ((c3_m * (t_m_r_scaled - df_model) / df_model) / (c3_b * (t_b_r_scaled - df_base) / df_base))
    rmsea_r = _rmsea_point(t_m_r_base, df_model, n_r, c_hat=c3_m)
    rmsea_ci_r = _rmsea_ci(t_m_r_scaled, df_model, n_r, c_hat=c3_m)

    robust_col = {
        "chisq": t_m_scaled,
        "df": float(df_model),
        "pvalue": p_scaled,
        "scaling_factor": c_model,
        "shift_parameter": b_model,
        "cfi": cfi_r,
        "tli": tli_r,
        "rmsea": rmsea_r,
        "rmsea_ci_lower": rmsea_ci_r[0],
        "rmsea_ci_upper": rmsea_ci_r[1],
        "srmr": srmr,
    }

    return pd.DataFrame({"Unscaled": unscaled_col, "Scaled": scaled_col, "Robust": robust_col})