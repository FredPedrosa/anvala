import numpy as np
import pandas as pd


def bifactor_indices(
    loadings: np.ndarray | pd.DataFrame,
    unidimensional_loadings: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    theta: np.ndarray | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Calcula os índices psicométricos bifatoriais (Dueber, 2021; Rodriguez et al., 2016).
    Paridade numérica exata com o pacote R BifactorIndicesCalculator.
    """
    if isinstance(loadings, pd.DataFrame):
        item_names = list(loadings.index)
        factor_names = list(loadings.columns)
        lam = loadings.to_numpy(dtype=np.float64)
    else:
        lam = np.asarray(loadings, dtype=np.float64)
        p_items, n_factors = lam.shape
        item_names = [f"item_{i+1}" for i in range(p_items)]
        factor_names = ["G"] + [f"S{j}" for j in range(1, n_factors)]

    p, k = lam.shape
    if phi is None:
        phi = np.eye(k, dtype=np.float64)
    else:
        phi = np.asarray(phi, dtype=np.float64)

    common_cov = lam @ phi @ lam.T
    communalities = np.diag(common_cov)

    if theta is None:
        uniqueness = 1.0 - communalities
    else:
        uniqueness = np.asarray(theta, dtype=np.float64)

    sigma = common_cov + np.diag(uniqueness)

    g = lam[:, 0]
    s_mat = lam[:, 1:]

    # --- 1. Índices em Nível de Modelo ---
    ss_g = np.sum(g ** 2)
    ss_s_total = np.sum(s_mat ** 2)
    ss_common_total = ss_g + ss_s_total
    ecv_g = ss_g / ss_common_total

    indicator_s = np.abs(s_mat) > 1e-8
    shared_specific = (indicator_s @ indicator_s.T) > 0
    upper_idx = np.triu_indices(p, k=1)
    contaminated_pairs = np.sum(shared_specific[upper_idx])
    total_pairs = p * (p - 1) / 2.0
    puc = 1.0 - (contaminated_pairs / total_pairs)

    # --- 2. Índices por Fator ---
    var_total = (np.sum(g) ** 2) + np.sum(np.sum(s_mat, axis=0) ** 2) + np.sum(uniqueness)
    omega_total = ((np.sum(g) ** 2) + np.sum(np.sum(s_mat, axis=0) ** 2)) / var_total
    omega_h_g = (np.sum(g) ** 2) / var_total

    h_vals = []
    for j in range(k):
        col = lam[:, j]
        ratio = np.sum((col ** 2) / (1.0 - (col ** 2)))
        h_vals.append(1.0 / (1.0 + (1.0 / ratio)) if ratio > 0 else 0.0)

    sigma_inv_lam_phi = np.linalg.solve(sigma, lam @ phi)
    fd_vals = np.sqrt(np.diag(phi @ lam.T @ sigma_inv_lam_phi))

    factor_rows = [
        {
            "Factor": factor_names[0],
            "ECV_SS": ecv_g,
            "ECV_SG": ecv_g,
            "ECV_GS": ecv_g,
            "Omega": omega_total,
            "OmegaH": omega_h_g,
            "Relative_Omega": omega_h_g / omega_total,
            "H": h_vals[0],
            "FD": fd_vals[0],
        }
    ]

    for idx_s in range(k - 1):
        col_s = s_mat[:, idx_s]
        mask = np.abs(col_s) > 1e-8

        g_sub = g[mask]
        s_sub = col_s[mask]
        u_sub = uniqueness[mask]

        var_sub = (np.sum(g_sub) ** 2) + (np.sum(s_sub) ** 2) + np.sum(u_sub)
        omega_s = ((np.sum(g_sub) ** 2) + (np.sum(s_sub) ** 2)) / var_sub
        omega_hs = (np.sum(s_sub) ** 2) / var_sub

        ss_gk = np.sum(g_sub ** 2)
        ss_sk = np.sum(s_sub ** 2)
        ecv_ss = ss_sk / (ss_gk + ss_sk)
        ecv_sg = ss_sk / ss_common_total
        ecv_gs = ss_gk / (ss_gk + ss_sk)

        factor_rows.append(
            {
                "Factor": factor_names[idx_s + 1],
                "ECV_SS": ecv_ss,
                "ECV_SG": ecv_sg,
                "ECV_GS": ecv_gs,
                "Omega": omega_s,
                "OmegaH": omega_hs,
                "Relative_Omega": omega_hs / omega_s if omega_s > 0 else np.nan,
                "H": h_vals[idx_s + 1],
                "FD": fd_vals[idx_s + 1],
            }
        )

    factor_df = pd.DataFrame(factor_rows).set_index("Factor")

    # --- 3. Índices por Item ---
    iecv = (g ** 2) / (g ** 2 + np.sum(s_mat ** 2, axis=1))
    item_data = {"IECV": iecv}

    arpb = np.nan
    if unidimensional_loadings is not None:
        uni = np.asarray(unidimensional_loadings, dtype=np.float64)
        rpb = np.abs((uni - g) / g)
        item_data["RPB"] = rpb
        arpb = float(np.mean(rpb))

    item_df = pd.DataFrame(item_data, index=item_names)

    return {
        "model": pd.Series(
            {
                "ECV": ecv_g,
                "PUC": puc,
                "Omega": omega_total,
                "OmegaH": omega_h_g,
                "ARPB": arpb,
            }
        ),
        "factors": factor_df,
        "items": item_df,
    }