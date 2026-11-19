from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import optimize, special

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * special.erfc(-x / np.sqrt(2.0))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (x ** 2)) / _SQRT_2PI


def _bvn_pdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """Densidade normal bivariada phi_2(h, k; rho), retornando 0 nos limites infinitos."""
    out = np.zeros(np.broadcast_shapes(np.shape(h), np.shape(k)), dtype=np.float64)
    h_b, k_b = np.broadcast_arrays(np.asarray(h, dtype=np.float64), np.asarray(k, dtype=np.float64))
    finite = np.isfinite(h_b) & np.isfinite(k_b)
    if not np.any(finite):
        return out
    hf = h_b[finite]
    kf = k_b[finite]
    one_m_r2 = 1.0 - rho * rho
    z = (hf * hf - 2.0 * rho * hf * kf + kf * kf) / (2.0 * one_m_r2)
    out[finite] = np.exp(-z) / (2.0 * np.pi * np.sqrt(one_m_r2))
    return out


def _bvn_cdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """CDF normal bivariada Phi_2(h, k; rho) via identidade de Plackett + Gauss-Legendre em arcsin(rho)."""
    h_b, k_b = np.broadcast_arrays(np.asarray(h, dtype=np.float64), np.asarray(k, dtype=np.float64))
    out = np.zeros_like(h_b, dtype=np.float64)

    mask_h_inf = np.isposinf(h_b) & np.isfinite(k_b)
    mask_k_inf = np.isposinf(k_b) & np.isfinite(h_b)
    mask_both_inf = np.isposinf(h_b) & np.isposinf(k_b)
    mask_fin = np.isfinite(h_b) & np.isfinite(k_b)

    out[mask_both_inf] = 1.0
    if np.any(mask_h_inf):
        out[mask_h_inf] = _norm_cdf(k_b[mask_h_inf])
    if np.any(mask_k_inf):
        out[mask_k_inf] = _norm_cdf(h_b[mask_k_inf])

    if np.any(mask_fin):
        hf = h_b[mask_fin]
        kf = k_b[mask_fin]
        base = _norm_cdf(hf) * _norm_cdf(kf)
        if abs(rho) < 1e-14:
            out[mask_fin] = base
        else:
            theta_max = np.arcsin(np.clip(rho, -0.999999, 0.999999))
            half_t = 0.5 * theta_max
            thetas = half_t * (_GL_X + 1.0)
            sin_t = np.sin(thetas)[None, :]
            cos2_t = (np.cos(thetas) ** 2)[None, :]

            h2_k2 = (hf * hf + kf * kf)[:, None]
            hk2 = (2.0 * hf * kf)[:, None]
            exponent = -(h2_k2 - hk2 * sin_t) / (2.0 * cos2_t)
            integral = (half_t / (2.0 * np.pi)) * (np.exp(exponent) @ _GL_W)
            out[mask_fin] = np.clip(base + integral, 0.0, 1.0)

    return out


def _cell_probs(tau_i: np.ndarray, tau_j: np.ndarray, rho: float) -> np.ndarray:
    """Calcula a matriz (C_i x C_j) de probabilidades de célula sob normal bivariada."""
    ext_i = np.concatenate(([-np.inf], tau_i, [np.inf]))
    ext_j = np.concatenate(([-np.inf], tau_j, [np.inf]))
    cdf_grid = _bvn_cdf(ext_i[:, None], ext_j[None, :], rho)
    probs = cdf_grid[1:, 1:] - cdf_grid[:-1, 1:] - cdf_grid[1:, :-1] + cdf_grid[:-1, :-1]
    return np.maximum(probs, 1e-15)


def _dprobs_drho(tau_i: np.ndarray, tau_j: np.ndarray, rho: float) -> np.ndarray:
    """Derivada exata d(pi_ab)/d(rho)."""
    ext_i = np.concatenate(([-np.inf], tau_i, [np.inf]))
    ext_j = np.concatenate(([-np.inf], tau_j, [np.inf]))
    pdf_grid = _bvn_pdf(ext_i[:, None], ext_j[None, :], rho)
    return pdf_grid[1:, 1:] - pdf_grid[:-1, 1:] - pdf_grid[1:, :-1] + pdf_grid[:-1, :-1]


def _dprobs_dtau_i(tau_i: np.ndarray, tau_j: np.ndarray, rho: float) -> np.ndarray:
    """Derivada exata d(pi_ab)/d(tau_{i,c}), shape (n_tau_i, C_i, C_j)."""
    n_ti = len(tau_i)
    c_i = n_ti + 1
    c_j = len(tau_j) + 1
    out = np.zeros((n_ti, c_i, c_j), dtype=np.float64)
    if n_ti == 0:
        return out

    sqrt_1mr2 = np.sqrt(1.0 - rho * rho)
    ext_j = np.concatenate(([-np.inf], tau_j, [np.inf]))

    for c in range(n_ti):
        t_c = tau_i[c]
        phi_tc = float(_norm_pdf(np.array(t_c)))
        d_vec = np.zeros(c_j + 1, dtype=np.float64)
        d_vec[-1] = phi_tc
        fin_j = np.isfinite(ext_j)
        d_vec[fin_j] = phi_tc * _norm_cdf((ext_j[fin_j] - rho * t_c) / sqrt_1mr2)
        diff_j = d_vec[1:] - d_vec[:-1]
        out[c, c, :] = diff_j
        out[c, c + 1, :] = -diff_j

    return out


def _dprobs_dtau_j(tau_i: np.ndarray, tau_j: np.ndarray, rho: float) -> np.ndarray:
    """Derivada exata d(pi_ab)/d(tau_{j,d}), shape (n_tau_j, C_i, C_j)."""
    n_tj = len(tau_j)
    c_i = len(tau_i) + 1
    c_j = n_tj + 1
    out = np.zeros((n_tj, c_i, c_j), dtype=np.float64)
    if n_tj == 0:
        return out

    sqrt_1mr2 = np.sqrt(1.0 - rho * rho)
    ext_i = np.concatenate(([-np.inf], tau_i, [np.inf]))

    for d in range(n_tj):
        t_d = tau_j[d]
        phi_td = float(_norm_pdf(np.array(t_d)))
        d_vec = np.zeros(c_i + 1, dtype=np.float64)
        d_vec[-1] = phi_td
        fin_i = np.isfinite(ext_i)
        d_vec[fin_i] = phi_td * _norm_cdf((ext_i[fin_i] - rho * t_d) / sqrt_1mr2)
        diff_i = d_vec[1:] - d_vec[:-1]
        out[d, :, d] = diff_i
        out[d, :, d + 1] = -diff_i

    return out


@dataclass
class PolychoricResult:
    corr: pd.DataFrame
    thresholds: pd.DataFrame
    thresholds_list: list[np.ndarray]
    gamma: np.ndarray
    n_obs: int

    def __repr__(self) -> str:
        return f"PolychoricResult(n_obs={self.n_obs}, n_vars={self.corr.shape[0]})\n{self.corr.round(4)}"


def polychoric(data: pd.DataFrame) -> PolychoricResult:
    """
    Estima limiares marginais, matriz de correlação policórica em 2 passos (Olsson, 1979)
    e matriz de covariância assintótica Gamma_poly (paridade exata com lavaan:::muthen1984).
    """
    df_clean = data.dropna()
    var_names = list(df_clean.columns)
    n_obs, p = df_clean.shape

    codes_mat = np.zeros((n_obs, p), dtype=np.int64)
    n_cats = []
    thresholds_list: list[np.ndarray] = []
    thresh_rows = []

    for j, col in enumerate(var_names):
        _, inv = np.unique(df_clean[col].to_numpy(), return_inverse=True)
        codes_mat[:, j] = inv
        c_j = int(inv.max()) + 1
        if c_j < 2:
            raise ValueError(f"A variável ordinal '{col}' possui menos de 2 categorias observadas.")
        n_cats.append(c_j)
        counts = np.bincount(inv, minlength=c_j)
        cum_props = np.cumsum(counts[:-1]) / float(n_obs)
        taus = special.ndtri(cum_props)
        thresholds_list.append(taus)
        for t_idx, t_val in enumerate(taus):
            thresh_rows.append({"item": col, "threshold": f"t{t_idx + 1}", "est": float(t_val)})

    thresholds_df = pd.DataFrame(thresh_rows)

    # Passo 1 (muthen1984): matriz de escores marginais dos limiares sc_th (N x n_tau_total)
    tau_offsets = np.cumsum([0] + [c - 1 for c in n_cats])
    n_tau_total = int(tau_offsets[-1])
    sc_th = np.zeros((n_obs, n_tau_total), dtype=np.float64)

    for j in range(p):
        taus = thresholds_list[j]
        off = tau_offsets[j]
        ext_t = np.concatenate(([-np.inf], taus, [np.inf]))
        cdf_t = _norm_cdf(ext_t)
        cat_probs = np.maximum(cdf_t[1:] - cdf_t[:-1], 1e-15)
        y_j = codes_mat[:, j]

        for c_idx, t_val in enumerate(taus):
            phi_tc = float(_norm_pdf(np.array(t_val)))
            sc_th[y_j == c_idx, off + c_idx] = phi_tc / cat_probs[c_idx]
            sc_th[y_j == (c_idx + 1), off + c_idx] = -phi_tc / cat_probs[c_idx + 1]

    # Passo 2 (muthen1984): correlações policóricas e blocos A21, A22, sc_cor
    r_mat = np.eye(p, dtype=np.float64)
    n_pairs = p * (p - 1) // 2
    sc_cor = np.zeros((n_obs, n_pairs), dtype=np.float64)
    a21 = np.zeros((n_pairs, n_tau_total), dtype=np.float64)
    a22_diag = np.zeros(n_pairs, dtype=np.float64)

    pair_idx = 0
    for c in range(p):
        for r in range(c + 1, p):
            c_i, c_j = n_cats[r], n_cats[c]
            tau_i, tau_j = thresholds_list[r], thresholds_list[c]
            y_i, y_j = codes_mat[:, r], codes_mat[:, c]

            table = np.bincount(y_i * c_j + y_j, minlength=c_i * c_j).reshape(c_i, c_j).astype(np.float64)

            def neg_loglik(rho_val: float) -> float:
                probs = _cell_probs(tau_i, tau_j, rho_val)
                return -float(np.sum(table * np.log(probs)))

            opt = optimize.minimize_scalar(
                neg_loglik, bounds=(-0.9995, 0.9995), method="bounded", options={"xatol": 1e-11}
            )
            rho_hat = float(opt.x)
            r_mat[r, c] = rho_hat
            r_mat[c, r] = rho_hat

            probs = _cell_probs(tau_i, tau_j, rho_hat)
            dp_drho = _dprobs_drho(tau_i, tau_j, rho_hat)
            dp_dti = _dprobs_dtau_i(tau_i, tau_j, rho_hat)
            dp_dtj = _dprobs_dtau_j(tau_i, tau_j, rho_hat)

            dx_rho_cell = dp_drho / probs
            dx_rho_sub = dx_rho_cell[y_i, y_j]
            sc_cor[:, pair_idx] = dx_rho_sub
            a22_diag[pair_idx] = float(np.sum(dx_rho_sub * dx_rho_sub))

            n_ti = len(tau_i)
            if n_ti > 0:
                dx_ti_sub = np.column_stack([(dp_dti[m] / probs)[y_i, y_j] for m in range(n_ti)])
                a21[pair_idx, tau_offsets[r]:tau_offsets[r + 1]] = dx_rho_sub @ dx_ti_sub

            n_tj = len(tau_j)
            if n_tj > 0:
                dx_tj_sub = np.column_stack([(dp_dtj[m] / probs)[y_i, y_j] for m in range(n_tj)])
                a21[pair_idx, tau_offsets[c]:tau_offsets[c + 1]] = dx_rho_sub @ dx_tj_sub

            pair_idx += 1

    # Sanduíche muthen1984: wls_w = b_inv @ (sc^T @ sc) @ b_inv^T ; Gamma = N * wls_w
    sc = np.hstack((sc_th, sc_cor))
    inner = sc.T @ sc

    a11 = np.zeros((n_tau_total, n_tau_total), dtype=np.float64)
    for j in range(p):
        s_idx, e_idx = tau_offsets[j], tau_offsets[j + 1]
        a11[s_idx:e_idx, s_idx:e_idx] = inner[s_idx:e_idx, s_idx:e_idx]

    a11_inv = np.linalg.inv(a11)
    a22_inv = np.diag(1.0 / a22_diag)
    a21_inv = -a22_inv @ a21 @ a11_inv

    b_inv = np.block([
        [a11_inv, np.zeros((n_tau_total, n_pairs), dtype=np.float64)],
        [a21_inv, a22_inv],
    ])
    wls_w = b_inv @ inner @ b_inv.T
    gamma_poly = float(n_obs) * wls_w

    corr_df = pd.DataFrame(r_mat, index=var_names, columns=var_names)
    return PolychoricResult(
        corr=corr_df,
        thresholds=thresholds_df,
        thresholds_list=thresholds_list,
        gamma=gamma_poly,
        n_obs=n_obs,
    )