from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from scipy import optimize, stats
import torch

from anvala.bifactor import bifactor_indices
from anvala.fit_measures import compute_fit_triad
from anvala.parser import ModelSpec, parse_lavaan_syntax
from anvala.polychoric import PolychoricResult, polychoric


def _vech_col_major_indices(p: int, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Índices do triângulo inferior em ordem column-major (vech ou vechs do lavaan)."""
    rows, cols = [], []
    for c in range(p):
        start_r = c + 1 if strict else c
        for r in range(start_r, p):
            rows.append(r)
            cols.append(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def _duplication_matrix(p: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Constrói a matriz de duplicação D_p tal que vec(A) = D_p @ vech(A) em column-major."""
    p_star = len(rows)
    d_mat = np.zeros((p * p, p_star), dtype=np.float64)
    for m, (r, c) in enumerate(zip(rows, cols)):
        d_mat[c * p + r, m] = 1.0
        if r != c:
            d_mat[r * p + c, m] = 1.0
    return d_mat


def _compute_gamma_continuous(x_centered: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Calcula a matriz de covariância assintótica de 4ª ordem Gamma (NACOV do lavaan)."""
    n = x_centered.shape[0]
    outer_prods = x_centered[:, rows] * x_centered[:, cols]
    w_bar = np.mean(outer_prods, axis=0, keepdims=True)
    w_diff = outer_prods - w_bar
    return (w_diff.T @ w_diff) / float(n)


@dataclass
class AnvalaFit:
    spec: ModelSpec
    estimator: str
    n_obs: int
    converged: bool
    theta_hat: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    lambda_hat: np.ndarray
    beta_hat: np.ndarray
    psi_hat: np.ndarray
    theta_res_hat: np.ndarray
    sigma_hat: np.ndarray
    sample_cov: np.ndarray
    fit_table: pd.DataFrame
    data_raw: pd.DataFrame | None = None
    poly_result: PolychoricResult | None = None

    def fit_measures(self) -> pd.DataFrame:
        """Retorna a tabela Unscaled | Scaled | Robust de índices de ajuste."""
        return self.fit_table

    def polychoric(self) -> pd.DataFrame:
        """Retorna a matriz de correlação policórica amostral para fácil inspeção."""
        if self.poly_result is None:
            raise ValueError("Este modelo não foi estimado com variáveis ordinais. Use av.polychoric(data) ou ordered=True.")
        return self.poly_result.corr

    def residuals(self) -> pd.DataFrame:
        """Retorna a matriz de resíduos (matriz amostral - matriz implicada pelo modelo)."""
        res_mat = self.sample_cov - self.sigma_hat
        return pd.DataFrame(res_mat, index=self.spec.ov_names, columns=self.spec.ov_names)

    def inspect_sampstat(self) -> dict[str, pd.DataFrame]:
        """Equivalente a lavInspect(fit, 'sampstat')."""
        samp_df = pd.DataFrame(self.sample_cov, index=self.spec.ov_names, columns=self.spec.ov_names)
        if self.poly_result is not None:
            return {"cor": samp_df, "thresholds": self.poly_result.thresholds}
        return {"cov": samp_df}

    def parameter_estimates(self) -> pd.DataFrame:
        """Retorna a tabela estilo parameterEstimates() do lavaan com est, se, z, pvalue e std.all."""
        ov_sd = np.sqrt(np.diag(self.sigma_hat))
        k = len(self.spec.lv_names)
        if k > 0:
            i_minus_b_inv = np.linalg.inv(np.eye(k) - self.beta_hat)
            lv_cov = i_minus_b_inv @ self.psi_hat @ i_minus_b_inv.T
            lv_sd = np.sqrt(np.diag(lv_cov))
        else:
            lv_sd = np.array([])

        ov_idx = {name: i for i, name in enumerate(self.spec.ov_names)}
        lv_idx = {name: i for i, name in enumerate(self.spec.lv_names)}

        rows = []
        for p_spec in self.spec.params:
            if self.spec.ordered and p_spec.op == "~~" and p_spec.lhs == p_spec.rhs and p_spec.lhs in ov_idx:
                i_ov = ov_idx[p_spec.lhs]
                est = float(self.theta_res_hat[i_ov, i_ov])
            else:
                est = float(p_spec.value) if p_spec.fixed else float(self.theta_hat[p_spec.free_idx])

            se_val = 0.0 if p_spec.fixed else float(self.se[p_spec.free_idx])
            z_val = est / se_val if se_val > 0 else np.nan
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_val)))) if se_val > 0 else np.nan

            if p_spec.op == "=~":
                i_ov, j_lv = ov_idx[p_spec.rhs], lv_idx[p_spec.lhs]
                std_all = est * lv_sd[j_lv] / ov_sd[i_ov]
            elif p_spec.op == "~":
                i_to, j_from = lv_idx[p_spec.lhs], lv_idx[p_spec.rhs]
                std_all = est * lv_sd[j_from] / lv_sd[i_to]
            elif p_spec.op == "~~":
                if p_spec.lhs in ov_idx:
                    i1, i2 = ov_idx[p_spec.lhs], ov_idx[p_spec.rhs]
                    std_all = est / (ov_sd[i1] * ov_sd[i2])
                else:
                    j1, j2 = lv_idx[p_spec.lhs], lv_idx[p_spec.rhs]
                    std_all = est / (lv_sd[j1] * lv_sd[j2])
            else:
                std_all = np.nan

            rows.append(
                {
                    "lhs": p_spec.lhs,
                    "op": p_spec.op,
                    "rhs": p_spec.rhs,
                    "label": p_spec.label,
                    "est": est,
                    "se": se_val,
                    "z": z_val,
                    "pvalue": p_val,
                    "std_all": float(std_all),
                }
            )

        if self.poly_result is not None:
            for _, t_row in self.poly_result.thresholds.iterrows():
                rows.append(
                    {
                        "lhs": t_row["item"],
                        "op": "|",
                        "rhs": t_row["threshold"],
                        "label": "",
                        "est": float(t_row["est"]),
                        "se": np.nan,
                        "z": np.nan,
                        "pvalue": np.nan,
                        "std_all": float(t_row["est"]),
                    }
                )

        return pd.DataFrame(rows)

    def inspect_std(self) -> dict[str, pd.DataFrame]:
        """Equivalente a lavInspect(fit, 'std')."""
        ov_sd = np.sqrt(np.diag(self.sigma_hat))
        k = len(self.spec.lv_names)
        i_minus_b_inv = np.linalg.inv(np.eye(k) - self.beta_hat)
        lv_cov = i_minus_b_inv @ self.psi_hat @ i_minus_b_inv.T
        lv_sd = np.sqrt(np.diag(lv_cov))

        lam_std = (self.lambda_hat * lv_sd[None, :]) / ov_sd[:, None]
        psi_std = self.psi_hat / np.outer(lv_sd, lv_sd)
        theta_std = self.theta_res_hat / np.outer(ov_sd, ov_sd)

        return {
            "lambda": pd.DataFrame(lam_std, index=self.spec.ov_names, columns=self.spec.lv_names),
            "psi": pd.DataFrame(psi_std, index=self.spec.lv_names, columns=self.spec.lv_names),
            "theta": pd.DataFrame(theta_std, index=self.spec.ov_names, columns=self.spec.ov_names),
        }

    def bifactor_indices(self, compute_arpb: bool = True) -> dict[str, pd.DataFrame | pd.Series]:
        """Calcula todos os índices de Dueber (2021) diretamente a partir do modelo ajustado."""
        std = self.inspect_std()
        uni_loadings = None
        if compute_arpb and self.data_raw is not None:
            items_str = " + ".join(self.spec.ov_names)
            uni_fit = cfa(
                f"_U =~ {items_str}",
                data=self.data_raw,
                estimator=self.estimator,
                std_lv=True,
                ordered=self.spec.ordered,
            )
            uni_loadings = uni_fit.inspect_std()["lambda"].to_numpy().ravel()

        return bifactor_indices(
            loadings=std["lambda"],
            unidimensional_loadings=uni_loadings,
            phi=std["psi"].to_numpy(),
            theta=np.diag(std["theta"].to_numpy()),
        )


def cfa(
    model: str,
    data: pd.DataFrame,
    estimator: str | None = None,
    std_lv: bool = False,
    orthogonal: bool = False,
    ordered: bool = False,
) -> AnvalaFit:
    """
    Ajusta um modelo CFA/SEM com paridade numérica ao lavaan.

    - Dados contínuos: estimator='ML', 'MLR' ou 'ULS'
    - Dados ordinais (ordered=True): estimator='WLSMV' (padrão) ou 'ULSMV'
    """
    if estimator is None:
        est_upper = "WLSMV" if ordered else "ML"
    else:
        est_upper = estimator.upper()
        if est_upper in ("WLSMV", "ULSMV", "DWLS"):
            ordered = True

    if est_upper not in ("ML", "MLR", "ULS", "WLSMV", "ULSMV", "DWLS"):
        raise ValueError(f"Estimador '{estimator}' não suportado.")

    clean_syntax = "\n".join(line.split("#", 1)[0] for line in model.splitlines())
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", clean_syntax)
    all_cols_set = set(data.columns)
    used_ovs = []
    for tok in tokens:
        if tok in all_cols_set and tok not in used_ovs:
            used_ovs.append(tok)

    df_clean = data[used_ovs].dropna()
    n_obs = df_clean.shape[0]
    p = len(used_ovs)

    poly_res: PolychoricResult | None = None
    if ordered:
        poly_res = polychoric(df_clean)
        s_mat = poly_res.corr.to_numpy(dtype=np.float64)
        sample_var = np.ones(p, dtype=np.float64)
    else:
        x_mat = df_clean.to_numpy(dtype=np.float64)
        x_centered = x_mat - np.mean(x_mat, axis=0, keepdims=True)
        s_mat = (x_centered.T @ x_centered) / float(n_obs)
        sample_var = np.diag(s_mat)

    spec = parse_lavaan_syntax(
        model,
        observed_vars=used_ovs,
        std_lv=std_lv,
        orthogonal=orthogonal,
        sample_var=sample_var,
        ordered=ordered,
    )

    k = len(spec.lv_names)
    vech_r, vech_c = _vech_col_major_indices(p, strict=ordered)
    p_star = len(vech_r)

    s_torch = torch.tensor(s_mat, dtype=torch.float64)
    vech_r_t = torch.tensor(vech_r, dtype=torch.long)
    vech_c_t = torch.tensor(vech_c, dtype=torch.long)
    s_vech_t = s_torch[vech_r_t, vech_c_t]
    logdet_s = torch.linalg.slogdet(s_torch)[1]
    eye_k = torch.eye(k, dtype=torch.float64)

    # Pesos diagonais para estimadores WLS/DWLS/ULS ordinais
    if ordered:
        n_tau = poly_res.thresholds.shape[0]
        gamma_rr = poly_res.gamma[n_tau:, n_tau:]
        if est_upper in ("WLSMV", "DWLS"):
            w_rr_diag = 1.0 / np.diag(gamma_rr)
        else:  # ULSMV
            w_rr_diag = np.ones(p_star, dtype=np.float64)
        w_rr_t = torch.tensor(w_rr_diag, dtype=torch.float64)
    else:
        w_rr_diag = None
        w_rr_t = None

    def build_matrices(theta_t: torch.Tensor):
        lam = torch.zeros((p, k), dtype=torch.float64)
        beta = torch.zeros((k, k), dtype=torch.float64)
        psi = torch.zeros((k, k), dtype=torch.float64)
        theta_res = torch.zeros((p, p), dtype=torch.float64)

        for r, c, f_idx, val in spec.lambda_map:
            lam[r, c] = theta_t[f_idx] if f_idx >= 0 else val
        for r, c, f_idx, val in spec.beta_map:
            beta[r, c] = theta_t[f_idx] if f_idx >= 0 else val
        for r, c, f_idx, val in spec.psi_map:
            v = theta_t[f_idx] if f_idx >= 0 else val
            psi[r, c] = v
            psi[c, r] = v
        for r, c, f_idx, val in spec.theta_map:
            v = theta_t[f_idx] if f_idx >= 0 else val
            theta_res[r, c] = v
            theta_res[c, r] = v

        if k > 0:
            inv_ib = torch.linalg.solve(eye_k - beta, eye_k)
            common = lam @ inv_ib @ psi @ inv_ib.T @ lam.T
        else:
            common = torch.zeros((p, p), dtype=torch.float64)

        if ordered:
            # Parametrização delta: diag(Sigma*) = 1 => diag(Theta) = 1 - diag(common)
            diag_theta = 1.0 - torch.diag(common)
            theta_res = theta_res - torch.diag(torch.diag(theta_res)) + torch.diag(diag_theta)

        sigma = common + theta_res
        return lam, beta, psi, theta_res, sigma

    def sigma_vech_fn(theta_t: torch.Tensor) -> torch.Tensor:
        _, _, _, _, sigma = build_matrices(theta_t)
        return sigma[vech_r_t, vech_c_t]

    def objective_half_f(theta_t: torch.Tensor) -> torch.Tensor:
        _, _, _, _, sigma = build_matrices(theta_t)
        if est_upper in ("ML", "MLR"):
            sign, logdet_sig = torch.linalg.slogdet(sigma)
            if sign <= 0:
                return torch.tensor(1e12, dtype=torch.float64)
            tr_term = torch.trace(torch.linalg.solve(sigma, s_torch))
            f_ml = logdet_sig + tr_term - logdet_s - float(p)
            return 0.5 * f_ml
        elif ordered:
            diff = s_vech_t - sigma[vech_r_t, vech_c_t]
            return 0.5 * torch.sum(w_rr_t * (diff ** 2))
        else:  # ULS contínuo
            diff = s_vech_t - sigma[vech_r_t, vech_c_t]
            return 0.5 * torch.sum(diff ** 2)

    def scipy_obj_and_grad(theta_np: np.ndarray):
        t_var = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        loss = objective_half_f(t_var)
        if not torch.isfinite(loss) or not loss.requires_grad:
            return 1e12, np.zeros_like(theta_np)
        loss.backward()
        return float(loss.detach().numpy()), t_var.grad.detach().numpy().copy()

    res = optimize.minimize(
        scipy_obj_and_grad,
        spec.theta_init,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1000},
    )

    theta_hat = res.x
    theta_hat_t = torch.tensor(theta_hat, dtype=torch.float64)
    lam_t, beta_t, psi_t, theta_res_t, sigma_t = build_matrices(theta_hat_t)

    lambda_hat = lam_t.numpy()
    beta_hat = beta_t.numpy()
    psi_hat = psi_t.numpy()
    theta_res_hat = theta_res_t.numpy()
    sigma_hat = sigma_t.numpy()

    delta_mat = torch.autograd.functional.jacobian(sigma_vech_fn, theta_hat_t).numpy()
    h_obs = torch.autograd.functional.hessian(objective_half_f, theta_hat_t).numpy()

    if ordered:
        df_model = p_star - spec.n_free
        df_base = p_star
        n_wls = float(n_obs - 1)

        r_obs_vech = s_mat[vech_r, vech_c]
        r_hat_vech = sigma_hat[vech_r, vech_c]
        diff_r = r_obs_vech - r_hat_vech

        t_model = n_wls * float(np.sum(w_rr_diag * (diff_r ** 2)))
        t_base = n_wls * float(np.sum(w_rr_diag * (r_obs_vech ** 2)))

        w_rr = np.diag(w_rr_diag)
        dtwd = delta_mat.T @ w_rr @ delta_mat
        dtwd_inv = np.linalg.inv(dtwd)
        meat = delta_mat.T @ w_rr @ gamma_rr @ w_rr @ delta_mat
        vcov = (dtwd_inv @ meat @ dtwd_inv) / n_wls
        se = np.sqrt(np.maximum(0.0, np.diag(vcov)))

        # Escalonamento e deslocamento de Asparouhov & Muthén (2010) / lavaan WLSMV
        u_mat = w_rr - w_rr @ delta_mat @ dtwd_inv @ delta_mat.T @ w_rr
        m_mat = u_mat @ gamma_rr
        tr_m = float(np.trace(m_mat))
        tr_m2 = float(np.trace(m_mat @ m_mat))
        a_scale = np.sqrt(float(df_model) / tr_m2)
        c_model = 1.0 / a_scale
        b_model = float(df_model) - a_scale * tr_m

        m_base = w_rr @ gamma_rr
        tr_mb = float(np.trace(m_base))
        tr_mb2 = float(np.trace(m_base @ m_base))
        a_base = np.sqrt(float(df_base) / tr_mb2)
        c_base = 1.0 / a_base
        b_base = float(df_base) - a_base * tr_mb

        # Categorical ML (Savalei, 2021 — lav_fit_catml_dwls) para CFI/TLI/RMSEA Robust
        full_r, full_c = _vech_col_major_indices(p, strict=False)
        d_full = _duplication_matrix(p, full_r, full_c)
        off_idx = np.array([m for m, (r, c) in enumerate(zip(full_r, full_c)) if r != c], dtype=np.int64)
        sigma_inv = np.linalg.inv(sigma_hat)
        v_full = 0.5 * (d_full.T @ np.kron(sigma_inv, sigma_inv) @ d_full)
        v_ml = v_full[np.ix_(off_idx, off_idx)]

        wi_u = np.eye(p_star) - delta_mat @ dtwd_inv @ delta_mat.T @ w_rr
        ks = float(np.trace(wi_u.T @ v_ml @ wi_u @ gamma_rr))
        c_hat3_m = ks / float(df_model)
        c_hat3_b = float(np.trace(gamma_rr)) / float(df_base)

        sign_sig, logdet_sig = np.linalg.slogdet(sigma_hat)
        logdet_r = float(logdet_s.numpy())
        f_catml = float(logdet_sig + np.trace(np.linalg.solve(sigma_hat, s_mat)) - logdet_r - float(p))
        t_m_catml = float(n_obs) * f_catml
        t_b_catml = float(n_obs) * (-logdet_r)

        fit_table = compute_fit_triad(
            t_model,
            df_model,
            t_base,
            df_base,
            n_obs,
            s_mat,
            sigma_hat,
            c_model=c_model,
            c_base=c_base,
            b_model=b_model,
            b_base=b_base,
            scaled_shifted=True,
            n_rmsea=n_wls,
            t_m_catml=t_m_catml,
            t_b_catml=t_b_catml,
            c_hat3_m=c_hat3_m,
            c_hat3_b=c_hat3_b,
        )

    else:
        d_mat = _duplication_matrix(p, vech_r, vech_c)
        sigma_inv = np.linalg.inv(sigma_hat)
        v_sigma = 0.5 * (d_mat.T @ np.kron(sigma_inv, sigma_inv) @ d_mat)

        df_model = p_star - spec.n_free
        df_base = p_star - p

        if est_upper == "ML":
            t_model = float(n_obs) * (2.0 * float(res.fun))
            t_base = float(n_obs) * (np.sum(np.log(np.diag(s_mat))) - float(logdet_s.numpy()))
            i_exp = delta_mat.T @ v_sigma @ delta_mat
            vcov = np.linalg.inv(i_exp) / float(n_obs)
            se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
            fit_table = compute_fit_triad(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat)

        elif est_upper == "MLR":
            t_model = float(n_obs) * (2.0 * float(res.fun))
            t_base = float(n_obs) * (np.sum(np.log(np.diag(s_mat))) - float(logdet_s.numpy()))
            gamma = _compute_gamma_continuous(x_centered, vech_r, vech_c)

            h_inv = np.linalg.inv(h_obs)
            b_meat = delta_mat.T @ v_sigma @ gamma @ v_sigma @ delta_mat
            vcov = (h_inv @ b_meat @ h_inv) / float(n_obs)
            se = np.sqrt(np.maximum(0.0, np.diag(vcov)))

            s_inv = np.linalg.inv(s_mat)
            v_s = 0.5 * (d_mat.T @ np.kron(s_inv, s_inv) @ d_mat)
            tr_vs_gamma = np.trace(v_s @ gamma)
            c_model = 1.0 if df_model <= 0 else float((tr_vs_gamma - np.trace(h_inv @ b_meat)) / float(df_model))

            diag_idx = np.array([m for m, (r, c) in enumerate(zip(vech_r, vech_c)) if r == c])
            sigma_b_inv = np.diag(1.0 / np.diag(s_mat))
            v_b = 0.5 * (d_mat.T @ np.kron(sigma_b_inv, sigma_b_inv) @ d_mat)
            delta_b = np.zeros((p_star, p), dtype=np.float64)
            delta_b[diag_idx, np.arange(p)] = 1.0
            h_b = delta_b.T @ v_b @ delta_b
            h_b_inv = np.linalg.inv(h_b)
            b_b_meat = delta_b.T @ v_b @ gamma @ v_b @ delta_b
            c_base = float((tr_vs_gamma - np.trace(h_b_inv @ b_b_meat)) / float(df_base))

            fit_table = compute_fit_triad(
                t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat, c_model=c_model, c_base=c_base
            )

        else:  # ULS contínuo
            diff_vech = s_mat[vech_r, vech_c] - sigma_hat[vech_r, vech_c]
            t_model = float(n_obs) * float(np.sum(diff_vech ** 2))
            s_off = s_mat.copy()
            np.fill_diagonal(s_off, 0.0)
            t_base = float(n_obs) * float(np.sum(s_off[vech_r, vech_c] ** 2))
            i_uls = delta_mat.T @ delta_mat
            vcov = np.linalg.inv(i_uls) / float(n_obs)
            se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
            fit_table = compute_fit_triad(t_model, df_model, t_base, df_base, n_obs, s_mat, sigma_hat)

    return AnvalaFit(
        spec=spec,
        estimator=est_upper,
        n_obs=n_obs,
        converged=bool(res.success or (np.isfinite(res.fun) and np.max(np.abs(res.jac)) < 1e-5)),
        theta_hat=theta_hat,
        se=se,
        vcov=vcov,
        lambda_hat=lambda_hat,
        beta_hat=beta_hat,
        psi_hat=psi_hat,
        theta_res_hat=theta_res_hat,
        sigma_hat=sigma_hat,
        sample_cov=s_mat,
        fit_table=fit_table,
        data_raw=df_clean,
        poly_result=poly_res,
    )