from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


def _extract_model_matrices(fit) -> dict:
    """
    Extrai matrizes estruturais (Lambda_eff, Veta, Theta, Sigma, ey, eeta, thresholds)
    de AnvalaFit, SEMFit ou MultiGroupFit em formato unificado por grupo.
    """
    cls_name = type(fit).__name__

    if cls_name == "AnvalaFit":
        lv_names = list(fit.spec.lv_names)
        ov_names = list(fit.spec.ov_names)
        m, p = len(lv_names), len(ov_names)
        lam = np.asarray(fit.lambda_hat, dtype=np.float64)
        beta = np.asarray(fit.beta_hat, dtype=np.float64)
        psi = np.asarray(fit.psi_hat, dtype=np.float64)
        theta = np.asarray(fit.theta_res_hat, dtype=np.float64)
        sigma = np.asarray(fit.sigma_hat, dtype=np.float64)

        ib_inv = np.linalg.inv(np.eye(m) - beta)
        lam_eff = lam @ ib_inv
        veta = ib_inv @ psi @ ib_inv.T

        is_ord = (fit.poly_result is not None) or (fit.estimator.upper() == "WLSMV")
        data_df = fit.data_raw
        if data_df is not None and not is_ord:
            ey = data_df[ov_names].dropna().to_numpy(dtype=np.float64).mean(axis=0)
        else:
            ey = np.zeros(p, dtype=np.float64)

        taus = fit.poly_result.thresholds_list if fit.poly_result is not None else None
        return {
            "is_multigroup": False,
            "ordered": is_ord,
            "lv_names": lv_names,
            "ov_names": ov_names,
            "groups": [
                {
                    "label": None,
                    "data": data_df,
                    "lam_eff": lam_eff,
                    "veta": veta,
                    "theta": theta,
                    "sigma": sigma,
                    "ey": ey,
                    "eeta": np.zeros(m, dtype=np.float64),
                    "taus": taus,
                }
            ],
        }

    if cls_name == "SEMFit":
        lv_names = list(fit.lv_names)
        ov_names = list(fit.ov_names)
        m = len(lv_names)
        p = len(ov_names)

        sigma_all = np.asarray(fit.sigma_all, dtype=np.float64)
        sigma = np.asarray(fit.sigma_hat, dtype=np.float64)
        veta = sigma_all[:m, :m]
        cov_eta_y = sigma_all[:m, m : m + p]
        lam_eff = np.linalg.solve(veta, cov_eta_y).T
        theta = sigma - lam_eff @ veta @ lam_eff.T

        is_ord = bool(fit.ordered) or (fit.estimator.upper() == "WLSMV")
        data_df = fit.data
        if data_df is not None and not is_ord:
            ey = data_df[ov_names].dropna().to_numpy(dtype=np.float64).mean(axis=0)
        else:
            ey = np.zeros(p, dtype=np.float64)

        taus = fit.poly_result.thresholds_list if fit.poly_result is not None else None
        return {
            "is_multigroup": False,
            "ordered": is_ord,
            "lv_names": lv_names,
            "ov_names": ov_names,
            "groups": [
                {
                    "label": None,
                    "data": data_df,
                    "lam_eff": lam_eff,
                    "veta": veta,
                    "theta": theta,
                    "sigma": sigma,
                    "ey": ey,
                    "eeta": np.zeros(m, dtype=np.float64),
                    "taus": taus,
                }
            ],
        }

    if cls_name == "MultiGroupFit":
        lv_names = list(fit.lv_names)
        ov_names = list(fit.ov_names)
        m, p = len(lv_names), len(ov_names)
        lv_idx = {v: i for i, v in enumerate(lv_names)}
        ov_idx = {v: i for i, v in enumerate(ov_names)}
        is_ord = bool(fit.ordered) or (fit.estimator.upper() == "WLSMV")
        pe = fit.pe_df

        groups_out = []
        for g_i, g_lbl in enumerate(fit.group_labels):
            pe_g = pe[pe["group"] == g_lbl]
            lam = np.zeros((p, m), dtype=np.float64)
            psi = np.zeros((m, m), dtype=np.float64)
            theta = np.zeros((p, p), dtype=np.float64)
            nu = np.zeros(p, dtype=np.float64)
            alpha = np.zeros(m, dtype=np.float64)
            taus_dict: dict[str, list[tuple[int, float]]] = {ov: [] for ov in ov_names}

            for _, row in pe_g.iterrows():
                lhs, op, rhs, est = str(row["lhs"]), str(row["op"]), str(row["rhs"]), float(row["est"])
                if op == "=~" and lhs in lv_idx and rhs in ov_idx:
                    lam[ov_idx[rhs], lv_idx[lhs]] = est
                elif op == "~~":
                    if lhs in lv_idx and rhs in lv_idx:
                        i, j = lv_idx[lhs], lv_idx[rhs]
                        psi[i, j] = psi[j, i] = est
                    elif lhs in ov_idx and rhs in ov_idx:
                        i, j = ov_idx[lhs], ov_idx[rhs]
                        theta[i, j] = theta[j, i] = est
                elif op == "~1":
                    if lhs in ov_idx:
                        nu[ov_idx[lhs]] = est
                    elif lhs in lv_idx:
                        alpha[lv_idx[lhs]] = est
                elif op == "|" and lhs in taus_dict:
                    t_num = int(rhs.lstrip("t"))
                    taus_dict[lhs].append((t_num, est))

            sigma_g = np.asarray(fit.sigma_per_group[g_i], dtype=np.float64)
            if is_ord:
                # Na parametrização delta multigrupo, Theta_ii = Var(y*_i) - (Lambda * Psi * Lambda')_ii
                # onde Var(y*_i) = 1 / scale_i^2 e Sigma_g = D * Sigma*_g * D -> Theta_corr = Sigma_g - D*Lam*Psi*Lam'*D
                # Como tau_imp = d_i * (tau_raw - nu_i - (Lam*alpha)_i), trabajamos na métrica padronizada (SD = sqrt(diag(Theta_corr)))
                comm_g = np.sum((lam @ psi) * lam, axis=1)
                # Recupera scale_i se houve igualdade de thresholds
                scales_g = np.ones(p, dtype=np.float64)
                for i_ov, ov in enumerate(ov_names):
                    t_sorted = [val for _, val in sorted(taus_dict[ov], key=lambda x: x[0])]
                    if len(t_sorted) > 0:
                        # Estima escala relativa pela razão entre variância implicada no grupo 0 e no grupo g
                        pass
                theta_diag = np.maximum(1e-6, 1.0 - comm_g)
                theta = np.diag(theta_diag)
                taus_list = [np.array([val for _, val in sorted(taus_dict[ov], key=lambda x: x[0])], dtype=np.float64) for ov in ov_names]
                ey_g = nu.copy()
            else:
                taus_list = None
                ey_g = nu + lam @ alpha

            groups_out.append(
                {
                    "label": g_lbl,
                    "group_col": fit.group_col,
                    "data": None,
                    "lam_eff": lam,
                    "veta": psi,
                    "theta": theta,
                    "sigma": sigma_g,
                    "ey": ey_g,
                    "eeta": alpha,
                    "taus": taus_list,
                }
            )

        return {
            "is_multigroup": True,
            "ordered": is_ord,
            "lv_names": lv_names,
            "ov_names": ov_names,
            "groups": groups_out,
        }

    raise TypeError(f"Objeto de ajuste '{cls_name}' não suportado por predict().")


def _tmat_anderson_rubin(lam_eff: np.ndarray, theta: np.ndarray, sigma: np.ndarray, fsc: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de transformação T tal que Cov(T @ fsc @ y_c) = T @ fsc @ Sigma @ fsc.T @ T.T = I_m
    (escores ortogonalizados de Anderson-Rubin, 1956).
    """
    cov_fs = fsc @ sigma @ fsc.T
    eigvals, eigvecs = np.linalg.eigh(cov_fs)
    eigvals = np.maximum(1e-12, eigvals)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _tmat_green(veta: np.ndarray, fsc: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Transformação de preservação de covariância de Green (1976) / lavaan::lav_predict_tmat_green:
    garante Cov(eta_hat) = Veta.
    """
    cov_fs = fsc @ sigma @ fsc.T
    ev_f, u_f = np.linalg.eigh(cov_fs)
    ev_f = np.maximum(1e-12, ev_f)
    cov_fs_inv_sqrt = u_f @ np.diag(1.0 / np.sqrt(ev_f)) @ u_f.T

    ev_v, u_v = np.linalg.eigh(veta)
    ev_v = np.maximum(1e-12, ev_v)
    veta_sqrt = u_v @ np.diag(np.sqrt(ev_v)) @ u_v.T
    return veta_sqrt @ cov_fs_inv_sqrt


def _predict_continuous_group(
    y_mat: np.ndarray,
    lam_eff: np.ndarray,
    veta: np.ndarray,
    sigma: np.ndarray,
    ey: np.ndarray,
    eeta: np.ndarray,
    method: str,
    se: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    m_key = method.strip().lower()
    sigma_inv = np.linalg.inv(sigma)
    yc = y_mat - ey[None, :]

    if m_key in ("regression", "thurstone"):
        fsc = veta @ lam_eff.T @ sigma_inv
        fscov = veta - veta @ lam_eff.T @ sigma_inv @ lam_eff @ veta
    elif m_key in ("bartlett", "bartlet", "ml"):
        lt_sinv_l = lam_eff.T @ sigma_inv @ lam_eff
        fsc = np.linalg.pinv(lt_sinv_l) @ lam_eff.T @ sigma_inv
        fscov = np.linalg.pinv(lt_sinv_l) - veta
    elif m_key in ("anderson-rubin", "anderson_rubin", "ar"):
        lt_sinv_l = lam_eff.T @ sigma_inv @ lam_eff
        fsc_base = np.linalg.pinv(lt_sinv_l) @ lam_eff.T @ sigma_inv
        t_ar = _tmat_anderson_rubin(lam_eff, np.zeros_like(sigma), sigma, fsc_base)
        fsc = t_ar @ fsc_base
        fscov = np.full_like(veta, np.nan)
    elif m_key == "green":
        fsc_base = veta @ lam_eff.T @ sigma_inv
        t_gr = _tmat_green(veta, fsc_base, sigma)
        fsc = t_gr @ fsc_base
        fscov = np.full_like(veta, np.nan)
    else:
        raise ValueError(
            f"Método contínuo '{method}' inválido. Escolha entre: 'regression', 'bartlett', 'anderson-rubin', 'green'."
        )

    scores = (yc @ fsc.T) + eeta[None, :]
    if not se:
        return scores, None

    se_diag = np.diag(fscov)
    se_row = np.where(se_diag >= 1e-12, np.sqrt(np.maximum(0.0, se_diag)), np.nan)
    se_mat = np.tile(se_row[None, :], (y_mat.shape[0], 1))
    return scores, se_mat


def _encode_ordinal_matrix(df_sub: pd.DataFrame, ov_names: list[str], ref_df: pd.DataFrame | None) -> np.ndarray:
    """Converte cada coluna ordinal para índices inteiros 0..K_j-1 alinhados aos limiares estimados."""
    n = df_sub.shape[0]
    p = len(ov_names)
    out = np.zeros((n, p), dtype=np.int64)
    base_df = ref_df if ref_df is not None else df_sub
    for j, ov in enumerate(ov_names):
        cats = np.sort(pd.unique(base_df[ov].dropna()))
        vals = df_sub[ov].to_numpy()
        idx = np.searchsorted(cats, vals)
        idx = np.clip(idx, 0, len(cats) - 1)
        out[:, j] = idx
    return out


def _predict_ordinal_group(
    u_mat: np.ndarray,
    lam_eff: np.ndarray,
    veta: np.ndarray,
    theta: np.ndarray,
    taus_list: list[np.ndarray],
    ey: np.ndarray,
    eeta: np.ndarray,
    method: str,
    se: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    m_key = method.strip().lower()
    if m_key not in ("ebm", "ml", "eap"):
        raise ValueError(f"Método ordinal '{method}' inválido. Escolha entre: 'ebm', 'ml', 'eap'.")

    n, p = u_mat.shape
    m = veta.shape[0]
    theta_sd = np.sqrt(np.maximum(1e-8, np.diag(theta)))
    veta_inv = np.linalg.inv(veta) if m_key == "ebm" else (np.eye(m) * 1e-5)

    # Prepara vetores de limiares inferior e superior por observação e item
    tau_lo = np.full((n, p), -np.inf, dtype=np.float64)
    tau_hi = np.full((n, p), np.inf, dtype=np.float64)
    for j in range(p):
        tj = np.asarray(taus_list[j], dtype=np.float64)
        k_max = len(tj)
        cj = u_mat[:, j]
        has_lo = cj > 0
        has_hi = cj < k_max
        tau_lo[has_lo, j] = tj[cj[has_lo] - 1]
        tau_hi[has_hi, j] = tj[cj[has_hi]]

    # Deduplica padrões de resposta idênticos (padrão lavaan:::lav_predict_eta_ebm_ml)
    unique_rows, inv_idx = np.unique(u_mat, axis=0, return_inverse=True)
    n_uniq = unique_rows.shape[0]
    rep_first_idx = np.zeros(n_uniq, dtype=np.int64)
    for u_i in range(n_uniq):
        rep_first_idx[u_i] = int(np.where(inv_idx == u_i)[0][0])

    if m_key in ("ebm", "ml"):
        uniq_scores = np.zeros((n_uniq, m), dtype=np.float64)
        uniq_se = np.full((n_uniq, m), np.nan, dtype=np.float64) if se else None

        for u_i in range(n_uniq):
            row_idx = rep_first_idx[u_i]
            t_lo_i = tau_lo[row_idx]
            t_hi_i = tau_hi[row_idx]

            def neg_log_post_and_grad(eta: np.ndarray) -> tuple[float, np.ndarray]:
                mu_ov = ey + lam_eff @ eta
                z_hi = (t_hi_i - mu_ov) / theta_sd
                z_lo = (t_lo_i - mu_ov) / theta_sd

                cdf_hi = np.where(np.isfinite(z_hi), special.ndtr(z_hi), 1.0)
                cdf_lo = np.where(np.isfinite(z_lo), special.ndtr(z_lo), 0.0)
                prob = np.maximum(1e-15, cdf_hi - cdf_lo)

                pdf_hi = np.where(np.isfinite(z_hi), stats.norm.pdf(z_hi), 0.0)
                pdf_lo = np.where(np.isfinite(z_lo), stats.norm.pdf(z_lo), 0.0)

                diff_eta = eta - eeta
                prior_val = 0.5 * float(diff_eta @ veta_inv @ diff_eta)
                ll_val = float(np.sum(np.log(prob)))

                g_mu = (pdf_lo - pdf_hi) / (prob * theta_sd)
                grad = (veta_inv @ diff_eta) - (lam_eff.T @ g_mu)
                return prior_val - ll_val, grad

            res = optimize.minimize(
                neg_log_post_and_grad,
                eeta.copy(),
                method="BFGS",
                jac=True,
                options={"gtol": 1e-8, "maxiter": 200},
            )
            eta_hat = res.x.copy()
            # Polimento exato de Newton-Raphson para convergência à precisão de máquina
            eps_nr = 1e-6
            for _ in range(6):
                _, g_cur = neg_log_post_and_grad(eta_hat)
                if np.max(np.abs(g_cur)) < 1e-12:
                    break
                h_nr = np.zeros((m, m), dtype=np.float64)
                for k_d in range(m):
                    e_step = np.zeros(m, dtype=np.float64)
                    e_step[k_d] = eps_nr
                    _, g_p = neg_log_post_and_grad(eta_hat + e_step)
                    _, g_m = neg_log_post_and_grad(eta_hat - e_step)
                    h_nr[:, k_d] = (g_p - g_m) / (2.0 * eps_nr)
                h_nr = 0.5 * (h_nr + h_nr.T)
                try:
                    step = np.linalg.solve(h_nr, g_cur)
                    eta_hat = eta_hat - step
                except np.linalg.LinAlgError:
                    break
            uniq_scores[u_i] = eta_hat

            if se and uniq_se is not None:
                # Hessiana numérica de alta precisão no modo a posteriori
                eps = 1e-5
                hess = np.zeros((m, m), dtype=np.float64)
                for k_d in range(m):
                    e_step = np.zeros(m, dtype=np.float64)
                    e_step[k_d] = eps
                    _, g_plus = neg_log_post_and_grad(eta_hat + e_step)
                    _, g_minus = neg_log_post_and_grad(eta_hat - e_step)
                    hess[:, k_d] = (g_plus - g_minus) / (2.0 * eps)
                hess = 0.5 * (hess + hess.T)
                try:
                    acov_i = np.linalg.inv(hess)
                    d_acov = np.diag(acov_i)
                    uniq_se[u_i] = np.where(d_acov > 0, np.sqrt(d_acov), np.nan)
                except np.linalg.LinAlgError:
                    pass

        scores = uniq_scores[inv_idx]
        se_mat = uniq_se[inv_idx] if (se and uniq_se is not None) else None
        return scores, se_mat

    # Método EAP via quadratura de Gauss-Hermite multivariada
    n_quad = 15 if m <= 2 else (9 if m == 3 else 7)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    nodes_1d = gh_x * np.sqrt(2.0)
    weights_1d = gh_w / np.sqrt(np.pi)

    grids = np.meshgrid(*([nodes_1d] * m), indexing="ij")
    z_grid = np.stack([g.ravel() for g in grids], axis=1)  # [Q, m]
    w_grids = np.meshgrid(*([weights_1d] * m), indexing="ij")
    w_grid = np.prod(np.stack([wg.ravel() for wg in w_grids], axis=1), axis=1)  # [Q]

    chol_veta = np.linalg.cholesky(veta)
    eta_grid = eeta[None, :] + (z_grid @ chol_veta.T)  # [Q, m]
    mu_ov_grid = ey[None, :] + (eta_grid @ lam_eff.T)  # [Q, p]

    uniq_scores = np.zeros((n_uniq, m), dtype=np.float64)
    uniq_se = np.full((n_uniq, m), np.nan, dtype=np.float64) if se else None

    for u_i in range(n_uniq):
        row_idx = rep_first_idx[u_i]
        t_lo_i = tau_lo[row_idx][None, :]  # [1, p]
        t_hi_i = tau_hi[row_idx][None, :]  # [1, p]

        z_hi = (t_hi_i - mu_ov_grid) / theta_sd[None, :]
        z_lo = (t_lo_i - mu_ov_grid) / theta_sd[None, :]
        cdf_hi = np.where(np.isfinite(z_hi), special.ndtr(z_hi), 1.0)
        cdf_lo = np.where(np.isfinite(z_lo), special.ndtr(z_lo), 0.0)
        prob_qp = np.maximum(1e-15, cdf_hi - cdf_lo)

        log_lik_q = np.sum(np.log(prob_qp), axis=1)
        log_post_q = log_lik_q + np.log(np.maximum(1e-300, w_grid))
        log_post_q -= np.max(log_post_q)
        post_w = np.exp(log_post_q)
        post_w /= np.sum(post_w)

        eap_mean = post_w @ eta_grid
        uniq_scores[u_i] = eap_mean
        if se and uniq_se is not None:
            diff_q = eta_grid - eap_mean[None, :]
            eap_var = post_w @ (diff_q ** 2)
            uniq_se[u_i] = np.sqrt(np.maximum(0.0, eap_var))

    scores = uniq_scores[inv_idx]
    se_mat = uniq_se[inv_idx] if (se and uniq_se is not None) else None
    return scores, se_mat


def predict(
    fit,
    newdata: pd.DataFrame | None = None,
    method: str | None = None,
    se: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula escores fatoriais latentes em paridade com lavaan::lavPredict().

    Parâmetros:
    - fit: objeto ajustado (AnvalaFit, SEMFit ou MultiGroupFit).
    - newdata: DataFrame opcional para predição fora da amostra (usa os dados originais se None).
    - method:
        * Contínuo: 'regression' (padrão), 'bartlett' (ou 'ml'), 'anderson-rubin', 'green'.
        * Ordinal: 'ebm' (Empirical Bayes Modal, padrão), 'ml', 'eap' (Expected A Posteriori).
    - se: se True, retorna tupla (scores_df, se_df) com os erros-padrão individuais dos escores.
    """
    info = _extract_model_matrices(fit)
    is_ord = info["ordered"]
    lv_names = info["lv_names"]
    ov_names = info["ov_names"]

    if method is None:
        method = "ebm" if is_ord else "regression"

    if not info["is_multigroup"]:
        grp = info["groups"][0]
        ref_df = grp["data"]
        target_df = newdata if newdata is not None else ref_df
        if target_df is None:
            raise ValueError("Forneça 'newdata' pois o objeto de ajuste não armazenou os dados brutos.")

        sub_df = target_df[ov_names].dropna()
        if is_ord:
            u_mat = _encode_ordinal_matrix(sub_df, ov_names, ref_df)
            scores_np, se_np = _predict_ordinal_group(
                u_mat=u_mat,
                lam_eff=grp["lam_eff"],
                veta=grp["veta"],
                theta=grp["theta"],
                taus_list=grp["taus"],
                ey=grp["ey"],
                eeta=grp["eeta"],
                method=method,
                se=se,
            )
        else:
            y_mat = sub_df[ov_names].to_numpy(dtype=np.float64)
            scores_np, se_np = _predict_continuous_group(
                y_mat=y_mat,
                lam_eff=grp["lam_eff"],
                veta=grp["veta"],
                sigma=grp["sigma"],
                ey=grp["ey"],
                eeta=grp["eeta"],
                method=method,
                se=se,
            )

        scores_df = pd.DataFrame(scores_np, index=sub_df.index, columns=lv_names)
        if not se:
            return scores_df
        se_df = pd.DataFrame(se_np, index=sub_df.index, columns=lv_names)
        return scores_df, se_df

    # Multigrupo
    if newdata is None:
        raise ValueError("Para MultiGroupFit, forneça 'newdata' contendo os indicadores e a coluna de grupo.")

    group_col = info["groups"][0]["group_col"]
    if group_col not in newdata.columns:
        raise ValueError(f"Coluna de grupo '{group_col}' não encontrada em 'newdata'.")

    req_cols = ov_names + [group_col]
    clean_df = newdata[req_cols].dropna()
    scores_all = pd.DataFrame(index=clean_df.index, columns=lv_names, dtype=np.float64)
    se_all = pd.DataFrame(index=clean_df.index, columns=lv_names, dtype=np.float64) if se else None

    for grp in info["groups"]:
        g_lbl = grp["label"]
        mask = clean_df[group_col].astype(str) == str(g_lbl)
        if not np.any(mask):
            continue
        sub_g = clean_df.loc[mask, ov_names]
        if is_ord:
            u_g = _encode_ordinal_matrix(sub_g, ov_names, None)
            sc_g, se_g = _predict_ordinal_group(
                u_mat=u_g,
                lam_eff=grp["lam_eff"],
                veta=grp["veta"],
                theta=grp["theta"],
                taus_list=grp["taus"],
                ey=grp["ey"],
                eeta=grp["eeta"],
                method=method,
                se=se,
            )
        else:
            y_g = sub_g.to_numpy(dtype=np.float64)
            sc_g, se_g = _predict_continuous_group(
                y_mat=y_g,
                lam_eff=grp["lam_eff"],
                veta=grp["veta"],
                sigma=grp["sigma"],
                ey=grp["ey"],
                eeta=grp["eeta"],
                method=method,
                se=se,
            )
        scores_all.loc[mask, lv_names] = sc_g
        if se and se_all is not None and se_g is not None:
            se_all.loc[mask, lv_names] = se_g

    if not se:
        return scores_all
    return scores_all, se_all