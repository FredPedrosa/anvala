from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _vech_indices(p: int, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = [], []
    for c in range(p):
        start_r = c + 1 if strict else c
        for r in range(start_r, p):
            rows.append(r)
            cols.append(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def _duplication_matrix(p: int, vech_r: np.ndarray, vech_c: np.ndarray) -> np.ndarray:
    p_star = len(vech_r)
    d = np.zeros((p * p, p_star), dtype=np.float64)
    for idx, (r, c) in enumerate(zip(vech_r, vech_c)):
        d[c * p + r, idx] = 1.0
        d[r * p + c, idx] = 1.0
    return d


def _extract_model_state(fit) -> dict:
    """Extrai matrizes observadas, implicadas e parâmetros estimados de AnvalaFit ou SEMFit."""
    pe = fit.parameter_estimates()
    if hasattr(fit, "spec"):
        ov_names = list(fit.spec.ov_names)
        lv_names = list(fit.spec.lv_names)
    else:
        ov_names = list(fit.ov_names)
        lv_names = list(fit.lv_names)

    est_upper = getattr(fit, "estimator", "ML").upper()
    poly_res = getattr(fit, "poly_result", None)
    is_ordered = poly_res is not None or bool(getattr(fit, "ordered", False))

    s_mat = None
    for attr in ("s_mat", "sample_cov", "obs_cov"):
        val = getattr(fit, attr, None)
        if val is not None:
            s_mat = np.asarray(val, dtype=np.float64)
            break

    data_df = getattr(fit, "data", None)
    if s_mat is None:
        if is_ordered:
            if poly_res is not None:
                s_mat = poly_res.corr.to_numpy(dtype=np.float64)
            else:
                s_mat = fit.polychoric().to_numpy(dtype=np.float64)
        elif data_df is not None:
            s_mat = np.cov(data_df[ov_names].to_numpy(dtype=np.float64), rowvar=False, bias=True)
        else:
            raise ValueError("Não foi possível recuperar a matriz de covariância/correlação amostral do modelo.")

    n_obs = getattr(fit, "n_obs", None)
    if n_obs is None and data_df is not None:
        n_obs = int(data_df.dropna(subset=ov_names).shape[0])

    # Reconstrói matrizes RAM (B e Psi) no ponto estimado
    all_vars = lv_names + ov_names
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    n_lv = len(lv_names)
    n_ov = len(ov_names)
    k_tot = len(all_vars)

    b_base = np.zeros((k_tot, k_tot), dtype=np.float64)
    psi_base = np.zeros((k_tot, k_tot), dtype=np.float64)

    free_specs: list[tuple[str, int, int, str]] = []
    label_to_free_idx: dict[str, int] = {}

    for _, row in pe.iterrows():
        op = str(row["op"])
        if op == ":=":
            continue
        lhs = str(row["lhs"])
        rhs = str(row["rhs"])
        est_val = float(row["est"])
        se_val = float(row["se"]) if pd.notna(row["se"]) else 0.0
        lbl = str(row["label"]) if ("label" in row and pd.notna(row["label"])) else ""

        if op == "=~":
            r_i, c_i = var_to_idx[rhs], var_to_idx[lhs]
            b_base[r_i, c_i] = est_val
            mat_tag = "B"
        elif op == "~":
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            b_base[r_i, c_i] = est_val
            mat_tag = "B"
        elif op == "~~":
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            psi_base[r_i, c_i] = est_val
            psi_base[c_i, r_i] = est_val
            mat_tag = "Psi"
        else:
            continue

        # Em modelos ordinais WLSMV, variâncias residuais dos OVs são determinadas por 1 - comunalidade
        is_ord_ov_var = is_ordered and op == "~~" and lhs == rhs and lhs in ov_names
        if se_val > 1e-12 and not is_ord_ov_var:
            if lbl and lbl in label_to_free_idx:
                f_idx = label_to_free_idx[lbl]
            else:
                f_idx = len(label_to_free_idx) if lbl else -1
                if lbl:
                    label_to_free_idx[lbl] = f_idx
            free_specs.append((mat_tag, r_i, c_i, lbl))

    # Mapeia índices únicos de parâmetros livres
    unique_keys: list[str] = []
    param_index_map: list[int] = []
    for idx_s, (mat_tag, r_i, c_i, lbl) in enumerate(free_specs):
        key = f"lbl:{lbl}" if lbl else f"pos:{idx_s}"
        if key not in unique_keys:
            unique_keys.append(key)
        param_index_map.append(unique_keys.index(key))

    n_free = len(unique_keys)
    theta_hat = np.zeros(n_free, dtype=np.float64)
    for (mat_tag, r_i, c_i, _), p_idx in zip(free_specs, param_index_map):
        theta_hat[p_idx] = b_base[r_i, c_i] if mat_tag == "B" else psi_base[r_i, c_i]

    def build_sigma_all(th_t: torch.Tensor, extra_b: tuple[int, int, torch.Tensor] | None = None, extra_psi: tuple[int, int, torch.Tensor] | None = None) -> torch.Tensor:
        b_t = torch.tensor(b_base, dtype=torch.float64)
        psi_t = torch.tensor(psi_base, dtype=torch.float64)
        for (mat_tag, r_i, c_i, _), p_idx in zip(free_specs, param_index_map):
            if mat_tag == "B":
                b_t[r_i, c_i] = th_t[p_idx]
            else:
                psi_t[r_i, c_i] = th_t[p_idx]
                psi_t[c_i, r_i] = th_t[p_idx]

        if extra_b is not None:
            er, ec, eval_t = extra_b
            b_t[er, ec] = eval_t
        if extra_psi is not None:
            er, ec, eval_t = extra_psi
            psi_t[er, ec] = eval_t
            psi_t[ec, er] = eval_t

        eye_k = torch.eye(k_tot, dtype=torch.float64)
        inv_ib = torch.linalg.solve(eye_k - b_t, eye_k)

        if is_ordered:
            # Garante diagonal unitária nas variáveis observadas latentes (parametrização delta)
            psi_lv = psi_t[:n_lv, :n_lv]
            b_lv = b_t[:n_lv, :n_lv]
            inv_ib_lv = torch.linalg.solve(torch.eye(n_lv, dtype=torch.float64) - b_lv, torch.eye(n_lv, dtype=torch.float64))
            cov_lv = inv_ib_lv @ psi_lv @ inv_ib_lv.T
            lam_ov = b_t[n_lv:, :n_lv]
            comm_ov = torch.sum((lam_ov @ cov_lv) * lam_ov, dim=1)
            ov_res = 1.0 - comm_ov
            psi_t = psi_t.clone()
            idx_ov = torch.arange(n_lv, n_lv + n_ov)
            psi_t[idx_ov, idx_ov] = ov_res
            for i_ov in range(n_ov):
                psi_t[n_lv + i_ov, n_lv + i_ov] = ov_res[i_ov]
        return inv_ib @ psi_t @ inv_ib.T

    with torch.no_grad():
        sigma_all_np = build_sigma_all(torch.tensor(theta_hat, dtype=torch.float64)).numpy()
    sigma_hat = sigma_all_np[n_lv:, n_lv:]

    psi_all_np = psi_base.copy()
    for (mat_tag, r_i, c_i, _), p_idx in zip(free_specs, param_index_map):
        if mat_tag == "Psi":
            psi_all_np[r_i, c_i] = theta_hat[p_idx]
            psi_all_np[c_i, r_i] = theta_hat[p_idx]
    if is_ordered:
        b_all_np = b_base.copy()
        for (mat_tag, r_i, c_i, _), p_idx in zip(free_specs, param_index_map):
            if mat_tag == "B":
                b_all_np[r_i, c_i] = theta_hat[p_idx]
        inv_ib_lv = np.linalg.inv(np.eye(n_lv) - b_all_np[:n_lv, :n_lv])
        cov_lv = inv_ib_lv @ psi_all_np[:n_lv, :n_lv] @ inv_ib_lv.T
        lam_ov = b_all_np[n_lv:, :n_lv]
        comm_ov = np.sum((lam_ov @ cov_lv) * lam_ov, axis=1)
        for i_ov in range(n_ov):
            psi_all_np[n_lv + i_ov, n_lv + i_ov] = 1.0 - comm_ov[i_ov]

    return {
        "pe": pe,
        "ov_names": ov_names,
        "lv_names": lv_names,
        "var_to_idx": var_to_idx,
        "n_lv": n_lv,
        "n_ov": n_ov,
        "s_mat": s_mat,
        "sigma_hat": sigma_hat,
        "sigma_all": sigma_all_np,
        "psi_all": psi_all_np,
        "n_obs": int(n_obs),
        "estimator": est_upper,
        "is_ordered": is_ordered,
        "poly_res": poly_res,
        "data_df": data_df,
        "theta_hat": theta_hat,
        "build_sigma_all": build_sigma_all,
        "vcov": getattr(fit, "vcov", None),
    }


def residuals(fit, type: str = "raw") -> pd.DataFrame:
    """
    Calcula resíduos locais do modelo em paridade com lavaan::residuals.

    Parâmetros
    ----------
    fit : AnvalaFit | SEMFit
        Modelo ajustado por cfa() ou sem().
    type : {"raw", "standardized", "cor", "cor.bollen"}
        Tipo de resíduo:
        - "raw": S - Sigma_hat
        - "cor" / "cor.bollen": R_obs - R_imp (resíduos de correlação de Bollen)
        - "standardized": resíduos normalizados pelo erro-padrão assintótico (S - Sigma_hat) / SE(S - Sigma_hat)
    """
    st = _extract_model_state(fit)
    ov_names = st["ov_names"]
    s_mat = st["s_mat"]
    sigma_hat = st["sigma_hat"]
    p = len(ov_names)
    res_type = type.lower().strip()

    if res_type == "raw":
        res_mat = s_mat - sigma_hat
        return pd.DataFrame(res_mat, index=ov_names, columns=ov_names)

    if res_type in ("cor", "cor.bollen"):
        sd_s = np.sqrt(np.maximum(1e-12, np.diag(s_mat)))
        sd_sig = np.sqrt(np.maximum(1e-12, np.diag(sigma_hat)))
        r_obs = s_mat / np.outer(sd_s, sd_s)
        r_imp = sigma_hat / np.outer(sd_sig, sd_sig)
        return pd.DataFrame(r_obs - r_imp, index=ov_names, columns=ov_names)

    if res_type == "standardized":
        n_obs = st["n_obs"]
        n_lv = st["n_lv"]
        th_t = torch.tensor(st["theta_hat"], dtype=torch.float64)
        build_sigma_all = st["build_sigma_all"]

        if st["is_ordered"]:
            vech_r, vech_c = _vech_indices(p, strict=True)
            def fn_vech(tv: torch.Tensor) -> torch.Tensor:
                sig = build_sigma_all(tv)[n_lv:, n_lv:]
                return sig[vech_r, vech_c]

            jac = torch.autograd.functional.jacobian(fn_vech, th_t).numpy()
            poly_res = st["poly_res"]
            n_tau = sum(len(t) for t in poly_res.thresholds_list)
            gamma_rr = poly_res.gamma[n_tau:, n_tau:]
            w_mat = np.diag(1.0 / np.maximum(1e-12, np.diag(gamma_rr)))
            dtwd_inv = np.linalg.pinv(jac.T @ w_mat @ jac)
            q_proj = np.eye(len(vech_r)) - jac @ dtwd_inv @ jac.T @ w_mat
            acov_obs = gamma_rr / float(n_obs - 1)
            acov_res = q_proj @ acov_obs @ q_proj.T
        else:
            vech_r, vech_c = _vech_indices(p, strict=False)
            d_mat = _duplication_matrix(p, vech_r, vech_c)

            def fn_vech_full(tv: torch.Tensor) -> torch.Tensor:
                sig = build_sigma_all(tv)[n_lv:, n_lv:]
                return sig[vech_r, vech_c]

            jac = torch.autograd.functional.jacobian(fn_vech_full, th_t).numpy()
            s_inv = np.linalg.inv(s_mat)
            w_h1 = 0.5 * (d_mat.T @ np.kron(s_inv, s_inv) @ d_mat)
            sigma_inv = np.linalg.inv(sigma_hat)
            w_ml = 0.5 * (d_mat.T @ np.kron(sigma_inv, sigma_inv) @ d_mat)

            if st["data_df"] is not None:
                y_c = st["data_df"][ov_names].dropna().to_numpy(dtype=np.float64)
                y_c = y_c - np.mean(y_c, axis=0)
                s_vech = s_mat[vech_r, vech_c]
                yi_s = np.einsum("ni,nj->nij", y_c, y_c)[:, vech_r, vech_c] - s_vech[None, :]
                acov_obs = ((yi_s.T @ yi_s) / float(n_obs)) / float(n_obs)
            else:
                acov_obs = np.linalg.pinv(w_h1) / float(n_obs)

            if st["vcov"] is not None and st["estimator"] == "ML":
                vcov_n = st["vcov"] * float(n_obs)
                q_proj = np.eye(len(vech_r)) - jac @ vcov_n @ jac.T @ w_ml
            else:
                a0_inv = np.linalg.pinv(jac.T @ w_h1 @ jac)
                q_proj = np.eye(len(vech_r)) - jac @ a0_inv @ jac.T @ w_h1

            acov_res = q_proj @ acov_obs @ q_proj.T

        diag_acov = np.diag(acov_res)
        se_vec = np.sqrt(np.maximum(0.0, diag_acov))
        diff_vec = (s_mat - sigma_hat)[vech_r, vech_c]
        z_mat = np.zeros((p, p), dtype=np.float64)
        for idx, (r, c) in enumerate(zip(vech_r, vech_c)):
            se_use = 1.0 if abs(diff_vec[idx]) < 1e-5 else se_vec[idx]
            z_val = diff_vec[idx] / se_use if se_use > 1e-12 else 0.0
            z_mat[r, c] = z_val
            z_mat[c, r] = z_val
        return pd.DataFrame(z_mat, index=ov_names, columns=ov_names)

    raise ValueError(f"Tipo de resíduo '{type}' não suportado. Use 'raw', 'standardized' ou 'cor.bollen'.")


def modindices(
    fit,
    sort: bool = True,
    minimum_value: float = 0.0,
    op: str | None = None,
) -> pd.DataFrame:
    """
    Calcula Índices de Modificação (Sörbom, 1989) e Mudança Esperada de Parâmetro (EPC, SEPC.lv, SEPC.all)
    em paridade com lavaan::modindices.
    """
    st = _extract_model_state(fit)
    pe = st["pe"]
    ov_names = st["ov_names"]
    lv_names = st["lv_names"]
    var_to_idx = st["var_to_idx"]
    n_lv = st["n_lv"]
    p = len(ov_names)
    s_mat = st["s_mat"]
    sigma_hat = st["sigma_hat"]
    sigma_all = st["sigma_all"]
    psi_all = st["psi_all"]
    n_obs = st["n_obs"]
    is_ordered = st["is_ordered"]
    th_t = torch.tensor(st["theta_hat"], dtype=torch.float64)
    build_sigma_all = st["build_sigma_all"]

    # Conjunto de relações já presentes no modelo
    existing = set()
    has_regressions = False
    for _, row in pe.iterrows():
        cur_op = str(row["op"])
        if cur_op == "~":
            has_regressions = True
        if cur_op in ("=~", "~", "~~"):
            l_s, r_s = str(row["lhs"]), str(row["rhs"])
            existing.add((l_s, cur_op, r_s))
            if cur_op == "~~":
                existing.add((r_s, "~~", l_s))

    # Gera candidatos (cross-loadings, covariâncias residuais, caminhos estruturais e covariâncias latentes)
    candidates: list[tuple[str, str, str]] = []
    for lv in lv_names:
        for ov in ov_names:
            if (lv, "=~", ov) not in existing:
                candidates.append((lv, "=~", ov))

    if has_regressions and len(lv_names) > 1:
        for lv1 in lv_names:
            for lv2 in lv_names:
                if lv1 != lv2 and (lv1, "~", lv2) not in existing:
                    candidates.append((lv1, "~", lv2))
        for i in range(len(lv_names)):
            for j in range(i + 1, len(lv_names)):
                l1, l2 = lv_names[i], lv_names[j]
                if (l1, "~~", l2) not in existing:
                    candidates.append((l1, "~~", l2))

    for i in range(p):
        for j in range(i + 1, p):
            o1, o2 = ov_names[i], ov_names[j]
            if (o1, "~~", o2) not in existing:
                candidates.append((o1, "~~", o2))

    if op is not None:
        candidates = [c for c in candidates if c[1] == op]

    if not candidates:
        return pd.DataFrame(columns=["lhs", "op", "rhs", "mi", "epc", "sepc_lv", "sepc_all"])

    if is_ordered:
        vech_r, vech_c = _vech_indices(p, strict=True)
        def fn_obs(tv: torch.Tensor, eb=None, ep=None) -> torch.Tensor:
            sig = build_sigma_all(tv, extra_b=eb, extra_psi=ep)[n_lv:, n_lv:]
            return sig[vech_r, vech_c]

        poly_res = st["poly_res"]
        n_tau = sum(len(t) for t in poly_res.thresholds_list)
        gamma_rr = poly_res.gamma[n_tau:, n_tau:]
        w_mat = np.diag(1.0 / np.maximum(1e-12, np.diag(gamma_rr)))
        diff_v = (s_mat - sigma_hat)[vech_r, vech_c]
        n_eff = float((n_obs - 1) ** 2) / float(n_obs)
    else:
        vech_r, vech_c = _vech_indices(p, strict=False)
        d_mat = _duplication_matrix(p, vech_r, vech_c)
        def fn_obs(tv: torch.Tensor, eb=None, ep=None) -> torch.Tensor:
            sig = build_sigma_all(tv, extra_b=eb, extra_psi=ep)[n_lv:, n_lv:]
            return sig[vech_r, vech_c]

        sigma_inv = np.linalg.inv(sigma_hat)
        w_mat = 0.5 * (d_mat.T @ np.kron(sigma_inv, sigma_inv) @ d_mat)
        diff_v = (s_mat - sigma_hat)[vech_r, vech_c]
        n_eff = float(n_obs)

    jac_free = torch.autograd.functional.jacobian(lambda tv: fn_obs(tv), th_t).numpy()
    i_tt = jac_free.T @ w_mat @ jac_free
    i_tt_inv = np.linalg.pinv(i_tt)
    w_diff = w_mat @ diff_v
    w_jac = w_mat @ jac_free

    rows_out = []
    zero_t = torch.tensor(0.0, dtype=torch.float64)

    for lhs, cand_op, rhs in candidates:
        if cand_op == "=~":
            r_i, c_i = var_to_idx[rhs], var_to_idx[lhs]
            delta_psi = torch.autograd.functional.jacobian(
                lambda z: fn_obs(th_t, eb=(r_i, c_i, z)), zero_t
            ).numpy()
        elif cand_op == "~":
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            delta_psi = torch.autograd.functional.jacobian(
                lambda z: fn_obs(th_t, eb=(r_i, c_i, z)), zero_t
            ).numpy()
        else:
            r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
            delta_psi = torch.autograd.functional.jacobian(
                lambda z: fn_obs(th_t, ep=(r_i, c_i, z)), zero_t
            ).numpy()

        d_psi = float(delta_psi @ w_diff)
        i_pt = delta_psi @ w_jac
        i_pp = float(delta_psi @ (w_mat @ delta_psi))
        v_psi = i_pp - float(i_pt @ i_tt_inv @ i_pt)

        if v_psi <= 1e-10:
            continue

        mi_val = n_eff * (d_psi ** 2) / v_psi
        epc_scale = (float(n_obs - 1) / float(n_obs)) if is_ordered else 1.0
        epc_val = (d_psi / v_psi) * epc_scale

        if mi_val < minimum_value:
            continue

        idx_l = var_to_idx[lhs]
        idx_r = var_to_idx[rhs]
        sd_lv_l = np.sqrt(max(1e-12, sigma_all[idx_l, idx_l])) if lhs in lv_names else 1.0
        sd_lv_r = np.sqrt(max(1e-12, sigma_all[idx_r, idx_r])) if rhs in lv_names else 1.0
        sd_all_l = np.sqrt(max(1e-12, sigma_all[idx_l, idx_l]))
        sd_all_r = np.sqrt(max(1e-12, sigma_all[idx_r, idx_r]))

        if cand_op == "=~":
            sepc_lv = epc_val * sd_lv_l
            sepc_all = epc_val * (sd_lv_l / sd_all_r)
        elif cand_op == "~":
            sepc_lv = epc_val * (sd_lv_r / sd_lv_l)
            sepc_all = epc_val * (sd_all_r / sd_all_l)
        else:
            sd_res_l = np.sqrt(max(1e-12, psi_all[idx_l, idx_l]))
            sd_res_r = np.sqrt(max(1e-12, psi_all[idx_r, idx_r]))
            sepc_lv = epc_val / (sd_lv_l * sd_lv_r)
            sepc_all = epc_val / (sd_res_l * sd_res_r)

        rows_out.append(
            {
                "lhs": lhs,
                "op": cand_op,
                "rhs": rhs,
                "mi": float(mi_val),
                "epc": float(epc_val),
                "sepc_lv": float(sepc_lv),
                "sepc_all": float(sepc_all),
            }
        )

    df_mi = pd.DataFrame(rows_out, columns=["lhs", "op", "rhs", "mi", "epc", "sepc_lv", "sepc_all"])
    if sort and not df_mi.empty:
        df_mi = df_mi.sort_values(by="mi", ascending=False).reset_index(drop=True)
    return df_mi