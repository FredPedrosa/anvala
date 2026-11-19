from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats

from anvala.sem import _parse_sem_syntax


@dataclass
class BootstrapResult:
    table: pd.DataFrame
    boot_samples: pd.DataFrame
    n_boot: int
    conf_level: float
    method: str
    device: str

    def summary(self) -> str:
        lines = [
            f"anvala Nonparametric Bootstrap (B={self.n_boot}, {int(self.conf_level * 100)}% {self.method.upper()}, device={self.device})",
            self.table.round(4).to_string(index=False),
        ]
        return "\n".join(lines)


def _extract_sem_spec_for_batch(fit) -> dict:
    """
    Reconstrói a estrutura paramétrica (B_fixed, Psi_fixed, slots livres, parâmetros definidos :=)
    a partir de SEMFit ou AnvalaFit para otimização vetorial 3D em GPU/CPU.
    """
    cls_name = type(fit).__name__

    if cls_name == "SEMFit":
        data_df = fit.data
        if data_df is None:
            raise ValueError("SEMFit não possui os dados brutos necessários para bootstrap não-paramétrico.")
        ov_names = list(fit.ov_names)
        lv_names = list(fit.lv_names)
        pe = fit.pe_df.copy()

        # Reconstrói sintaxe a partir de pe_df preservando fixos, livres, rótulos e :=
        lines: list[str] = []
        free_entries: list[dict] = []
        defined_rows: list[tuple[str, str]] = []

        all_vars = lv_names + ov_names
        var_to_idx = {v: i for i, v in enumerate(all_vars)}
        n_lv = len(lv_names)
        p = len(ov_names)
        k_tot = len(all_vars)

        b_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)
        psi_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)
        label_to_slot: dict[str, int] = {}
        free_slots: list[dict] = []
        init_raw: list[float] = []
        is_log_var: list[bool] = []
        param_meta: list[dict] = []

        for _, row in pe.iterrows():
            lhs = str(row["lhs"])
            op = str(row["op"])
            rhs = str(row["rhs"])
            lbl = str(row.get("label", "")) if pd.notna(row.get("label", "")) else ""
            est = float(row["est"])
            se_val = float(row["se"]) if pd.notna(row["se"]) else 0.0

            if op == ":=":
                defined_rows.append((lhs, rhs))
                continue

            is_fixed = (se_val == 0.0) and (lbl == "" or lbl not in label_to_slot)
            # Checa se realmente é fixo (ex: marcador 1.0 ou variância fixa em 1.0)
            if se_val == 0.0 and (est in (0.0, 1.0)):
                if op == "=~":
                    b_fixed[var_to_idx[rhs], var_to_idx[lhs]] = est
                elif op == "~":
                    b_fixed[var_to_idx[lhs], var_to_idx[rhs]] = est
                elif op == "~~":
                    r_i, c_i = var_to_idx[lhs], var_to_idx[rhs]
                    psi_fixed[r_i, c_i] = psi_fixed[c_i, r_i] = est
                continue

            if lbl != "" and lbl in label_to_slot:
                slot = label_to_slot[lbl]
            else:
                slot = len(init_raw)
                log_v = (op == "~~") and (lhs == rhs) and (est > 1e-6)
                is_log_var.append(log_v)
                init_raw.append(np.log(est) if log_v else est)
                if lbl != "":
                    label_to_slot[lbl] = slot

            if op == "=~":
                free_slots.append({"slot": slot, "mat": "B", "r": var_to_idx[rhs], "c": var_to_idx[lhs]})
            elif op == "~":
                free_slots.append({"slot": slot, "mat": "B", "r": var_to_idx[lhs], "c": var_to_idx[rhs]})
            elif op == "~~":
                free_slots.append({"slot": slot, "mat": "Psi", "r": var_to_idx[lhs], "c": var_to_idx[rhs]})

            param_meta.append({"lhs": lhs, "op": op, "rhs": rhs, "label": lbl, "slot": slot, "est": est})

        return {
            "data": data_df[ov_names].dropna().to_numpy(dtype=np.float64),
            "ov_names": ov_names,
            "k_tot": k_tot,
            "n_lv": n_lv,
            "p": p,
            "b_fixed": b_fixed,
            "psi_fixed": psi_fixed,
            "free_slots": free_slots,
            "init_raw": np.array(init_raw, dtype=np.float64),
            "is_log_var": np.array(is_log_var, dtype=bool),
            "label_to_slot": label_to_slot,
            "param_meta": param_meta,
            "defined_rows": defined_rows,
        }

    if cls_name == "AnvalaFit":
        if fit.data_raw is None:
            raise ValueError("AnvalaFit não possui 'data_raw' para bootstrap não-paramétrico.")
        ov_names = list(fit.spec.ov_names)
        lv_names = list(fit.spec.lv_names)
        n_lv = len(lv_names)
        p = len(ov_names)
        k_tot = n_lv + p

        b_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)
        psi_fixed = np.zeros((k_tot, k_tot), dtype=np.float64)
        free_slots: list[dict] = []
        init_raw: list[float] = []
        is_log_var: list[bool] = []
        param_meta: list[dict] = []

        lam = np.asarray(fit.lambda_hat, dtype=np.float64)
        beta = np.asarray(fit.beta_hat, dtype=np.float64)
        psi = np.asarray(fit.psi_hat, dtype=np.float64)
        theta_res = np.asarray(fit.theta_res_hat, dtype=np.float64)

        for i_ov in range(p):
            for j_lv in range(n_lv):
                val = lam[i_ov, j_lv]
                if val == 0.0:
                    continue
                if val == 1.0 and i_ov == fit.spec.ov_names.index(fit.spec.loadings[lv_names[j_lv]][0]):
                    b_fixed[n_lv + i_ov, j_lv] = 1.0
                else:
                    sl = len(init_raw)
                    init_raw.append(val)
                    is_log_var.append(False)
                    free_slots.append({"slot": sl, "mat": "B", "r": n_lv + i_ov, "c": j_lv})
                    param_meta.append({"lhs": lv_names[j_lv], "op": "=~", "rhs": ov_names[i_ov], "label": "", "slot": sl, "est": val})

        for i_lv in range(n_lv):
            for j_lv in range(i_lv, n_lv):
                val = psi[i_lv, j_lv]
                if i_lv == j_lv and val == 1.0 and np.all(fit.se == 0.0):
                    psi_fixed[i_lv, i_lv] = 1.0
                elif val != 0.0:
                    sl = len(init_raw)
                    log_v = (i_lv == j_lv) and (val > 1e-6)
                    init_raw.append(np.log(val) if log_v else val)
                    is_log_var.append(log_v)
                    free_slots.append({"slot": sl, "mat": "Psi", "r": i_lv, "c": j_lv})
                    param_meta.append({"lhs": lv_names[i_lv], "op": "~~", "rhs": lv_names[j_lv], "label": "", "slot": sl, "est": val})

        for i_ov in range(p):
            for j_ov in range(i_ov, p):
                val = theta_res[i_ov, j_ov]
                if val != 0.0:
                    sl = len(init_raw)
                    log_v = (i_ov == j_ov) and (val > 1e-6)
                    init_raw.append(np.log(val) if log_v else val)
                    is_log_var.append(log_v)
                    free_slots.append({"slot": sl, "mat": "Psi", "r": n_lv + i_ov, "c": n_lv + j_ov})
                    param_meta.append({"lhs": ov_names[i_ov], "op": "~~", "rhs": ov_names[j_ov], "label": "", "slot": sl, "est": val})

        return {
            "data": fit.data_raw[ov_names].dropna().to_numpy(dtype=np.float64),
            "ov_names": ov_names,
            "k_tot": k_tot,
            "n_lv": n_lv,
            "p": p,
            "b_fixed": b_fixed,
            "psi_fixed": psi_fixed,
            "free_slots": free_slots,
            "init_raw": np.array(init_raw, dtype=np.float64),
            "is_log_var": np.array(is_log_var, dtype=bool),
            "label_to_slot": {},
            "param_meta": param_meta,
            "defined_rows": [],
        }

    raise TypeError(f"Tipo '{cls_name}' não suportado por bootstrap_ci().")


def _solve_batch_sem_torch(
    s_batch: torch.Tensor,
    spec: dict,
    max_iter: int = 18,
) -> torch.Tensor:
    """
    Otimiza simultaneamente B modelos SEM/CFA sobre o tensor 3D s_batch [B, p, p]
    usando Fisher Scoring vetorial em lote (GPU CUDA ou CPU).
    Retorna os parâmetros naturais [B, n_free].
    """
    dev = s_batch.device
    dtype = s_batch.dtype
    b_size, p, _ = s_batch.shape
    k_tot = spec["k_tot"]
    n_lv = spec["n_lv"]
    init_raw = torch.tensor(spec["init_raw"], device=dev, dtype=dtype)
    n_free = init_raw.shape[0]
    is_log_t = torch.tensor(spec["is_log_var"], device=dev, dtype=torch.bool)

    b_fixed_t = torch.tensor(spec["b_fixed"], device=dev, dtype=dtype).unsqueeze(0).expand(b_size, -1, -1)
    psi_fixed_t = torch.tensor(spec["psi_fixed"], device=dev, dtype=dtype).unsqueeze(0).expand(b_size, -1, -1)
    eye_k = torch.eye(k_tot, device=dev, dtype=dtype).unsqueeze(0).expand(b_size, -1, -1)
    eye_f = torch.eye(n_free, device=dev, dtype=dtype).unsqueeze(0)

    raw_b = init_raw.unsqueeze(0).repeat(b_size, 1)  # [B, n_free]

    def _raw_to_nat(r_t: torch.Tensor) -> torch.Tensor:
        return torch.where(is_log_t.unsqueeze(0), torch.exp(torch.clamp(r_t, -15.0, 15.0)), r_t)

    def _unpack_sigma(nat_t: torch.Tensor) -> torch.Tensor:
        b_mat = b_fixed_t.clone()
        psi_mat = psi_fixed_t.clone()
        for fs in spec["free_slots"]:
            col_v = nat_t[:, fs["slot"]]
            if fs["mat"] == "B":
                b_mat[:, fs["r"], fs["c"]] = col_v
            else:
                psi_mat[:, fs["r"], fs["c"]] = col_v
                psi_mat[:, fs["c"], fs["r"]] = col_v
        inv_ib = torch.linalg.solve(eye_k - b_mat, eye_k)
        sigma_all = torch.bmm(torch.bmm(inv_ib, psi_mat), inv_ib.transpose(1, 2))
        return sigma_all[:, n_lv:, n_lv:]

    def _loss_vec(r_t: torch.Tensor) -> torch.Tensor:
        sig = _unpack_sigma(_raw_to_nat(r_t))
        sign, logdet = torch.linalg.slogdet(sig)
        tr_term = torch.sum(torch.linalg.solve(sig, s_batch) * torch.eye(p, device=dev, dtype=dtype).unsqueeze(0), dim=(1, 2))
        val = 0.5 * (logdet + tr_term - float(p))
        return torch.where(sign > 0, val, torch.full_like(val, 1e12))

    # Otimização Fisher Scoring em lote com busca de linha vetorizada
    for _ in range(max_iter):
        r_req = raw_b.detach().requires_grad_(True)
        nat_req = _raw_to_nat(r_req)
        sig_req = _unpack_sigma(nat_req)

        with torch.no_grad():
            sig_inv = torch.linalg.inv(sig_req)
            # Gradiente matricial exato dF/dSigma = 0.5 * (Sigma^-1 - Sigma^-1 S Sigma^-1)
            df_dsig = 0.5 * (sig_inv - torch.bmm(torch.bmm(sig_inv, s_batch), sig_inv))

        grad_b = torch.autograd.grad(torch.sum(sig_req * df_dsig), r_req)[0]  # [B, n_free]
        if float(torch.max(torch.abs(grad_b))) < 1e-8:
            break

        # Informação esperada de Fisher por diferenças finitas vetoriais em dSigma/dtheta
        eps = 1e-6
        dsig_list = []
        with torch.no_grad():
            for u in range(n_free):
                r_p = raw_b.clone()
                r_m = raw_b.clone()
                r_p[:, u] += eps
                r_m[:, u] -= eps
                ds_u = (_unpack_sigma(_raw_to_nat(r_p)) - _unpack_sigma(_raw_to_nat(r_m))) / (2.0 * eps)
                # Pré-multiplica por Sigma^-1: [B, p, p]
                dsig_list.append(torch.bmm(sig_inv, ds_u))

            info_b = torch.zeros((b_size, n_free, n_free), device=dev, dtype=dtype)
            for u in range(n_free):
                for v in range(u, n_free):
                    # 0.5 * tr((Sigma^-1 dSigma_u) @ (Sigma^-1 dSigma_v))
                    tr_uv = 0.5 * torch.sum(dsig_list[u] * dsig_list[v].transpose(1, 2), dim=(1, 2))
                    info_b[:, u, v] = tr_uv
                    info_b[:, v, u] = tr_uv

            step_b = torch.linalg.solve(info_b + 1e-6 * eye_f, grad_b.unsqueeze(2)).squeeze(2)
            f_cur = _loss_vec(raw_b)

            alpha = torch.ones((b_size, 1), device=dev, dtype=dtype)
            for _ in range(10):
                cand = raw_b - alpha * step_b
                f_cand = _loss_vec(cand)
                improve = torch.isfinite(f_cand) & (f_cand <= f_cur + 1e-10)
                if bool(torch.all(improve)):
                    break
                alpha = torch.where(improve.unsqueeze(1), alpha, 0.5 * alpha)

            cand_final = raw_b - alpha * step_b
            f_final = _loss_vec(cand_final)
            accept = torch.isfinite(f_final) & (f_final <= f_cur + 1e-10)
            raw_b = torch.where(accept.unsqueeze(1), cand_final, raw_b)

    return _raw_to_nat(raw_b).detach()


def bootstrap_ci(
    fit,
    n_boot: int = 1000,
    conf_level: float = 0.95,
    method: str = "bca",
    seed: int = 42,
    device: str = "auto",
) -> BootstrapResult:
    """
    Calcula intervalos de confiança bootstrap não-paramétricos ('bca' ou 'percentile')
    acelerados em GPU (CUDA) para CFA e SEM (incluindo efeitos indiretos/mediação ':=').
    """
    m_key = method.strip().lower()
    if m_key not in ("bca", "percentile"):
        raise ValueError("method deve ser 'bca' ou 'percentile'.")

    spec = _extract_sem_spec_for_batch(fit)
    x_np = spec["data"]
    n_obs, p = x_np.shape

    if device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (device.lower() == "cuda") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    x_t = torch.tensor(x_np, device=dev, dtype=torch.float64)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    # Reamostragem não-paramétrica e covariâncias em lote na GPU
    chunk_size = min(n_boot, 512)
    nat_chunks: list[np.ndarray] = []

    for start_b in range(0, n_boot, chunk_size):
        b_cur = min(chunk_size, n_boot - start_b)
        idx = torch.randint(0, n_obs, (b_cur, n_obs), generator=gen, device=dev)
        x_b = x_t[idx]  # [b_cur, n_obs, p]
        x_c = x_b - torch.mean(x_b, dim=1, keepdim=True)
        s_b = torch.bmm(x_c.transpose(1, 2), x_c) / float(n_obs)
        nat_b = _solve_batch_sem_torch(s_b, spec)
        nat_chunks.append(nat_b.cpu().numpy())

    nat_boot = np.vstack(nat_chunks)  # [n_boot, n_free]

    # Jackknife leave-one-out em lote na GPU para fator de aceleração BCa
    if m_key == "bca":
        loo_s_list = []
        for i in range(n_obs):
            xi = np.delete(x_np, i, axis=0)
            loo_s_list.append(np.cov(xi, rowvar=False, bias=True))
        s_loo_t = torch.tensor(np.stack(loo_s_list, axis=0), device=dev, dtype=torch.float64)
        nat_loo = _solve_batch_sem_torch(s_loo_t, spec).cpu().numpy()  # [n_obs, n_free]
    else:
        nat_loo = None

    # Monta séries de parâmetros livres e definidos (:=)
    label_to_slot = spec["label_to_slot"]

    def _eval_defined(nat_mat: np.ndarray, expr: str) -> np.ndarray:
        env = {lbl: nat_mat[:, sl] for lbl, sl in label_to_slot.items()}
        return np.asarray(eval(expr, {"__builtins__": {}}, env), dtype=np.float64)

    all_series: list[dict] = []
    for pm in spec["param_meta"]:
        sl = pm["slot"]
        all_series.append({
            "lhs": pm["lhs"],
            "op": pm["op"],
            "rhs": pm["rhs"],
            "label": pm["label"],
            "est": float(pm["est"]),
            "boot": nat_boot[:, sl],
            "loo": nat_loo[:, sl] if nat_loo is not None else None,
        })

    for def_lhs, def_expr in spec["defined_rows"]:
        est_row = fit.pe_df[(fit.pe_df["lhs"] == def_lhs) & (fit.pe_df["op"] == ":=")]
        est_d = float(est_row["est"].iloc[0]) if not est_row.empty else float("nan")
        all_series.append({
            "lhs": def_lhs,
            "op": ":=",
            "rhs": def_expr,
            "label": def_lhs,
            "est": est_d,
            "boot": _eval_defined(nat_boot, def_expr),
            "loo": _eval_defined(nat_loo, def_expr) if nat_loo is not None else None,
        })

    alpha = 1.0 - conf_level
    z_low = float(stats.norm.ppf(alpha / 2.0))
    z_high = float(stats.norm.ppf(1.0 - alpha / 2.0))

    rows_out = []
    boot_cols: dict[str, np.ndarray] = {}

    for item in all_series:
        b_vec = item["boot"]
        est_0 = item["est"]
        col_name = item["label"] if item["label"] else f"{item['lhs']}{item['op']}{item['rhs']}"
        boot_cols[col_name] = b_vec

        se_b = float(np.std(b_vec, ddof=1))
        bias_b = float(np.mean(b_vec) - est_0)

        if m_key == "percentile":
            ci_lo, ci_hi = np.percentile(b_vec, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])
        else:
            prop_less = np.clip(np.mean(b_vec < est_0), 1.0 / (2.0 * n_boot), 1.0 - 1.0 / (2.0 * n_boot))
            z0 = float(stats.norm.ppf(prop_less))
            loo_vec = item["loo"]
            loo_diff = np.mean(loo_vec) - loo_vec
            denom = 6.0 * (np.sum(loo_diff ** 2) ** 1.5)
            acc = float(np.sum(loo_diff ** 3) / denom) if denom > 1e-15 else 0.0

            p1 = float(stats.norm.cdf(z0 + (z0 + z_low) / max(1e-6, 1.0 - acc * (z0 + z_low))))
            p2 = float(stats.norm.cdf(z0 + (z0 + z_high) / max(1e-6, 1.0 - acc * (z0 + z_high))))
            p1 = float(np.clip(p1, 0.001, 0.999))
            p2 = float(np.clip(p2, 0.001, 0.999))
            ci_lo, ci_hi = np.quantile(b_vec, [p1, p2])

        p_boot = 2.0 * min(float(np.mean(b_vec <= 0.0)), float(np.mean(b_vec >= 0.0)))
        rows_out.append({
            "lhs": item["lhs"],
            "op": item["op"],
            "rhs": item["rhs"],
            "label": item["label"],
            "est": est_0,
            "boot_se": se_b,
            "bias": bias_b,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "pvalue": float(min(1.0, p_boot)),
        })

    return BootstrapResult(
        table=pd.DataFrame(rows_out),
        boot_samples=pd.DataFrame(boot_cols),
        n_boot=n_boot,
        conf_level=conf_level,
        method=m_key,
        device=str(dev),
    )