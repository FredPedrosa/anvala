from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

import anvala as av

warnings.filterwarnings("ignore")


def _std_chisq3(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Gera variáveis padronizadas com assimetria ~1.63 e excesso de curtose = 4.0."""
    return (rng.chisquare(df=3.0, size=shape) - 3.0) / np.sqrt(6.0)


def _summarize_replications(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    rows = []
    for (scenario, n_obs, param), grp in df.groupby(["scenario", "N", "parameter"], sort=False):
        conv_rate = float(grp["converged"].mean())
        valid = grp[grp["converged"]].copy()
        true_val = float(valid["true_val"].iloc[0])
        est = valid["est"].to_numpy(dtype=float)
        se = valid["se"].to_numpy(dtype=float)
        cov = valid["covered"].to_numpy(dtype=bool)

        mean_est = float(np.mean(est))
        rel_bias = 100.0 * (mean_est - true_val) / true_val if abs(true_val) > 1e-8 else np.nan
        emp_sd = float(np.std(est, ddof=1))
        mean_se = float(np.mean(se))
        se_sd_ratio = mean_se / emp_sd if emp_sd > 0 else np.nan
        rmse = float(np.sqrt(np.mean((est - true_val) ** 2)))
        coverage_95 = float(np.mean(cov))

        rows.append({
            "scenario": scenario,
            "N": int(n_obs),
            "parameter": param,
            "true_val": round(true_val, 4),
            "mean_est": round(mean_est, 4),
            "rel_bias_pct": round(rel_bias, 2),
            "emp_sd": round(emp_sd, 4),
            "mean_se": round(mean_se, 4),
            "se_sd_ratio": round(se_sd_ratio, 3),
            "rmse": round(rmse, 4),
            "coverage_95": round(coverage_95, 3),
            "conv_rate": round(conv_rate, 3),
        })
    return pd.DataFrame(rows)


def run_cfa_mlr_study(
    n_list: list[int] = [150, 300, 600],
    n_reps: int = 200,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estudo 2A: CFA Contínua Não-Normal com estimador robusto MLR."""
    rng = np.random.default_rng(seed)
    model = """
        F1 =~ x1 + x2 + x3
        F2 =~ x4 + x5 + x6
    """
    # Covariância latente: Var(F1)=Var(F2)=0.64 (SD=0.80), Cov(F1,F2)=0.256 (r=0.40)
    psi = np.array([[0.64, 0.256], [0.256, 0.64]])
    l_psi = np.linalg.cholesky(psi)

    lam = np.array([
        [1.00, 0.00],
        [0.90, 0.00],
        [0.80, 0.00],
        [0.00, 1.00],
        [0.00, 0.90],
        [0.00, 0.80],
    ])
    theta_var = 1.0 - (lam.sum(axis=1) ** 2) * 0.64
    theta_sd = np.sqrt(theta_var)

    targets = [
        ("F1", "=~", "x2", "F1 =~ x2", 0.90),
        ("F1", "=~", "x3", "F1 =~ x3", 0.80),
        ("F1", "~~", "F2", "F1 ~~ F2", 0.256),
        ("F1", "~~", "F1", "F1 ~~ F1", 0.64),
        ("x2", "~~", "x2", "x2 ~~ x2", float(theta_var[1])),
    ]

    records = []
    for n_obs in n_list:
        print(f"[Estudo 2A | CFA MLR Não-Normal] N = {n_obs} ({n_reps} réplicas)...", flush=True)
        for rep in range(n_reps):
            z_f = _std_chisq3(rng, (n_obs, 2)) @ l_psi.T
            z_e = _std_chisq3(rng, (n_obs, 6)) * theta_sd
            x = z_f @ lam.T + z_e
            df = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(6)])

            fit = av.cfa(model, data=df, estimator="MLR")
            pe = fit.parameter_estimates()
            for lhs, op, rhs, label, t_val in targets:
                row = pe[(pe["lhs"] == lhs) & (pe["op"] == op) & (pe["rhs"] == rhs)].iloc[0]
                est = float(row["est"])
                se = float(row["se"])
                ci_low = est - 1.959964 * se
                ci_high = est + 1.959964 * se
                records.append({
                    "scenario": "CFA_NonNormal_MLR",
                    "N": n_obs,
                    "rep": rep,
                    "parameter": label,
                    "true_val": t_val,
                    "est": est,
                    "se": se,
                    "covered": bool(ci_low <= t_val <= ci_high),
                    "converged": bool(fit.converged) and np.isfinite(se) and se > 0,
                })

    return _summarize_replications(records)


def run_cfa_ordinal_study(
    n_list: list[int] = [150, 300, 600],
    n_reps: int = 200,
    seed: int = 3026,
) -> pd.DataFrame:
    """Estudo 2B: CFA Ordinal (4 categorias) com estimador WLSMV."""
    rng = np.random.default_rng(seed)
    model = "F1 =~ u1 + u2 + u3 + u4 + u5"
    loadings = np.array([0.65, 0.70, 0.75, 0.80, 0.85])
    err_sd = np.sqrt(1.0 - loadings**2)
    taus = np.array([-1.1, 0.0, 1.1])

    targets = [
        ("F1", "=~", "u1", "F1 =~ u1", 0.65),
        ("F1", "=~", "u3", "F1 =~ u3", 0.75),
        ("F1", "=~", "u5", "F1 =~ u5", 0.85),
    ]

    records = []
    for n_obs in n_list:
        print(f"[Estudo 2B | CFA Ordinal WLSMV] N = {n_obs} ({n_reps} réplicas)...", flush=True)
        for rep in range(n_reps):
            f = rng.normal(size=(n_obs, 1))
            y_star = f @ loadings[None, :] + rng.normal(scale=err_sd, size=(n_obs, 5))
            u = np.searchsorted(taus, y_star) + 1
            df = pd.DataFrame(u, columns=[f"u{i + 1}" for i in range(5)])

            fit = av.cfa(model, data=df, estimator="WLSMV", ordered=True, std_lv=True)
            pe = fit.parameter_estimates()
            for lhs, op, rhs, label, t_val in targets:
                row = pe[(pe["lhs"] == lhs) & (pe["op"] == op) & (pe["rhs"] == rhs)].iloc[0]
                est = float(row["est"])
                se = float(row["se"])
                ci_low = est - 1.959964 * se
                ci_high = est + 1.959964 * se
                records.append({
                    "scenario": "CFA_Ordinal_WLSMV",
                    "N": n_obs,
                    "rep": rep,
                    "parameter": label,
                    "true_val": t_val,
                    "est": est,
                    "se": se,
                    "covered": bool(ci_low <= t_val <= ci_high),
                    "converged": bool(fit.converged) and np.isfinite(se) and se > 0,
                })

    return _summarize_replications(records)


def run_mediation_bca_study(
    n_list: list[int] = [150, 300, 600],
    n_reps: int = 200,
    n_boot: int = 400,
    seed: int = 4026,
) -> pd.DataFrame:
    """Estudo 2C: Mediação SEM com intervalos de confiança Bootstrap BCa."""
    rng = np.random.default_rng(seed)
    model = """
        M ~ a*X
        Y ~ b*M + c*X
        indirect := a*b
    """
    a_true, b_true, c_true = 0.50, 0.45, 0.20
    ind_true = a_true * b_true

    targets = [
        ("a", "a (X -> M)", a_true),
        ("b", "b (M -> Y)", b_true),
        ("c", "c' (X -> Y)", c_true),
        ("indirect", "indirect (a*b)", ind_true),
    ]

    records = []
    for n_obs in n_list:
        print(f"[Estudo 2C | Mediação BCa (B={n_boot})] N = {n_obs} ({n_reps} réplicas)...", flush=True)
        for rep in range(n_reps):
            x = rng.normal(size=n_obs)
            m = a_true * x + rng.normal(size=n_obs)
            y = b_true * m + c_true * x + rng.normal(size=n_obs)
            df = pd.DataFrame({"X": x, "M": m, "Y": y})

            fit = av.sem(model, data=df, estimator="ML")
            b_res = fit.bootstrap_ci(n_boot=n_boot, method="bca", seed=seed + rep, device="cpu").table

            for lbl, p_name, t_val in targets:
                row = b_res[b_res["label"] == lbl].iloc[0]
                est = float(row["est"])
                se = float(row["boot_se"])
                ci_low = float(row["ci_lower"])
                ci_high = float(row["ci_upper"])
                records.append({
                    "scenario": "SEM_Mediation_BCa",
                    "N": n_obs,
                    "rep": rep,
                    "parameter": p_name,
                    "true_val": t_val,
                    "est": est,
                    "se": se,
                    "covered": bool(ci_low <= t_val <= ci_high),
                    "converged": bool(fit.converged) and np.isfinite(ci_low) and np.isfinite(ci_high),
                })

    return _summarize_replications(records)


if __name__ == "__main__":
    os.makedirs("simulations/results", exist_ok=True)
    t_start = time.perf_counter()

    df_mlr = run_cfa_mlr_study()
    df_ord = run_cfa_ordinal_study()
    df_med = run_mediation_bca_study()

    df_all = pd.concat([df_mlr, df_ord, df_med], ignore_index=True)
    out_path = "simulations/results/mc_parameter_recovery.csv"
    df_all.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t_start
    print(f"\nConcluído em {elapsed:.1f}s! Tabela salva em {out_path}\n")
    print(df_all.to_string(index=False))