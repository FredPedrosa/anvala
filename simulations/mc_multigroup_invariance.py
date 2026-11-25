from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

import anvala as av

warnings.filterwarnings("ignore")

MODEL_MG = "F1 =~ x1 + x2 + x3 + x4"
LAM_BASE = np.array([0.80, 0.75, 0.70, 0.75])
NU_BASE = np.array([0.0, 0.0, 0.0, 0.0])
THETA_SD = np.sqrt(1.0 - LAM_BASE**2)


def _sim_two_groups(
    rng: np.random.Generator,
    n_per_group: int,
    lam2: np.ndarray,
    nu2: np.ndarray,
    kappa2: float = 0.0,
) -> pd.DataFrame:
    f1 = rng.normal(loc=0.0, scale=1.0, size=(n_per_group, 1))
    x1 = NU_BASE + f1 @ LAM_BASE[None, :] + rng.normal(scale=THETA_SD, size=(n_per_group, 4))
    df1 = pd.DataFrame(x1, columns=["x1", "x2", "x3", "x4"])
    df1["grp"] = "G1"

    f2 = rng.normal(loc=kappa2, scale=1.0, size=(n_per_group, 1))
    x2 = nu2 + f2 @ lam2[None, :] + rng.normal(scale=THETA_SD, size=(n_per_group, 4))
    df2 = pd.DataFrame(x2, columns=["x1", "x2", "x3", "x4"])
    df2["grp"] = "G2"

    return pd.concat([df1, df2], ignore_index=True)


def run_invariance_error_and_power(
    n_list: list[int] = [150, 300],
    n_reps: int = 200,
    seed: int = 5026,
) -> pd.DataFrame:
    """Estudos 3A (Erro Tipo I) e 3B (Poder sob não-invariância métrica/escalar)."""
    rng = np.random.default_rng(seed)
    conditions = [
        ("3A_Full_Invariance_TypeI", LAM_BASE.copy(), NU_BASE.copy()),
        ("3B_NonInvariance_Power", np.array([0.80, 0.75, 0.40, 0.75]), np.array([0.0, 0.0, 0.50, 0.0])),
    ]

    records = []
    for cond_name, lam2, nu2 in conditions:
        for n_grp in n_list:
            print(f"[{cond_name}] N_grupo = {n_grp} ({n_reps} réplicas)...", flush=True)
            for rep in range(n_reps):
                df = _sim_two_groups(rng, n_grp, lam2=lam2, nu2=nu2, kappa2=0.0)
                inv = av.multigroup.measurement_invariance(MODEL_MG, df, group="grp", estimator="ML")
                tbl = inv.table.set_index("model")

                for lvl in ["metric", "scalar", "strict"]:
                    row = tbl.loc[lvl]
                    p_val = float(row["pvalue"])
                    d_cfi = float(row["delta_cfi"])
                    d_rmsea = float(row["delta_rmsea"])
                    records.append({
                        "condition": cond_name,
                        "N_per_group": n_grp,
                        "transition": lvl,
                        "mean_delta_chisq": float(row["chisq_diff"]),
                        "reject_lrt_05": float(p_val < 0.05),
                        "mean_delta_cfi": d_cfi,
                        "reject_chen_cfi_01": float(d_cfi < -0.010),
                        "mean_delta_rmsea": d_rmsea,
                        "reject_chen_rmsea_015": float(d_rmsea > 0.015),
                    })

    raw_df = pd.DataFrame(records)
    summary = (
        raw_df.groupby(["condition", "N_per_group", "transition"], sort=False)
        .mean()
        .reset_index()
    )
    for col in summary.columns:
        if col not in ("condition", "N_per_group", "transition"):
            summary[col] = summary[col].round(4)
    return summary


def run_partial_invariance_study(
    n_list: list[int] = [150, 300],
    n_reps: int = 200,
    seed: int = 6026,
) -> pd.DataFrame:
    """Estudo 3C: Recuperação de média latente (kappa2 = 0.50) sob Escalar Completa vs. Parcial."""
    rng = np.random.default_rng(seed)
    lam2 = np.array([0.80, 0.75, 0.40, 0.75])
    nu2 = np.array([0.0, 0.0, 0.50, 0.0])
    kappa2_dgp = 0.50
    kappa2_true = 0.80 * kappa2_dgp  # 0.40 na m?trica do indicador marcador x1 (lambda_1 = 0.80)

    records = []
    for n_grp in n_list:
        print(f"[3C_Partial_Invariance_LatentMean] N_grupo = {n_grp} ({n_reps} réplicas)...", flush=True)
        for rep in range(n_reps):
            df = _sim_two_groups(rng, n_grp, lam2=lam2, nu2=nu2, kappa2=kappa2_dgp)

            specs = [
                ("Full_Scalar_Misspecified", None),
                ("Partial_Scalar_Correct", ["F1 =~ x3", "x3 ~ 1"]),
            ]
            for spec_label, partial_terms in specs:
                fit = av.multigroup.cfa_multigroup(
                    MODEL_MG,
                    df,
                    group="grp",
                    estimator="ML",
                    group_equal=["loadings", "intercepts"],
                    group_partial=partial_terms,
                )
                pe = fit.parameter_estimates()
                f_means = pe[(pe["lhs"] == "F1") & (pe["op"] == "~1")]
                free_means = f_means[f_means["se"] > 0]
                row_k2 = free_means.iloc[0] if len(free_means) > 0 else f_means.iloc[-1]

                est = float(row_k2["est"])
                se = float(row_k2["se"])
                ci_low = est - 1.959964 * se
                ci_high = est + 1.959964 * se

                records.append({
                    "model_spec": spec_label,
                    "N_per_group": n_grp,
                    "true_kappa2": kappa2_true,
                    "est": est,
                    "se": se,
                    "covered": bool(ci_low <= kappa2_true <= ci_high),
                })

    raw_df = pd.DataFrame(records)
    rows = []
    for (spec_label, n_grp), grp in raw_df.groupby(["model_spec", "N_per_group"], sort=False):
        est = grp["est"].to_numpy(dtype=float)
        se = grp["se"].to_numpy(dtype=float)
        mean_est = float(np.mean(est))
        rel_bias = 100.0 * (mean_est - kappa2_true) / kappa2_true
        rmse = float(np.sqrt(np.mean((est - kappa2_true) ** 2)))
        rows.append({
            "model_spec": spec_label,
            "N_per_group": int(n_grp),
            "true_kappa2": kappa2_true,
            "mean_est": round(mean_est, 4),
            "rel_bias_pct": round(rel_bias, 2),
            "mean_se": round(float(np.mean(se)), 4),
            "rmse": round(rmse, 4),
            "coverage_95": round(float(grp["covered"].mean()), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("simulations/results", exist_ok=True)
    t0 = time.perf_counter()

    df_inv = run_invariance_error_and_power()
    path_inv = "simulations/results/mc_invariance_rates.csv"
    df_inv.to_csv(path_inv, index=False)

    df_part = run_partial_invariance_study()
    path_part = "simulations/results/mc_partial_invariance.csv"
    df_part.to_csv(path_part, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nEstudo 3 concluído em {elapsed:.1f}s!\n")
    print("=== Estudo 3A/3B: Taxas de Erro Tipo I e Poder ===")
    print(df_inv.to_string(index=False))
    print("\n=== Estudo 3C: Recuperação de Média Latente (Escalar Completa vs. Parcial) ===")
    print(df_part.to_string(index=False))