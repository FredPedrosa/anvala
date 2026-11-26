from __future__ import annotations

import os
import sys

# Configura o R 4.6.1 no Windows antes de importar o rpy2
if sys.platform == "win32":
    r_home = r"C:\Program Files\R\R-4.6.1"
    r_bin = os.path.join(r_home, "bin", "x64")
    if os.path.isdir(r_home):
        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_bin + os.pathsep + os.environ.get("PATH", "")
        os.environ["LANGUAGE"] = "en"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(r_bin)

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av
from anvala.bifactor import bifactor_indices


def main() -> None:
    os.makedirs("simulations/results", exist_ok=True)
    rows: list[dict[str, object]] = []

    def add_row(
        domain: str,
        dataset: str,
        r_pkg: str,
        metric: str,
        r_val: float | str,
        py_val: float | str,
        max_abs_diff: float,
    ) -> None:
        rows.append(
            {
                "domain": domain,
                "dataset": dataset,
                "r_reference": r_pkg,
                "metric": metric,
                "R_value": round(r_val, 6) if isinstance(r_val, (float, np.floating)) else r_val,
                "anvala_value": round(py_val, 6) if isinstance(py_val, (float, np.floating)) else py_val,
                "max_abs_diff": f"{max_abs_diff:.2e}",
            }
        )

    # ------------------------------------------------------------------
    # 1. CFA Continuous (ML) - HolzingerSwineford1939
    # ------------------------------------------------------------------
    ro.r("""
        library(lavaan)
        mod_hs <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit_ml <- cfa(mod_hs, data = HolzingerSwineford1939, estimator = "ML")
        pe_ml <- parameterEstimates(fit_ml, standardized = TRUE)
        fm_ml <- fitMeasures(fit_ml, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("HolzingerSwineford1939"))

    syntax_hs = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py_ml = av.cfa(syntax_hs, data=hs_df, estimator="ML")
    pe_py_ml = fit_py_ml.parameter_estimates()
    fm_py_ml = fit_py_ml.fit_measures()["Unscaled"]

    r_est_ml = np.array(ro.r("pe_ml$est"))
    r_se_ml = np.array(ro.r("pe_ml$se"))
    r_fm_ml = np.array(ro.r("as.numeric(fm_ml)"))
    py_fm_ml_vec = np.array([
        fm_py_ml["chisq"], fm_py_ml["df"], fm_py_ml["pvalue"],
        fm_py_ml["cfi"], fm_py_ml["tli"], fm_py_ml["rmsea"], fm_py_ml["srmr"],
    ])
    max_diff_ml = max(
        float(np.max(np.abs(pe_py_ml["est"].to_numpy() - r_est_ml))),
        float(np.max(np.abs(pe_py_ml["se"].to_numpy() - r_se_ml))),
        float(np.max(np.abs(py_fm_ml_vec - r_fm_ml))),
    )
    add_row("CFA (ML)", "HS1939 (N=301)", "lavaan 0.7-2", "lambda(visual=~x2)", float(r_est_ml[1]), float(pe_py_ml["est"].iloc[1]), max_diff_ml)
    add_row("CFA (ML)", "HS1939 (N=301)", "lavaan 0.7-2", "Chi-square (df=24)", float(r_fm_ml[0]), float(fm_py_ml["chisq"]), max_diff_ml)
    add_row("CFA (ML)", "HS1939 (N=301)", "lavaan 0.7-2", "CFI / RMSEA", f"{r_fm_ml[3]:.5f} / {r_fm_ml[5]:.5f}", f"{fm_py_ml['cfi']:.5f} / {fm_py_ml['rmsea']:.5f}", max_diff_ml)

    # ------------------------------------------------------------------
    # 2. CFA Continuous Robust (MLR) - HolzingerSwineford1939
    # ------------------------------------------------------------------
    ro.r("""
        fit_mlr <- cfa(mod_hs, data = HolzingerSwineford1939, estimator = "MLR")
        pe_mlr <- parameterEstimates(fit_mlr, standardized = TRUE)
        fm_mlr <- fitMeasures(fit_mlr, c("chisq.scaled", "chisq.scaling.factor", "cfi.robust", "tli.robust", "rmsea.robust", "srmr"))
    """)
    fit_py_mlr = av.cfa(syntax_hs, data=hs_df, estimator="MLR")
    pe_py_mlr = fit_py_mlr.parameter_estimates()
    ft_mlr = fit_py_mlr.fit_measures()

    r_est_mlr = np.array(ro.r("pe_mlr$est"))
    r_se_mlr = np.array(ro.r("pe_mlr$se"))
    r_fm_mlr = np.array(ro.r("as.numeric(fm_mlr)"))
    py_fm_mlr_vec = np.array([
        ft_mlr.loc["chisq", "Scaled"], ft_mlr.loc["scaling_factor", "Scaled"],
        ft_mlr.loc["cfi", "Robust"], ft_mlr.loc["tli", "Robust"],
        ft_mlr.loc["rmsea", "Robust"], ft_mlr.loc["srmr", "Unscaled"],
    ])
    max_diff_mlr = max(
        float(np.max(np.abs(pe_py_mlr["est"].to_numpy() - r_est_mlr))),
        float(np.max(np.abs(pe_py_mlr["se"].to_numpy() - r_se_mlr))),
        float(np.max(np.abs(py_fm_mlr_vec - r_fm_mlr))),
    )
    add_row("CFA (MLR)", "HS1939 (N=301)", "lavaan 0.7-2", "Sandwich SE(visual=~x2)", float(r_se_mlr[1]), float(pe_py_mlr["se"].iloc[1]), max_diff_mlr)
    add_row("CFA (MLR)", "HS1939 (N=301)", "lavaan 0.7-2", "Scaled Chi-sq / Robust CFI", f"{r_fm_mlr[0]:.5f} / {r_fm_mlr[2]:.5f}", f"{ft_mlr.loc['chisq', 'Scaled']:.5f} / {ft_mlr.loc['cfi', 'Robust']:.5f}", max_diff_mlr)

    # ------------------------------------------------------------------
    # 3. Structural Equation Model (ML) - PoliticalDemocracy
    # ------------------------------------------------------------------
    ro.r("""
        mod_pd <- '
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8
        '
        fit_pd <- sem(mod_pd, data = PoliticalDemocracy, estimator = "ML")
        pe_pd <- parameterEstimates(fit_pd)
        fm_pd <- fitMeasures(fit_pd, c("chisq", "cfi", "tli", "rmsea", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_df = ro.conversion.rpy2py(ro.r("PoliticalDemocracy"))
        r_pe_pd_df = ro.conversion.rpy2py(ro.r("pe_pd[, c('lhs', 'op', 'rhs', 'est', 'se')]"))

    syntax_pd = """
        ind60 =~ x1 + x2 + x3
        dem60 =~ y1 + y2 + y3 + y4
        dem65 =~ y5 + y6 + y7 + y8
        dem60 ~ ind60
        dem65 ~ ind60 + dem60
        y1 ~~ y5
        y2 ~~ y4 + y6
        y3 ~~ y7
        y4 ~~ y8
        y6 ~~ y8
    """
    fit_py_pd = av.sem(syntax_pd, data=pd_df, estimator="ML")
    pe_py_pd = fit_py_pd.parameter_estimates()
    fm_py_pd = fit_py_pd.fit_measures()["Unscaled"]

    merged_pd = pd.merge(
        pe_py_pd[["lhs", "op", "rhs", "est", "se"]],
        r_pe_pd_df,
        on=["lhs", "op", "rhs"],
        suffixes=("_py", "_r"),
    )
    r_fm_pd = np.array(ro.r("as.numeric(fm_pd)"))
    py_fm_pd_vec = np.array([
        fm_py_pd["chisq"], fm_py_pd["cfi"], fm_py_pd["tli"], fm_py_pd["rmsea"], fm_py_pd["srmr"],
    ])
    max_diff_pd = max(
        float(np.max(np.abs(merged_pd["est_py"] - merged_pd["est_r"]))),
        float(np.max(np.abs(merged_pd["se_py"] - merged_pd["se_r"]))),
        float(np.max(np.abs(py_fm_pd_vec - r_fm_pd))),
    )
    row_path = merged_pd[(merged_pd["lhs"] == "dem65") & (merged_pd["rhs"] == "dem60")].iloc[0]
    add_row("SEM (ML)", "PolDem (N=75)", "lavaan 0.7-2", "Path dem65 ~ dem60 (SE)", f"{row_path['est_r']:.5f} ({row_path['se_r']:.5f})", f"{row_path['est_py']:.5f} ({row_path['se_py']:.5f})", max_diff_pd)
    add_row("SEM (ML)", "PolDem (N=75)", "lavaan 0.7-2", "Chi-sq / CFI", f"{r_fm_pd[0]:.5f} / {r_fm_pd[1]:.5f}", f"{fm_py_pd['chisq']:.5f} / {fm_py_pd['cfi']:.5f}", max_diff_pd)

    # ------------------------------------------------------------------
    # 4. Ordinal CFA (WLSMV, Delta) - Discretized HS1939 (5 categories)
    # ------------------------------------------------------------------
    ro.r("""
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        hs_ord <- as.data.frame(lapply(hs, function(col) {
            as.integer(cut(col, breaks = quantile(col, probs = seq(0, 1, 0.2)), include.lowest = TRUE))
        }))
        fit_w <- cfa(mod_hs, data = hs_ord, ordered = TRUE, estimator = "WLSMV")
        pe_w <- parameterEstimates(fit_w, standardized = TRUE)
        pe_w_struct <- pe_w[pe_w$op %in% c("=~", "~~") & !(pe_w$op == "~~" & pe_w$lhs == pe_w$rhs & pe_w$lhs %in% paste0("x", 1:9)), ]
        pe_w_thresh <- pe_w[pe_w$op == "|", ]
        fm_w <- fitMeasures(fit_w, c("chisq.scaled", "cfi.robust", "rmsea.robust", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    fit_py_w = av.cfa(syntax_hs, data=hs_ord_df, estimator="WLSMV", ordered=True)
    pe_py_w = fit_py_w.parameter_estimates()
    pe_py_w_struct = pe_py_w[
        pe_py_w["op"].isin(["=~", "~~"])
        & ~((pe_py_w["op"] == "~~") & (pe_py_w["lhs"] == pe_py_w["rhs"]) & (pe_py_w["lhs"].isin([f"x{i}" for i in range(1, 10)])))
    ]
    poly_out = av.polychoric(hs_ord_df)
    ft_w = fit_py_w.fit_measures()

    r_est_w = np.array(ro.r("pe_w_struct$est"))
    r_se_w = np.array(ro.r("pe_w_struct$se"))
    r_thr_w = np.array(ro.r("pe_w_thresh$est"))
    r_fm_w = np.array(ro.r("as.numeric(fm_w)"))
    py_fm_w_vec = np.array([
        ft_w.loc["chisq", "Scaled"], ft_w.loc["cfi", "Robust"],
        ft_w.loc["rmsea", "Robust"], ft_w.loc["srmr", "Unscaled"],
    ])
    max_diff_w_param = max(
        float(np.max(np.abs(pe_py_w_struct["est"].to_numpy() - r_est_w))),
        float(np.max(np.abs(pe_py_w_struct["se"].to_numpy() - r_se_w))),
        float(np.max(np.abs(poly_out.thresholds["est"].to_numpy() - r_thr_w))),
    )
    max_diff_w_fm = float(np.max(np.abs(py_fm_w_vec - r_fm_w)))
    add_row("Ordinal CFA (WLSMV)", "HS1939-Ord", "lavaan 0.7-2", "Polychoric lambda(visual=~x2)", float(r_est_w[1]), float(pe_py_w_struct["est"].iloc[1]), max_diff_w_param)
    add_row("Ordinal CFA (WLSMV)", "HS1939-Ord", "lavaan 0.7-2", "Scaled Chi-sq / Robust CFI", f"{r_fm_w[0]:.5f} / {r_fm_w[1]:.5f}", f"{ft_w.loc['chisq', 'Scaled']:.5f} / {ft_w.loc['cfi', 'Robust']:.5f}", max_diff_w_fm)

    # ------------------------------------------------------------------
    # 5. Multi-Group Invariance (MLR, Scalar) - HS1939 by school
    # ------------------------------------------------------------------
    ro.r("""
        hs_mg <- HolzingerSwineford1939[, c("school", paste0("x", 1:9))]
        fit_mg_sca <- cfa(mod_hs, data = hs_mg, group = "school", estimator = "MLR", group.equal = c("loadings", "intercepts"))
        fm_mg_sca <- fitMeasures(fit_mg_sca, c("chisq.scaled", "cfi.robust", "rmsea.robust", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_mg_df = ro.conversion.rpy2py(ro.r("hs_mg"))

    inv_py = av.measurement_invariance(syntax_hs, data=hs_mg_df, group="school", estimator="MLR")
    ft_mg = inv_py.fits["scalar"].fit_measures()
    r_fm_mg = np.array(ro.r("as.numeric(fm_mg_sca)"))
    py_fm_mg_vec = np.array([
        ft_mg.loc["chisq", "Scaled"], ft_mg.loc["cfi", "Robust"],
        ft_mg.loc["rmsea", "Robust"], ft_mg.loc["srmr", "Unscaled"],
    ])
    max_diff_mg = float(np.max(np.abs(py_fm_mg_vec - r_fm_mg)))
    add_row("Multi-Group (MLR)", "HS1939 (G=2)", "lavaan 0.7-2", "Scalar Scaled Chi-sq / Robust CFI", f"{r_fm_mg[0]:.5f} / {r_fm_mg[1]:.5f}", f"{ft_mg.loc['chisq', 'Scaled']:.5f} / {ft_mg.loc['cfi', 'Robust']:.5f}", max_diff_mg)

    # ------------------------------------------------------------------
    # 6. Bifactor Indices & EGA Network Psychometrics
    # ------------------------------------------------------------------
    ro.r("""
        library(BifactorIndicesCalculator)
        mod_bi <- '
            G  =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            S1 =~ x1 + x2 + x3
            S2 =~ x4 + x5 + x6
            S3 =~ x7 + x8 + x9
        '
        fit_bi <- cfa(mod_bi, data = HolzingerSwineford1939, orthogonal = TRUE, std.lv = TRUE)
        mod_uni <- 'U =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9'
        fit_uni <- cfa(mod_uni, data = HolzingerSwineford1939, std.lv = TRUE)
        bi_r <- bifactorIndices(fit_bi, UniLambda = fit_uni)
        lam_r <- lavInspect(fit_bi, "std")$lambda
        uni_r <- as.numeric(lavInspect(fit_uni, "std")$lambda)
        m_vals_r <- as.numeric(bi_r$ModelLevelIndices)
        m_names_r <- names(bi_r$ModelLevelIndices)
    """)
    lam_py = np.array(ro.r("lam_r"))
    uni_py = np.array(ro.r("uni_r"))
    r_bi_series = pd.Series(np.array(ro.r("m_vals_r")), index=list(ro.r("m_names_r")))
    py_bi = bifactor_indices(lam_py, unidimensional_loadings=uni_py)["model"]
    max_diff_bi = float(np.max(np.abs(py_bi.to_numpy() - r_bi_series.to_numpy())))
    ecv_i = next(i for i, k in enumerate(r_bi_series.index) if k.startswith("ECV"))
    puc_i = next(i for i, k in enumerate(r_bi_series.index) if k.startswith("PUC"))
    omh_i = next(i for i, k in enumerate(r_bi_series.index) if k.startswith("OmegaH"))
    add_row(
        "Bifactor Indices",
        "HS1939 (N=301)",
        "BifactorIndicesCalc 0.2-2",
        "ECV / PUC / OmegaH",
        f"{r_bi_series.iloc[ecv_i]:.5f} / {r_bi_series.iloc[puc_i]:.5f} / {r_bi_series.iloc[omh_i]:.5f}",
        f"{py_bi.iloc[ecv_i]:.5f} / {py_bi.iloc[puc_i]:.5f} / {py_bi.iloc[omh_i]:.5f}",
        max_diff_bi,
    )

    ro.r("""
        library(EGAnet)
        hs_ega <- HolzingerSwineford1939[, paste0("x", 1:9)]
        e_r <- EGA(hs_ega, model = "glasso", algorithm = "walktrap", plot.EGA = FALSE)
        r_ndim <- as.integer(e_r$n.dim)
        r_tefi <- as.numeric(e_r$TEFI)
        r_pcor <- as.matrix(e_r$network)
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_ega_df = ro.conversion.rpy2py(ro.r("hs_ega"))

    ega_py = av.ega(hs_ega_df, algorithm="walktrap")
    r_ndim = int(ro.r("r_ndim")[0])
    r_tefi = float(ro.r("r_tefi")[0])
    r_pcor = np.array(ro.r("r_pcor"))
    max_diff_tefi = abs(float(ega_py.tefi) - r_tefi)
    max_diff_pcor = float(np.max(np.abs(ega_py.partial_corr().to_numpy() - r_pcor)))
    add_row("Network (EGA)", "HS1939 (N=301)", "EGAnet 2.4.1", "Dimensions (K) / TEFI", f"{r_ndim} / {r_tefi:.5f}", f"{ega_py.n_dim} / {ega_py.tefi:.5f}", max_diff_tefi)
    add_row("Network (EGA)", "HS1939 (N=301)", "EGAnet 2.4.1", "Partial Corr. Matrix (EBICglasso)", "9x9 matrix", "9x9 matrix", max_diff_pcor)

    df_out = pd.DataFrame(rows)
    out_path = "simulations/results/parity_benchmark.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n=== Tabela de Paridade Numérica ({out_path}) ===")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()