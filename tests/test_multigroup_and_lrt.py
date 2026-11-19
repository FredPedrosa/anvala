import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av


def test_compare_models_lrt_ml_and_mlr_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        mod_3f <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        mod_1f <- '
            g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
        '
        f3_ml <- cfa(mod_3f, data = hs, estimator = "ML")
        f1_ml <- cfa(mod_1f, data = hs, estimator = "ML")
        lrt_ml <- lavTestLRT(f3_ml, f1_ml)

        f3_mlr <- cfa(mod_3f, data = hs, estimator = "MLR")
        f1_mlr <- cfa(mod_1f, data = hs, estimator = "MLR")
        lrt_mlr <- lavTestLRT(f3_mlr, f1_mlr)
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))
        r_lrt_ml = ro.conversion.rpy2py(ro.r("as.data.frame(lrt_ml)"))
        r_lrt_mlr = ro.conversion.rpy2py(ro.r("as.data.frame(lrt_mlr)"))

    s_3f = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    s_1f = "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"

    py_3f_ml = av.cfa(s_3f, data=hs_df, estimator="ML")
    py_1f_ml = av.cfa(s_1f, data=hs_df, estimator="ML")
    cmp_ml = av.compare_models(py_3f_ml, py_1f_ml, names=["3F", "1F"])
    np.testing.assert_allclose(cmp_ml.loc[1, "chisq_diff"], float(r_lrt_ml["Chisq diff"].iloc[1]), atol=1e-3)
    np.testing.assert_allclose(cmp_ml.loc[1, "pvalue"], float(r_lrt_ml["Pr(>Chisq)"].iloc[1]), atol=1e-5)

    py_3f_mlr = av.cfa(s_3f, data=hs_df, estimator="MLR")
    py_1f_mlr = av.cfa(s_1f, data=hs_df, estimator="MLR")
    cmp_mlr = av.compare_models(py_3f_mlr, py_1f_mlr, names=["3F", "1F"])
    np.testing.assert_allclose(cmp_mlr.loc[1, "chisq_diff"], float(r_lrt_mlr["Chisq diff"].iloc[1]), atol=1e-2)
    assert cmp_mlr.loc[1, "df_diff"] == 3.0


def test_measurement_invariance_continuous_and_ordinal_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs_full <- HolzingerSwineford1939[, c("school", paste0("x", 1:9))]
        hs_ord <- hs_full
        for (col in paste0("x", 1:9)) {
            hs_ord[[col]] <- as.integer(cut(hs_full[[col]], breaks = quantile(hs_full[[col]], probs = seq(0, 1, 0.25)), include.lowest = TRUE))
        }
        mod <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit_conf <- cfa(mod, data = hs_full, group = "school", estimator = "MLR")
        fit_met  <- cfa(mod, data = hs_full, group = "school", estimator = "MLR", group.equal = c("loadings"))
        fit_sca  <- cfa(mod, data = hs_full, group = "school", estimator = "MLR", group.equal = c("loadings", "intercepts"))
        fit_str  <- cfa(mod, data = hs_full, group = "school", estimator = "MLR", group.equal = c("loadings", "intercepts", "residuals"))
        lrt_inv  <- lavTestLRT(fit_conf, fit_met, fit_sca, fit_str)

        fm_conf <- fitMeasures(fit_conf, c("chisq", "chisq.scaled", "cfi.robust", "rmsea.robust", "srmr"))
        fm_sca  <- fitMeasures(fit_sca, c("chisq", "chisq.scaled", "cfi.robust", "rmsea.robust", "srmr"))

        # Ordinal WLSMV Invariance
        fit_w_conf <- cfa(mod, data = hs_ord, group = "school", ordered = TRUE, estimator = "WLSMV")
        fit_w_met  <- cfa(mod, data = hs_ord, group = "school", ordered = TRUE, estimator = "WLSMV", group.equal = c("loadings"))
        fit_w_sca  <- cfa(mod, data = hs_ord, group = "school", ordered = TRUE, estimator = "WLSMV", group.equal = c("loadings", "thresholds"))
        fm_w_sca   <- fitMeasures(fit_w_sca, c("chisq", "chisq.scaled", "cfi.robust", "rmsea.robust", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs_full"))
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))
        r_lrt_inv = ro.conversion.rpy2py(ro.r("as.data.frame(lrt_inv)"))

    r_fm_conf = np.array(ro.r("as.numeric(fm_conf)"))
    r_fm_sca = np.array(ro.r("as.numeric(fm_sca)"))
    r_fm_w_sca = np.array(ro.r("as.numeric(fm_w_sca)"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    inv_py = av.measurement_invariance(syntax, data=hs_df, group="school", estimator="MLR")
    tbl = inv_py.table

    assert list(tbl["model"]) == ["configural", "metric", "scalar", "strict"]
    assert list(tbl["df"]) == [48, 54, 60, 69]

    # Paridade dos índices de ajuste Configural e Escalar (MLR)
    ft_c = inv_py.fits["configural"].fit_measures()
    py_fm_conf = np.array([ft_c.loc["chisq", "Unscaled"], ft_c.loc["chisq", "Scaled"], ft_c.loc["cfi", "Robust"], ft_c.loc["rmsea", "Robust"], ft_c.loc["srmr", "Unscaled"]])
    np.testing.assert_allclose(py_fm_conf, r_fm_conf, atol=5e-4)

    ft_s = inv_py.fits["scalar"].fit_measures()
    py_fm_sca = np.array([ft_s.loc["chisq", "Unscaled"], ft_s.loc["chisq", "Scaled"], ft_s.loc["cfi", "Robust"], ft_s.loc["rmsea", "Robust"], ft_s.loc["srmr", "Unscaled"]])
    np.testing.assert_allclose(py_fm_sca, r_fm_sca, atol=5e-4)

    # Paridade do Delta Chi-quadrado Satorra-Bentler (2001) entre níveis
    np.testing.assert_allclose(tbl["chisq_diff"].to_numpy()[1:], r_lrt_inv["Chisq diff"].to_numpy()[1:], atol=2e-2)

    # Invariância parcial liberando intercepto de x3
    inv_partial = av.measurement_invariance(
        syntax, data=hs_df, group="school", estimator="MLR",
        levels=("metric", "scalar"), group_partial=["x3 ~ 1"],
    )
    assert inv_partial.table.loc[1, "df"] == 59
    assert inv_partial.table.loc[1, "chisq"] < tbl.loc[2, "chisq"]

    # Invariância Ordinal WLSMV (Configural -> Métrica -> Escalar)
    inv_ord = av.measurement_invariance(
        syntax, data=hs_ord_df, group="school", estimator="WLSMV", ordered=True,
        levels=("configural", "metric", "scalar"),
    )
    ft_ws = inv_ord.fits["scalar"].fit_measures()
    py_fm_w_sca = np.array([ft_ws.loc["chisq", "Unscaled"], ft_ws.loc["chisq", "Scaled"], ft_ws.loc["cfi", "Robust"], ft_ws.loc["rmsea", "Robust"], ft_ws.loc["srmr", "Unscaled"]])
    np.testing.assert_allclose(py_fm_w_sca, r_fm_w_sca, atol=2e-3)