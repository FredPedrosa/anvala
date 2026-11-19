import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av


def test_residuals_and_modindices_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        hs_ord <- as.data.frame(lapply(hs, function(col) {
            as.integer(cut(col, breaks = quantile(col, probs = seq(0, 1, 0.2)), include.lowest = TRUE))
        }))
        model <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit_ml <- cfa(model, data = hs, estimator = "ML")
        res_raw <- residuals(fit_ml, type = "raw")$cov
        res_std <- residuals(fit_ml, type = "standardized")$cov
        res_cor <- residuals(fit_ml, type = "cor.bollen")$cov
        mi_ml <- modindices(fit_ml, sort = TRUE)

        fit_w <- cfa(model, data = hs_ord, ordered = TRUE, estimator = "WLSMV")
        mi_w <- modindices(fit_w, sort = TRUE)
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))
        r_mi_ml = ro.conversion.rpy2py(ro.r("mi_ml"))
        r_mi_w = ro.conversion.rpy2py(ro.r("mi_w"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa(syntax, data=hs_df, estimator="ML")

    # 1. Resíduos raw, standardized e cor.bollen
    np.testing.assert_allclose(fit_py.residuals("raw").to_numpy(), np.array(ro.r("res_raw")), atol=1e-4)
    np.testing.assert_allclose(fit_py.residuals("cor.bollen").to_numpy(), np.array(ro.r("res_cor")), atol=1e-4)
    np.testing.assert_allclose(fit_py.residuals("standardized").to_numpy(), np.array(ro.r("res_std")), atol=2e-3)

    # 2. Índices de modificação (MI, EPC, SEPC.lv, SEPC.all) - ML
    mi_py = fit_py.modindices(sort=True)
    assert len(mi_py) == len(r_mi_ml)
    np.testing.assert_allclose(mi_py["mi"].to_numpy()[:10], r_mi_ml["mi"].to_numpy()[:10], atol=1e-2)
    np.testing.assert_allclose(mi_py["epc"].to_numpy()[:10], r_mi_ml["epc"].to_numpy()[:10], atol=1e-3)
    np.testing.assert_allclose(mi_py["sepc_lv"].to_numpy()[:10], r_mi_ml["sepc.lv"].to_numpy()[:10], atol=1e-3)
    np.testing.assert_allclose(mi_py["sepc_all"].to_numpy()[:10], r_mi_ml["sepc.all"].to_numpy()[:10], atol=1e-3)

    # 3. Índices de modificação - WLSMV Ordinal
    fit_py_w = av.cfa(syntax, data=hs_ord_df, estimator="WLSMV", ordered=True)
    mi_py_w = fit_py_w.modindices(sort=True)
    np.testing.assert_allclose(mi_py_w["mi"].to_numpy()[:8], r_mi_w["mi"].to_numpy()[:8], atol=5e-2)
    np.testing.assert_allclose(mi_py_w["epc"].to_numpy()[:8], r_mi_w["epc"].to_numpy()[:8], atol=2e-3)


def test_sem_structural_and_defined_mediation_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        pd_data <- PoliticalDemocracy
        sem_mod <- '
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            dem60 ~ a * ind60
            dem65 ~ c * ind60 + b * dem60
            y1 ~~ y5
            y2 ~~ y4 + y6
            y3 ~~ y7
            y4 ~~ y8
            y6 ~~ y8
            indirect := a * b
            total := c + (a * b)
        '
        fit_sem <- sem(sem_mod, data = pd_data, estimator = "MLR")
        pe_sem <- parameterEstimates(fit_sem, standardized = TRUE)
        fm_sem <- fitMeasures(fit_sem, c("chisq", "chisq.scaled", "cfi", "cfi.robust", "rmsea", "rmsea.robust", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_df = ro.conversion.rpy2py(ro.r("pd_data"))
        r_pe = ro.conversion.rpy2py(ro.r("pe_sem"))

    r_fm = np.array(ro.r("as.numeric(fm_sem)"))

    sem_syntax = """
        ind60 =~ x1 + x2 + x3
        dem60 =~ y1 + y2 + y3 + y4
        dem65 =~ y5 + y6 + y7 + y8
        dem60 ~ a * ind60
        dem65 ~ c * ind60 + b * dem60
        y1 ~~ y5
        y2 ~~ y4 + y6
        y3 ~~ y7
        y4 ~~ y8
        y6 ~~ y8
        indirect := a * b
        total := c + (a * b)
    """
    fit_py = av.sem(sem_syntax, data=pd_df, estimator="MLR")
    assert fit_py.converged is True

    pe_py = fit_py.parameter_estimates()
    # Compara caminhos estruturais (~) e parâmetros definidos (:=)
    py_reg = pe_py[pe_py["op"].isin(["~", ":="])].reset_index(drop=True)
    r_reg = r_pe[r_pe["op"].isin(["~", ":="])].reset_index(drop=True)

    np.testing.assert_allclose(py_reg["est"].to_numpy(), r_reg["est"].to_numpy(), atol=1e-3)
    np.testing.assert_allclose(py_reg["se"].to_numpy(), r_reg["se"].to_numpy(), atol=2e-3)
    np.testing.assert_allclose(py_reg["std_all"].to_numpy(), r_reg["std.all"].to_numpy(), atol=1e-3)

    ft = fit_py.fit_measures()
    py_fm = np.array([
        ft.loc["chisq", "Unscaled"], ft.loc["chisq", "Scaled"],
        ft.loc["cfi", "Unscaled"], ft.loc["cfi", "Robust"],
        ft.loc["rmsea", "Unscaled"], ft.loc["rmsea", "Robust"],
        ft.loc["srmr", "Unscaled"],
    ])
    np.testing.assert_allclose(py_fm, r_fm, atol=2e-4)