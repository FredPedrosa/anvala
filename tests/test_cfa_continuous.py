import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anvala as av


def test_cfa_ml_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        model <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit <- cfa(model, data = HolzingerSwineford1939, estimator = "ML")
        pe <- parameterEstimates(fit, standardized = TRUE)
        fm <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr"))
    """)

    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("HolzingerSwineford1939"))

    r_est = np.array(ro.r("pe$est"))
    r_se = np.array(ro.r("pe$se"))
    r_std = np.array(ro.r("pe$std.all"))
    r_fm = np.array(ro.r("as.numeric(fm)"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa(syntax, data=hs_df, estimator="ML")
    pe_py = fit_py.parameter_estimates()
    fm_py = fit_py.fit_measures()["Unscaled"]

    np.testing.assert_allclose(pe_py["est"].to_numpy(), r_est, atol=1e-5)
    np.testing.assert_allclose(pe_py["se"].to_numpy(), r_se, atol=1e-5)
    np.testing.assert_allclose(pe_py["std_all"].to_numpy(), r_std, atol=1e-5)

    py_fm_vec = np.array([
        fm_py["chisq"], fm_py["df"], fm_py["pvalue"],
        fm_py["cfi"], fm_py["tli"],
        fm_py["rmsea"], fm_py["rmsea_ci_lower"], fm_py["rmsea_ci_upper"],
        fm_py["srmr"],
    ])
    np.testing.assert_allclose(py_fm_vec, r_fm, atol=1e-5)


def test_cfa_mlr_unscaled_scaled_robust_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        model <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit <- cfa(model, data = HolzingerSwineford1939, estimator = "MLR")
        pe <- parameterEstimates(fit, standardized = TRUE)
        fm <- fitMeasures(fit, c(
            "chisq", "chisq.scaled", "chisq.scaling.factor",
            "cfi", "cfi.scaled", "cfi.robust",
            "tli", "tli.scaled", "tli.robust",
            "rmsea", "rmsea.scaled", "rmsea.robust",
            "rmsea.ci.lower", "rmsea.ci.upper",
            "rmsea.ci.lower.scaled", "rmsea.ci.upper.scaled",
            "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
            "srmr"
        ))
    """)

    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("HolzingerSwineford1939"))

    r_est = np.array(ro.r("pe$est"))
    r_se = np.array(ro.r("pe$se"))
    r_fm = np.array(ro.r("as.numeric(fm)"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa(syntax, data=hs_df, estimator="MLR")
    pe_py = fit_py.parameter_estimates()
    ft = fit_py.fit_measures()

    np.testing.assert_allclose(pe_py["est"].to_numpy(), r_est, atol=1e-5)
    np.testing.assert_allclose(pe_py["se"].to_numpy(), r_se, atol=1e-5)

    py_fm_vec = np.array([
        ft.loc["chisq", "Unscaled"], ft.loc["chisq", "Scaled"], ft.loc["scaling_factor", "Scaled"],
        ft.loc["cfi", "Unscaled"], ft.loc["cfi", "Scaled"], ft.loc["cfi", "Robust"],
        ft.loc["tli", "Unscaled"], ft.loc["tli", "Scaled"], ft.loc["tli", "Robust"],
        ft.loc["rmsea", "Unscaled"], ft.loc["rmsea", "Scaled"], ft.loc["rmsea", "Robust"],
        ft.loc["rmsea_ci_lower", "Unscaled"], ft.loc["rmsea_ci_upper", "Unscaled"],
        ft.loc["rmsea_ci_lower", "Scaled"], ft.loc["rmsea_ci_upper", "Scaled"],
        ft.loc["rmsea_ci_lower", "Robust"], ft.loc["rmsea_ci_upper", "Robust"],
        ft.loc["srmr", "Unscaled"],
    ])
    np.testing.assert_allclose(py_fm_vec, r_fm, atol=1e-5)