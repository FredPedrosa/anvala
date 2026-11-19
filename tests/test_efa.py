import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anvala as av


def test_dimtests_horn_pa_gold_standard(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))

    dt = av.dimtests(hs_df, tests=["RAWPAR", "EMPKC", "HULL", "SMT", "MAP", "NEVALSGT1"], seed=42)
    assert dt.primary_method == "RAWPAR"
    assert dt.n_factors == 3
    assert dt.adequacy["kmo"] > 0.70
    assert dt.adequacy["bartlett_pvalue"] < 1e-10
    
    res_by_method = dt.results.set_index("method")["n_factors"]
    assert res_by_method["RAWPAR (Horn Parallel Analysis)"] == 3
    assert res_by_method["EMPKC (Empirical Kaiser)"] == 3
    assert res_by_method["HULL (PAF/CAF)"] == 3
    assert res_by_method["NEVALSGT1 (Kaiser > 1)"] == 3


def test_efa_ml_and_mlr_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        fit_mlr <- efa(hs, nfactors = 3, estimator = "MLR")
        f1 <- fit_mlr[[1]]
        pe <- parameterEstimates(f1, standardized = TRUE)
        pe_load <- pe[pe$op == "=~", ]
        pe_phi <- pe[pe$op == "~~" & pe$lhs != pe$rhs, ]
        fm <- fitMeasures(f1, c("chisq", "chisq.scaled", "chisq.scaling.factor", "cfi", "cfi.scaled", "cfi.robust", "rmsea", "rmsea.scaled", "rmsea.robust", "srmr"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))

    r_est = np.array(ro.r("pe_load$est"))
    r_se = np.array(ro.r("pe_load$se"))
    r_std = np.array(ro.r("pe_load$std.all"))
    r_phi = np.array(ro.r("pe_phi$est"))
    r_phi_se = np.array(ro.r("pe_phi$se"))
    r_fm = np.array(ro.r("as.numeric(fm)"))

    fit_py = av.efa(hs_df, n_factors=3, estimator="MLR", rotation="geomin", seed=42)
    assert fit_py.converged is True

    pe_py = fit_py.parameter_estimates()
    py_load = pe_py[pe_py["op"] == "=~"]
    py_phi = pe_py[pe_py["op"] == "~~"]

    np.testing.assert_allclose(np.abs(py_load["est"].to_numpy()), np.abs(r_est), atol=1e-3)
    np.testing.assert_allclose(py_load["se"].to_numpy(), r_se, atol=6e-3)
    np.testing.assert_allclose(np.abs(py_load["std_all"].to_numpy()), np.abs(r_std), atol=1e-3)
    np.testing.assert_allclose(np.abs(py_phi["est"].to_numpy()), np.abs(r_phi), atol=1e-3)
    np.testing.assert_allclose(py_phi["se"].to_numpy(), r_phi_se, atol=2e-3)

    ft = fit_py.fit_measures()
    py_fm_vec = np.array([
        ft.loc["chisq", "Unscaled"], ft.loc["chisq", "Scaled"], ft.loc["scaling_factor", "Scaled"],
        ft.loc["cfi", "Unscaled"], ft.loc["cfi", "Scaled"], ft.loc["cfi", "Robust"],
        ft.loc["rmsea", "Unscaled"], ft.loc["rmsea", "Scaled"], ft.loc["rmsea", "Robust"],
        ft.loc["srmr", "Unscaled"],
    ])
    np.testing.assert_allclose(py_fm_vec, r_fm, atol=1e-4)


def test_efa_wlsmv_ordinal_and_rotations_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        hs_ord <- as.data.frame(lapply(hs, function(col) {
            as.integer(cut(col, breaks = quantile(col, probs = seq(0, 1, 0.2)), include.lowest = TRUE))
        }))
        fit_wlsmv <- efa(hs_ord, nfactors = 3, ordered = TRUE, estimator = "WLSMV")
        f_w <- fit_wlsmv[[1]]
        pe_w <- parameterEstimates(f_w, standardized = TRUE)
        pe_w_load <- pe_w[pe_w$op == "=~", ]
        fm_w <- fitMeasures(f_w, c("chisq", "chisq.scaled", "chisq.scaling.factor", "cfi", "cfi.scaled", "cfi.robust", "rmsea", "rmsea.scaled", "rmsea.robust", "srmr"))

        fit_varimax <- efa(hs, nfactors = 3, estimator = "ML", rotation = "varimax")
        pe_v <- parameterEstimates(fit_varimax[[1]], standardized = TRUE)
        pe_v_load <- pe_v[pe_v$op == "=~", ]
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    # 1. EFA Ordinal com WLSMV
    fit_py_w = av.efa(hs_ord_df, n_factors=3, estimator="WLSMV", ordered=True, seed=42)
    assert fit_py_w.converged is True

    pe_py_w = fit_py_w.parameter_estimates()
    py_w_load = pe_py_w[pe_py_w["op"] == "=~"]
    np.testing.assert_allclose(np.abs(py_w_load["std_all"].to_numpy()), np.abs(np.array(ro.r("pe_w_load$std.all"))), atol=2e-3)
    np.testing.assert_allclose(py_w_load["se"].to_numpy(), np.array(ro.r("pe_w_load$se")), atol=7e-3)
    
    ft_w = fit_py_w.fit_measures()
    py_fm_w = np.array([
        ft_w.loc["chisq", "Unscaled"], ft_w.loc["chisq", "Scaled"], ft_w.loc["scaling_factor", "Scaled"],
        ft_w.loc["cfi", "Unscaled"], ft_w.loc["cfi", "Scaled"], ft_w.loc["cfi", "Robust"],
        ft_w.loc["rmsea", "Unscaled"], ft_w.loc["rmsea", "Scaled"], ft_w.loc["rmsea", "Robust"],
        ft_w.loc["srmr", "Unscaled"],
    ])
    np.testing.assert_allclose(py_fm_w, np.array(ro.r("as.numeric(fm_w)")), atol=2e-4)

    # 2. Rotação ortogonal Varimax + Oblimin
    fit_py_v = av.efa(hs_df, n_factors=3, estimator="ML", rotation="varimax", seed=42)
    py_v_load = fit_py_v.parameter_estimates()[fit_py_v.parameter_estimates()["op"] == "=~"]
    np.testing.assert_allclose(np.abs(py_v_load["std_all"].to_numpy()), np.abs(np.array(ro.r("pe_v_load$std.all"))), atol=1e-3)
    np.testing.assert_allclose(fit_py_v.factor_corr.to_numpy(), np.eye(3), atol=1e-10)

    fit_py_obl = av.efa(hs_df, n_factors=3, estimator="ML", rotation="oblimin", seed=42)
    assert fit_py_obl.converged is True