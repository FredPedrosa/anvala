import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anvala as av


def test_polychoric_and_wlsmv_parity_with_lavaan(r_env):
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
        fit <- cfa(model, data = hs_ord, ordered = TRUE, estimator = "WLSMV")

        r_poly <- lavInspect(fit, "sampstat")$cov
        pe <- parameterEstimates(fit, standardized = TRUE)
        pe_struct <- pe[pe$op %in% c("=~", "~~") & !(pe$op == "~~" & pe$lhs == pe$rhs & pe$lhs %in% paste0("x", 1:9)), ]
        pe_thresh <- pe[pe$op == "|", ]

        fm <- fitMeasures(fit, c(
            "chisq", "chisq.scaled", "chisq.scaling.factor",
            "cfi", "cfi.scaled", "cfi.robust",
            "tli", "tli.scaled", "tli.robust",
            "rmsea", "rmsea.scaled", "rmsea.robust",
            "srmr"
        ))
        shift_r <- fit@test[[2]]$shift.parameter
    """)

    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    r_poly = np.array(ro.r("r_poly"))
    r_est = np.array(ro.r("pe_struct$est"))
    r_se = np.array(ro.r("pe_struct$se"))
    r_std = np.array(ro.r("pe_struct$std.all"))
    r_thresh = np.array(ro.r("pe_thresh$est"))
    r_fm = np.array(ro.r("as.numeric(fm)"))
    r_shift = float(ro.r("shift_r")[0])

    # 1. Verifica av.polychoric() diretamente
    poly_out = av.polychoric(hs_ord_df)
    np.testing.assert_allclose(poly_out.corr.to_numpy(), r_poly, atol=1e-4)
    np.testing.assert_allclose(poly_out.thresholds["est"].to_numpy(), r_thresh, atol=1e-5)

    # 2. Verifica av.cfa(..., estimator="WLSMV")
    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa(syntax, data=hs_ord_df, estimator="WLSMV", ordered=True)
    assert fit_py.converged is True

    np.testing.assert_allclose(fit_py.polychoric().to_numpy(), r_poly, atol=1e-4)

    pe_py = fit_py.parameter_estimates()
    pe_py_struct = pe_py[
        pe_py["op"].isin(["=~", "~~"])
        & ~((pe_py["op"] == "~~") & (pe_py["lhs"] == pe_py["rhs"]) & (pe_py["lhs"].isin([f"x{i}" for i in range(1, 10)])))
    ]

    np.testing.assert_allclose(pe_py_struct["est"].to_numpy(), r_est, atol=1e-5)
    np.testing.assert_allclose(pe_py_struct["se"].to_numpy(), r_se, atol=1e-5)
    np.testing.assert_allclose(pe_py_struct["std_all"].to_numpy(), r_std, atol=1e-5)

    ft = fit_py.fit_measures()
    np.testing.assert_allclose(ft.loc["shift_parameter", "Scaled"], r_shift, atol=1e-4)

    py_fm_vec = np.array([
        ft.loc["chisq", "Unscaled"], ft.loc["chisq", "Scaled"], ft.loc["scaling_factor", "Scaled"],
        ft.loc["cfi", "Unscaled"], ft.loc["cfi", "Scaled"], ft.loc["cfi", "Robust"],
        ft.loc["tli", "Unscaled"], ft.loc["tli", "Scaled"], ft.loc["tli", "Robust"],
        ft.loc["rmsea", "Unscaled"], ft.loc["rmsea", "Scaled"], ft.loc["rmsea", "Robust"],
        ft.loc["srmr", "Unscaled"],
    ])
    np.testing.assert_allclose(py_fm_vec, r_fm, atol=1e-4)