import numpy as np
import pandas as pd
from anvala.bifactor import bifactor_indices


def test_parity_with_r_bifactor_indices_calculator(r_env):
    ro = r_env["ro"]

    ro.r("""
        library(lavaan)
        library(BifactorIndicesCalculator)

        model_bi <- '
            G  =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            S1 =~ x1 + x2 + x3
            S2 =~ x4 + x5 + x6
            S3 =~ x7 + x8 + x9
        '
        fit_bi <- cfa(model_bi, data = HolzingerSwineford1939, orthogonal = TRUE, std.lv = TRUE)
        bi_res <- bifactorIndices(fit_bi)

        model_uni <- 'U =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9'
        fit_uni <- cfa(model_uni, data = HolzingerSwineford1939, std.lv = TRUE)
        bi_arpb <- bifactorIndices(fit_bi, UniLambda = fit_uni)

        lam_r <- lavInspect(fit_bi, "std")$lambda
        uni_r <- as.numeric(lavInspect(fit_uni, "std")$lambda)

        m_vals <- as.numeric(bi_arpb$ModelLevelIndices)
        m_names <- names(bi_arpb$ModelLevelIndices)

        f_vals <- as.matrix(bi_res$FactorLevelIndices)
        f_cols <- colnames(bi_res$FactorLevelIndices)
        i_vals <- as.matrix(bi_arpb$ItemLevelIndices)
    """)

    lam_py = np.array(ro.r("lam_r"))
    uni_py = np.array(ro.r("uni_r"))

    r_model = pd.Series(np.array(ro.r("m_vals")), index=list(ro.r("m_names")))
    r_factor_cols = list(ro.r("f_cols"))
    r_factor_table = np.array(ro.r("f_vals"))
    r_item_table = np.array(ro.r("i_vals"))

    py_res = bifactor_indices(lam_py, unidimensional_loadings=uni_py)

    np.testing.assert_allclose(py_res["model"].to_numpy(), r_model.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(
        py_res["factors"][r_factor_cols].to_numpy(),
        r_factor_table,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(py_res["items"].to_numpy(), r_item_table, atol=1e-6, equal_nan=True)