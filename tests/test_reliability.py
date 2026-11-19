import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anvala as av


def test_model_based_reliability_omega_ave_htmt_and_no_default_alpha(r_env):
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
        fit_wlsmv <- cfa(model, data = hs_ord, ordered = TRUE, estimator = "WLSMV")

        has_semtools <- requireNamespace("semTools", quietly = TRUE)
        if (has_semtools) {
            rel_ml <- semTools::compRelSEM(fit_ml)
            ave_ml <- semTools::AVE(fit_ml)
            htmt_ml <- semTools::htmt(model, data = hs)
            rel_ord <- semTools::compRelSEM(fit_wlsmv, ord.scale = TRUE)
        }
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """

    # 1. CFA Contínua: por padrão NÃO inclui alpha
    fit_py_ml = av.cfa(syntax, data=hs_df, estimator="ML")
    rel_py = av.reliability(fit_py_ml, data=hs_df)
    assert "alpha_raw" not in rel_py.constructs.columns
    assert "alpha_std" not in rel_py.constructs.columns
    assert "omega_total" in rel_py.constructs.columns
    assert "ave" in rel_py.constructs.columns
    assert rel_py.htmt_matrix is not None

    # Se explicitamente solicitado, alpha aparece
    rel_py_with_alpha = av.reliability(fit_py_ml, data=hs_df, include_alpha=True)
    assert "alpha_raw" in rel_py_with_alpha.constructs.columns

    # 2. CFA Ordinal WLSMV: Ômega categórico de Green & Yang (2009)
    fit_py_w = av.cfa(syntax, data=hs_ord_df, estimator="WLSMV", ordered=True)
    rel_py_w = av.reliability(fit_py_w, data=hs_ord_df)
    assert "omega_categorical" in rel_py_w.constructs.columns
    assert "alpha_raw" not in rel_py_w.constructs.columns

    if bool(ro.r("has_semtools")[0]):
        np.testing.assert_allclose(rel_py.constructs["omega_total"].to_numpy(), np.array(ro.r("as.numeric(rel_ml)")), atol=1e-4)
        np.testing.assert_allclose(rel_py.constructs["ave"].to_numpy(), np.array(ro.r("as.numeric(ave_ml)")), atol=1e-4)
        np.testing.assert_allclose(rel_py.htmt_matrix.to_numpy(), np.array(ro.r("as.matrix(htmt_ml)")), atol=1e-4)
        np.testing.assert_allclose(rel_py_w.constructs["omega_categorical"].to_numpy(), np.array(ro.r("as.numeric(rel_ord)")), atol=2e-3)

    # 3. Fidedignidade diretamente de uma EFA
    efa_py = av.efa(hs_df, n_factors=3, estimator="MLR")
    rel_efa = av.reliability(efa_py, data=hs_df)
    assert "omega_total" in rel_efa.constructs.columns
    assert "alpha_raw" not in rel_efa.constructs.columns