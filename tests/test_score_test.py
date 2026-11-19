import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av


def test_score_test_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, c("school", paste0("x", 1:9))]
        mod <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit_sca <- cfa(mod, data = hs, group = "school", estimator = "MLR", group.equal = c("loadings", "intercepts"))
        sc <- lavTestScore(fit_sca)
        tot_df <- sc$test
        uni_df <- sc$uni
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))
        r_tot = ro.conversion.rpy2py(ro.r("tot_df"))
        r_uni = ro.conversion.rpy2py(ro.r("uni_df"))

    mod = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa_multigroup(mod, data=hs_df, group="school", estimator="MLR", group_equal=["loadings", "intercepts"])
    tot_py, uni_py = fit_py.score_test(sort=False, return_total=True)

    # Teste global (score, score.scaled, score.adjusted, score.robust)
    np.testing.assert_allclose(tot_py["x2"].to_numpy(), r_tot["X2"].to_numpy(), atol=5e-2)
    np.testing.assert_allclose(tot_py["df"].to_numpy(), r_tot["df"].to_numpy(), atol=5e-2)

    # Testes univariados (X2 e X2.scaled)
    np.testing.assert_allclose(uni_py["x2"].to_numpy(), r_uni["X2"].to_numpy(), atol=5e-2)
    np.testing.assert_allclose(uni_py["x2_scaled"].to_numpy(), r_uni["X2.scaled"].to_numpy(), atol=5e-2)

    # Confirma que os maiores violadores da invariância escalar são x3~1 e x7~1
    uni_sorted = fit_py.score_test(sort=True)
    top2 = set(zip(uni_sorted.iloc[:2]["lhs"], uni_sorted.iloc[:2]["op"]))
    assert ("x3", "~1") in top2
    assert ("x7", "~1") in top2