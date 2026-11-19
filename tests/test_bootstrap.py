import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av


def test_bootstrap_ci_sem_mediation_and_ega_cuda(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        set.seed(123)
        X <- rnorm(250)
        M <- 0.55 * X + rnorm(250)
        Y <- 0.45 * M + 0.20 * X + rnorm(250)
        df_med <- data.frame(X = X, M = M, Y = Y)
        mod <- '
            M ~ a*X
            Y ~ b*M + c*X
            indirect := a*b
            total := c + a*b
        '
        fit <- sem(mod, data = df_med, se = "bootstrap", bootstrap = 300)
        pe_b <- parameterEstimates(fit, boot.ci.type = "bca.simple")
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_med = ro.conversion.rpy2py(ro.r("df_med"))
        r_pe = ro.conversion.rpy2py(ro.r("pe_b"))

    mod = """
        M ~ a*X
        Y ~ b*M + c*X
        indirect := a*b
        total := c + a*b
    """
    fit_py = av.sem(mod, data=df_med, estimator="ML")
    boot_res = fit_py.bootstrap_ci(n_boot=400, conf_level=0.95, method="bca", seed=42)

    ind_py = boot_res.table[boot_res.table["lhs"] == "indirect"].iloc[0]
    ind_r = r_pe[r_pe["lhs"] == "indirect"].iloc[0]

    np.testing.assert_allclose(ind_py["est"], float(ind_r["est"]), atol=1e-4)
    np.testing.assert_allclose(ind_py["boot_se"], float(ind_r["se"]), atol=0.025)
    assert ind_py["ci_lower"] > 0.10
    assert ind_py["ci_upper"] < 0.45

    # Testa boot_ega em GPU/auto
    boot_ega_res = av.boot_ega(df_med, n_boot=50, boot_type="nonparametric", seed=42)
    assert boot_ega_res.summary["n_boot"].iloc[0] == 50
    assert 0.0 <= boot_ega_res.dimension_stability <= 1.0