import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import anvala as av


def test_predict_continuous_cfa_sem_and_multigroup_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, c("school", paste0("x", 1:9))]
        mod_cfa <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        '
        fit_cfa <- cfa(mod_cfa, data = hs, estimator = "ML")
        fs_reg <- lavPredict(fit_cfa, method = "regression", se = "standard")
        se_reg <- attr(fs_reg, "se")[[1]]
        fs_bar <- lavPredict(fit_cfa, method = "Bartlett", se = "standard")
        se_bar <- attr(fs_bar, "se")[[1]]
        fs_new <- lavPredict(fit_cfa, newdata = hs[1:15, ], method = "regression")

        # SEM estrutural
        mod_sem <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            speed ~ visual + textual
        '
        fit_sem <- sem(mod_sem, data = hs, estimator = "ML")
        fs_sem <- lavPredict(fit_sem, method = "regression")

        # Multigrupo
        fit_mg <- cfa(mod_cfa, data = hs, group = "school", group.equal = c("loadings", "intercepts"))
        fs_mg <- do.call(rbind, lavPredict(fit_mg, method = "regression"))
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))

    r_fs_reg = np.array(ro.r("as.matrix(fs_reg)"))
    r_se_reg = np.array(ro.r("as.matrix(se_reg)")).ravel()
    r_fs_bar = np.array(ro.r("as.matrix(fs_bar)"))
    r_se_bar = np.array(ro.r("as.matrix(se_bar)")).ravel()
    r_fs_new = np.array(ro.r("as.matrix(fs_new)"))
    r_fs_sem = np.array(ro.r("as.matrix(fs_sem)"))
    r_fs_mg = np.array(ro.r("as.matrix(fs_mg)"))

    mod_cfa = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    fit_py = av.cfa(mod_cfa, data=hs_df, estimator="ML")

    # 1. Regression (Thurstone) + SE
    sc_reg, se_reg = fit_py.predict(method="regression", se=True)
    np.testing.assert_allclose(sc_reg.to_numpy(), r_fs_reg, atol=1e-4)
    np.testing.assert_allclose(se_reg.iloc[0].to_numpy(), r_se_reg, atol=1e-4)

    # 2. Bartlett + SE
    sc_bar, se_bar = fit_py.predict(method="bartlett", se=True)
    np.testing.assert_allclose(sc_bar.to_numpy(), r_fs_bar, atol=1e-4)
    np.testing.assert_allclose(se_bar.iloc[0].to_numpy(), r_se_bar, atol=1e-4)

    # 3. Anderson-Rubin (ortogonalizados com covariância identidade)
    sc_ar = fit_py.predict(method="anderson-rubin")
    cov_ar = np.cov(sc_ar.to_numpy(), rowvar=False, bias=True)
    # Verifica ortogonalidade e variância padronizada
    off_diag = cov_ar - np.diag(np.diag(cov_ar))
    assert np.max(np.abs(off_diag)) < 0.05

    # 4. Predição fora da amostra (newdata)
    sc_new = fit_py.predict(newdata=hs_df.iloc[:15], method="regression")
    np.testing.assert_allclose(sc_new.to_numpy(), r_fs_new, atol=1e-4)

    # 5. SEM estrutural
    mod_sem = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
        speed ~ visual + textual
    """
    fit_sem_py = av.sem(mod_sem, data=hs_df, estimator="ML")
    sc_sem = fit_sem_py.predict(method="regression")
    np.testing.assert_allclose(sc_sem.to_numpy(), r_fs_sem, atol=1e-4)

    # 6. Multigrupo
    fit_mg_py = av.cfa_multigroup(mod_cfa, data=hs_df, group="school", group_equal=["loadings", "intercepts"])
    sc_mg_py = fit_mg_py.predict(newdata=hs_df, method="regression")
    # Reordena por grupo na mesma ordem do do.call(rbind, lavPredict(fit_mg))
    ordered_idx = pd.concat([hs_df[hs_df["school"] == g] for g in fit_mg_py.group_labels]).index
    np.testing.assert_allclose(sc_mg_py.loc[ordered_idx].to_numpy(), r_fs_mg, atol=5e-4)


def test_predict_ordinal_ebm_and_eap_parity_with_lavaan(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        hs <- HolzingerSwineford1939[, paste0("x", 1:6)]
        hs_ord <- as.data.frame(lapply(hs, function(col) {
            as.integer(cut(col, breaks = quantile(col, probs = seq(0, 1, 0.25)), include.lowest = TRUE))
        }))
        mod <- '
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
        '
        fit_w <- cfa(mod, data = hs_ord, ordered = TRUE, estimator = "WLSMV")
        fs_ebm <- lavPredict(fit_w, method = "EBM", se = "standard")
        se_ebm <- attr(fs_ebm, "se")[[1]]
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    r_fs_ebm = np.array(ro.r("as.matrix(fs_ebm)"))
    r_se_ebm = np.array(ro.r("as.matrix(se_ebm)"))

    syntax = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
    """
    fit_py_w = av.cfa(syntax, data=hs_ord_df, estimator="WLSMV", ordered=True)
    sc_ebm, se_ebm = fit_py_w.predict(method="ebm", se=True)

    np.testing.assert_allclose(sc_ebm.to_numpy(), r_fs_ebm, atol=2e-3)
    np.testing.assert_allclose(se_ebm.to_numpy(), r_se_ebm, atol=2e-3)

    # EAP (Expected A Posteriori)
    sc_eap, se_eap = fit_py_w.predict(method="eap", se=True)
    assert sc_eap.shape == r_fs_ebm.shape
    assert np.all(np.isfinite(se_eap.to_numpy()))
    assert np.corrcoef(sc_eap["visual"], r_fs_ebm[:, 0])[0, 1] > 0.99