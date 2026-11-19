import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import anvala as av


def test_ebic_glasso_and_ega_continuous_parity_with_eganet(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        library(EGAnet)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        e <- EGA(hs, model = "glasso", algorithm = "walktrap", plot.EGA = FALSE)
        r_ndim <- e$n.dim
        r_wc <- as.integer(e$wc)
        r_pcor <- as.matrix(e$network)
        r_tefi <- as.numeric(e$TEFI)
        r_loads <- as.matrix(net.loads(e, ordered = "variable")$std)
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_df = ro.conversion.rpy2py(ro.r("hs"))

    r_ndim = int(ro.r("r_ndim")[0])
    r_wc = np.array(ro.r("r_wc"), dtype=np.int64)
    r_pcor = np.array(ro.r("r_pcor"))
    r_tefi = float(ro.r("r_tefi")[0])
    r_loads = np.array(ro.r("r_loads"))

    # 1. Matriz de correlações parciais EBICglasso
    glasso_res = av.ebic_glasso(hs_df, gamma=0.5)
    pcor = glasso_res.partial_corr.to_numpy()
    np.testing.assert_allclose(pcor, r_pcor, atol=1e-3)

    # 2. EGA completo (Walktrap + Louvain, rede parcial, TEFI e net.loads)
    ega_res = av.ega(hs_df, algorithm="walktrap")
    assert ega_res.n_dim == r_ndim == 3
    np.testing.assert_array_equal(ega_res.communities["dimension"].to_numpy(), r_wc)
    np.testing.assert_allclose(ega_res.partial_corr().to_numpy(), r_pcor, atol=1e-3)
    np.testing.assert_allclose(ega_res.tefi, r_tefi, atol=1e-4)
    np.testing.assert_allclose(ega_res.net_loads().to_numpy(), r_loads, atol=1e-3)

    ega_louvain = av.ega(hs_df, algorithm="louvain")
    assert ega_louvain.n_dim == 3


def test_ega_ordinal_default_polychoric_uva_and_boot_ega(r_env):
    ro = r_env["ro"]
    ro.r("""
        library(lavaan)
        library(EGAnet)
        hs <- HolzingerSwineford1939[, paste0("x", 1:9)]
        hs_ord <- as.data.frame(lapply(hs, function(col) {
            as.integer(cut(col, breaks = quantile(col, probs = seq(0, 1, 0.2)), include.lowest = TRUE))
        }))
        e_ord <- EGA(hs_ord, model = "glasso", algorithm = "walktrap", plot.EGA = FALSE)
        r_ndim_ord <- e_ord$n.dim
        r_wc_ord <- as.integer(e_ord$wc)
        r_pcor_ord <- as.matrix(e_ord$network)
        r_tefi_ord <- as.numeric(e_ord$TEFI)
    """)
    with localconverter(ro.default_converter + pandas2ri.converter):
        hs_ord_df = ro.conversion.rpy2py(ro.r("hs_ord"))

    # Sem passar corr=..., o padrão corr="auto" usa correlação policórica automaticamente
    ega_ord = av.ega(hs_ord_df, algorithm="walktrap")
    assert ega_ord.n_dim == int(ro.r("r_ndim_ord")[0]) == 3
    np.testing.assert_array_equal(
        ega_ord.communities["dimension"].to_numpy(),
        np.array(ro.r("r_wc_ord"), dtype=np.int64),
    )
    np.testing.assert_allclose(ega_ord.partial_corr().to_numpy(), np.array(ro.r("r_pcor_ord")), atol=1e-3)
    np.testing.assert_allclose(ega_ord.tefi, float(ro.r("r_tefi_ord")[0]), atol=1e-3)

    # UVA com padrão corr="auto" (policórica) em item redundante
    rng = np.random.default_rng(123)
    hs_red = hs_ord_df.copy()
    hs_red["x1_dup"] = np.clip(hs_red["x1"] + rng.choice([0, 0, 0, 1, -1], size=len(hs_red)), 1, 5)

    uva_res = av.uva(hs_red, cut_off=0.25)
    assert len(uva_res.remove) >= 1
    assert any(item in uva_res.remove for item in ("x1", "x1_dup"))

    # bootEGA paramétrico e não-paramétrico sobre itens ordinais (policórica)
    boot_param = av.boot_ega(hs_ord_df, n_boot=25, boot_type="parametric", seed=42)
    assert boot_param.empirical.n_dim == 3
    assert 0.0 < boot_param.dimension_stability <= 1.0
    # Itens dos fatores Visual (x1-x3) e Textual (x4-x6) mantêm alta estabilidade
    assert np.all(boot_param.item_stability.loc[["x1", "x2", "x3", "x4", "x5", "x6"], "stability"] >= 0.85)
    assert boot_param.typical_graph.shape == (9, 9)

    boot_nonparam = av.boot_ega(hs_ord_df, n_boot=15, boot_type="nonparametric", seed=42)
    assert boot_nonparam.empirical.n_dim == 3
    assert np.isclose(boot_nonparam.dimension_frequencies["proportion"].sum(), 1.0)