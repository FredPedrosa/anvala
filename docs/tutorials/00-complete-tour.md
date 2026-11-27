# `anvala` Complete Tour: Every Psychometric Workflow in One Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)

This self-contained notebook demonstrates **every major capability** of [`anvala`](https://github.com/FredPedrosa/anvala) on reproducible synthetic datasets:

1. **Continuous & Ordinal CFA** (`cfa`, `ML`, `MLR`, `WLSMV`, `polychoric`)
2. **Factor Scores & Local Diagnostics** (`predict`, `residuals`, `modindices`)
3. **Structural Equation Modeling, Mediation & GPU BCa Bootstrap** (`sem`, `:=`, `bootstrap_ci`)
4. **Multi-Group Measurement Invariance & Score Tests** (`cfa_multigroup`, `measurement_invariance`, `compare_models`, `score_test`, `group_partial`)
5. **EFA, Dimensionality Tests, Reliability, HTMT & Bifactor Indices** (`kmo_bartlett`, `dimtests`, `efa`, `reliability`, `htmt`, `cronbach_alpha`, `bifactor_indices`)
6. **Network Psychometrics** (`ebic_glasso`, `ega`, `uva`, `compute_net_loads`, `compute_tefi`, `boot_ega`)

> **Google Colab GPU tip:** Select **Runtime -> Change runtime type -> T4 GPU** to test CUDA-accelerated `bootstrap_ci` and `boot_ega`.

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Generate Reproducible 3-Factor Continuous & Ordinal Data

??? example "Show data simulation code (click to expand)"
    ```python
    rng = np.random.default_rng(2026)
    n = 400

    # True 3-factor correlated latent variables
    phi = np.array([
        [1.00, 0.45, 0.35],
        [0.45, 1.00, 0.40],
        [0.35, 0.40, 1.00],
    ])
    eta = rng.multivariate_normal(np.zeros(3), phi, size=n)

    lam = np.array([
        [0.80, 0.00, 0.00],
        [0.75, 0.00, 0.00],
        [0.70, 0.00, 0.00],
        [0.00, 0.82, 0.00],
        [0.00, 0.76, 0.00],
        [0.00, 0.72, 0.00],
        [0.00, 0.00, 0.78],
        [0.00, 0.00, 0.74],
        [0.00, 0.00, 0.68],
    ])
    err_sd = np.sqrt(1.0 - np.sum(lam ** 2, axis=1))
    y_cont = eta @ lam.T + rng.normal(0.0, 1.0, size=(n, 9)) * err_sd

    cols = [f"x{i}" for i in range(1, 10)]
    df_cont = pd.DataFrame(y_cont, columns=cols)

    # Discretize into 5-point Likert items (1..5)
    cuts = np.array([-1.2, -0.4, 0.4, 1.2])
    df_ord = pd.DataFrame(np.digitize(y_cont, cuts) + 1, columns=cols)

    print("Continuous sample shape:", df_cont.shape)
    print("Ordinal value counts (x1):", df_ord["x1"].value_counts().sort_index().to_dict())
    ```

```text
Continuous sample shape: (400, 9)
Ordinal value counts (x1): {1: 44, 2: 82, 3: 134, 4: 99, 5: 41}
```

---

## 2. Continuous CFA (`MLR`), Local Diagnostics & Factor Scores

```python
cfa_model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit_mlr = av.cfa(cfa_model, data=df_cont, estimator="MLR")
print("=== MLR Fit Measures (Unscaled / Scaled / Robust) ===")
print(fit_mlr.fit_measures().round(4))

print("\n=== Standardized Parameter Estimates (first 9 loadings) ===")
pe = fit_mlr.parameter_estimates()
print(pe[pe["op"] == "=~"][["lhs", "op", "rhs", "est", "se", "pvalue", "std_all"]].round(4).to_string(index=False))

print("\n=== Top 5 Modification Indices ===")
print(fit_mlr.modindices(minimum_value=0.0).head(5).round(4).to_string(index=False))

scores_reg, se_reg = fit_mlr.predict(method="regression", se=True)
print("\n=== Regression Factor Scores (first 5 rows) ===")
print(scores_reg.head().round(4))
```

???+ quote "Output"
    ```text
    === MLR Fit Measures (Unscaled / Scaled / Robust) ===
                     Unscaled   Scaled   Robust
    chisq             31.2317  31.8041  31.8041
    df                24.0000  24.0000  24.0000
    pvalue             0.1472   0.1319   0.1319
    scaling_factor     1.0000   0.9820   0.9820
    shift_parameter    0.0000   0.0000   0.0000
    cfi                0.9938   0.9935   0.9935
    tli                0.9908   0.9902   0.9902
    rmsea              0.0274   0.0285   0.0283
    rmsea_ci_lower     0.0000   0.0000   0.0000
    rmsea_ci_upper     0.0519   0.0528   0.0522
    srmr               0.0301   0.0301   0.0301

    === Standardized Parameter Estimates (first 9 loadings) ===
        lhs op rhs    est     se  pvalue  std_all
     visual =~  x1 1.0000 0.0000     NaN   0.7874
     visual =~  x2 0.9433 0.0823     0.0   0.7350
     visual =~  x3 0.9390 0.0793     0.0   0.7062
    textual =~  x4 1.0000 0.0000     NaN   0.8316
    textual =~  x5 0.8793 0.0649     0.0   0.7492
    textual =~  x6 0.8428 0.0595     0.0   0.7185
      speed =~  x7 1.0000 0.0000     NaN   0.7577
      speed =~  x8 0.9153 0.0847     0.0   0.7030
      speed =~  x9 0.9672 0.0909     0.0   0.7270

    === Top 5 Modification Indices ===
      lhs op rhs      mi     epc  sepc_lv  sepc_all
    speed =~  x5 12.4830  0.2524   0.1911    0.1944
    speed =~  x6  8.7770 -0.2129  -0.1612   -0.1641
       x4 ~~  x6  6.5469  0.1475   0.1475    0.3858
       x4 ~~  x5  6.1828 -0.1554  -0.1554   -0.4266
       x6 ~~  x9  5.7314 -0.0723  -0.0723   -0.1530

    === Regression Factor Scores (first 5 rows) ===
       visual  textual   speed
    0  0.6779   1.3141 -0.3572
    1  0.0418  -1.1588 -0.8586
    2  0.5038   0.4928  0.0948
    3  0.8896   0.1741 -0.2619
    4 -0.2329  -0.0718  0.0360
    ```

---

## 3. Ordinal CFA (`WLSMV`), Polychoric Correlations & EBM/EAP Scoring

```python
fit_ord = av.cfa(cfa_model, data=df_ord, estimator="WLSMV", ordered=True)
print("=== WLSMV Fit Measures ===")
print(fit_ord.fit_measures().round(4))

poly = av.polychoric(df_ord)
print("\n=== Polychoric Correlation Matrix (first 5 items) ===")
print(poly.corr.iloc[:5, :5].round(3))

scores_ebm, se_ebm = fit_ord.predict(method="ebm", se=True)
print("\n=== Ordinal EBM Factor Scores (first 5 rows) ===")
print(scores_ebm.head().round(4))
```

???+ quote "Output"
    ```text
    === WLSMV Fit Measures ===
                     Unscaled   Scaled   Robust
    chisq             16.9044  30.1226  30.1226
    df                24.0000  24.0000  24.0000
    pvalue             0.8527   0.1807   0.1807
    scaling_factor     1.0000   0.6388   0.6388
    shift_parameter    0.0000   3.6580   3.6580
    cfi                1.0000   0.9959   1.0000
    tli                1.0045   0.9939   1.0024
    rmsea              0.0000   0.0253   0.0000
    rmsea_ci_lower     0.0000   0.0000   0.0000
    rmsea_ci_upper     0.0230   0.0504   0.0451
    srmr               0.0296   0.0296   0.0296

    === Polychoric Correlation Matrix (first 5 items) ===
           x1     x2     x3     x4     x5
    x1  1.000  0.580  0.546  0.304  0.249
    x2  0.580  1.000  0.512  0.267  0.241
    x3  0.546  0.512  1.000  0.240  0.184
    x4  0.304  0.267  0.240  1.000  0.604
    x5  0.249  0.241  0.184  0.604  1.000

    === Ordinal EBM Factor Scores (first 5 rows) ===
       visual  textual   speed
    0  0.6151   1.2079 -0.4161
    1  0.0965  -1.2834 -0.7145
    2  0.2771   0.4568  0.0740
    3  0.7945   0.1132 -0.1606
    4 -0.0963  -0.0249 -0.0079
    ```

---

## 4. SEM Mediation (`:=`) & GPU-Accelerated `BCa` Bootstrap

```python
rng_m = np.random.default_rng(7)
n_m = 300
X = rng_m.normal(size=n_m)
M = 0.55 * X + rng_m.normal(scale=0.8, size=n_m)
Y = 0.45 * M + 0.20 * X + rng_m.normal(scale=0.8, size=n_m)
df_med = pd.DataFrame({"X": X, "M": M, "Y": Y})

med_model = """
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total    := c + a*b
"""

fit_sem = av.sem(med_model, data=df_med, estimator="ML")
print("=== SEM Parameter Estimates (Delta-Method SEs) ===")
print(fit_sem.parameter_estimates()[["lhs", "op", "rhs", "label", "est", "se", "pvalue", "std_all"]].round(4).to_string(index=False))

boot_res = fit_sem.bootstrap_ci(n_boot=500, conf_level=0.95, method="bca", seed=42)
print("\n=== BCa Bootstrap Confidence Intervals ===")
print(boot_res.summary())
```

???+ quote "Output"
    ```text
    === SEM Parameter Estimates (Delta-Method SEs) ===
         lhs op     rhs    label    est     se  pvalue  std_all
           M  ~       X        a 0.5212 0.0460  0.0000   0.5471
           Y  ~       M        b 0.4720 0.0608  0.0000   0.4581
           Y  ~       X        c 0.0968 0.0579  0.0946   0.0986
           X ~~       X          0.8506 0.0000     NaN   1.0000
           M ~~       M          0.5410 0.0442  0.0000   0.7007
           Y ~~       Y          0.5993 0.0489  0.0000   0.7311
    indirect :=     a*b indirect 0.2460 0.0384  0.0000   0.2506
       total := c + a*b    total 0.3428 0.0531  0.0000   0.3492

    === BCa Bootstrap Confidence Intervals ===
    anvala Nonparametric Bootstrap (B=500, 95% BCA, device=cuda)
         lhs op     rhs    label    est  boot_se    bias  ci_lower  ci_upper  pvalue
           M  ~       X        a 0.5212   0.0493 -0.0025    0.4300    0.6320   0.000
           Y  ~       M        b 0.4720   0.0616  0.0038    0.3503    0.5925   0.000
           Y  ~       X        c 0.0968   0.0572 -0.0005   -0.0078    0.2036   0.096
           X ~~       X          0.8506   0.0672 -0.0053    0.7098    0.9797   0.000
           M ~~       M          0.5410   0.0412 -0.0050    0.4642    0.6253   0.000
           Y ~~       Y          0.5993   0.0486 -0.0029    0.5151    0.7024   0.000
    indirect :=     a*b indirect 0.2460   0.0378  0.0005    0.1779    0.3274   0.000
       total := c + a*b    total 0.3428   0.0569 -0.0001    0.2389    0.4478   0.000
    ```

---

## 5. Multi-Group Measurement Invariance, Score Tests & Partial Invariance

```python
rng_g = np.random.default_rng(101)
n_g = 220

def sim_group(mean_shift, x5_intercept_bias=0.0):
    f = rng_g.multivariate_normal([0.0, mean_shift], [[1.0, 0.4], [0.4, 1.0]], size=n_g)
    L = np.array([
        [0.80, 0.00], [0.75, 0.00], [0.72, 0.00],
        [0.00, 0.82], [0.00, 0.76], [0.00, 0.70],
    ])
    y = f @ L.T + rng_g.normal(0.0, 0.6, size=(n_g, 6))
    y[:, 4] += x5_intercept_bias  # Non-invariant intercept on x5
    return pd.DataFrame(y, columns=[f"x{i}" for i in range(1, 7)])

df_g1 = sim_group(mean_shift=0.0, x5_intercept_bias=0.0)
df_g1["school"] = "Grant-White"
df_g2 = sim_group(mean_shift=0.45, x5_intercept_bias=0.65)
df_g2["school"] = "Pasteur"
df_mg = pd.concat([df_g1, df_g2], ignore_index=True)

mg_model = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6
"""

inv = av.measurement_invariance(mg_model, data=df_mg, group="school", estimator="MLR")
print(inv.summary())

print("\n=== Scalar Invariance Score Test (Lagrange Multipliers) ===")
print(inv.fits["scalar"].score_test().head(5).round(4).to_string(index=False))

# Re-fit partial scalar invariance freeing x5 intercept
fit_partial = av.cfa_multigroup(
    mg_model, data=df_mg, group="school", estimator="MLR",
    group_equal=["loadings", "intercepts"],
    group_partial=["x5 ~ 1"],
)
print("\n=== Compare Metric vs. Partial Scalar vs. Full Scalar ===")
print(av.compare_models(inv.fits["metric"], fit_partial, inv.fits["scalar"], names=["metric", "partial_scalar", "full_scalar"])[["model", "df", "chisq_scaled", "chisq_diff", "pvalue", "cfi", "delta_cfi", "rmsea"]].round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    anvala Measurement Invariance (MLR, groups=['Grant-White', 'Pasteur'])
         model  df      aic      bic   chisq  chisq_scaled  chisq_diff  df_diff  pvalue    cfi  delta_cfi    tli  delta_tli  rmsea  delta_rmsea   srmr  delta_srmr
    configural  16  97.2967 252.5942 21.2967       21.2916         NaN      NaN     NaN 0.9946        NaN 0.9898        NaN 0.0388          NaN 0.0281         NaN
        metric  20 101.7173 257.0148 25.7173       25.9393      4.6229      4.0  0.3282 0.9940    -0.0006 0.9908     0.0010 0.0366      -0.0022 0.0354      0.0073
        scalar  24 169.7465 333.2175 89.7465       92.5092     74.1426      4.0  0.0000 0.9318    -0.0621 0.9120    -0.0788 0.1122       0.0756 0.0641      0.0287
        strict  30 174.3684 337.8393 94.3684       96.3083      4.5369      6.0  0.6044 0.9334     0.0015 0.9319     0.0199 0.0992      -0.0130 0.0676      0.0035

    === Scalar Invariance Score Test (Lagrange Multipliers) ===
    lhs op rhs      x2  df  pvalue  x2_scaled  pvalue_scaled
     x5 ~1   1 64.8411   1  0.0000    44.4824         0.0000
     x4 ~1   1 27.9933   1  0.0000    28.6306         0.0000
     x6 ~1   1 15.3391   1  0.0001    14.8021         0.0001
     f2 =~  x5  5.9829   1  0.0144     4.5982         0.0320
     x3 ~1   1  0.8219   1  0.3646     0.8164         0.3662

    === Compare Metric vs. Partial Scalar vs. Full Scalar ===
             model  df  chisq_scaled  chisq_diff  pvalue    cfi  delta_cfi  rmsea
            metric  20       25.9393         NaN     NaN 0.9940        NaN 0.0366
    partial_scalar  23       27.0141      1.1093  0.7748 0.9959     0.0020 0.0281
       full_scalar  24       92.5092    143.8022  0.0000 0.9318    -0.0641 0.1122
    ```

---

## 6. EFA, Dimensionality Tests, Reliability, HTMT & Bifactor Indices

```python
kmo = av.kmo_bartlett(df_cont)
print(f"Overall KMO = {kmo['kmo']:.4f} | Bartlett chi2({kmo['bartlett_df']}) = {kmo['bartlett_chisq']:.2f}, p = {kmo['bartlett_pvalue']:.2e}")

dt = av.dimtests(df_cont, tests=["RAWPAR", "EMPKC", "HULL", "MAP"], n_sim=50, seed=42)
print("\n=== Dimensionality Retention Tests ===")
print(dt)

fit_efa = av.efa(df_cont, n_factors=3, estimator="MLR", rotation="geomin", rstarts=10, seed=42)
print("\n=== EFA Standardized Loadings (MLR + Geomin) ===")
print(fit_efa.loadings().round(3))

rel = av.reliability(fit_mlr, data=df_cont, include_alpha=True)
print("\n=== Model-Based Reliability & Discriminant Validity (HTMT) ===")
print(rel)

# Bifactor CFA & Dueber (2021) indices
bf_model = """
    G  =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
    S1 =~ x1 + x2 + x3
    S2 =~ x4 + x5 + x6
    S3 =~ x7 + x8 + x9
"""
fit_bf = av.cfa(bf_model, data=df_cont, estimator="ML", std_lv=True, orthogonal=True)
bf_idx = fit_bf.bifactor_indices(compute_arpb=True)
print("\n=== Bifactor Model-Level Indices ===")
print(bf_idx["model"].round(4))
print("\n=== Bifactor Factor-Level Indices ===")
print(bf_idx["factors"].round(4))
```

???+ quote "Output"
    ```text
    Overall KMO = 0.7856 | Bartlett chi2(36) = 1195.11, p = 0.00e+00

    === Dimensionality Retention Tests ===
    DimTestsResult(recommended_n_factors=3 [padrão ouro: RAWPAR], KMO=0.786)
                             method  n_factors                        detail
    RAWPAR (Horn Parallel Analysis)          3          FA p95=3 | PCA p95=3
           EMPKC (Empirical Kaiser)          3      Primeiro autovalor=3.401
                     HULL (PAF/CAF)          3 st_max=288219251265261696.000
                      MAP (Velicer)          3       min_avg_sq_pcor=0.06484

    === EFA Standardized Loadings (MLR + Geomin) ===
           f1     f2     f3
    x1  0.787  0.011 -0.017
    x2  0.723  0.019 -0.002
    x3  0.727 -0.045  0.016
    x4  0.029  0.807  0.011
    x5  0.001  0.667  0.154
    x6 -0.007  0.797 -0.106
    x7 -0.032  0.030  0.735
    x8  0.010 -0.044  0.734
    x9  0.013  0.004  0.727

    === Model-Based Reliability & Discriminant Validity (HTMT) ===
    ReliabilityResult(estimator='MLR', ordered=False)
    construct  n_items  omega_total  omega_implied  omega_std     cr    ave  ave_implied  alpha_raw  alpha_std
       visual        3       0.7861         0.7864     0.7874 0.7864 0.5512       0.5530     0.7863     0.7868
      textual        3       0.8116         0.8120     0.8112 0.8120 0.5914       0.5897     0.8101     0.8099
        speed        3       0.7733         0.7735     0.7733 0.7735 0.5326       0.5323     0.7731     0.7731

    Validade Discriminante (HTMT):
             visual  textual   speed
    visual   1.0000   0.4152  0.2427
    textual  0.4152   1.0000  0.4479
    speed    0.2427   0.4479  1.0000

    === Bifactor Model-Level Indices ===
    ECV       0.3861
    PUC       0.7500
    Omega     0.8930
    OmegaH    0.5868
    ARPB      0.1835
    dtype: float64

    === Bifactor Factor-Level Indices ===
            ECV_SS  ECV_SG  ECV_GS   Omega  OmegaH  Relative_Omega       H      FD
    Factor                                                                        
    G       0.3861  0.3861  0.3861  0.8930  0.5868          0.6571  0.8046  0.8706
    S1      0.8128  0.2339  0.1872  0.7878  0.6413          0.8140  0.7119  0.8503
    S2      0.4262  0.1854  0.5738  0.9249  0.2800          0.3028  0.0000  1.1633
    S3      0.7018  0.1946  0.2982  0.7740  0.5433          0.7020  0.6442  0.8110
    ```

---

## 7. Network Psychometrics: `ebic_glasso`, `ega`, `uva` & GPU `boot_ega`

```python
# Add a near-duplicate item to illustrate Unique Variable Analysis (UVA)
df_net = df_cont.copy()
df_net["x1_dup"] = 0.92 * df_net["x1"] + rng.normal(0, 0.25, size=len(df_net))

uva_res = av.uva(df_net, cut_off=0.25, corr="pearson")
print("=== Unique Variable Analysis (UVA) ===")
print(uva_res)
print("Redundant pairs:\n", uva_res.redundant_pairs.round(4).to_string(index=False))

ega_res = av.ega(uva_res.reduced_data, algorithm="walktrap", corr="pearson")
print("\n=== Exploratory Graph Analysis (EGA) ===")
print(ega_res)
print("\nNetwork Loadings:\n", ega_res.loadings.round(3))

boot_ega_res = av.boot_ega(uva_res.reduced_data, n_boot=100, boot_type="parametric", corr="pearson", seed=42)
print("\n=== Bootstrap EGA Summary ===")
print(boot_ega_res.summary.round(4).to_string(index=False))
print("\n=== Item Stability ===")
print(boot_ega_res.item_stability.round(3))
```

???+ quote "Output"
    ```text
    === Unique Variable Analysis (UVA) ===
    UVAResult(cut_off=0.25, n_keep=3, n_remove=7)
    Sugeridos para remoção: ['x1', 'x4', 'x5', 'x7', 'x9', 'x1_dup', 'x2']
    Redundant pairs:
     item_1 item_2    wto
        x1 x1_dup 0.7181
        x4     x6 0.3449
        x7     x8 0.2838
        x4     x5 0.2821
        x7     x9 0.2766
        x8     x9 0.2519

    === Exploratory Graph Analysis (EGA) ===
    EGAResult(n_dim=1, algorithm='walktrap', tefi=0.0000)
    item  dimension
      x3          1
      x6          1
      x8          1

    Network Loadings:
         Dim_1
    x3  0.345
    x6  0.402
    x8  0.271

    === Bootstrap EGA Summary ===
     n_boot  empirical_dim  median_dim  se_dim  ci_lower  ci_upper  dimension_stability
        100              1         1.0     0.0       1.0       1.0                  1.0

    === Item Stability ===
        empirical_dim  Dim_1  stability
    x3              1    1.0        1.0
    x6              1    1.0        1.0
    x8              1    1.0        1.0
    ```

---

## Related Reference Pages

- [`cfa()`](../reference/cfa.md) · [`sem()`](../reference/sem.md) · [`measurement_invariance()`](../reference/multigroup.md) · [`efa()`](../reference/efa.md) · [`reliability()`](../reference/reliability-bifactor.md) · [`ega()` & `boot_ega()`](../reference/network.md)
- Interactive notebook: [`00_anvala_complete_tour.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)
