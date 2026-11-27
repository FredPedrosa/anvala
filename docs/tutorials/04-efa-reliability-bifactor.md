# Tutorial 4: EFA, Dimensionality Tests, Reliability, HTMT & Bifactor Indices

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/04_efa_reliability_bifactor.ipynb)

This tutorial demonstrates the full exploratory and reliability toolkit in `anvala`:
- Sampling adequacy: `kmo_bartlett()` (overall KMO, item-level MSA, Bartlett's test)
- Dimensionality retention battery: `dimtests()` (`RAWPAR`, `EMPKC`, `HULL`, `SMT`, `MAP`, `CD`, `NEVALSGT1`, `SESCREE`)
- Exploratory Factor Analysis: `efa()` with continuous (`MLR`) & ordinal (`WLSMV`) estimators and oblique/orthogonal rotations (`geomin`, `oblimin`, `promax`, `varimax`)
- Model-based reliability & discriminant validity: `reliability()`, `htmt()`, `cronbach_alpha()`
- Bifactor evaluation: `bifactor_indices()` (ECV, PUC, $\omega$, $\omega_H$, $H$, $FD$, IECV, ARPB)

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Simulate a 3-Factor Dataset (Continuous & Ordinal)

??? example "Show data simulation code (click to expand)"
    ```python
    rng = np.random.default_rng(2026)
    n = 450

    phi = np.array([
        [1.00, 0.40, 0.30],
        [0.40, 1.00, 0.35],
        [0.30, 0.35, 1.00],
    ])
    eta = rng.multivariate_normal(np.zeros(3), phi, size=n)

    L = np.zeros((12, 3))
    L[0:4, 0] = [0.82, 0.78, 0.74, 0.70]
    L[4:8, 1] = [0.84, 0.79, 0.73, 0.69]
    L[8:12, 2] = [0.80, 0.76, 0.72, 0.68]

    y = eta @ L.T + rng.normal(0.0, 1.0, size=(n, 12)) * np.sqrt(1.0 - np.sum(L ** 2, axis=1))
    cols = [f"item_{i:02d}" for i in range(1, 13)]
    df_cont = pd.DataFrame(y, columns=cols)
    df_ord = pd.DataFrame(np.digitize(y, [-1.2, -0.4, 0.4, 1.2]) + 1, columns=cols)
    ```

---

## 2. KMO, Bartlett & Dimensionality Retention (`dimtests`)

```python
kb = av.kmo_bartlett(df_cont)
print(f"Overall KMO: {kb['kmo']:.4f}")
print(f"Bartlett's test: chi2({kb['bartlett_df']}) = {kb['bartlett_chisq']:.2f}, p = {kb['bartlett_pvalue']:.2e}")
print("Lowest 3 item MSAs:\n", kb["msa"].sort_values().head(3).round(4))

dt = av.dimtests(
    df_cont,
    tests=["RAWPAR", "EMPKC", "HULL", "SMT", "MAP", "NEVALSGT1", "SESCREE"],
    n_sim=100,
    seed=42,
)
print("\n=== Dimensionality Battery ===")
print(dt)
```

???+ quote "Output"
    ```text
    Overall KMO: 0.8458
    Bartlett's test: chi2(66) = 2247.20, p = 0.00e+00
    Lowest 3 item MSAs:
     item_06    0.8253
    item_09    0.8265
    item_03    0.8287
    Name: MSA, dtype: float64

    === Dimensionality Battery ===
    DimTestsResult(recommended_n_factors=4 [padrão ouro: RAWPAR], KMO=0.846)
                             method  n_factors                   detail
    RAWPAR (Horn Parallel Analysis)          4     FA p95=4 | PCA p95=3
           EMPKC (Empirical Kaiser)          3 Primeiro autovalor=4.214
                     HULL (PAF/CAF)          3           st_max=261.944
     SMT (Sequential Model Testing)          3                 p=0.5367
                      MAP (Velicer)          3  min_avg_sq_pcor=0.03251
             NEVALSGT1 (Kaiser > 1)          3                 ev > 1.0
             SESCREE (Zoski & Jurs)          3                 SE scree
    ```

---

## 3. Exploratory Factor Analysis (`efa`) — Continuous `MLR` & Ordinal `WLSMV`

```python
efa_mlr = av.efa(df_cont, n_factors=3, estimator="MLR", rotation="geomin", rstarts=15, seed=42)
print("=== Continuous EFA (MLR + Geomin) ===")
print(efa_mlr)
print("\nFactor Correlations:\n", efa_mlr.factor_corr.round(3))
print("\nVariance Explained:\n", efa_mlr.variance_explained.round(3))

# Ordinal EFA with WLSMV + Oblimin
efa_ord = av.efa(df_ord, n_factors=3, estimator="WLSMV", ordered=True, rotation="oblimin", rstarts=10, seed=42)
print("\n=== Ordinal EFA (WLSMV + Oblimin) Fit Measures ===")
print(efa_ord.fit_measures().round(4))
```

???+ quote "Output"
    ```text
    === Continuous EFA (MLR + Geomin) ===
    EFAFit(n_factors=3, estimator='MLR', rotation='geomin', chisq=31.984 (df=33), CFI=1.000, RMSEA=0.000)
                f1     f2     f3
    item_01  0.814  0.066  0.027
    item_02  0.731  0.065 -0.012
    item_03  0.774 -0.036 -0.029
    item_04  0.671 -0.036  0.010
    item_05  0.001  0.828  0.051
    item_06 -0.004  0.822 -0.024
    item_07 -0.006  0.735 -0.011
    item_08  0.059  0.660  0.007
    item_09 -0.032  0.015  0.793
    item_10  0.013  0.004  0.778
    item_11  0.046 -0.042  0.723
    item_12 -0.020  0.012  0.689

    Factor Correlations:
            f1     f2     f3
    f1  1.000  0.305  0.270
    f2  0.305  1.000  0.357
    f3  0.270  0.357  1.000

    Variance Explained:
         ss_loadings  proportion_var  cumulative_var  proportion_explained
    f1        2.279           0.190           0.190                 0.330
    f2        2.377           0.198           0.388                 0.345
    f3        2.240           0.187           0.575                 0.325

    === Ordinal EFA (WLSMV + Oblimin) Fit Measures ===
                     Unscaled   Scaled   Robust
    chisq              9.3169  32.7213  32.7213
    df                33.0000  33.0000  33.0000
    pvalue             1.0000   0.4809   0.4809
    scaling_factor     1.0000   0.3218   0.3218
    shift_parameter    0.0000   3.7678   3.7678
    cfi                1.0000   1.0000   1.0000
    tli                1.0085   1.0002   1.0027
    rmsea              0.0000   0.0000   0.0000
    rmsea_ci_lower     0.0000   0.0000   0.0000
    rmsea_ci_upper     0.0000   0.0342   0.0394
    srmr               0.0155   0.0155   0.0155
    ```

---

## 4. Reliability (McDonald's Omega, Green & Yang Categorical Omega, AVE) & HTMT

```python
cfa_syntax = """
    F1 =~ item_01 + item_02 + item_03 + item_04
    F2 =~ item_05 + item_06 + item_07 + item_08
    F3 =~ item_09 + item_10 + item_11 + item_12
"""

fit_cfa_cont = av.cfa(cfa_syntax, data=df_cont, estimator="MLR")
fit_cfa_ord = av.cfa(cfa_syntax, data=df_ord, estimator="WLSMV", ordered=True)

rel_cont = av.reliability(fit_cfa_cont, data=df_cont, include_alpha=True)
print("=== Continuous Reliability & HTMT ===")
print(rel_cont)

rel_ord = av.reliability(fit_cfa_ord, data=df_ord)
print("\n=== Ordinal Reliability (includes Green & Yang omega_categorical) ===")
print(rel_ord.constructs.round(4).to_string(index=False))

# Standalone HTMT / HTMT2 and Cronbach's alpha
factors = {"F1": cols[0:4], "F2": cols[4:8], "F3": cols[8:12]}
print("\n=== Standalone HTMT2 (geometric=True) ===")
print(av.htmt(df_cont, factors=factors, geometric=True).round(4))

print("\n=== Standalone Cronbach's Alpha (F1) ===")
print(av.cronbach_alpha(df_cont[cols[0:4]]).round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Continuous Reliability & HTMT ===
    ReliabilityResult(estimator='MLR', ordered=False)
    construct  n_items  omega_total  omega_implied  omega_std     cr    ave  ave_implied  alpha_raw  alpha_std
           F1        4       0.8359         0.8378     0.8390 0.8378 0.5650       0.5677     0.8371     0.8377
           F2        4       0.8522         0.8519     0.8520 0.8519 0.5919       0.5917     0.8494     0.8498
           F3        4       0.8324         0.8324     0.8333 0.8324 0.5544       0.5564     0.8318     0.8324

    Validade Discriminante (HTMT):
            F1      F2      F3
    F1  1.0000  0.3232  0.2660
    F2  0.3232  1.0000  0.3608
    F3  0.2660  0.3608  1.0000

    === Ordinal Reliability (includes Green & Yang omega_categorical) ===
    construct  n_items  omega_total  omega_implied  omega_std     cr    ave  ave_implied  omega_categorical
           F1        4       0.8468         0.8468     0.8468 0.8468 0.5825       0.5825             0.8181
           F2        4       0.8534         0.8534     0.8534 0.8534 0.5943       0.5943             0.8309
           F3        4       0.8363         0.8363     0.8363 0.8363 0.5617       0.5617             0.8105

    === Standalone HTMT2 (geometric=True) ===
            F1      F2      F3
    F1  1.0000  0.3232  0.2660
    F2  0.3232  1.0000  0.3608
    F3  0.2660  0.3608  1.0000

    === Standalone Cronbach's Alpha (F1) ===
     n_items  alpha_raw  alpha_std
           4     0.8371     0.8377
    ```

---

## 5. Bifactor CFA & Dueber (2021) Bifactor Indices

```python
# Simulate a true bifactor structure (strong General factor + 3 Specific factors)
rng_bf = np.random.default_rng(77)
n_bf = 450
factors_bf = rng_bf.normal(0.0, 1.0, size=(n_bf, 4))  # [G, S1, S2, S3]
L_bf = np.zeros((12, 4))
L_bf[:, 0] = [0.72, 0.68, 0.70, 0.66, 0.74, 0.69, 0.71, 0.65, 0.73, 0.67, 0.70, 0.64]
L_bf[0:4, 1] = [0.42, 0.38, 0.40, 0.36]
L_bf[4:8, 2] = [0.45, 0.40, 0.38, 0.35]
L_bf[8:12, 3] = [0.40, 0.44, 0.37, 0.39]

y_bf = factors_bf @ L_bf.T + rng_bf.normal(0.0, 1.0, size=(n_bf, 12)) * np.sqrt(1.0 - np.sum(L_bf ** 2, axis=1))
df_bf = pd.DataFrame(y_bf, columns=cols)

bf_syntax = """
    G  =~ item_01 + item_02 + item_03 + item_04 + item_05 + item_06 + item_07 + item_08 + item_09 + item_10 + item_11 + item_12
    S1 =~ item_01 + item_02 + item_03 + item_04
    S2 =~ item_05 + item_06 + item_07 + item_08
    S3 =~ item_09 + item_10 + item_11 + item_12
"""

fit_bf = av.cfa(bf_syntax, data=df_bf, estimator="MLR", std_lv=True, orthogonal=True)
print("=== Bifactor CFA Fit Measures ===")
print(fit_bf.fit_measures().round(4))

bf_indices = fit_bf.bifactor_indices(compute_arpb=True)
print("\n=== Model-Level Bifactor Indices ===")
print(bf_indices["model"].round(4))
print("\n=== Factor-Level Bifactor Indices (ECV, Omega, OmegaH, H, FD) ===")
print(bf_indices["factors"].round(4))
print("\n=== Item-Level IECV & Relative Parameter Bias (RPB) ===")
print(bf_indices["items"].round(4))
```

???+ quote "Output"
    ```text
    === Bifactor CFA Fit Measures ===
                     Unscaled   Scaled   Robust
    chisq             52.9205  55.0230  55.0230
    df                42.0000  42.0000  42.0000
    pvalue             0.1204   0.0858   0.0858
    scaling_factor     1.0000   0.9618   0.9618
    shift_parameter    0.0000   0.0000   0.0000
    cfi                0.9965   0.9959   0.9960
    tli                0.9945   0.9936   0.9937
    rmsea              0.0240   0.0262   0.0257
    rmsea_ci_lower     0.0000   0.0000   0.0000
    rmsea_ci_upper     0.0422   0.0442   0.0430
    srmr               0.0173   0.0173   0.0173

    === Model-Level Bifactor Indices ===
    ECV       0.7512
    PUC       0.7273
    Omega     0.9435
    OmegaH    0.8505
    ARPB      0.0444
    dtype: float64

    === Factor-Level Bifactor Indices (ECV, Omega, OmegaH, H, FD) ===
            ECV_SS  ECV_SG  ECV_GS   Omega  OmegaH  Relative_Omega       H      FD
    Factor                                                                        
    G       0.7512  0.7512  0.7512  0.9435  0.8505          0.9014  0.9156  0.9247
    S1      0.1921  0.0633  0.8079  0.8663  0.1662          0.1918  0.3510  0.6074
    S2      0.3054  0.1027  0.6946  0.8721  0.2623          0.3007  0.4961  0.7423
    S3      0.2476  0.0827  0.7524  0.8706  0.2157          0.2477  0.4246  0.6743

    === Item-Level IECV & Relative Parameter Bias (RPB) ===
               IECV     RPB
    item_01  0.8331  0.0150
    item_02  0.7831  0.0282
    item_03  0.7944  0.0215
    item_04  0.8201  0.0232
    item_05  0.6155  0.0759
    item_06  0.7323  0.0567
    item_07  0.6928  0.0648
    item_08  0.7638  0.0539
    item_09  0.7718  0.0366
    item_10  0.7001  0.0650
    item_11  0.7655  0.0443
    item_12  0.7671  0.0480
    ```

---

## Related Reference Pages

- [`kmo_bartlett()`, `dimtests()`, `efa()`](../reference/efa.md) · [`reliability()`, `htmt()`, `cronbach_alpha()`, `bifactor_indices()`](../reference/reliability-bifactor.md)
- Interactive notebook: [`04_efa_reliability_bifactor.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/04_efa_reliability_bifactor.ipynb)
