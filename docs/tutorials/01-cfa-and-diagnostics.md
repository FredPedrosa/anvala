# Tutorial 1: Continuous & Ordinal CFA, Factor Scores & Local Diagnostics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/01_cfa_ordinal_and_scoring.ipynb)

In this tutorial you will learn how to:
- Fit continuous CFA models with `ML` and robust `MLR` (Yuan--Bentler / Satorra--Bentler corrections)
- Fit ordered-categorical (Likert) CFA models with `WLSMV` (delta parameterization) and inspect polychoric correlations & thresholds
- Inspect local misfit with `residuals(type=...)` and `modindices()`
- Compute latent factor scores and standard errors with `predict()` across all continuous (`regression`, `bartlett`, `anderson-rubin`, `green`) and ordinal (`ebm`, `ml`, `eap`) methods

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Simulate Continuous and Likert Data with a Subtle Cross-Loading

??? example "Show data simulation code (click to expand)"
    ```python
    rng = np.random.default_rng(123)
    n = 450

    phi = np.array([
        [1.00, 0.50, 0.35],
        [0.50, 1.00, 0.40],
        [0.35, 0.40, 1.00],
    ])
    eta = rng.multivariate_normal(np.zeros(3), phi, size=n)

    # Intentional cross-loading of x9 on Visual (0.32) so modindices() detects it!
    lam = np.array([
        [0.80, 0.00, 0.00],
        [0.76, 0.00, 0.00],
        [0.72, 0.00, 0.00],
        [0.00, 0.84, 0.00],
        [0.00, 0.78, 0.00],
        [0.00, 0.74, 0.00],
        [0.00, 0.00, 0.80],
        [0.00, 0.00, 0.74],
        [0.32, 0.00, 0.62],
    ])
    # Add mild skewness via standardized chi-square errors to showcase MLR vs ML
    raw_err = (rng.chisquare(df=5, size=(n, 9)) - 5.0) / np.sqrt(10.0)
    comm = np.sum((lam @ phi) * lam, axis=1)
    y = eta @ lam.T + raw_err * np.sqrt(np.maximum(0.05, 1.0 - comm))

    cols = [f"x{i}" for i in range(1, 10)]
    df_cont = pd.DataFrame(y, columns=cols)
    df_likert = pd.DataFrame(np.digitize(y, [-1.3, -0.45, 0.45, 1.3]) + 1, columns=cols)
    df_cont.head(3).round(3)
    ```

```text
x1     x2     x3     x4     x5     x6     x7     x8     x9
0  0.859  1.206  1.736 -0.738  1.621 -0.784  0.731  1.206  1.035
1 -0.213  1.902 -0.112 -0.167 -0.043  0.130 -0.482 -0.155  0.583
2  0.262  0.577  0.461  0.923  0.382  0.787  0.610 -0.391  0.151
```

---

## 2. Continuous CFA: `ML` vs `MLR`

```python
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit_ml = av.cfa(model, data=df_cont, estimator="ML")
fit_mlr = av.cfa(model, data=df_cont, estimator="MLR")

print("=== Fit Measures (MLR: Unscaled | Scaled | Robust) ===")
print(fit_mlr.fit_measures().round(4))

print("\n=== Standardized Matrices (inspect_std) ===")
std_mats = fit_mlr.inspect_std()
print("Lambda (standardized):\n", std_mats["lambda"].round(3))
print("\nPsi (factor correlations):\n", std_mats["psi"].round(3))
```

???+ quote "Output"
    ```text
    === Fit Measures (MLR: Unscaled | Scaled | Robust) ===
                     Unscaled   Scaled   Robust
    chisq             69.0562  67.4912  67.4912
    df                24.0000  24.0000  24.0000
    pvalue             0.0000   0.0000   0.0000
    scaling_factor     1.0000   1.0232   1.0232
    shift_parameter    0.0000   0.0000   0.0000
    cfi                0.9722   0.9696   0.9724
    tli                0.9582   0.9545   0.9586
    rmsea              0.0646   0.0635   0.0642
    rmsea_ci_lower     0.0471   0.0461   0.0464
    rmsea_ci_upper     0.0827   0.0814   0.0825
    srmr               0.0385   0.0385   0.0385

    === Standardized Matrices (inspect_std) ===
    Lambda (standardized):
         visual  textual  speed
    x1   0.795    0.000  0.000
    x2   0.753    0.000  0.000
    x3   0.667    0.000  0.000
    x4   0.000    0.806  0.000
    x5   0.000    0.781  0.000
    x6   0.000    0.762  0.000
    x7   0.000    0.000  0.715
    x8   0.000    0.000  0.701
    x9   0.000    0.000  0.871

    Psi (factor correlations):
              visual  textual  speed
    visual    1.000    0.484  0.613
    textual   0.484    1.000  0.557
    speed     0.613    0.557  1.000
    ```

---

## 3. Local Diagnostics: Residuals & Modification Indices

```python
print("=== Correlation Residuals (Bollen: R_obs - R_imp) ===")
print(fit_mlr.residuals(type="cor.bollen").round(3))

print("\n=== Top Modification Indices (MI >= 3.84) ===")
mi = fit_mlr.modindices(sort=True, minimum_value=3.84)
print(mi.head(8).round(4).to_string(index=False))

# Re-specify model adding the substantively justified cross-loading visual =~ x9
model_respec = """
    visual  =~ x1 + x2 + x3 + x9
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""
fit_mlr_2 = av.cfa(model_respec, data=df_cont, estimator="MLR")
print("\n=== Nested Model Comparison (Satorra-Bentler Scaled LRT) ===")
print(av.compare_models(fit_mlr, fit_mlr_2, names=["Original", "With visual=~x9"])[["model", "df", "chisq_scaled", "chisq_diff", "pvalue", "cfi", "rmsea", "srmr"]].round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Correlation Residuals (Bollen: R_obs - R_imp) ===
           x1     x2     x3     x4     x5     x6     x7     x8     x9
    x1  0.000 -0.000 -0.010  0.051  0.021  0.012 -0.103 -0.057  0.066
    x2 -0.000  0.000  0.014 -0.038 -0.041 -0.049 -0.110 -0.034  0.062
    x3 -0.010  0.014  0.000  0.001  0.014  0.010 -0.078 -0.022  0.030
    x4  0.051 -0.038  0.001  0.000 -0.004 -0.002  0.004  0.019  0.013
    x5  0.021 -0.041  0.014 -0.004  0.000  0.007 -0.044 -0.040  0.022
    x6  0.012 -0.049  0.010 -0.002  0.007 -0.000  0.010 -0.018 -0.013
    x7 -0.103 -0.110 -0.078  0.004 -0.044  0.010  0.000  0.092 -0.011
    x8 -0.057 -0.034 -0.022  0.019 -0.040 -0.018  0.092 -0.000 -0.026
    x9  0.066  0.062  0.030  0.013  0.022 -0.013 -0.011 -0.026  0.000

    === Top Modification Indices (MI >= 3.84) ===
        lhs op rhs      mi     epc  sepc_lv  sepc_all
     visual =~  x9 45.3863  0.5685   0.4616    0.4621
         x7 ~~  x8 36.3654  0.2287   0.2287    0.4401
     visual =~  x7 28.8755 -0.4182  -0.3396   -0.3361
         x8 ~~  x9 19.0913 -0.2085  -0.2085   -0.5780
         x1 ~~  x9  6.9318  0.0663   0.0663    0.2185
         x2 ~~  x9  6.3832  0.0595   0.0595    0.1958
    textual =~  x2  5.4059 -0.1538  -0.1192   -0.1264
         x2 ~~  x7  5.1462 -0.0594  -0.0594   -0.1355

    === Nested Model Comparison (Satorra-Bentler Scaled LRT) ===
              model  df  chisq_scaled  chisq_diff  pvalue    cfi  rmsea   srmr
    With visual=~x9  23       24.0346         NaN     NaN 0.9993 0.0102 0.0222
           Original  24       67.4912     73.7445     0.0 0.9724 0.0642 0.0385
    ```

---

## 4. Ordinal CFA (`WLSMV`) & Polychoric Thresholds

```python
fit_wlsmv = av.cfa(model_respec, data=df_likert, estimator="WLSMV", ordered=True)
print("=== Ordinal WLSMV Fit Measures ===")
print(fit_wlsmv.fit_measures().round(4))

print("\n=== Sample Polychoric Thresholds (first 8) ===")
print(fit_wlsmv.inspect_sampstat()["thresholds"].head(8).round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Ordinal WLSMV Fit Measures ===
                     Unscaled   Scaled   Robust
    chisq             19.4890  38.5347  38.5347
    df                23.0000  23.0000  23.0000
    pvalue             0.6725   0.0223   0.0223
    scaling_factor     1.0000   0.5555   0.5555
    shift_parameter    0.0000   3.4512   3.4512
    cfi                1.0000   0.9941   0.9894
    tli                1.0012   0.9907   0.9834
    rmsea              0.0000   0.0388   0.0421
    rmsea_ci_lower     0.0000   0.0147   0.0079
    rmsea_ci_upper     0.0316   0.0596   0.0677
    srmr               0.0289   0.0289   0.0289

    === Sample Polychoric Thresholds (first 8) ===
    item threshold     est
      x1        t1 -1.4051
      x1        t2 -0.5308
      x1        t3  0.3525
      x1        t4  1.1862
      x2        t1 -1.4356
      x2        t2 -0.4307
      x2        t3  0.4615
      x2        t4  1.4202
    ```

---

## 5. Factor Scoring (`predict`) for Continuous & Ordinal Models

```python
# Continuous methods
sc_reg, se_reg = fit_mlr_2.predict(method="regression", se=True)
sc_bart = fit_mlr_2.predict(method="bartlett")
sc_ar = fit_mlr_2.predict(method="anderson-rubin")
sc_green = fit_mlr_2.predict(method="green")

print("Correlation between Regression and Bartlett 'visual' scores:",
      round(sc_reg["visual"].corr(sc_bart["visual"]), 4))
print("Anderson-Rubin score covariance (should be ~Identity):\n", sc_ar.cov().round(3))

# Ordinal methods: EBM vs EAP
sc_ebm, se_ebm = fit_wlsmv.predict(method="ebm", se=True)
sc_eap, se_eap = fit_wlsmv.predict(method="eap", se=True)

comp_ord = pd.DataFrame({
    "EBM_visual": sc_ebm["visual"].head(6),
    "EBM_se": se_ebm["visual"].head(6),
    "EAP_visual": sc_eap["visual"].head(6),
    "EAP_se": se_eap["visual"].head(6),
})
print("\n=== Ordinal EBM vs EAP Scores & SEs (first 6 respondents) ===")
print(comp_ord.round(4))
```

???+ quote "Output"
    ```text
    Correlation between Regression and Bartlett 'visual' scores: 0.9901
    Anderson-Rubin score covariance (should be ~Identity):
              visual  textual  speed
    visual    1.002    0.000  0.000
    textual   0.000    1.002  0.000
    speed     0.000    0.000  1.002

    === Ordinal EBM vs EAP Scores & SEs (first 6 respondents) ===
       EBM_visual  EBM_se  EAP_visual  EAP_se
    0      0.8775  0.3512      0.8543  0.3059
    1      0.5612  0.3467      0.6009  0.3635
    2      0.4140  0.3454      0.4214  0.4052
    3     -0.3682  0.3466     -0.3465  0.4031
    4     -0.3882  0.3439     -0.3792  0.4052
    5     -0.0540  0.3480     -0.0356  0.2863
    ```

---

## Related Reference Pages

- [`cfa()` & `AnvalaFit`](../reference/cfa.md) · [`predict()`](../reference/predict.md) · [`residuals()` & `modindices()`](../reference/diagnostics.md) · [`polychoric()`](../reference/polychoric.md)
- Interactive notebook: [`01_cfa_ordinal_and_scoring.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/01_cfa_ordinal_and_scoring.ipynb)
