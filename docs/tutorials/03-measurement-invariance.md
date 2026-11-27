# Tutorial 3: Multi-Group Measurement Invariance, Score Tests & Partial Invariance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/03_multigroup_invariance.ipynb)

This tutorial walks through a complete multi-group invariance workflow:
- Automated `configural -> metric -> scalar -> strict` invariance testing with `measurement_invariance()`
- Pinpointing non-invariant parameters using Lagrange multiplier tests (`score_test()`, matching `lavaan::lavTestScore`)
- Fitting partial invariance models (`group_partial`) and verifying unbiased latent mean difference recovery
- Running **ordinal (`WLSMV`) multi-group invariance** and multi-group factor scoring (`predict`)

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Simulate Two-Group Data with Known Scalar Non-Invariance on `x5`

??? example "Show data simulation code (click to expand)"
    ```python
    rng = np.random.default_rng(42)
    n_g = 300

    def gen_group(latent_means, intercept_shift):
        eta = rng.multivariate_normal(latent_means, [[1.0, 0.45], [0.45, 1.0]], size=n_g)
        L = np.array([
            [0.80, 0.00], [0.75, 0.00], [0.72, 0.00],
            [0.00, 0.82], [0.00, 0.78], [0.00, 0.70],
        ])
        y = eta @ L.T + rng.normal(0.0, 0.60, size=(n_g, 6)) + intercept_shift
        return pd.DataFrame(y, columns=[f"x{i}" for i in range(1, 7)])

    g_ref = gen_group([0.0, 0.0], np.zeros(6))
    g_ref["group"] = "Control"

    # True latent mean difference on f2 = +0.50, PLUS item x5 intercept bias = +0.75
    g_trt = gen_group([0.0, 0.50], np.array([0.0, 0.0, 0.0, 0.0, 0.75, 0.0]))
    g_trt["group"] = "Treatment"

    df_mg = pd.concat([g_ref, g_trt], ignore_index=True)

    model = """
        f1 =~ x1 + x2 + x3
        f2 =~ x4 + x5 + x6
    """
    print("Group sample sizes:", df_mg["group"].value_counts().to_dict())
    ```

```text
Group sample sizes: {'Control': 300, 'Treatment': 300}
```

---

## 2. Automated Measurement Invariance Sequence (`MLR`)

```python
inv = av.measurement_invariance(model, data=df_mg, group="group", estimator="MLR")
print(inv.summary())
```

???+ quote "Output"
    ```text
    anvala Measurement Invariance (MLR, groups=['Control', 'Treatment'])
         model  df      aic      bic    chisq  chisq_scaled  chisq_diff  df_diff  pvalue   cfi  delta_cfi    tli  delta_tli  rmsea  delta_rmsea   srmr  delta_srmr
    configural  16  84.0710 251.1543   8.0710        8.1976         NaN      NaN     NaN 1.000        NaN 1.0101        NaN 0.0000          NaN 0.0127         NaN
        metric  20  85.1313 252.2146   9.1313        9.3186      1.1030      4.0  0.8938 1.000      0.000 1.0110     0.0010 0.0000       0.0000 0.0160      0.0032
        scalar  24 209.6882 385.5654 129.6882      137.6766    160.2313      4.0  0.0000 0.925     -0.075 0.9021    -0.1089 0.1220       0.1220 0.0748      0.0588
        strict  30 212.5388 388.4160 132.5388      140.3008      2.9835      6.0  0.8109 0.927      0.002 0.9240     0.0219 0.1076      -0.0144 0.0750      0.0002
    ```

---

## 3. Diagnose Non-Invariant Items with `score_test()`

```python
tot_score, uni_score = inv.fits["scalar"].score_test(sort=True, return_total=True)
print("=== Omnibus Score Test (All Scalar Equality Constraints) ===")
print(tot_score.round(4).to_string(index=False))

print("\n=== Univariate Score Tests (Top 6 Constraints to Consider Releasing) ===")
print(uni_score.head(6).round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Omnibus Score Test (All Scalar Equality Constraints) ===
              test       x2      df  pvalue
             score 128.7778 10.0000     0.0
      score.scaled 124.8957 10.0000     0.0
    score.adjusted 110.7271  8.8656     0.0
      score.robust  87.3485 10.0000     0.0

    === Univariate Score Tests (Top 6 Constraints to Consider Releasing) ===
    lhs op rhs       x2  df  pvalue  x2_scaled  pvalue_scaled
     x5 ~1   1 125.8139   1  0.0000    78.0709         0.0000
     x4 ~1   1  60.1526   1  0.0000    63.6106         0.0000
     f2 =~  x5  36.2894   1  0.0000    23.6587         0.0000
     x6 ~1   1  28.8043   1  0.0000    29.1674         0.0000
     f2 =~  x6   7.0505   1  0.0079     8.3270         0.0039
     x1 ~1   1   0.9333   1  0.3340     0.9402         0.3322
    ```

---

## 4. Fit Partial Scalar Invariance (`group_partial=['x5 ~ 1']`) & Compare Latent Mean Recovery

```python
fit_partial = av.cfa_multigroup(
    model,
    data=df_mg,
    group="group",
    estimator="MLR",
    group_equal=["loadings", "intercepts"],
    group_partial=["x5 ~ 1"],
)

print("=== Model Comparison: Metric vs. Partial Scalar vs. Full Scalar ===")
comp = av.compare_models(
    inv.fits["metric"], fit_partial, inv.fits["scalar"],
    names=["metric", "partial_scalar", "full_scalar"],
)
print(comp[["model", "df", "chisq_scaled", "chisq_diff", "pvalue", "cfi", "delta_cfi", "rmsea", "srmr"]].round(4).to_string(index=False))

# Inspect Treatment group latent mean estimate for f2 (true population shift = +0.50)
pe_full = inv.fits["scalar"].parameter_estimates()
pe_part = fit_partial.parameter_estimates()

est_full = pe_full.query("group == 'Treatment' and lhs == 'f2' and op == '~1'")[["est", "se", "std_lv"]]
est_part = pe_part.query("group == 'Treatment' and lhs == 'f2' and op == '~1'")[["est", "se", "std_lv"]]
print("\nTrue standardized latent mean shift on f2: +0.50")
print("Full scalar (misspecified):\n", est_full.round(4).to_string(index=False))
print("Partial scalar (x5 ~ 1 freed):\n", est_part.round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Model Comparison: Metric vs. Partial Scalar vs. Full Scalar ===
             model  df  chisq_scaled  chisq_diff  pvalue   cfi  delta_cfi  rmsea   srmr
            metric  20        9.3186         NaN     NaN 1.000        NaN  0.000 0.0160
    partial_scalar  23       11.2580      1.9369  0.5856 1.000      0.000  0.000 0.0176
       full_scalar  24      137.6766   2181.5107  0.0000 0.925     -0.075  0.122 0.0748

    True standardized latent mean shift on f2: +0.50
    Full scalar (misspecified):
       est     se  std_lv
    0.773 0.0624  1.1619
    Partial scalar (x5 ~ 1 freed):
       est     se  std_lv
    0.489 0.0764  0.5953
    ```

---

## 5. Ordinal (`WLSMV`) Multi-Group Invariance & Multi-Group Factor Scoring

```python
cuts = np.array([-1.0, 0.0, 1.0])
df_mg_ord = df_mg.copy()
for col in [f"x{i}" for i in range(1, 7)]:
    df_mg_ord[col] = np.digitize(df_mg[col].to_numpy(), cuts) + 1

inv_ord = av.measurement_invariance(
    model,
    data=df_mg_ord,
    group="group",
    estimator="WLSMV",
    ordered=True,
    levels=("configural", "metric", "scalar"),
)
print(inv_ord.summary())

# Predict multi-group factor scores (note: pass newdata for MultiGroupFit)
mg_scores = fit_partial.predict(newdata=df_mg, method="regression")
mg_scores["group"] = df_mg["group"].values
print("\n=== Mean Predicted Factor Scores by Group ===")
print(mg_scores.groupby("group")[["f1", "f2"]].mean().round(4))
```

???+ quote "Output"
    ```text
    anvala Measurement Invariance (WLSMV, groups=['Control', 'Treatment'])
         model  df  aic  bic   chisq  chisq_scaled  chisq_diff  df_diff  pvalue  cfi  delta_cfi    tli  delta_tli  rmsea  delta_rmsea   srmr  delta_srmr
    configural  16  NaN  NaN  5.2499       11.0011         NaN      NaN     NaN  1.0        NaN 1.0112        NaN    0.0          NaN 0.0207         NaN
        metric  20  NaN  NaN  7.3981       13.9542      2.5101      4.0  0.6428  1.0        0.0 1.0115     0.0004    0.0          0.0 0.0245      0.0039
        scalar  26  NaN  NaN 13.8217       25.1437     12.3857      6.0  0.0539  1.0        0.0 1.0203     0.0087    0.0          0.0 0.0240     -0.0005

    === Mean Predicted Factor Scores by Group ===
                   f1     f2
    group                   
    Control   -0.0000 -0.000
    Treatment -0.0493  0.489
    ```

---

## Related Reference Pages

- [`cfa_multigroup()`](../reference/multigroup.md#cfa_multigroup) · [`measurement_invariance()`](../reference/multigroup.md#measurement_invariance) · [`compare_models()`](../reference/multigroup.md#compare_models) · [`predict()`](../reference/predict.md)
- Interactive notebook: [`03_multigroup_invariance.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/03_multigroup_invariance.ipynb)
