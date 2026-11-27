# Tutorial 2: Structural Equation Modeling, Mediation & GPU BCa Bootstrap

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/02_sem_mediation_gpu_bootstrap.ipynb)

This tutorial covers:
- Specifying full latent SEM models (`=~`, `~`, `~~`) in `lavaan` syntax
- Labeling paths (`a*X`, `b*M`) and defining nonlinear functions (`indirect := a*b`, `total := c + a*b`)
- Comparing analytical Delta-method standard errors vs. GPU-accelerated `BCa` and `percentile` bootstrap confidence intervals (`bootstrap_ci`)
- Applying parameter equality constraints by reusing parameter labels

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Observed-Variable Mediation & GPU BCa Bootstrap

```python
rng = np.random.default_rng(2026)
n = 350

X = rng.normal(0.0, 1.0, size=n)
M = 0.50 * X + rng.normal(0.0, 0.85, size=n)
Y = 0.45 * M + 0.18 * X + rng.normal(0.0, 0.85, size=n)
df_med = pd.DataFrame({"X": X, "M": M, "Y": Y})

med_syntax = """
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total    := c + a*b
    prop_med := (a*b) / (c + a*b)
"""

fit_med = av.sem(med_syntax, data=df_med, estimator="ML")
print("=== Analytical Estimates (Delta Method for :=) ===")
print(fit_med.parameter_estimates()[["lhs", "op", "rhs", "label", "est", "se", "pvalue", "std_all"]].round(4).to_string(index=False))

# GPU-accelerated BCa Bootstrap
boot_bca = fit_med.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca", seed=42)
print("\n=== BCa Bootstrap 95% CIs ===")
print(boot_bca.summary())

# Percentile Bootstrap for comparison
boot_pct = fit_med.bootstrap_ci(n_boot=1000, conf_level=0.95, method="percentile", seed=42)
print("\nIndirect effect CI comparison:")
print("  BCa       :", boot_bca.table.query("lhs == 'indirect'")[["est", "boot_se", "ci_lower", "ci_upper"]].round(4).to_dict("records")[0])
print("  Percentile:", boot_pct.table.query("lhs == 'indirect'")[["est", "boot_se", "ci_lower", "ci_upper"]].round(4).to_dict("records")[0])
```

???+ quote "Output"
    ```text
    === Analytical Estimates (Delta Method for :=) ===
         lhs op               rhs    label    est     se  pvalue  std_all
           M  ~                 X        a 0.4983 0.0446  0.0000   0.5129
           Y  ~                 M        b 0.3789 0.0525  0.0000   0.3824
           Y  ~                 X        c 0.2047 0.0510  0.0001   0.2127
           X ~~                 X          1.0915 0.0000     NaN   1.0000
           M ~~                 M          0.7590 0.0574  0.0000   0.7369
           Y ~~                 Y          0.7330 0.0554  0.0000   0.7251
    indirect :=               a*b indirect 0.1888 0.0312  0.0000   0.1962
       total :=           c + a*b    total 0.3935 0.0469  0.0000   0.4088
    prop_med := (a*b) / (c + a*b) prop_med 0.4798 0.0882  0.0000   0.4798

    === BCa Bootstrap 95% CIs ===
    anvala Nonparametric Bootstrap (B=1000, 95% BCA, device=cuda)
         lhs op               rhs    label    est  boot_se    bias  ci_lower  ci_upper  pvalue
           M  ~                 X        a 0.4983   0.0441 -0.0014    0.4184    0.5963     0.0
           Y  ~                 M        b 0.3789   0.0530 -0.0008    0.2857    0.4887     0.0
           Y  ~                 X        c 0.2047   0.0546  0.0012    0.0987    0.3072     0.0
           X ~~                 X          1.0915   0.0896 -0.0011    0.9286    1.2870     0.0
           M ~~                 M          0.7590   0.0589 -0.0032    0.6598    0.8888     0.0
           Y ~~                 Y          0.7330   0.0519 -0.0069    0.6384    0.8470     0.0
    indirect :=               a*b indirect 0.1888   0.0344 -0.0004    0.1306    0.2619     0.0
       total :=           c + a*b    total 0.3935   0.0512  0.0008    0.2874    0.4924     0.0
    prop_med := (a*b) / (c + a*b) prop_med 0.4798   0.0975  0.0038    0.3166    0.7016     0.0

    Indirect effect CI comparison:
      BCa       : {'est': 0.1888, 'boot_se': 0.0344, 'ci_lower': 0.1306, 'ci_upper': 0.2619}
      Percentile: {'est': 0.1888, 'boot_se': 0.0344, 'ci_lower': 0.1282, 'ci_upper': 0.2572}
    ```

---

## 2. Full Latent-Variable Structural Mediation Model

```python
rng2 = np.random.default_rng(99)
n2 = 400

F_X = rng2.normal(0.0, 1.0, size=n2)
F_M = 0.55 * F_X + rng2.normal(0.0, 0.8, size=n2)
F_Y = 0.50 * F_M + 0.20 * F_X + rng2.normal(0.0, 0.8, size=n2)

def make_indicators(factor, loadings, prefix):
    out = {}
    for i, l in enumerate(loadings, start=1):
        out[f"{prefix}{i}"] = l * factor + rng2.normal(0.0, 0.55, size=len(factor))
    return out

df_lsem = pd.DataFrame({
    **make_indicators(F_X, [0.80, 0.78, 0.74], "x"),
    **make_indicators(F_M, [0.82, 0.76, 0.72], "m"),
    **make_indicators(F_Y, [0.85, 0.79, 0.73], "y"),
})

latent_sem_syntax = """
    # Measurement model
    FX =~ x1 + x2 + x3
    FM =~ m1 + m2 + m3
    FY =~ y1 + y2 + y3

    # Structural model
    FM ~ a*FX
    FY ~ b*FM + c*FX

    # Defined parameters
    indirect := a*b
    total    := c + a*b
"""

fit_lsem = av.sem(latent_sem_syntax, data=df_lsem, estimator="MLR")
print("=== Latent SEM Fit Measures (MLR) ===")
print(fit_lsem.fit_measures().round(4))

print("\n=== Structural Paths & Defined Parameters ===")
pe_lsem = fit_lsem.parameter_estimates()
print(pe_lsem[pe_lsem["op"].isin(["~", ":="])][["lhs", "op", "rhs", "label", "est", "se", "pvalue", "std_all"]].round(4).to_string(index=False))

# GPU BCa bootstrap for the full latent mediation model
boot_lsem = fit_lsem.bootstrap_ci(n_boot=500, conf_level=0.95, method="bca", seed=42)
print("\n=== Latent SEM BCa Bootstrap (Structural & Indirect Paths) ===")
print(boot_lsem.table[boot_lsem.table["op"].isin(["~", ":="])].round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    === Latent SEM Fit Measures (MLR) ===
                    Unscaled   Scaled  Robust
    chisq            21.2029  21.8155     NaN
    df               24.0000  24.0000     NaN
    pvalue            0.6267   0.5903     NaN
    scaling_factor    1.0000   0.9719     NaN
    cfi               1.0000   1.0000  1.0000
    tli               1.0022   1.0017  1.0017
    rmsea             0.0000   0.0000  0.0000
    srmr              0.0224   0.0224  0.0224

    === Structural Paths & Defined Parameters ===
         lhs op     rhs    label    est     se  pvalue  std_all
          FM  ~      FX        a 0.5848 0.0545  0.0000   0.6035
          FY  ~      FM        b 0.4695 0.0682  0.0000   0.4552
          FY  ~      FX        c 0.2451 0.0715  0.0006   0.2453
    indirect :=     a*b indirect 0.2746 0.0474  0.0000   0.2747
       total := c + a*b    total 0.5197 0.0614  0.0000   0.5200

    === Latent SEM BCa Bootstrap (Structural & Indirect Paths) ===
         lhs op     rhs    label    est  boot_se    bias  ci_lower  ci_upper  pvalue
          FM  ~      FX        a 0.5848   0.0562 -0.0005    0.4802    0.7101     0.0
          FY  ~      FM        b 0.4695   0.0698 -0.0025    0.3441    0.6216     0.0
          FY  ~      FX        c 0.2451   0.0713  0.0013    0.1131    0.3794     0.0
    indirect :=     a*b indirect 0.2746   0.0472 -0.0020    0.1953    0.3842     0.0
       total := c + a*b    total 0.5197   0.0600 -0.0007    0.4124    0.6540     0.0
    ```

---

## 3. Equality Constraints & Nested Model Comparison

```python
# Test whether a == b using shared parameter label 'eq_ab'
constrained_syntax = """
    FX =~ x1 + x2 + x3
    FM =~ m1 + m2 + m3
    FY =~ y1 + y2 + y3

    FM ~ eq_ab*FX
    FY ~ eq_ab*FM + c*FX
"""
fit_lsem_eq = av.sem(constrained_syntax, data=df_lsem, estimator="MLR")
print(av.compare_models(fit_lsem, fit_lsem_eq, names=["Free (a, b)", "Constrained (a = b)"])[["model", "df", "chisq_scaled", "chisq_diff", "df_diff", "pvalue", "cfi", "rmsea"]].round(4).to_string(index=False))
```

???+ quote "Output"
    ```text
    model  df  chisq_scaled  chisq_diff  df_diff  pvalue  cfi  rmsea
            Free (a, b)  24       21.8155         NaN      NaN     NaN  1.0    0.0
    Constrained (a = b)  25       23.4816      1.7855      1.0  0.1815  1.0    0.0
    ```

---

## Related Reference Pages

- [`sem()` & `SEMFit`](../reference/sem.md) · [`bootstrap_ci()` & `BootstrapResult`](../reference/bootstrap.md) · [`compare_models()`](../reference/multigroup.md#compare_models)
- Interactive notebook: [`02_sem_mediation_gpu_bootstrap.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/02_sem_mediation_gpu_bootstrap.ipynb)
