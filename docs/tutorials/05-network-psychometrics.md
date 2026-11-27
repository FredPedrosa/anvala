# Tutorial 5: Network Psychometrics (`ebic_glasso`, `uva`, `ega` & GPU `boot_ega`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/05_network_psychometrics_ega.ipynb)

This tutorial covers the full Network Psychometrics pipeline in `anvala` (matching `EGAnet` 2.4.1):
- Detecting and resolving local dependence / item redundancy with Unique Variable Analysis (`uva`)
- Estimating sparse Gaussian Graphical Models via `ebic_glasso`
- Discovering communities, network loadings, and Total Entropy Fit Index (`ega`, `compute_net_loads`, `compute_tefi`)
- Evaluating structural & item stability with GPU-accelerated `boot_ega` (both `parametric` and `nonparametric`)

!!! tip "100% Reproducible Examples"
    Every snippet below generates its own synthetic dataset and displays the **exact output** produced by `anvala`. Copy and paste directly into Python, or click **Open in Colab** above to run interactively (with free T4 GPU support).

## Setup

```python
import numpy as np
import pandas as pd
import anvala as av
```

---

## 1. Simulate a 3-Community Psychometric Network with One Redundant Item Pair

??? example "Show data simulation code (click to expand)"
    ```python
    rng = np.random.default_rng(2026)
    n = 500

    phi = np.array([
        [1.00, 0.35, 0.30],
        [0.35, 1.00, 0.32],
        [0.30, 0.32, 1.00],
    ])
    eta = rng.multivariate_normal(np.zeros(3), phi, size=n)

    L = np.zeros((12, 3))
    L[0:4, 0] = [0.68, 0.65, 0.62, 0.60]
    L[4:8, 1] = [0.69, 0.64, 0.61, 0.59]
    L[8:12, 2] = [0.67, 0.63, 0.60, 0.58]

    y = eta @ L.T + rng.normal(0.0, 1.0, size=(n, 12)) * np.sqrt(1.0 - np.sum(L ** 2, axis=1))
    cols = [f"v{i:02d}" for i in range(1, 13)]
    df = pd.DataFrame(y, columns=cols)

    # Inject a locally dependent / redundant item mirroring v01
    df["v01_redundant"] = 0.92 * df["v01"] + rng.normal(0.0, 0.20, size=n)
    print("Dataset shape (with redundant item):", df.shape)
    ```

```text
Dataset shape (with redundant item): (500, 13)
```

---

## 2. Unique Variable Analysis (`uva`)

```python
uva_res = av.uva(df, cut_off=0.25, gamma=0.5, corr="pearson")
print(uva_res)
print("\n=== Redundant Pairs Exceeding wTO > 0.25 ===")
print(uva_res.redundant_pairs.round(4).to_string(index=False))

df_clean = uva_res.reduced_data
print("\nRetained items:", uva_res.keep)
```

???+ quote "Output"
    ```text
    UVAResult(cut_off=0.25, n_keep=12, n_remove=1)
    Sugeridos para remoção: ['v01']

    === Redundant Pairs Exceeding wTO > 0.25 ===
    item_1        item_2    wto
       v01 v01_redundant 0.7204

    Retained items: ['v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'v10', 'v11', 'v12', 'v01_redundant']
    ```

---

## 3. Sparse Network Estimation (`ebic_glasso`) & Exploratory Graph Analysis (`ega`)

```python
glasso = av.ebic_glasso(df_clean, gamma=0.5, n_lambda=100, corr="pearson")
print(f"EBICglasso -> lambda_opt={glasso.lambda_opt:.4f}, EBIC={glasso.ebic:.2f}, edges={glasso.n_edges}, sparsity={glasso.sparsity:.1%}")

ega_wt = av.ega(df_clean, algorithm="walktrap", gamma=0.5, corr="pearson", steps=4)
print("\n=== EGA (Walktrap) ===")
print(ega_wt)

print("\n=== Standardized Network Loadings ===")
print(ega_wt.net_loads().round(3))

# Compare Walktrap vs Louvain using TEFI (lower = better fit)
ega_lv = av.ega(df_clean, algorithm="louvain", gamma=0.5, corr="pearson")
print(f"\nTEFI comparison -> Walktrap: {ega_wt.tefi:.4f} (n_dim={ega_wt.n_dim}) | Louvain: {ega_lv.tefi:.4f} (n_dim={ega_lv.n_dim})")

# Direct compute_tefi / compute_net_loads on the recovered partition
mem_wt = ega_wt.communities["dimension"].to_numpy()
tefi_direct = av.compute_tefi(df_clean.corr().to_numpy(), mem_wt)
loads_direct = av.compute_net_loads(glasso.partial_corr, mem_wt)
print(f"Direct compute_tefi check: {tefi_direct:.4f} (matches ega_wt.tefi={ega_wt.tefi:.4f})")
```

???+ quote "Output"
    ```text
    EBICglasso -> lambda_opt=0.0611, EBIC=5273.38, edges=42, sparsity=36.4%

    === EGA (Walktrap) ===
    EGAResult(n_dim=3, algorithm='walktrap', tefi=-8.6478)
             item  dimension
              v02          1
              v03          1
              v04          1
              v05          2
              v06          2
              v07          2
              v08          2
              v09          3
              v10          3
              v11          3
              v12          3
    v01_redundant          1

    === Standardized Network Loadings ===
                   Dim_1  Dim_2  Dim_3
    v02            0.431  0.021  0.085
    v03            0.456  0.051  0.018
    v04            0.441  0.055  0.030
    v05            0.070  0.464  0.068
    v06            0.085  0.438  0.086
    v07            0.008  0.449  0.016
    v08            0.040  0.444  0.000
    v09            0.053  0.073  0.469
    v10            0.030  0.050  0.434
    v11            0.037  0.042  0.398
    v12            0.023  0.000  0.431
    v01_redundant  0.478  0.076  0.017

    TEFI comparison -> Walktrap: -8.6478 (n_dim=3) | Louvain: -8.6478 (n_dim=3)
    Direct compute_tefi check: -8.6478 (matches ega_wt.tefi=-8.6478)
    ```

---

## 4. GPU-Accelerated Bootstrap EGA (`boot_ega`)

```python
boot_res = av.boot_ega(
    df_clean,
    n_boot=200,
    boot_type="parametric",
    algorithm="walktrap",
    gamma=0.5,
    corr="pearson",
    seed=42,
    device="auto",
)

print("=== Bootstrap EGA Summary ===")
print(boot_res.summary.round(4).to_string(index=False))

print("\n=== Dimension Frequency Table ===")
print(boot_res.dimension_frequencies.round(4).to_string(index=False))

print("\n=== Item Stability Across Bootstrap Replicates ===")
print(boot_res.item_stability.round(3))

print("\n=== Typical Graph (Median Partial Correlations, first 6 items) ===")
print(boot_res.typical_graph.iloc[:6, :6].round(3))
```

???+ quote "Output"
    ```text
    === Bootstrap EGA Summary ===
     n_boot  empirical_dim  median_dim  se_dim  ci_lower  ci_upper  dimension_stability
        200              3         3.0     0.0       3.0       3.0                  1.0

    === Dimension Frequency Table ===
     n_dim  frequency  proportion
         3        200         1.0

    === Item Stability Across Bootstrap Replicates ===
                   empirical_dim  Dim_1  Dim_2  Dim_3  stability
    v02                        1    1.0    0.0    0.0        1.0
    v03                        1    1.0    0.0    0.0        1.0
    v04                        1    1.0    0.0    0.0        1.0
    v05                        2    0.0    1.0    0.0        1.0
    v06                        2    0.0    1.0    0.0        1.0
    v07                        2    0.0    1.0    0.0        1.0
    v08                        2    0.0    1.0    0.0        1.0
    v09                        3    0.0    0.0    1.0        1.0
    v10                        3    0.0    0.0    1.0        1.0
    v11                        3    0.0    0.0    1.0        1.0
    v12                        3    0.0    0.0    1.0        1.0
    v01_redundant              1    1.0    0.0    0.0        1.0

    === Typical Graph (Median Partial Correlations, first 6 items) ===
           v02    v03    v04    v05    v06    v07
    v02  0.000  0.157  0.187  0.000  0.027  0.000
    v03  0.157  0.000  0.206  0.052  0.000  0.000
    v04  0.187  0.206  0.000  0.013  0.061  0.008
    v05  0.000  0.052  0.013  0.000  0.200  0.177
    v06  0.027  0.000  0.061  0.200  0.000  0.199
    v07  0.000  0.000  0.008  0.177  0.199  0.000
    ```

---

## Related Reference Pages

- [`ebic_glasso()`, `ega()`, `uva()`, `boot_ega()`, `compute_net_loads()`, `compute_tefi()`](../reference/network.md)
- Interactive notebook: [`05_network_psychometrics_ega.ipynb`](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/05_network_psychometrics_ega.ipynb)
