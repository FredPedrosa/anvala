# Polychoric Correlations & Thresholds (`polychoric`, `PolychoricResult`)

*Module:* `anvala.polychoric` · *R equivalents:* `lavaan::lavCor(ordered=...)`, `lavaan:::muthen1984()`

---

## `polychoric`

### Description

Estimates marginal item thresholds $\hat{\boldsymbol{\tau}}$, the two-step bivariate normal maximum likelihood **polychoric correlation matrix** $R$ (Olsson, 1979) using Plackett's identity with 32-point Gauss--Legendre quadrature, and the exact fourth-order asymptotic covariance matrix $\Gamma_{\mathrm{poly}}$ of thresholds and correlations in numerical parity with `lavaan:::muthen1984()`.

### Usage

```python
anvala.polychoric(data: pd.DataFrame) -> PolychoricResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | DataFrame of ordered-categorical (e.g., binary or Likert) items. Each column must have at least 2 observed categories. Rows containing missing values are listwise-deleted. |

### Details

1. **Step 1 (Thresholds):** Marginal thresholds for item $j$ with $C_j$ categories are computed as $\hat{\tau}_{j,c} = \Phi^{-1}(\hat{F}_{j,c})$ for $c = 1, \dots, C_j - 1$.
2. **Step 2 (Polychoric correlations):** Each pairwise correlation $\hat{\rho}_{ij} \in (-0.9995, 0.9995)$ maximizes the bivariate normal contingency-table log-likelihood via bounded scalar optimization (`xatol=1e-11`).
3. **Asymptotic covariance ($\Gamma$):** Exact analytical derivatives $\partial \pi_{ab} / \partial \rho$ and $\partial \pi_{ab} / \partial \tau$ form the block-triangular sandwich matrix $N \cdot B^{-1} (S_{\mathrm{sc}}^\top S_{\mathrm{sc}}) B^{-\top}$ used by `cfa(..., estimator="WLSMV")`, `sem(..., estimator="WLSMV")`, `cfa_multigroup(..., estimator="WLSMV")`, and `efa(..., estimator="WLSMV")`.

### Value (`PolychoricResult`)

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `corr` | `pd.DataFrame` | $p \times p$ polychoric correlation matrix labeled by variable names. |
| `thresholds` | `pd.DataFrame` | Long-format threshold table with columns `item`, `threshold` (`"t1"`, `"t2"`, ...), and `est`. |
| `thresholds_list` | `list[np.ndarray]` | List of per-item 1D threshold arrays aligned with `corr.columns`. |
| `gamma` | `np.ndarray` | $(n_\tau + p(p-1)/2) \times (n_\tau + p(p-1)/2)$ asymptotic covariance matrix $\Gamma$ (thresholds first, then column-major strictly lower-triangular correlations). |
| `n_obs` | `int` | Effective sample size after listwise deletion. |

### Examples

```python
import anvala as av

poly = av.polychoric(df_likert)
print(poly.corr.round(3))
print(poly.thresholds.head(8))
print("Gamma shape:", poly.gamma.shape)
```

### References

- Muthén, B. (1984). A general structural equation model with dichotomous, ordered categorical, and continuous latent variable indicators. *Psychometrika*, 49(1), 115--132.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika*, 44(4), 443--460.
