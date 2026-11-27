# Local Diagnostics (`residuals`, `modindices`)

*Module:* `anvala.diagnostics` · *R equivalents:* `lavaan::residuals()`, `lavaan::modindices()`

Both functions are attached as convenience methods on `AnvalaFit` and `SEMFit` (`fit.residuals(...)`, `fit.modindices(...)`) and can also be imported directly from `anvala.diagnostics`.

---

## `residuals`

### Description

Computes local model residuals comparing observed sample moments $S$ against model-implied moments $\hat{\Sigma}$ to diagnose localized misfit hidden by global fit indices.

### Usage

```python
fit.residuals(type: str = "raw") -> pd.DataFrame
# or:
from anvala.diagnostics import residuals
residuals(fit, type: str = "raw") -> pd.DataFrame
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fit` | `AnvalaFit \| SEMFit` | *required* | Fitted CFA or SEM model object. |
| `type` | `str` | `"raw"` | Residual metric:<br>- `"raw"`: Unstandardized covariance residuals $S - \hat{\Sigma}$.<br>- `"cor"` or `"cor.bollen"`: Correlation residuals $R_{\mathrm{obs}} - R_{\mathrm{imp}}$ (Bollen, 1989). Values with $|r| > 0.10$ commonly indicate local misfit.<br>- `"standardized"`: Asymptotic $z$-residuals $(S - \hat{\Sigma}) / \mathrm{SE}(S - \hat{\Sigma})$ using the analytical projection matrix $Q = I - \Delta(\Delta^\top W \Delta)^{-1}\Delta^\top W$. |

### Value

A symmetric $p \times p$ `pd.DataFrame` indexed and labeled by observed indicator names.

---

## `modindices`

### Description

Computes Sörbom's (1989) Lagrange multiplier **Modification Indices (MI)** and **Expected Parameter Changes** (`epc`, `sepc_lv`, `sepc_all`) for all omitted cross-loadings (`=~`), residual covariances (`~~`), and—when structural regressions are present—omitted latent paths (`~`) and latent covariances (`~~`). Exact partial derivatives with respect to each candidate parameter fixed at zero are evaluated via PyTorch automatic differentiation.

### Usage

```python
fit.modindices(
    sort: bool = True,
    minimum_value: float = 0.0,
    op: str | None = None,
) -> pd.DataFrame
# or:
from anvala.diagnostics import modindices
modindices(fit, sort=True, minimum_value=0.0, op=None) -> pd.DataFrame
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fit` | `AnvalaFit \| SEMFit` | *required* | Fitted continuous or ordinal (`WLSMV`) model object. |
| `sort` | `bool` | `True` | If `True`, sorts candidate modifications in descending order of `mi`. |
| `minimum_value` | `float` | `0.0` | Minimum MI threshold for inclusion (e.g., `3.84` for $\alpha = 0.05$, `df = 1`, or `10.0` for conservative screening). |
| `op` | `str \| None` | `None` | Optional operator filter: `"=~"` (cross-loadings only), `"~~"` (error/latent covariances only), or `"~"` (structural paths only). |

### Value

A `pd.DataFrame` with columns:

| Column | Description |
| :--- | :--- |
| `lhs`, `op`, `rhs` | Candidate parameter specification (`lhs op rhs`). |
| `mi` | Expected decrease in model $\chi^2$ (1 df) if the parameter is freed. |
| `epc` | Expected unstandardized parameter value if freed. |
| `sepc_lv` | Expected parameter change standardizing latent variables only. |
| `sepc_all` | Completely standardized expected parameter change (`std.all`). |

### Examples

```python
import anvala as av

fit = av.cfa(model, data=df, estimator="MLR")

# Correlation residuals (Bollen) and standardized z-residuals
print(fit.residuals(type="cor.bollen").round(3))
print(fit.residuals(type="standardized").round(2))

# Modification indices >= 3.84
print(fit.modindices(sort=True, minimum_value=3.84).head(10))

# Inspect only candidate cross-loadings
print(fit.modindices(minimum_value=3.84, op="=~"))
```

### References

- Sörbom, D. (1989). Model modification. *Psychometrika*, 54(3), 371--384.
