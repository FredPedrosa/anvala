# Confirmatory Factor Analysis (`cfa`, `AnvalaFit`)

*Module:* `anvala.cfa` · *R equivalent:* `lavaan::cfa()`

---

## `cfa`

### Description

Fits a Confirmatory Factor Analysis (CFA) or latent measurement model using `lavaan` syntax with exact double-precision (`float64`) PyTorch automatic differentiation and BFGS quasi-Newton optimization. Supports continuous normal-theory (`ML`), Yuan--Bentler / Satorra--Bentler robust (`MLR`), unweighted least squares (`ULS`), and ordered-categorical polychoric (`WLSMV`, `ULSMV`, `DWLS`) estimation under delta parameterization.

### Usage

```python
anvala.cfa(
    model: str,
    data: pd.DataFrame,
    estimator: str | None = None,
    std_lv: bool = False,
    orthogonal: bool = False,
    ordered: bool = False,
) -> AnvalaFit
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | *required* | Model specification in `lavaan` syntax (`=~`, `~`, `~~`, parameter labels, fixed values, `NA*`). |
| `data` | `pd.DataFrame` | *required* | Input dataset containing the observed indicator columns referenced in `model`. Rows with missing values on referenced indicators are listwise-deleted. |
| `estimator` | `str \| None` | `None` | Estimation method. If `None`, defaults to `"WLSMV"` when `ordered=True` and `"ML"` otherwise. Supported values: `"ML"`, `"MLR"`, `"ULS"` (continuous); `"WLSMV"`, `"ULSMV"`, `"DWLS"` (ordinal; automatically sets `ordered=True`). |
| `std_lv` | `bool` | `False` | If `True`, exogenous latent variances are fixed to `1.0` and all factor loadings are freely estimated. If `False` (default), the first indicator loading of each latent variable is fixed to `1.0`. |
| `orthogonal` | `bool` | `False` | If `True`, covariances among exogenous latent factors are fixed to `0.0` (useful for orthogonal bifactor models). |
| `ordered` | `bool` | `False` | If `True`, treats observed indicators as ordered-categorical, estimates two-step polychoric correlations and thresholds via `polychoric()`, and applies delta parameterization ($\operatorname{diag}(\Sigma^*) = \mathbf{1}_p$). |

### Details

- **Continuous `MLR`:** Computes Huber--White sandwich standard errors $\operatorname{ACov}(\hat{\theta}) = N^{-1} H^{-1} J H^{-1}$ using the fourth-order asymptotic covariance matrix $\Gamma$, Satorra--Bentler / Yuan--Bentler scaled $\chi^2$, and Brosseau-Liard & Savalei (2014) robust CFI, TLI, and RMSEA (with 90% CI).
- **Ordinal `WLSMV`:** Minimizes diagonally weighted least squares $F_{\mathrm{DWLS}}(\theta) = \frac{1}{2}(s - \sigma(\theta))^\top W (s - \sigma(\theta))$ with $W = \operatorname{diag}(\Gamma)^{-1}$, computes Asparouhov & Muthén (2010) mean- and variance-adjusted (scaled-shifted) $\chi^2$, and Savalei (2021) categorical-ML (`catML`) robust CFI, TLI, and RMSEA.

### Value

Returns an [`AnvalaFit`](#anvalafit) object.

### Examples

```python
import anvala as av

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Continuous robust ML
fit = av.cfa(model, data=df, estimator="MLR")
print(fit.fit_measures().round(4))
print(fit.parameter_estimates())

# Ordinal Likert CFA (WLSMV)
fit_ord = av.cfa(model, data=df_likert, estimator="WLSMV", ordered=True)
print(fit_ord.polychoric().round(3))
```

---

## `AnvalaFit`

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `spec` | `ModelSpec` | Compiled model specification (`ov_names`, `lv_names`, `params`, `n_free`, matrix maps). |
| `estimator` | `str` | Uppercase name of the estimator used (`"ML"`, `"MLR"`, `"WLSMV"`, etc.). |
| `n_obs` | `int` | Effective sample size after listwise deletion. |
| `converged` | `bool` | `True` if optimization satisfied convergence criteria. |
| `theta_hat` | `np.ndarray` | Vector of estimated free parameters. |
| `se` | `np.ndarray` | Standard errors of free parameters. |
| `vcov` | `np.ndarray` | Asymptotic variance--covariance matrix of free parameters. |
| `lambda_hat` | `np.ndarray` | Estimated factor loading matrix $\hat{\Lambda}$ ($p \times k$). |
| `beta_hat` | `np.ndarray` | Estimated latent regression matrix $\hat{B}$ ($k \times k$). |
| `psi_hat` | `np.ndarray` | Estimated latent covariance matrix $\hat{\Psi}$ ($k \times k$). |
| `theta_res_hat` | `np.ndarray` | Estimated residual covariance matrix $\hat{\Theta}$ ($p \times p$). |
| `sigma_hat` | `np.ndarray` | Model-implied covariance/correlation matrix $\hat{\Sigma}$ ($p \times p$). |
| `sample_cov` | `np.ndarray` | Observed sample covariance (or polychoric correlation) matrix $S$. |
| `fit_table` | `pd.DataFrame` | Fit measures table (`Unscaled`, and `Scaled` / `Robust` for robust estimators). |
| `data_raw` | `pd.DataFrame \| None` | Cleaned observed dataset retained for downstream scoring and diagnostics. |
| `poly_result` | `PolychoricResult \| None` | Full polychoric estimation result if `ordered=True`, else `None`. |

### Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `fit_measures` | `fit.fit_measures() -> pd.DataFrame` | Returns the `Unscaled | Scaled | Robust` fit measures DataFrame (`chisq`, `df`, `pvalue`, `scaling_factor`, `shift_parameter`, `cfi`, `tli`, `rmsea`, `rmsea_ci_lower`, `rmsea_ci_upper`, `srmr`). |
| `parameter_estimates` | `fit.parameter_estimates() -> pd.DataFrame` | Returns a `lavaan`-style parameter table (`lhs`, `op`, `rhs`, `label`, `est`, `se`, `z`, `pvalue`, `std_all`), including item thresholds (`op="|"`) for ordinal models. |
| `inspect_std` | `fit.inspect_std() -> dict[str, pd.DataFrame]` | Returns standardized matrices `{"lambda", "psi", "theta"}` (equivalent to `lavInspect(fit, "std")`). |
| `inspect_sampstat` | `fit.inspect_sampstat() -> dict[str, pd.DataFrame]` | Returns `{"cov": DataFrame}` for continuous models or `{"cor": DataFrame, "thresholds": DataFrame}` for ordinal models. |
| `polychoric` | `fit.polychoric() -> pd.DataFrame` | Returns the sample polychoric correlation matrix (raises `ValueError` for continuous models). |
| `residuals` | `fit.residuals(type: str = "raw") -> pd.DataFrame` | Computes local residuals (`"raw"`, `"cor"`, `"cor.bollen"`, `"standardized"`). See [`residuals()`](diagnostics.md#residuals). |
| `modindices` | `fit.modindices(sort: bool = True, minimum_value: float = 0.0, op: str | None = None) -> pd.DataFrame` | Computes Lagrange multiplier modification indices and EPCs. See [`modindices()`](diagnostics.md#modindices). |
| `bifactor_indices` | `fit.bifactor_indices(compute_arpb: bool = True) -> dict` | Computes Dueber (2021) bifactor indices directly from the fitted orthogonal bifactor model. When `compute_arpb=True`, automatically fits a unidimensional comparison model to compute RPB/ARPB. |
| `predict` | `fit.predict(newdata=None, method=None, se=False)` | Computes latent factor scores and optional SEs. See [`predict()`](predict.md#predict). |
| `bootstrap_ci` | `fit.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca", seed=42, device="auto")` | Runs GPU-accelerated bootstrap CIs. See [`bootstrap_ci()`](bootstrap.md#bootstrap_ci). |

### References

- Asparouhov, T., & Muthén, B. (2010). *Simple second order chi-square correction* (Mplus Technical Appendix).
- Brosseau-Liard, P. E., & Savalei, V. (2014). Adjusting incremental fit indices for nonnormality. *Multivariate Behavioral Research*, 49(5), 460--470.
- Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. *Journal of Statistical Software*, 48(2), 1--36.
- Savalei, V. (2021). Improving fit indices in structural equation modeling with categorical data. *Multivariate Behavioral Research*, 56(3), 390--407.
