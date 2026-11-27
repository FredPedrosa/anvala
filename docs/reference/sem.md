# Structural Equation Modeling (`sem`, `SEMFit`)

*Module:* `anvala.sem` · *R equivalent:* `lavaan::sem()`

---

## `sem`

### Description

Estimates general Structural Equation Models (SEM), path analyses, mediation models, and CFAs using RAM matrix parameterization ($B, \Psi$), BFGS + damped Newton--Raphson polishing with exact `float64` PyTorch Hessians, parameter equality constraints, and nonlinear defined parameters (`:=`) with exact autograd Delta-method standard errors.

### Usage

```python
anvala.sem(
    model: str,
    data: pd.DataFrame,
    estimator: str = "ML",
    ordered: bool = False,
    std_lv: bool = False,
    orthogonal: bool = False,
    fixed_x: bool = True,
) -> SEMFit
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | *required* | Model specification in `lavaan` syntax supporting `=~`, `~`, `~~`, labels (`a*X`), equality constraints (reused labels), fixed values (`0.5*X`), marker release (`NA*x1`), and defined parameters (`:=`). |
| `data` | `pd.DataFrame` | *required* | Input dataset. Rows with missing values are listwise-deleted. |
| `estimator` | `str` | `"ML"` | Estimation method: `"ML"`, `"MLR"`, or `"WLSMV"` (automatically sets `ordered=True`). |
| `ordered` | `bool` | `False` | If `True`, uses two-step polychoric correlations and WLSMV delta parameterization. |
| `std_lv` | `bool` | `False` | If `True`, fixes exogenous latent variances to `1.0` instead of fixing first-indicator loadings. |
| `orthogonal` | `bool` | `False` | If `True`, constrains covariances among exogenous latent variables to `0.0`. |
| `fixed_x` | `bool` | `True` | If `True` (matching `lavaan::sem`), fixes variances and covariances among observed exogenous predictors to their sample values and adjusts model degrees of freedom accordingly. |

### Details

- **Defined parameters (`:=`):** Arbitrary algebraic expressions (`+`, `-`, `*`, `/`, `**`, unary `+`/`-`) combining labeled parameters are parsed safely via Python AST and differentiated with `torch.autograd.grad` to compute Delta-method standard errors $\mathrm{SE}(g(\hat{\theta})) = \sqrt{\nabla g(\hat{\theta})^\top \widehat{\operatorname{ACov}}(\hat{\theta}) \nabla g(\hat{\theta})}$.
- **Log-variance parameterization:** Free diagonal variance parameters are optimized on the log scale and transformed back via the chain rule for natural-scale `vcov` and `se`, preventing Heywood cases during optimization.

### Value

Returns an [`SEMFit`](#semfit) object.

### Examples

```python
import anvala as av

med_model = """
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total    := c + a*b
"""

fit = av.sem(med_model, data=df, estimator="ML")
print(fit.summary())

# GPU-accelerated BCa bootstrap confidence intervals for mediation
boot = fit.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca")
print(boot.table)
```

---

## `SEMFit`

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `lv_names` | `list[str]` | Latent variable names in specification order. |
| `ov_names` | `list[str]` | Observed indicator names used in the model. |
| `estimator` | `str` | Estimator name (`"ML"`, `"MLR"`, `"WLSMV"`). |
| `ordered` | `bool` | Whether ordinal estimation was used. |
| `n_obs` | `int` | Effective sample size. |
| `s_mat` | `np.ndarray` | Observed sample covariance (or polychoric correlation) matrix ($p \times p$). |
| `sigma_hat` | `np.ndarray` | Model-implied covariance matrix of observed variables ($p \times p$). |
| `sigma_all` | `np.ndarray` | Joint model-implied covariance matrix of all `[lv_names + ov_names]` variables. |
| `vcov` | `np.ndarray` | Asymptotic variance--covariance matrix of natural-scale free parameters. |
| `pe_df` | `pd.DataFrame` | Full parameter estimates table (including `:=` defined parameters, `std_lv`, and `std_all`). |
| `fit_table` | `pd.DataFrame` | `Unscaled | Scaled | Robust` fit measures table. |
| `data` | `pd.DataFrame \| None` | Cleaned input dataset. |
| `poly_result` | `PolychoricResult \| None` | Polychoric estimation result if `ordered=True`. |
| `converged` | `bool` | Convergence indicator. |

### Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `parameter_estimates` | `fit.parameter_estimates() -> pd.DataFrame` | Returns a copy of `pe_df` (`lhs`, `op`, `rhs`, `label`, `est`, `se`, `z`, `pvalue`, `std_lv`, `std_all`). |
| `fit_measures` | `fit.fit_measures() -> pd.DataFrame` | Returns a copy of `fit_table` (`chisq`, `df`, `pvalue`, `scaling_factor`, `cfi`, `tli`, `rmsea`, `srmr`). |
| `implied_cov` | `fit.implied_cov() -> pd.DataFrame` | Returns the labeled model-implied covariance matrix $\hat{\Sigma}$. |
| `residuals` | `fit.residuals(type: str = "raw") -> pd.DataFrame` | Computes local residuals (`"raw"`, `"cor"`, `"cor.bollen"`, `"standardized"`). |
| `modindices` | `fit.modindices(sort: bool = True, minimum_value: float = 0.0, op: str | None = None) -> pd.DataFrame` | Computes modification indices and EPCs. |
| `summary` | `fit.summary() -> str` | Formats a readable text summary of fit measures and parameter estimates. |
| `predict` | `fit.predict(newdata=None, method=None, se=False)` | Computes latent factor scores. See [`predict()`](predict.md#predict). |
| `bootstrap_ci` | `fit.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca", seed=42, device="auto")` | Computes GPU-accelerated bootstrap CIs. See [`bootstrap_ci()`](bootstrap.md#bootstrap_ci). |
