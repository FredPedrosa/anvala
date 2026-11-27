# GPU-Accelerated Bootstrap (`bootstrap_ci`, `BootstrapResult`)

*Module:* `anvala.bootstrap` · *R equivalents:* `lavaan::sem(..., se="bootstrap")`, `boot::boot.ci(type="bca")`

---

## `bootstrap_ci`

### Description

Computes nonparametric **bias-corrected and accelerated (`BCa`)** or **percentile** bootstrap confidence intervals for continuous SEM and CFA models—including nonlinear defined parameters (`:=`) such as indirect and total mediation effects. Resampling, covariance construction, batched Fisher scoring, and leave-one-out jackknife estimation (for the BCa acceleration constant) are vectorized over 3D tensors on CUDA GPUs with automatic CPU fallback.

### Usage

```python
anvala.bootstrap_ci(
    fit,
    n_boot: int = 1000,
    conf_level: float = 0.95,
    method: str = "bca",
    seed: int = 42,
    device: str = "auto",
) -> BootstrapResult
```

Also callable directly as `fit.bootstrap_ci(...)` on `SEMFit` and `AnvalaFit` objects.

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fit` | `SEMFit \| AnvalaFit` | *required* | Fitted continuous model object (recommended: fit with `av.sem()` when using `:=` defined parameters or equality labels). |
| `n_boot` | `int` | `1000` | Number of nonparametric bootstrap resamples $B$ (processed in chunks of up to 512 replicates to bound VRAM usage). |
| `conf_level` | `float` | `0.95` | Nominal confidence level (e.g., `0.95` for 95% CIs). |
| `method` | `str` | `"bca"` | Confidence interval method: `"bca"` (bias-corrected and accelerated; Efron, 1987) or `"percentile"`. |
| `seed` | `int` | `42` | Random seed for reproducible resampling via `torch.Generator`. |
| `device` | `str` | `"auto"` | Execution device: `"auto"` (uses CUDA GPU when `torch.cuda.is_available()`, else CPU), `"cuda"`, or `"cpu"`. |

### Details

For `method="bca"`, `bootstrap_ci()` computes:

1. Bias-correction factor $z_0 = \Phi^{-1}\!\left(\frac{1}{B}\sum_{b=1}^{B} \mathbb{I}(\hat{\theta}^*_b < \hat{\theta})\right)$ (clipped away from 0 and 1).
2. Acceleration factor $\hat{a} = \frac{\sum_{i=1}^{N} (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(-i)})^3}{6 \left[\sum_{i=1}^{N} (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(-i)})^2\right]^{3/2}}$ via batched leave-one-out jackknife estimation.
3. Adjusted quantile levels $\alpha_1, \alpha_2 = \Phi\!\left(z_0 + \frac{z_0 + z_{\alpha/2}}{1 - \hat{a}(z_0 + z_{\alpha/2})}\right)$ (and analogously for $z_{1-\alpha/2}$).

### Value

Returns a [`BootstrapResult`](#bootstrapresult) object.

---

## `BootstrapResult`

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `table` | `pd.DataFrame` | Summary table with columns: `lhs`, `op`, `rhs`, `label`, `est`, `boot_se`, `bias`, `ci_lower`, `ci_upper`, `pvalue`. |
| `boot_samples` | `pd.DataFrame` | Raw $B \times q$ matrix of bootstrap parameter draws (each column named by its label or `lhs+op+rhs`). |
| `n_boot` | `int` | Number of bootstrap replicates executed. |
| `conf_level` | `float` | Confidence level used. |
| `method` | `str` | `"bca"` or `"percentile"`. |
| `device` | `str` | PyTorch device string used (`"cuda"` or `"cpu"`). |

### Methods

- `summary() -> str`: Prints a concise formatted header and rounded parameter table.

### Examples

```python
import anvala as av

med_model = """
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total    := c + a*b
"""

fit_sem = av.sem(med_model, data=df, estimator="ML")
boot = fit_sem.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca", seed=42)
print(boot.summary())

# Inspect empirical distribution of the indirect effect
indirect_draws = boot.boot_samples["indirect"]
```

### References

- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical Association*, 82(397), 171--185.
