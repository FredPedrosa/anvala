# Factor Scores (`predict`)

*Module:* `anvala.predict` · *R equivalent:* `lavaan::lavPredict()`

---

## `predict`

### Description

Computes latent factor scores (and optional individual standard errors) from a fitted `AnvalaFit`, `SEMFit`, or `MultiGroupFit` object, either on the training sample or on new data (`newdata`). Also available directly as a method on fitted objects (`fit.predict(...)`).

### Usage

```python
anvala.predict(
    fit,
    newdata: pd.DataFrame | None = None,
    method: str | None = None,
    se: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fit` | `AnvalaFit \| SEMFit \| MultiGroupFit` | *required* | Fitted model object. |
| `newdata` | `pd.DataFrame \| None` | `None` | Optional DataFrame for out-of-sample scoring. If `None`, uses the original estimation dataset stored in `AnvalaFit` / `SEMFit`. **Required** for `MultiGroupFit` (must include the grouping column). |
| `method` | `str \| None` | `None` | Scoring algorithm. If `None`, defaults to `"regression"` for continuous models and `"ebm"` for ordinal models. See table below. |
| `se` | `bool` | `False` | If `True`, returns a tuple `(scores_df, se_df)` containing individual standard errors for every latent factor. |

### Supported Methods

#### Continuous Models

| Method | Aliases | Description | SE Available? |
| :--- | :--- | :--- | :---: |
| `"regression"` *(default)* | `"thurstone"` | Regression (Thurstone) factor scores: $\hat{\eta} = \Psi \Lambda^\top \Sigma^{-1}(y - \bar{y}) + \alpha$. | Yes |
| `"bartlett"` | `"bartlet"`, `"ml"` | Bartlett (Maximum Likelihood) unbiased factor scores: $(\Lambda^\top \Sigma^{-1}\Lambda)^{-1}\Lambda^\top \Sigma^{-1}(y - \bar{y}) + \alpha$. | Yes |
| `"anderson-rubin"` | `"anderson_rubin"`, `"ar"` | Orthogonalized Anderson--Rubin (1956) scores with $\operatorname{Cov}(\hat{\eta}) = I_m$. | `NaN` |
| `"green"` | — | Covariance-preserving Green (1976) transformation ensuring $\operatorname{Cov}(\hat{\eta}) = \Psi$. | `NaN` |

#### Ordinal Models (`WLSMV` / `ordered=True`)

| Method | Description | SE Available? |
| :--- | :--- | :---: |
| `"ebm"` *(default)* | **Empirical Bayes Modal (MAP):** Maximizes the log-posterior combining the multivariate normal prior $\mathcal{N}(\alpha, \Psi)$ with the graded-response likelihood via BFGS + Newton--Raphson polishing (deduplicating identical response patterns). | Yes (inverse posterior Hessian) |
| `"ml"` | **Maximum Likelihood:** Maximizes the graded-response likelihood with negligible prior precision. | Yes |
| `"eap"` | **Expected A Posteriori:** Evaluates posterior mean and SD via multivariate Gauss--Hermite quadrature ($Q=15$ nodes per dimension for $k \le 2$, $Q=9$ for $k=3$, $Q=7$ for $k \ge 4$). | Yes (posterior SD) |

### Value

- If `se=False`: `pd.DataFrame` indexed like the scored data, with columns equal to `lv_names`.
- If `se=True`: tuple `(scores_df, se_df)` of two `pd.DataFrame` objects with identical shape and labels.

### Examples

```python
import anvala as av

# Continuous CFA -> Regression & Bartlett scores
fit = av.cfa(model, data=df, estimator="MLR")
scores_reg, se_reg = fit.predict(method="regression", se=True)
scores_bart = av.predict(fit, method="bartlett")

# Ordinal CFA -> EBM & EAP scores
fit_ord = av.cfa(model, data=df_likert, estimator="WLSMV", ordered=True)
scores_ebm, se_ebm = fit_ord.predict(method="ebm", se=True)
scores_eap, se_eap = fit_ord.predict(method="eap", se=True)

# Multi-group scoring (requires newdata with group column)
mg_scores = fit_mg.predict(newdata=df_mg, method="regression")
```
