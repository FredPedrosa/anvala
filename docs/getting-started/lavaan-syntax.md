# `lavaan` Model Syntax in `anvala`

`anvala` adopts standard [`lavaan`](https://lavaan.ugent.be/tutorial/syntax1.html) model syntax so researchers migrating from R can reuse existing specifications without translation.

## Supported Operators

| Operator | Meaning | Example | Supported In |
| :---: | :--- | :--- | :--- |
| `=~` | Latent factor definition ("is measured by") | `visual =~ x1 + x2 + x3` | `cfa`, `sem`, `cfa_multigroup`, `measurement_invariance` |
| `~` | Structural regression ("is regressed on") | `Y ~ b*M + c*X` | `sem`, `cfa` (latent-on-latent) |
| `~~` | Variance or (residual) covariance | `x1 ~~ x2` | `cfa`, `sem`, `cfa_multigroup`, `measurement_invariance` |
| `:=` | Nonlinear defined parameter (Delta-method & Bootstrap SEs) | `indirect := a*b` | `sem` (and `bootstrap_ci`) |
| `#` | Inline or full-line comment | `# Measurement block` | All parsers |
| `;` | Statement separator (single-line specs) | `F1 =~ x1 + x2 + x3; F2 =~ x4 + x5 + x6` | All parsers |

## Modifiers: Labels, Fixed Values, Equality Constraints & Freeing Markers

Any right-hand-side term can be prefixed with `<modifier>*<variable>`:

| Syntax | Effect |
| :--- | :--- |
| `a*X` | Assigns the label `a` to the parameter (used in `:=` formulas or equality constraints). |
| `a*x2` and `a*x3` | **Equality constraint:** reusing the same label `a` constrains both parameters to a single shared free parameter. |
| `0.5*x2` or `0*x2` | Fixes the parameter to a constant numeric value (`se = 0`). |
| `NA*x1` | Frees a parameter that would otherwise be fixed by default (e.g., frees the first-indicator marker loading when `std_lv=False`). |

### Example: Full Latent Mediation with Constraints

```python
model = """
    # Measurement model (marker indicator fixed to 1.0 by default)
    FX =~ x1 + x2 + x3
    FM =~ m1 + m2 + m3
    FY =~ y1 + y2 + y3

    # Residual covariance between two similarly worded items
    x1 ~~ x2

    # Structural paths with parameter labels
    FM ~ a*FX
    FY ~ b*FM + c*FX

    # Defined parameters (exact autograd Delta-method SEs + BCa bootstrap)
    indirect := a*b
    total    := c + a*b
    prop_med := (a*b) / (c + a*b)
"""
```

## Automatic Defaults (Matching `lavaan`)

- **Latent scale identification:**
  - When `std_lv=False` (default), the first indicator of every latent variable has its loading fixed to `1.0`, and latent variances are freely estimated.
  - When `std_lv=True`, exogenous latent variances are fixed to `1.0` and all factor loadings are freely estimated.
- **Factor covariances:**
  - When `orthogonal=False` (default), all exogenous latent variables are freely allowed to covary.
  - When `orthogonal=True`, latent covariances are fixed to `0` (ideal for orthogonal bifactor models).
- **Observed exogenous variables in `sem()`:**
  - When `fixed_x=True` (default), variances and covariances among observed exogenous predictors are fixed to their sample moments (matching `lavaan::sem(..., fixed.x = TRUE)`).
- **Ordinal delta parameterization (`ordered=True` or `estimator="WLSMV"`):**
  - Latent response total variances are fixed to `1.0`, so residual variances are implicitly $\operatorname{diag}(\Theta) = \mathbf{1}_p - \operatorname{diag}(\Lambda (I-B)^{-1}\Psi(I-B)^{-\top}\Lambda^\top)$.

## Multi-Group `group_equal` and `group_partial` Syntax

In [`cfa_multigroup()`](../reference/multigroup.md#cfa_multigroup) and [`measurement_invariance()`](../reference/multigroup.md#measurement_invariance):

- **`group_equal` keywords** (aliases accepted):
  - `"loadings"` (alias `"lambda"`)
  - `"intercepts"` (alias `"nu"`)
  - `"thresholds"` (alias `"tau"`)
  - `"residuals"` (alias `"theta"`)
  - `"lv.variances"`, `"lv.covariances"`
  - `"means"` (alias `"alpha"`)
- **`group_partial` parameter strings** (free specific parameters across groups):
  - Factor loading: `"visual =~ x2"`
  - Item intercept: `"x5 ~ 1"`
  - Item threshold (ordinal): `"x5 | t2"`
  - Residual variance or covariance: `"x3 ~~ x3"`, `"x1 ~~ x2"`
