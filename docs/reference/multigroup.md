# Multi-Group CFA & Measurement Invariance

*Module:* `anvala.multigroup` · *R equivalents:* `lavaan::cfa(group=...)`, `lavaan::lavTestLRT()`, `lavaan::lavTestScore()`

---

## `cfa_multigroup`

### Description

Fits a Multi-Group Confirmatory Factor Analysis (MG-CFA) with mean/threshold structures, cross-group equality constraints (`group_equal`), partial invariance exemptions (`group_partial`), and exact Lagrange multiplier score test context (`score_test()`).

### Usage

```python
anvala.cfa_multigroup(
    model: str,
    data: pd.DataFrame,
    group: str,
    estimator: str = "ML",
    ordered: bool = False,
    group_equal: list[str] | tuple[str, ...] | str | None = None,
    group_partial: list[str] | tuple[str, ...] | str | None = None,
    std_lv: bool = False,
) -> MultiGroupFit
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | *required* | CFA syntax specifying factor loadings (`=~`) and optional residual covariances (`~~`). |
| `data` | `pd.DataFrame` | *required* | Dataset containing all observed indicators and the grouping column. |
| `group` | `str` | *required* | Column name in `data` defining at least 2 distinct groups. |
| `estimator` | `str` | `"ML"` | `"ML"`, `"MLR"` (continuous with mean structures), or `"WLSMV"` (ordinal delta parameterization; automatically sets `ordered=True`). |
| `ordered` | `bool` | `False` | If `True`, fits multi-group ordinal CFA using group-specific polychoric correlations and joint threshold constraints. |
| `group_equal` | `list[str] \| str \| None` | `None` | Parameter classes constrained equal across all groups. Accepted keys (and aliases): `"loadings"` (`"lambda"`), `"intercepts"` (`"nu"`), `"thresholds"` (`"tau"`), `"residuals"` (`"theta"`), `"lv.variances"`, `"lv.covariances"`, `"means"` (`"alpha"`). |
| `group_partial` | `list[str] \| str \| None` | `None` | Specific parameters freed from `group_equal` constraints across groups, e.g. `["visual =~ x2", "x5 ~ 1", "x3 ~~ x3", "x1 | t1"]`. |
| `std_lv` | `bool` | `False` | If `True`, fixes latent variances to `1.0` in the reference group (and frees them in subsequent groups when `"loadings"` are constrained equal). |

### Value

Returns a [`MultiGroupFit`](#multigroupfit) object.

---

## `measurement_invariance`

### Description

Automates hierarchical multi-group measurement invariance testing across `"configural"`, `"metric"`, `"scalar"`, and `"strict"` levels and compares consecutive models via `compare_models()`.

### Usage

```python
anvala.measurement_invariance(
    model: str,
    data: pd.DataFrame,
    group: str,
    estimator: str = "ML",
    ordered: bool = False,
    levels: tuple[str, ...] = ("configural", "metric", "scalar", "strict"),
    group_partial: list[str] | tuple[str, ...] | str | None = None,
    std_lv: bool = False,
) -> InvarianceResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model`, `data`, `group`, `estimator`, `ordered`, `group_partial`, `std_lv` | *same as `cfa_multigroup`* | — | Passed to each level's `cfa_multigroup()` fit. |
| `levels` | `tuple[str, ...]` | `("configural", "metric", "scalar", "strict")` | Ordered subset of invariance levels to fit. For continuous models: `configural` (`[]`), `metric` (`["loadings"]`), `scalar` (`["loadings", "intercepts"]`), `strict` (`["loadings", "intercepts", "residuals"]`). For ordinal (`WLSMV`) models, `"thresholds"` replaces `"intercepts"`. |

### Value

Returns an [`InvarianceResult`](#invarianceresult) object containing `.table` (comparison DataFrame) and `.fits` (dictionary mapping level name to `MultiGroupFit`).

---

## `compare_models`

### Description

Compares two or more nested `AnvalaFit`, `SEMFit`, or `MultiGroupFit` models sorted from least restricted (smallest `df`) to most restricted (largest `df`), using standard LRT for `ML` and the Satorra--Bentler (2001) scaled chi-square difference test for `MLR` and `WLSMV`, alongside $\Delta$CFI, $\Delta$TLI, $\Delta$RMSEA, and $\Delta$SRMR.

### Usage

```python
anvala.compare_models(*fits, names: list[str] | None = None) -> pd.DataFrame
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `*fits` | `AnvalaFit \| SEMFit \| MultiGroupFit` | *required* | Two or more fitted model objects. |
| `names` | `list[str] \| None` | `None` | Optional labels for each model (defaults to `["Model_1", "Model_2", ...]`). |

### Value

A `pd.DataFrame` with columns: `model`, `df`, `aic`, `bic`, `chisq`, `chisq_scaled`, `chisq_diff`, `df_diff`, `pvalue`, `cfi`, `delta_cfi`, `tli`, `delta_tli`, `rmsea`, `delta_rmsea`, `srmr`, `delta_srmr`.

---

## `MultiGroupFit`

### Key Attributes

- `group_col` (`str`), `group_labels` (`list[str]`), `n_obs_per_group` (`list[int]`), `n_obs` (`int`)
- `lv_names`, `ov_names`, `estimator`, `ordered`, `group_equal`, `group_partial`, `converged`
- `pe_df` (`pd.DataFrame` with `group` column), `fit_table` (`pd.DataFrame`), `group_chisq` (`dict[str, float]`), `sigma_per_group`, `s_per_group`, `vcov`

### Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `parameter_estimates` | `fit.parameter_estimates() -> pd.DataFrame` | Per-group parameter table (`group`, `lhs`, `op`, `rhs`, `est`, `se`, `z`, `pvalue`, `std_lv`, `std_all`). |
| `fit_measures` | `fit.fit_measures() -> pd.DataFrame` | Global multi-group fit table (`Unscaled | Scaled | Robust`), with lavaan's $\sqrt{G}$ multi-group RMSEA scaling and $N_g$-weighted SRMR. |
| `score_test` | `fit.score_test(sort: bool = True, return_total: bool = False)` | Lagrange multiplier tests (`lavTestScore`) for all cross-group equality constraints. Returns univariate `DataFrame` (`lhs`, `op`, `rhs`, `x2`, `df`, `pvalue`, `x2_scaled`, `pvalue_scaled`) when `return_total=False`, or `(total_df, uni_df)` when `return_total=True` (`total_df` includes `score`, `score.scaled`, `score.adjusted`, `score.robust`). |
| `summary` | `fit.summary() -> str` | Formatted summary string. |
| `predict` | `fit.predict(newdata, method=None, se=False)` | Computes group-specific factor scores (`newdata` containing indicators and `group_col` is required). |

---

## `InvarianceResult`

- **Attributes:** `table` (`pd.DataFrame`), `fits` (`dict[str, MultiGroupFit]`), `group_col`, `group_labels`, `estimator`, `ordered`.
- **Method:** `summary() -> str`.

### Examples

```python
import anvala as av

inv = av.measurement_invariance(model, data=df, group="school", estimator="MLR")
print(inv.summary())

# Diagnose scalar non-invariance
tot, uni = inv.fits["scalar"].score_test(return_total=True)
print(uni.head())

# Free the violating intercept
fit_partial = av.cfa_multigroup(
    model, data=df, group="school", estimator="MLR",
    group_equal=["loadings", "intercepts"],
    group_partial=["x5 ~ 1"],
)
print(av.compare_models(inv.fits["metric"], fit_partial, inv.fits["scalar"],
                        names=["metric", "partial_scalar", "full_scalar"]))
```
