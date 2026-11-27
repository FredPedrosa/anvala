# Reliability, Discriminant Validity & Bifactor Indices

*Modules:* `anvala.reliability`, `anvala.bifactor` · *R equivalents:* `semTools::compRelSEM()`, `semTools::AVE()`, `semTools::htmt()`, `BifactorIndicesCalculator::bifactorIndices()`

---

## `reliability`

### Description

Computes model-based scale reliability and convergent/discriminant validity indices directly from a fitted CFA (`AnvalaFit`) or EFA (`EFAFit`) object: **McDonald's $\omega$** (`omega_total`, `omega_implied`, `omega_std`), **Composite Reliability (`cr`)**, **Average Variance Extracted (`ave`, `ave_implied`)**, **Green & Yang (2009) categorical $\omega$** (`omega_categorical`, computed automatically for ordinal `WLSMV` CFA models), and **HTMT / HTMT2** discriminant validity matrices (when `data` is supplied).

In alignment with modern psychometric guidelines, Cronbach's $\alpha$ is **not** computed by default; set `include_alpha=True` or call [`cronbach_alpha()`](#cronbach_alpha) explicitly if needed.

### Usage

```python
anvala.reliability(
    fit: AnvalaFit | EFAFit,
    data: pd.DataFrame | None = None,
    loading_cutoff: float = 0.30,
    include_alpha: bool = False,
) -> ReliabilityResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fit` | `AnvalaFit \| EFAFit` | *required* | Fitted CFA or EFA object. |
| `data` | `pd.DataFrame \| None` | `None` | Raw dataset used to compute observed-denominator $\omega_{\mathrm{total}}$, observed-denominator AVE, HTMT/HTMT2 matrices, and optional Cronbach's $\alpha$. |
| `loading_cutoff` | `float` | `0.30` | Salient loading cutoff ($|\lambda| \ge \text{cutoff}$) used to assign items to factors when `fit` is an `EFAFit`. |
| `include_alpha` | `bool` | `False` | If `True` (and `data` is provided), appends `alpha_raw` (and `alpha_std` for CFA) to `constructs`. |

### Value (`ReliabilityResult`)

- `constructs` (`pd.DataFrame`): Per-construct table (`construct`, `n_items`, `omega_total`, `omega_implied`, `omega_std`, `cr`, `ave`, `ave_implied`, plus `omega_categorical` for ordinal CFA, `omega_scale_total` for EFA, and optional `alpha_raw`/`alpha_std`).
- `htmt_matrix` (`pd.DataFrame | None`): HTMT2 (geometric mean) matrix when `data` is provided and there are $\ge 2$ constructs.
- `htmt2_matrix` (`pd.DataFrame | None`): Complementary HTMT matrix (`geometric=False` for CFA).
- `estimator` (`str`), `ordered` (`bool`).

---

## `htmt`

### Description

Computes the **Heterotrait--Monotrait ratio of correlations**—either **HTMT2** (Roemer et al., 2021; `geometric=True`, default) or classic **HTMT** (Henseler et al., 2015; `geometric=False`)—in parity with `semTools::htmt()`.

### Usage

```python
anvala.htmt(
    data: pd.DataFrame,
    factors: dict[str, list[str]],
    corr: str = "auto",
    geometric: bool = True,
    absolute: bool = True,
) -> pd.DataFrame
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Raw item DataFrame. |
| `factors` | `dict[str, list[str]]` | *required* | Mapping from construct name to list of indicator column names (e.g., `{"F1": ["x1", "x2", "x3"], "F2": ["x4", "x5", "x6"]}`). |
| `corr` | `str` | `"auto"` | `"auto"` (polychoric for integer items with $\le 7$ unique values, Pearson otherwise), `"polychoric"`, or `"pearson"`. |
| `geometric` | `bool` | `True` | `True` computes HTMT2 (geometric mean); `False` computes classic HTMT (arithmetic mean). |
| `absolute` | `bool` | `True` | If `True`, takes absolute correlations before aggregation. |

### Value

Symmetric $m \times m$ `pd.DataFrame` of HTMT ratios (values $< 0.85$ support discriminant validity).

---

## `cronbach_alpha`

### Description

Explicitly computes raw and standardized Cronbach's $\alpha$ on a DataFrame of scale items.

### Usage

```python
anvala.cronbach_alpha(data: pd.DataFrame, corr: str = "pearson") -> pd.DataFrame
```

### Arguments

- `data` (`pd.DataFrame`): Scale items (listwise deletion applied).
- `corr` (`str`, default `"pearson"`): `"pearson"` or `"polychoric"` (ordinal alpha).

### Value

Single-row `pd.DataFrame` with columns `n_items`, `alpha_raw`, `alpha_std`.

---

## `bifactor_indices`

### Description

Computes the complete suite of psychometric bifactor indices (Rodriguez et al., 2016; Dueber, 2021) with exact numerical parity to R's `BifactorIndicesCalculator`. Can be called standalone on a loading matrix or directly via `fit.bifactor_indices(compute_arpb=True)` on a fitted orthogonal bifactor `AnvalaFit`.

### Usage

```python
anvala.bifactor_indices(
    loadings: np.ndarray | pd.DataFrame,
    unidimensional_loadings: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    theta: np.ndarray | None = None,
) -> dict[str, pd.DataFrame | pd.Series]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `loadings` | `np.ndarray \| pd.DataFrame` | *required* | Standardized $p \times k$ bifactor loading matrix. **Column 0 must be the General factor (`G`)**, and columns $1, \dots, k-1$ are Specific factors (`S1, S2, ...`). |
| `unidimensional_loadings` | `np.ndarray \| None` | `None` | Optional length-$p$ vector of standardized loadings from a unidimensional CFA used to compute item-level Relative Parameter Bias (`RPB`) and model-level Average RPB (`ARPB`). |
| `phi` | `np.ndarray \| None` | `None` | Optional $k \times k$ factor correlation matrix (defaults to $I_k$). |
| `theta` | `np.ndarray \| None` | `None` | Optional length-$p$ residual variance vector (defaults to $1 - h^2$). |

### Value

Dictionary with three entries:

- `"model"` (`pd.Series`): `ECV` (Explained Common Variance of General factor), `PUC` (Percentage of Uncontaminated Correlations), `Omega` ($\omega_{\mathrm{total}}$), `OmegaH` ($\omega_H$), `ARPB`.
- `"factors"` (`pd.DataFrame`, indexed by factor): `ECV_SS`, `ECV_SG`, `ECV_GS`, `Omega`, `OmegaH`, `Relative_Omega`, `H` (construct replicability), `FD` (Factor Determinacy).
- `"items"` (`pd.DataFrame`, indexed by item): `IECV` (Item Explained Common Variance) and, when `unidimensional_loadings` is supplied, `RPB`.

### Examples

```python
import anvala as av

# Reliability & HTMT from CFA
fit = av.cfa(model, data=df, estimator="MLR")
rel = av.reliability(fit, data=df)
print(rel.constructs)
print(rel.htmt_matrix)

# Bifactor indices directly from an orthogonal bifactor CFA
fit_bf = av.cfa(bf_model, data=df, estimator="MLR", std_lv=True, orthogonal=True)
bf = fit_bf.bifactor_indices(compute_arpb=True)
print(bf["model"])
print(bf["factors"])
```

### References

- Dueber, D. M. (2021). *BifactorIndicesCalculator: Bifactor indices calculator* [R package].
- Green, S. B., & Yang, Y. (2009). Reliability of summed item scores using structural equation modeling. *Psychometrika*, 74(1), 155--167.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods*, 21(2), 137--150.
- Roemer, E., Schuberth, F., & Henseler, J. (2021). HTMT2--an improved criterion for assessing discriminant validity in structural equation modeling. *Industrial Management & Data Systems*, 121(12), 2637--2650.
