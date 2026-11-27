# Exploratory Factor Analysis & Dimensionality Tests

*Module:* `anvala.efa` · *R equivalents:* `lavaan::efa()`, `EFA.dimensions::DIMTESTS()`, `psych::KMO()`

---

## `kmo_bartlett`

### Description

Computes the overall **Kaiser--Meyer--Olkin (KMO)** measure of sampling adequacy, item-level **Measures of Sampling Adequacy (MSA)**, and **Bartlett's test of sphericity**.

### Usage

```python
anvala.kmo_bartlett(data: pd.DataFrame, corkind: str = "auto") -> dict
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Item-level dataset (rows with missing values are dropped). |
| `corkind` | `str` | `"auto"` | Correlation type: `"auto"` (polychoric for integer items with $\le 7$ unique categories, Pearson otherwise), `"pearson"`, `"spearman"`, or `"polychoric"`. |

### Value

Dictionary containing:
- `"kmo"` (`float`): Overall KMO statistic ($0$ to $1$; values $\ge 0.80$ are considered meritorious).
- `"msa"` (`pd.Series`): Per-item MSA indexed by variable name.
- `"bartlett_chisq"` (`float`), `"bartlett_df"` (`int`), `"bartlett_pvalue"` (`float`): Bartlett's test of sphericity.

---

## `dimtests`

### Description

Runs a customizable battery of **dimensionality retention criteria** inspired by `EFA.dimensions::DIMTESTS` and Auerswald & Moshagen (2019), using Horn's Parallel Analysis (`RAWPAR`) as the default primary gold-standard criterion. Accepts either raw item data or a precomputed correlation matrix (with `n_cases`).

### Usage

```python
anvala.dimtests(
    data: pd.DataFrame,
    tests: list[str] | None = None,
    corkind: str = "auto",
    n_cases: int | None = None,
    hull_method: str = "PAF",
    hull_gof: str = "CAF",
    n_sim: int = 100,
    percentile: float = 95.0,
    seed: int = 42,
) -> DimTestsResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Raw item DataFrame or symmetric correlation DataFrame (if correlation matrix, `n_cases` is required). |
| `tests` | `list[str] \| None` | `None` | List of tests to execute (defaults to `["RAWPAR", "EMPKC", "HULL", "SMT"]`). Supported codes:<br>- `"RAWPAR"`: Horn's Parallel Analysis (common-factor SMC & PCA eigenvalues vs. simulated percentile)<br>- `"EMPKC"`: Empirical Kaiser Criterion (Braeken & van Assen, 2017)<br>- `"HULL"`: Hull Method (Lorenzo-Seva et al., 2011)<br>- `"SMT"`: Sequential Model Testing (sequential ML $\chi^2$ until $p > 0.05$)<br>- `"MAP"`: Velicer's Minimum Average Partial (1976)<br>- `"CD"`: Comparison Data (Ruscio & Roche, 2012; requires raw data)<br>- `"NEVALSGT1"`: Classic Kaiser eigenvalue $> 1$ rule<br>- `"SESCREE"`: Standard Error Scree (Zoski & Jurs, 1996) |
| `corkind` | `str` | `"auto"` | Correlation type (`"auto"`, `"pearson"`, `"spearman"`, `"polychoric"`). |
| `n_cases` | `int \| None` | `None` | Sample size (required only when `data` is a correlation matrix). |
| `hull_method` | `str` | `"PAF"` | Extraction label for Hull method (Principal Axis Factoring). |
| `hull_gof` | `str` | `"CAF"` | Goodness-of-fit index for Hull method: `"CAF"` (Common part Accounted For, $1 - \mathrm{KMO}_{\mathrm{res}}$) or `"SRMR"` ($1 - \mathrm{SRMR}$). |
| `n_sim` | `int` | `100` | Number of random datasets generated for Parallel Analysis (`RAWPAR`). |
| `percentile` | `float` | `95.0` | Eigenvalue reference percentile for `RAWPAR` (e.g., `95.0`). |
| `seed` | `int` | `42` | Random seed for reproducible simulations. |

### Value (`DimTestsResult`)

- `results` (`pd.DataFrame`): Columns `method`, `n_factors`, `detail`.
- `n_factors` (`int`): Recommended dimensionality from the first method in `tests`.
- `primary_method` (`str`): Code of the primary method.
- `eigenvalues` (`pd.DataFrame`): Observed and simulated parallel-analysis eigenvalues (`eig_pca`, `pa_mean_pca`, `pa_p95_pca`, `eig_fa`, `pa_mean_fa`, `pa_p95_fa`).
- `adequacy` (`dict`): Output of `kmo_bartlett()` (for raw data).

---

## `efa`

### Description

Fits an Exploratory Factor Analysis model in numerical parity with `lavaan::efa()`, supporting continuous (`ML`, `MLR`, `ULS`, `DWLS`) and ordinal (`WLSMV`) estimators, echelon-form PyTorch optimization with BFGS + Newton--Raphson polishing, multi-start **Gradient Projection Algorithm (GPA)** rotations (Bernaards & Jennrich, 2005), and rotated standard errors for factor loadings and factor correlations.

### Usage

```python
anvala.efa(
    data: pd.DataFrame,
    n_factors: int = 1,
    estimator: str = "MLR",
    ordered: bool = False,
    rotation: str = "geomin",
    orthogonal: bool = False,
    geomin_epsilon: float = 0.001,
    oblimin_gamma: float = 0.0,
    promax_kappa: float = 4.0,
    rstarts: int = 30,
    std_ov: bool = True,
    sample_cov: np.ndarray | None = None,
    sample_nobs: int | None = None,
    seed: int = 42,
) -> EFAFit
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Item-level DataFrame (or labeled template when `sample_cov` is provided). |
| `n_factors` | `int` | `1` | Number of latent factors $m$ to extract (must satisfy $(p-m)^2 - (p+m) \ge 0$). |
| `estimator` | `str` | `"MLR"` | `"ML"`, `"MLR"`, `"WLSMV"` (sets `ordered=True`), `"ULS"`, or `"DWLS"`. |
| `ordered` | `bool` | `False` | If `True`, estimates polychoric correlations and WLSMV delta-parameterized EFA. |
| `rotation` | `str` | `"geomin"` | Rotation criterion: `"geomin"`, `"oblimin"`, `"promax"`, `"varimax"`, `"quartimax"`, `"equamax"`, `"parsimax"`, or `"none"`. |
| `orthogonal` | `bool` | `False` | If `True`, constrains factors to be uncorrelated (automatically `True` for `varimax`/`quartimax`/`equamax`/`parsimax`). |
| `geomin_epsilon` | `float` | `0.001` | Small smoothing constant $\epsilon$ in the Geomin criterion. |
| `oblimin_gamma` | `float` | `0.0` | Complexity parameter $\gamma$ for direct Oblimin (`0.0` = quartimin). |
| `promax_kappa` | `float` | `4.0` | Power $\kappa$ for Promax target matrix. |
| `rstarts` | `int` | `30` | Number of random orthogonal starting matrices for multi-start GPA optimization. |
| `std_ov` | `bool` | `True` | Standardize observed indicators when reporting standardized loadings. |
| `sample_cov` | `np.ndarray \| None` | `None` | Optional precomputed covariance/correlation matrix (requires `sample_nobs`). |
| `sample_nobs` | `int \| None` | `None` | Sample size when passing `sample_cov`. |
| `seed` | `int` | `42` | Random seed for GPA random starts. |

---

## `EFAFit`

- **Attributes:** `n_factors`, `estimator`, `rotation`, `ordered`, `n_obs`, `var_names`, `factor_names`, `loadings_unstd`, `loadings_se`, `loadings_std`, `factor_corr`, `factor_corr_se`, `communalities`, `uniquenesses`, `variance_explained`, `eigenvalues`, `fit_table`, `thresholds`, `polychoric_corr`, `converged`.
- **Methods:**
  - `loadings(standardized: bool = True) -> pd.DataFrame`
  - `fit_measures() -> pd.DataFrame`
  - `parameter_estimates() -> pd.DataFrame` (long-format table with `est`, `se`, `z`, `pvalue`, `std_all` for loadings and factor correlations)

### Examples

```python
import anvala as av

kb = av.kmo_bartlett(df)
dt = av.dimtests(df, tests=["RAWPAR", "EMPKC", "HULL", "MAP"], n_sim=100)
print(dt)

fit_efa = av.efa(df, n_factors=dt.n_factors, estimator="MLR", rotation="geomin")
print(fit_efa.loadings().round(3))
print(fit_efa.factor_corr.round(3))
print(fit_efa.fit_measures().round(4))
```
