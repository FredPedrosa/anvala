# Network Psychometrics (`ebic_glasso`, `ega`, `uva`, `boot_ega`)

*Module:* `anvala.ega` · *R equivalents:* `qgraph::EBICglasso()`, `EGAnet::EGA()`, `EGAnet::UVA()`, `EGAnet::bootEGA()`, `EGAnet::net.loads()`, `EGAnet::tefi()`

---

## `ebic_glasso`

### Description

Estimates a sparse Gaussian Graphical Model (partial correlation network) using Graphical LASSO (Friedman et al., 2008) over a logarithmic grid of $L$ penalties and selects the network minimizing the Extended Bayesian Information Criterion ($\mathrm{EBIC}_\gamma$; Foygel & Drton, 2010).

### Usage

```python
anvala.ebic_glasso(
    data: pd.DataFrame,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
) -> GlassoResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Item-level DataFrame (listwise deletion applied). |
| `gamma` | `float` | `0.5` | EBIC hyperparameter $\gamma \in [0, 1]$ controlling sparsity (`0.5` is standard in psychometrics). |
| `n_lambda` | `int` | `100` | Number of $\lambda$ regularization values in the geometric grid. |
| `lambda_min_ratio` | `float` | `0.1` | Ratio $\lambda_{\min} / \lambda_{\max}$ defining the lower bound of the penalty path. |
| `corr` | `str` | `"auto"` | Correlation estimator: `"auto"` (polychoric for integer items with $\le 7$ categories, Pearson otherwise), `"pearson"`, `"spearman"`, or `"polychoric"`. |

### Value (`GlassoResult`)

- `partial_corr` (`pd.DataFrame`): Sparse $p \times p$ partial correlation network.
- `precision` (`pd.DataFrame`): Estimated precision matrix $\hat{\Omega}$.
- `lambda_opt` (`float`), `ebic` (`float`), `gamma` (`float`), `n_edges` (`int`), `sparsity` (`float`).

---

## `ega`

### Description

Performs **Exploratory Graph Analysis** (Golino & Epskamp, 2017; parity with `EGAnet` 2.4.1): estimates the sparse partial correlation network via `ebic_glasso()`, checks for unidimensionality via the Leading Eigenvalue criterion (Christensen et al., 2023), detects communities with Walktrap or Louvain, and computes standardized network loadings and TEFI.

### Usage

```python
anvala.ega(
    data: pd.DataFrame,
    algorithm: str = "walktrap",
    gamma: float = 0.5,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
    steps: int = 4,
    unidimensional_check: bool = True,
) -> EGAResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Item-level DataFrame. |
| `algorithm` | `str` | `"walktrap"` | Community detection algorithm: `"walktrap"` (Pons & Latapy, 2005) or `"louvain"` (Blondel et al., 2008). |
| `gamma` | `float` | `0.5` | EBIC sparsity hyperparameter. |
| `lambda_min_ratio` | `float` | `0.1` | Penalty grid minimum ratio. |
| `corr` | `str` | `"auto"` | `"auto"`, `"pearson"`, `"spearman"`, or `"polychoric"`. |
| `steps` | `int` | `4` | Random-walk step length for `"walktrap"`. |
| `unidimensional_check` | `bool` | `True` | If `True`, applies the Leading Eigenvalue unidimensionality check before community detection. |

### Value (`EGAResult`)

- **Attributes:** `n_dim` (`int`), `communities` (`pd.DataFrame` with `item`, `dimension`), `network` (`pd.DataFrame`), `glasso` (`GlassoResult`), `loadings` (`pd.DataFrame`), `tefi` (`float`), `algorithm` (`str`).
- **Methods:** `partial_corr() -> pd.DataFrame`, `net_loads() -> pd.DataFrame`.

---

## `uva`

### Description

Performs **Unique Variable Analysis** (Christensen, Garrido, & Golino, 2023) to identify locally dependent / redundant item pairs via signed **Weighted Topological Overlap (`wTO`)** on the `EBICglasso` network and iteratively removes redundant items until all remaining pairs satisfy $|\mathrm{wTO}| \le \text{cut\_off}$.

### Usage

```python
anvala.uva(
    data: pd.DataFrame,
    cut_off: float = 0.25,
    gamma: float = 0.5,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
) -> UVAResult
```

### Value (`UVAResult`)

- `wto_matrix` (`pd.DataFrame`): Initial $p \times p$ signed wTO matrix.
- `redundant_pairs` (`pd.DataFrame`): Pairs exceeding `cut_off` (`item_1`, `item_2`, `wto`), sorted by $|\mathrm{wTO}|$.
- `keep` (`list[str]`): Retained item names.
- `remove` (`list[str]`): Iteratively pruned redundant item names.
- `reduced_data` (`pd.DataFrame`): Cleaned DataFrame containing only `keep` columns.
- `cut_off` (`float`).

---

## `boot_ega`

### Description

Performs **Bootstrap Exploratory Graph Analysis** (Christensen & Golino, 2021) to evaluate structural consistency (dimension stability) and per-item replicability across parametric or nonparametric bootstrap samples. On CUDA-enabled GPUs (`device="auto"` or `"cuda"`, for non-polychoric `corr`), stacks correlation matrices in 3D tensors and solves the entire 100-$\lambda$ `EBICglasso` path simultaneously via batched ADMM (Boyd et al., 2011).

### Usage

```python
anvala.boot_ega(
    data: pd.DataFrame,
    n_boot: int = 500,
    boot_type: str = "parametric",
    algorithm: str = "walktrap",
    gamma: float = 0.5,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
    steps: int = 4,
    seed: int = 42,
    n_jobs: int = 4,
    device: str = "auto",
) -> BootEGAResult
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | *required* | Item-level DataFrame. |
| `n_boot` | `int` | `500` | Number of bootstrap replicates. |
| `boot_type` | `str` | `"parametric"` | `"parametric"` (multivariate normal + optional threshold discretization from the empirical correlation matrix) or `"nonparametric"` (row resampling with replacement). |
| `algorithm` | `str` | `"walktrap"` | `"walktrap"` or `"louvain"`. |
| `gamma`, `lambda_min_ratio`, `corr`, `steps` | *same as `ega`* | — | Network estimation and community detection settings. |
| `seed` | `int` | `42` | Random seed. |
| `n_jobs` | `int` | `4` | Number of worker threads for CPU fallback. |
| `device` | `str` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"`. |

### Value (`BootEGAResult`)

- `empirical` (`EGAResult`): Empirical EGA fit on the observed sample.
- `n_boot` (`int`), `boot_type` (`str`).
- `summary` (`pd.DataFrame`): `n_boot`, `empirical_dim`, `median_dim`, `se_dim`, `ci_lower`, `ci_upper`, `dimension_stability`.
- `dimension_frequencies` (`pd.DataFrame`): `n_dim`, `frequency`, `proportion`.
- `dimension_stability` (`float`): Proportion of replicates recovering the exact empirical number of dimensions.
- `item_stability` (`pd.DataFrame`): Indexed by item with `empirical_dim`, per-dimension assignment probabilities (`Dim_1`, ...), and overall `stability`.
- `typical_graph` (`pd.DataFrame`): Element-wise median partial correlation network across all bootstrap samples.

---

## `compute_net_loads` & `compute_tefi`

### `compute_net_loads`

```python
anvala.compute_net_loads(
    pcor_df: pd.DataFrame,
    memberships: np.ndarray,
    method: str = "revised",
    scaling: float = 2.0,
) -> pd.DataFrame
```

Computes standardized network loadings (parity with `EGAnet >= 2.0.7` `net.loads`). `method="revised"` uses the community-size-adjusted denominator; any other value uses the classic square-root normalization.

### `compute_tefi`

```python
anvala.compute_tefi(corr_mat: np.ndarray, memberships: np.ndarray) -> float
```

Computes the **Total Entropy Fit Index (TEFI)** (Golino et al., 2021; parity with `EGAnet::tefi`). Lower (more negative) values indicate better structural fit of the community partition `memberships` to the correlation matrix `corr_mat`.

### Examples

```python
import anvala as av

uva_res = av.uva(df, cut_off=0.25)
ega_res = av.ega(uva_res.reduced_data, algorithm="walktrap")
print(ega_res.communities)
print(ega_res.net_loads().round(3))
print("TEFI:", ega_res.tefi)

boot_res = av.boot_ega(uva_res.reduced_data, n_boot=500, boot_type="parametric")
print(boot_res.summary)
print(boot_res.item_stability.round(3))
```
