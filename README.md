# anvala

[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-indigo.svg)](https://fredpedrosa.github.io/anvala/)
[![Open Complete Tour in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)
[![Reference Manual PDF](https://img.shields.io/badge/Manual-CRAN--Style%20PDF-b31b1b.svg)](https://fredpedrosa.github.io/anvala/assets/anvala-reference-manual.pdf)
[![PyPI](https://img.shields.io/pypi/v/anvala.svg)](https://pypi.org/project/anvala/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24507587-blue.svg)](https://doi.org/10.5281/zenodo.24507587)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/FredPedrosa/anvala/blob/main/LICENSE)

**anvala** is a Python library for psychometrics, Structural Equation Modeling (SEM), and Network Psychometrics designed for numerical parity with R’s [`lavaan`](https://lavaan.ugent.be/), [`EGAnet`](https://github.com/hfgolino/EGAnet), and [`BifactorIndicesCalculator`](https://cran.r-project.org/package=BifactorIndicesCalculator), featuring PyTorch automatic differentiation and CUDA GPU acceleration for resampling and batched ADMM optimization.

## Documentation & Interactive Tutorials

- **Documentation Website:** [fredpedrosa.github.io/anvala](https://fredpedrosa.github.io/anvala/)
- **CRAN-Style Reference Manual (PDF):** [Download `anvala-reference-manual.pdf`](https://fredpedrosa.github.io/anvala/assets/anvala-reference-manual.pdf)
- **R (`lavaan` / `EGAnet` / `semTools`) to Python Rosetta Stone:** [Translation Guide](https://fredpedrosa.github.io/anvala/getting-started/r-to-python/)
- **Interactive Google Colab Notebooks (Free T4 GPU):**
  - [00 — Complete End-to-End Tour (All Functions)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)
  - [01 — Continuous & Ordinal CFA, Factor Scoring & Local Diagnostics](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/01_cfa_ordinal_and_scoring.ipynb)
  - [02 — SEM, Mediation & GPU BCa Bootstrap](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/02_sem_mediation_gpu_bootstrap.ipynb)
  - [03 — Multi-Group Measurement Invariance & Score Tests](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/03_multigroup_invariance.ipynb)
  - [04 — EFA, Dimensionality Tests, Reliability, HTMT & Bifactor Indices](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/04_efa_reliability_bifactor.ipynb)
  - [05 — Network Psychometrics (EGA, UVA, TEFI & GPU `boot_ega`)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/05_network_psychometrics_ega.ipynb)

## Key Features

- **CFA & SEM (`cfa`, `sem`)**: Continuous (`ML`, `MLR` with Yuan-Bentler / Satorra-Bentler robust corrections) and ordinal (`WLSMV` delta parameterization via polychoric correlations) estimation using `lavaan` syntax (`=~`, `~`, `~~`, parameter labels, and `:=` defined parameters), plus local diagnostics (`residuals`, `modindices`).
- **Multi-Group CFA & Measurement Invariance (`cfa_multigroup`, `measurement_invariance`)**: Automated `configural` -> `metric` -> `scalar` -> `strict` invariance testing, partial invariance (`group_partial`), scaled likelihood-ratio tests (`compare_models`), and Lagrange multiplier tests (`score_test`, matching `lavTestScore`).
- **Factor Scores (`predict`)**: Regression, Bartlett, Anderson-Rubin, and Green scores for continuous models, plus Empirical Bayes Modal (`EBM`), `ML`, and Expected A Posteriori (`EAP`) latent score estimation with standard errors for ordinal indicators.
- **EFA & Dimensionality Tests (`efa`, `dimtests`, `kmo_bartlett`)**: Continuous (`ML`, `MLR`, `ULS`, `DWLS`) and ordinal (`WLSMV`) EFA with multi-start GPA rotations (`geomin`, `oblimin`, `promax`, `varimax`, etc.), rotated SEs, KMO/Bartlett adequacy, and an 8-method dimensionality battery (`RAWPAR`, `EMPKC`, `HULL`, `SMT`, `MAP`, `CD`).
- **Network Psychometrics (`ega`, `boot_ega`, `uva`)**: Exploratory Graph Analysis via vectorized `EBICglasso` + Walktrap/Louvain, Unique Variable Analysis (`uva`), network loadings (`compute_net_loads`), `compute_tefi`, and GPU-accelerated `boot_ega`.
- **Reliability & Bifactor Indices (`reliability`, `bifactor_indices`, `htmt`)**: McDonald’s omega, Green & Yang (2009) categorical omega, optional Cronbach’s alpha (`cronbach_alpha`), Average Variance Extracted (AVE), HTMT/HTMT2 discriminant validity, and ECV/PUC/FD/H/IECV/ARPB bifactor indices.
- **GPU-Accelerated Bootstrap (`bootstrap_ci`, `boot_ega`)**: Vectorized 3D tensor optimization on CUDA (with automatic CPU fallback) for nonparametric and `BCa` confidence intervals, including mediation indirect effects.

## Installation

Install from PyPI:

```bash
pip install anvala
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/FredPedrosa/anvala.git
```

**Requirements:** Python >= 3.11, `numpy`, `scipy`, `pandas`, `torch`, and `igraph`.

## Quick Start

### 1. Confirmatory Factor Analysis (Continuous & Ordinal)

```python
import anvala as av

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Robust ML (MLR) for continuous indicators
fit = av.cfa(model, data=df, estimator="MLR")
print(fit.fit_table)
print(fit.parameter_estimates())

# WLSMV for ordinal (Likert) indicators + EBM factor scores
fit_ord = av.cfa(model, data=df_likert, estimator="WLSMV", ordered=True)
scores, se = fit_ord.predict(method="ebm", se=True)
```

### 2. SEM Mediation & GPU Bootstrap Confidence Intervals

```python
med_model = """
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total    := c + a*b
"""

fit_sem = av.sem(med_model, data=df, estimator="ML")
boot = fit_sem.bootstrap_ci(n_boot=1000, conf_level=0.95, method="bca")
print(boot.table)
```

### 3. Multi-Group Measurement Invariance & Score Tests

```python
inv = av.measurement_invariance(
    model,
    data=df,
    group="school",
    estimator="MLR",
)
print(inv.summary())

# Inspect parameter-level invariance violations (Lagrange multipliers)
print(inv.fits["scalar"].score_test())
```

### 4. Exploratory Graph Analysis (EGA) & GPU `boot_ega`

```python
ega_res = av.ega(df, corr="auto", algorithm="walktrap")
print(ega_res.communities)
print(ega_res.loadings)

# GPU-accelerated bootstrap EGA
boot_res = av.boot_ega(df, n_boot=500, boot_type="parametric")
print(boot_res.summary)
print(boot_res.item_stability)
```

## Computational Benchmarks & Monte Carlo Validation

Reproducible scripts and raw outputs are available in `simulations/`.

### Speed Benchmarks ($N = 400$)

| Task | R Reference (`lavaan` / `EGAnet`) | `anvala` CPU | `anvala` RTX 3060 | `anvala` NVIDIA L4 | Speedup vs. R |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SEM Mediation BCa Bootstrap** ($B = 5000$) | 100.66 s | 0.68 s | 1.24 s | 0.79 s | **81.3x – 148.4x** |
| **bootEGA** (12 items, $B = 1000$) | 38.11 s | 500.83 s | 22.70 s | 19.46 s | **1.68x – 1.96x** (up to **32.8x** vs. CPU) |

### Monte Carlo Validation Summary (200 Replications per Cell)

- **Parameter Recovery & Coverage ($N \in \{150, 300, 600\}$):** 100% convergence across non-normal continuous CFA (`MLR`, standardized $\chi^2_3$ indicators), 4-category ordinal CFA (`WLSMV`), and mediation (`BCa`), with relative bias $< 5\%$, $\text{SE}/\text{SD}$ ratios within $0.90\text{--}1.05$, and nominal 95% CI coverage between $90.0\%$ and $96.5\%$.
- **Multi-Group Invariance & Partial Scalar Recovery ($N_{\text{group}} = 300$):** Empirical Type I error at nominal $\alpha = 0.05$ ($0.040\text{--}0.050$), detection power of $95.5\%$ (metric) and $100.0\%$ (scalar), and unbiased latent mean recovery under `group_partial` ($+2.53\%$ relative bias, $94.5\%$ CI coverage vs. $+19.69\%$ bias and $78.5\%$ coverage under misspecified full scalar invariance).

## Citation

If you use **anvala** in your research, please cite:

> Pedrosa, F. G. (2026). *anvala: GPU-Accelerated Psychometrics and Structural Equation Modeling in Python* [Computer software]. Universidade Federal de Minas Gerais (UFMG). https://github.com/FredPedrosa/anvala. DOI: 10.5281/zenodo.24507587

## License

MIT License
