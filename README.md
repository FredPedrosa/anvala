# anvala

**anvala** is a Python library for psychometrics, Structural Equation Modeling (SEM), and Network Psychometrics designed for numerical parity with R's [`lavaan`](https://lavaan.ugent.be/) and [`EGAnet`](https://github.com/hfgolino/EGAnet), featuring PyTorch CUDA GPU acceleration for resampling and batch optimization.

## Key Features

- **CFA & SEM (`cfa`, `sem`)**: Continuous (`ML`, `MLR` with Yuan-Bentler / Satorra-Bentler robust corrections) and ordinal (`WLSMV` delta parameterization via polychoric correlations) estimation using `lavaan` syntax (`=~`, `~`, `~~`, parameter labels, and `:=` defined parameters).
- **Multi-Group CFA & Invariance (`cfa_multigroup`, `measurement_invariance`)**: Automated `configural` -> `metric` -> `scalar` -> `strict` invariance testing, partial invariance (`group_partial`), scaled likelihood-ratio tests (`compare_models`), and Lagrange multiplier tests (`score_test`, matching `lavTestScore`).
- **Factor Scores (`predict`)**: Empirical Bayes Modal (`EBM`) and Expected A Posteriori (`EAP`) latent score estimation with standard errors for continuous and ordinal indicators.
- **Network Psychometrics (`ega`, `boot_ega`, `uva`)**: Exploratory Graph Analysis via `EBICglasso` + Walktrap/Louvain, Unique Variable Analysis (`uva`), network loadings, TEFI, and GPU-accelerated `boot_ega`.
- **Reliability & Diagnostics (`reliability`, `modindices`)**: Cronbach's alpha, McDonald's omega (total and hierarchical), Average Variance Extracted (AVE), and modification indices.
- **GPU-Accelerated Bootstrap (`bootstrap_ci`, `boot_ega`)**: Vectorized 3D tensor optimization on CUDA (with automatic CPU fallback) for nonparametric and `BCa` confidence intervals, including mediation indirect effects.

## Installation

Install directly from GitHub:

```bash
pip install git+[https://github.com/FredPedrosa/anvala.git](https://github.com/FredPedrosa/anvala.git)
```

**Requirements:** Python >= 3.10, `numpy`, `scipy`, `pandas`, `torch`, and `igraph` (for EGA community detection).

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
print(fit.summary())
print(fit.parameter_estimates())
print(fit.fit_measures())

# WLSMV for ordinal (Likert) indicators + EBM factor scores
fit_ord = av.cfa(model, data=df_likert, estimator="WLSMV", ordered=True)
scores, se = fit_ord.predict(method="EBM", return_se=True)
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
print(boot.summary())
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

### 4. Exploratory Graph Analysis (EGA) & GPU `bootEGA`

```python
ega_res = av.ega(df, corr="auto", algorithm="walktrap")
print(ega_res.communities)
print(ega_res.loadings)

# GPU-accelerated bootstrap EGA
boot_ega = av.boot_ega(df, n_boot=500, boot_type="parametric")
print(boot_ega.summary)
print(boot_ega.item_stability)
```

## Citation

If you use **anvala** in your research, please cite:

> Pedrosa, F. G. (2026). *anvala: GPU-Accelerated Psychometrics and Structural Equation Modeling in Python* [Computer software]. Universidade Federal de Minas Gerais (UFMG). https://github.com/FredPedrosa/anvala

## License

MIT License
