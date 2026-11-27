# anvala

**GPU-Accelerated Psychometrics, Structural Equation Modeling, and Network Psychometrics in Python**

[![Open Complete Tour in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)
[![Download CRAN-Style PDF Manual](https://img.shields.io/badge/Manual-CRAN--Style%20PDF-b31b1b.svg)](assets/anvala-reference-manual.pdf)
[![PyPI](https://img.shields.io/pypi/v/anvala.svg)](https://pypi.org/project/anvala/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/FredPedrosa/anvala/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24507587-blue.svg)](https://doi.org/10.5281/zenodo.24507587)

---

**anvala** (*Analysis of Latent Variables*) is an open-source Python library built on PyTorch (`float64` automatic differentiation + CUDA GPU acceleration) designed for **numerical parity** with established R psychometric packages:

- [`lavaan`](https://lavaan.ugent.be/) — CFA, SEM, multi-group invariance, Lagrange multiplier score tests, local diagnostics, and factor scoring
- [`semTools`](https://github.com/simsem/semTools) — McDonald's $\omega$, Green & Yang (2009) categorical $\omega$, AVE, and HTMT/HTMT2 discriminant validity
- [`EGAnet`](https://github.com/hfgolino/EGAnet) — Exploratory Graph Analysis (`ega`), Unique Variable Analysis (`uva`), network loadings, TEFI, and GPU-accelerated `boot_ega`
- [`BifactorIndicesCalculator`](https://cran.r-project.org/package=BifactorIndicesCalculator) — ECV, PUC, $\omega_H$, $H$, Factor Determinacy ($FD$), IECV, and ARPB
- [`EFA.dimensions`](https://cran.r-project.org/package=EFA.dimensions) & [`psych`](https://cran.r-project.org/package=psych) — KMO, Bartlett's test, and 8 dimensionality retention criteria (`dimtests`)

## Quick Install

```bash
pip install anvala
```

No local GPU? Launch the interactive **Complete Tour** directly in Google Colab (with a free T4 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FredPedrosa/anvala/blob/main/notebooks/00_anvala_complete_tour.ipynb)

## What `anvala` Covers

| Domain | Public Functions | R Reference Parity |
| :--- | :--- | :--- |
| **CFA & SEM** | [`cfa()`](reference/cfa.md), [`sem()`](reference/sem.md) | `lavaan::cfa`, `lavaan::sem` (`ML`, `MLR`, `ULS`, `WLSMV`, `ULSMV`, `DWLS`) |
| **Multi-Group & Invariance** | [`cfa_multigroup()`](reference/multigroup.md), [`measurement_invariance()`](reference/multigroup.md), [`compare_models()`](reference/multigroup.md), `.score_test()` | `lavaan::cfa(group=...)`, `lavaan::lavTestLRT`, `lavaan::lavTestScore` |
| **Factor Scores** | [`predict()`](reference/predict.md) (`regression`, `bartlett`, `anderson-rubin`, `green`, `ebm`, `ml`, `eap`) | `lavaan::lavPredict` |
| **GPU Bootstrap CIs** | [`bootstrap_ci()`](reference/bootstrap.md) (`bca`, `percentile`) | `lavaan::sem(..., se="bootstrap")` / `boot::boot.ci` |
| **Local Diagnostics** | [`residuals()`](reference/diagnostics.md), [`modindices()`](reference/diagnostics.md) | `lavaan::residuals`, `lavaan::modindices` |
| **EFA & Dimensionality** | [`kmo_bartlett()`](reference/efa.md), [`dimtests()`](reference/efa.md), [`efa()`](reference/efa.md) | `lavaan::efa`, `EFA.dimensions::DIMTESTS`, `psych::KMO` |
| **Polychoric Estimation** | [`polychoric()`](reference/polychoric.md) | `lavaan::lavCor` / `lavaan:::muthen1984` |
| **Reliability & Bifactor** | [`reliability()`](reference/reliability-bifactor.md), [`htmt()`](reference/reliability-bifactor.md), [`cronbach_alpha()`](reference/reliability-bifactor.md), [`bifactor_indices()`](reference/reliability-bifactor.md) | `semTools::compRelSEM`, `semTools::htmt`, `BifactorIndicesCalculator` |
| **Network Psychometrics** | [`ebic_glasso()`](reference/network.md), [`ega()`](reference/network.md), [`uva()`](reference/network.md), [`boot_ega()`](reference/network.md), [`compute_net_loads()`](reference/network.md), [`compute_tefi()`](reference/network.md) | `EGAnet::EGA`, `EGAnet::UVA`, `EGAnet::bootEGA`, `EGAnet::net.loads`, `EGAnet::tefi` |

## Documentation Roadmap

- **New to `anvala`?** Start with [Installation & GPU Setup](getting-started/installation.md) and the [lavaan Model Syntax](getting-started/lavaan-syntax.md) guide.
- **Migrating from R?** See the [R to Python Rosetta Stone](getting-started/r-to-python.md) for side-by-side `lavaan`/`EGAnet`/`semTools` translations.
- **Want runnable examples?** Explore the [5 Step-by-Step Tutorials & Colab Notebooks](tutorials/01-cfa-and-diagnostics.md).
- **Looking up exact arguments and return attributes?** Browse the [CRAN-Style Reference Manual](reference/index.md) or [download the complete PDF Reference Manual](assets/anvala-reference-manual.pdf).

## Citation

If you use **anvala** in your research, please cite:

> Pedrosa, F. G. (2026). *anvala: GPU-Accelerated Psychometrics and Structural Equation Modeling in Python* [Computer software]. Universidade Federal de Minas Gerais (UFMG). https://github.com/FredPedrosa/anvala. DOI: `10.5281/zenodo.24507587`
