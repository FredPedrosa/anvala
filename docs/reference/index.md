# CRAN-Style Reference Manual

Welcome to the complete **CRAN-style Reference Manual** for **`anvala` v0.1.0**. Every public function, method, parameter, default value, return object attribute, R reference equivalent, and runnable example is documented here.

[![Download Complete PDF Manual](https://img.shields.io/badge/Download-CRAN--Style%20PDF%20Manual-b31b1b.svg)](../assets/anvala-reference-manual.pdf)

---

## Alphabetical Index of Public Functions

| Function | Module | Summary | R Equivalent |
| :--- | :--- | :--- | :--- |
| [`bifactor_indices()`](reliability-bifactor.md#bifactor_indices) | `anvala.bifactor` | Model-, factor-, and item-level bifactor indices (ECV, PUC, $\omega$, $\omega_H$, $H$, $FD$, IECV, ARPB) | `BifactorIndicesCalculator::bifactorIndices` |
| [`boot_ega()`](network.md#boot_ega) | `anvala.ega` | GPU-accelerated parametric & nonparametric Bootstrap EGA | `EGAnet::bootEGA` |
| [`bootstrap_ci()`](bootstrap.md#bootstrap_ci) | `anvala.bootstrap` | GPU-accelerated BCa & percentile bootstrap confidence intervals for SEM/CFA | `lavaan::sem(..., se="bootstrap")` |
| [`cfa()`](cfa.md#cfa) | `anvala.cfa` | Continuous (`ML`, `MLR`, `ULS`) & ordinal (`WLSMV`, `ULSMV`, `DWLS`) CFA | `lavaan::cfa` |
| [`cfa_multigroup()`](multigroup.md#cfa_multigroup) | `anvala.multigroup` | Multi-group CFA with equality constraints (`group_equal`) & partial invariance (`group_partial`) | `lavaan::cfa(..., group=...)` |
| [`compare_models()`](multigroup.md#compare_models) | `anvala.multigroup` | Nested model comparison with Satorra--Bentler scaled LRT and $\Delta$ fit indices | `lavaan::lavTestLRT` |
| [`compute_net_loads()`](network.md#compute_net_loads) | `anvala.ega` | Standardized network loadings from a partial correlation network and community vector | `EGAnet::net.loads` |
| [`compute_tefi()`](network.md#compute_tefi) | `anvala.ega` | Total Entropy Fit Index (TEFI) for evaluating community partitions | `EGAnet::tefi` |
| [`cronbach_alpha()`](reliability-bifactor.md#cronbach_alpha) | `anvala.reliability` | Raw and standardized Cronbach's $\alpha$ (Pearson or polychoric) | `psych::alpha` |
| [`dimtests()`](efa.md#dimtests) | `anvala.efa` | 8-method dimensionality retention battery (`RAWPAR`, `EMPKC`, `HULL`, `SMT`, `MAP`, `CD`, `NEVALSGT1`, `SESCREE`) | `EFA.dimensions::DIMTESTS` |
| [`ebic_glasso()`](network.md#ebic_glasso) | `anvala.ega` | Sparse partial correlation network estimation via EBIC-selected Graphical LASSO | `qgraph::EBICglasso` |
| [`efa()`](efa.md#efa) | `anvala.efa` | Exploratory Factor Analysis with multi-start GPA rotations and rotated SEs | `lavaan::efa` |
| [`ega()`](network.md#ega) | `anvala.ega` | Exploratory Graph Analysis (`EBICglasso` + Walktrap/Louvain + unidimensional check) | `EGAnet::EGA` |
| [`htmt()`](reliability-bifactor.md#htmt) | `anvala.reliability` | HTMT2 (geometric mean) and classic HTMT (arithmetic mean) discriminant validity | `semTools::htmt` |
| [`kmo_bartlett()`](efa.md#kmo_bartlett) | `anvala.efa` | Kaiser--Meyer--Olkin (KMO), item MSA, and Bartlett's test of sphericity | `psych::KMO`, `psych::cortest.bartlett` |
| [`measurement_invariance()`](multigroup.md#measurement_invariance) | `anvala.multigroup` | Automated `configural -> metric -> scalar -> strict` invariance sequence | `semTools::measEq.syntax` + `lavTestLRT` |
| [`modindices()`](diagnostics.md#modindices) | `anvala.diagnostics` | Lagrange multiplier modification indices (MI, EPC, SEPC.lv, SEPC.all) | `lavaan::modindices` |
| [`polychoric()`](polychoric.md#polychoric) | `anvala.polychoric` | Two-step Olsson (1979) polychoric correlations, thresholds, and Muthén $\Gamma$ | `lavaan::lavCor` / `lavaan:::muthen1984` |
| [`predict()`](predict.md#predict) | `anvala.predict` | Latent factor scores & SEs (`regression`, `bartlett`, `anderson-rubin`, `green`, `ebm`, `ml`, `eap`) | `lavaan::lavPredict` |
| [`reliability()`](reliability-bifactor.md#reliability) | `anvala.reliability` | McDonald's $\omega$, Green & Yang categorical $\omega$, CR, AVE, and HTMT from CFA/EFA fits | `semTools::compRelSEM`, `semTools::AVE` |
| [`residuals()`](diagnostics.md#residuals) | `anvala.diagnostics` | Local covariance/correlation residuals (`raw`, `cor`, `cor.bollen`, `standardized`) | `lavaan::residuals` |
| [`sem()`](sem.md#sem) | `anvala.sem` | Full Structural Equation Modeling & path analysis with `:=` defined parameters | `lavaan::sem` |
| [`uva()`](network.md#uva) | `anvala.ega` | Unique Variable Analysis (Weighted Topological Overlap redundancy detection & pruning) | `EGAnet::UVA` |

---

## Alphabetical Index of Result Classes

| Class | Returned By | Key Methods & Attributes |
| :--- | :--- | :--- |
| [`AnvalaFit`](cfa.md#anvalafit) | `cfa()` | `.fit_measures()`, `.parameter_estimates()`, `.inspect_std()`, `.inspect_sampstat()`, `.polychoric()`, `.residuals()`, `.modindices()`, `.bifactor_indices()`, `.predict()`, `.bootstrap_ci()` |
| [`BootEGAResult`](network.md#boot_ega) | `boot_ega()` | `.empirical`, `.summary`, `.dimension_frequencies`, `.dimension_stability`, `.item_stability`, `.typical_graph` |
| [`BootstrapResult`](bootstrap.md#bootstrapresult) | `bootstrap_ci()` | `.table`, `.boot_samples`, `.n_boot`, `.conf_level`, `.method`, `.device`, `.summary()` |
| [`DimTestsResult`](efa.md#dimtests) | `dimtests()` | `.results`, `.n_factors`, `.primary_method`, `.eigenvalues`, `.adequacy` |
| [`EFAFit`](efa.md#efafit) | `efa()` | `.loadings()`, `.fit_measures()`, `.parameter_estimates()`, `.factor_corr`, `.communalities`, `.uniquenesses`, `.variance_explained` |
| [`EGAResult`](network.md#ega) | `ega()` | `.n_dim`, `.communities`, `.network`, `.glasso`, `.loadings`, `.tefi`, `.partial_corr()`, `.net_loads()` |
| [`GlassoResult`](network.md#ebic_glasso) | `ebic_glasso()` | `.partial_corr`, `.precision`, `.lambda_opt`, `.ebic`, `.gamma`, `.n_edges`, `.sparsity` |
| [`InvarianceResult`](multigroup.md#invarianceresult) | `measurement_invariance()` | `.table`, `.fits`, `.group_col`, `.group_labels`, `.estimator`, `.ordered`, `.summary()` |
| [`MultiGroupFit`](multigroup.md#multigroupfit) | `cfa_multigroup()` | `.parameter_estimates()`, `.fit_measures()`, `.score_test()`, `.summary()`, `.predict()`, `.group_chisq` |
| [`PolychoricResult`](polychoric.md#polychoric) | `polychoric()` | `.corr`, `.thresholds`, `.thresholds_list`, `.gamma`, `.n_obs` |
| [`ReliabilityResult`](reliability-bifactor.md#reliability) | `reliability()` | `.constructs`, `.htmt_matrix`, `.htmt2_matrix`, `.estimator`, `.ordered` |
| [`SEMFit`](sem.md#semfit) | `sem()` | `.parameter_estimates()`, `.fit_measures()`, `.implied_cov()`, `.residuals()`, `.modindices()`, `.summary()`, `.predict()`, `.bootstrap_ci()` |
| [`UVAResult`](network.md#uva) | `uva()` | `.wto_matrix`, `.redundant_pairs`, `.keep`, `.remove`, `.reduced_data`, `.cut_off` |
