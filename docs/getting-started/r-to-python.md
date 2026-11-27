# R to Python Rosetta Stone

Designed for researchers transitioning from **R (`lavaan`, `semTools`, `EGAnet`, `psych`, `EFA.dimensions`, `BifactorIndicesCalculator`)** to **Python (`anvala`)**.

```python
import anvala as av
```

## 1. Confirmatory Factor Analysis & SEM (`lavaan`)

| Task | R (`lavaan`) | Python (`anvala`) |
| :--- | :--- | :--- |
| Continuous CFA (`MLR`) | `fit <- cfa(m, data=df, estimator="MLR")` | `fit = av.cfa(m, data=df, estimator="MLR")` |
| Standardized latent factors | `cfa(m, df, std.lv=TRUE, orthogonal=TRUE)` | `av.cfa(m, df, std_lv=True, orthogonal=True)` |
| Ordinal CFA (`WLSMV`) | `cfa(m, df, ordered=TRUE, estimator="WLSMV")` | `av.cfa(m, df, ordered=True, estimator="WLSMV")` |
| Full SEM & mediation (`:=`) | `fit <- sem(m, data=df, estimator="ML")` | `fit = av.sem(m, data=df, estimator="ML")` |
| Fit measures triad | `fitMeasures(fit)` | `fit.fit_measures()` (or `fit.fit_table`) |
| Parameter table (`std.all`) | `parameterEstimates(fit, standardized=TRUE)` | `fit.parameter_estimates()` |
| Standardized matrices | `lavInspect(fit, "std")` | `fit.inspect_std()` |
| Sample moments & thresholds | `lavInspect(fit, "sampstat")` | `fit.inspect_sampstat()` |
| Polychoric correlations | `lavCor(df, ordered=names(df))` | `av.polychoric(df)` |
| Raw / Bollen / Std residuals | `residuals(fit, type="cor.bollen")` | `fit.residuals(type="cor.bollen")` |
| Modification indices & EPC | `modindices(fit, sort=TRUE, minimum.value=3.84)` | `fit.modindices(sort=True, minimum_value=3.84)` |
| BCa bootstrap CIs (GPU) | `sem(m, df, se="bootstrap", bootstrap=1000)` | `fit.bootstrap_ci(n_boot=1000, method="bca")` |

## 2. Multi-Group Invariance & Factor Scoring (`lavaan`)

| Task | R (`lavaan`) | Python (`anvala`) |
| :--- | :--- | :--- |
| Multi-group CFA | `cfa(m, df, group="g", group.equal=c("loadings","intercepts"))` | `av.cfa_multigroup(m, df, group="g", group_equal=["loadings", "intercepts"])` |
| Partial invariance | `cfa(..., group.partial=c("x5 ~ 1", "f1 =~ x2"))` | `av.cfa_multigroup(..., group_partial=["x5 ~ 1", "f1 =~ x2"])` |
| Configural -> Strict pipeline | 4 manual `cfa()` calls + `lavTestLRT()` | `inv = av.measurement_invariance(m, df, group="g", estimator="MLR")` |
| Scaled LRT comparison | `lavTestLRT(fit1, fit2)` | `av.compare_models(fit1, fit2, names=["m1", "m2"])` |
| Lagrange multiplier score test | `lavTestScore(fit_scalar)` | `fit_scalar.score_test(return_total=True)` |
| Continuous factor scores | `lavPredict(fit, method="regression", se="standard")` | `fit.predict(method="regression", se=True)` |
| Bartlett / Anderson-Rubin / Green | `lavPredict(fit, method="Bartlett")` | `fit.predict(method="bartlett")` / `"anderson-rubin"` / `"green"` |
| Ordinal EBM / EAP scores | `lavPredict(fit_ord, method="EBM", se="standard")` | `fit_ord.predict(method="ebm", se=True)` (or `method="eap"`) |

## 3. EFA, Dimensionality, Reliability & Bifactor (`lavaan`, `EFA.dimensions`, `semTools`, `BifactorIndicesCalculator`)

| Task | R Equivalent | Python (`anvala`) |
| :--- | :--- | :--- |
| KMO & Bartlett's test | `psych::KMO(df); psych::cortest.bartlett(df)` | `av.kmo_bartlett(df)` |
| Dimensionality retention battery | `EFA.dimensions::DIMTESTS(df, tests=c("RAWPAR","EMPKC","HULL","SMT"))` | `av.dimtests(df, tests=["RAWPAR", "EMPKC", "HULL", "SMT"])` |
| Exploratory Factor Analysis | `lavaan::efa(df, nfactors=3, estimator="MLR", rotation="geomin")` | `av.efa(df, n_factors=3, estimator="MLR", rotation="geomin")` |
| McDonald's $\omega$ & AVE | `semTools::compRelSEM(fit); semTools::AVE(fit)` | `av.reliability(fit, data=df)` |
| Green & Yang ordinal $\omega$ | `semTools::compRelSEM(fit_ord, ord.scale=TRUE)` | `av.reliability(fit_ord, data=df_ord)` (automatic `omega_categorical`) |
| HTMT / HTMT2 discriminant validity | `semTools::htmt(model, data=df, htmt2=TRUE)` | `av.htmt(df, factors={"F1": [...], "F2": [...]})` |
| Cronbach's $\alpha$ (explicit) | `psych::alpha(df)` | `av.cronbach_alpha(df)` |
| Bifactor indices (ECV, PUC, $\omega_H$, H, FD, ARPB) | `BifactorIndicesCalculator::bifactorIndices(fit)` | `fit_bf.bifactor_indices(compute_arpb=True)` |

## 4. Network Psychometrics (`EGAnet`, `qgraph`)

| Task | R (`EGAnet` / `qgraph`) | Python (`anvala`) |
| :--- | :--- | :--- |
| Sparse partial correlation network | `qgraph::EBICglasso(cor(df), n=nrow(df), gamma=0.5)` | `av.ebic_glasso(df, gamma=0.5)` |
| Exploratory Graph Analysis | `EGAnet::EGA(df, algorithm="walktrap")` | `av.ega(df, algorithm="walktrap")` |
| Unique Variable Analysis (wTO) | `EGAnet::UVA(df, cut.off=0.25)` | `av.uva(df, cut_off=0.25)` |
| Standardized network loadings | `EGAnet::net.loads(ega_obj)` | `ega_res.net_loads()` or `av.compute_net_loads(pcor, mem)` |
| Total Entropy Fit Index (TEFI) | `EGAnet::tefi(cor_mat, structure)` | `ega_res.tefi` or `av.compute_tefi(cor_mat, mem)` |
| Bootstrap EGA (GPU-accelerated) | `EGAnet::bootEGA(df, iter=500, type="parametric")` | `av.boot_ega(df, n_boot=500, boot_type="parametric")` |

!!! note "Naming Convention"
    R arguments with dots (like `std.lv`, `group.equal`, `group.partial`, `cut.off`, `fixed.x`) become Python `snake_case` (`std_lv`, `group_equal`, `group_partial`, `cut_off`, `fixed_x`).
