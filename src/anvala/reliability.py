from dataclasses import dataclass
import numpy as np
import pandas as pd

from anvala.cfa import AnvalaFit
from anvala.efa import EFAFit
from anvala.polychoric import _bvn_cdf, _norm_cdf, polychoric


def _green_yang_categorical_omega(
    std_loadings: np.ndarray,
    thresholds_list: list[np.ndarray],
    r_poly: np.ndarray,
) -> float:
    """
    Ômega categórico / ordinal de Green & Yang (2009), em paridade com semTools::compRelSEM(..., ord.scale = TRUE).
    """
    k = len(std_loadings)
    if k < 2:
        return np.nan

    # Variâncias e covariâncias implícitas e observadas na escala categórica observada
    var_obs = np.zeros(k, dtype=np.float64)
    var_true = np.zeros(k, dtype=np.float64)

    for i in range(k):
        taus = thresholds_list[i]
        p_ge = _norm_cdf(-taus)  # P(X_i >= c)
        c_minus_1 = len(taus)
        v_i = 0.0
        for c in range(c_minus_1):
            v_i += p_ge[c] * (1.0 - p_ge[c])
            for d in range(c + 1, c_minus_1):
                v_i += 2.0 * p_ge[d] * (1.0 - p_ge[c])
        var_obs[i] = v_i

        # Variância verdadeira na escala ordinal usa rho_ii = lambda_i^2
        rho_ii = float(np.clip(std_loadings[i] ** 2, -0.9999, 0.9999))
        vt_i = 0.0
        for c in range(c_minus_1):
            for d in range(c_minus_1):
                p_joint = float(_bvn_cdf(np.array(-taus[c]), np.array(-taus[d]), rho_ii))
                vt_i += p_joint - float(p_ge[c] * p_ge[d])
        var_true[i] = vt_i

    cov_obs_sum = 0.0
    cov_mod_sum = 0.0
    for i in range(k):
        taus_i = thresholds_list[i]
        p_i = _norm_cdf(-taus_i)
        for j in range(i + 1, k):
            taus_j = thresholds_list[j]
            p_j = _norm_cdf(-taus_j)
            rho_obs_ij = float(np.clip(r_poly[i, j], -0.9999, 0.9999))
            rho_mod_ij = float(np.clip(std_loadings[i] * std_loadings[j], -0.9999, 0.9999))

            c_obs = 0.0
            c_mod = 0.0
            for ci in range(len(taus_i)):
                for cj in range(len(taus_j)):
                    prod_marg = float(p_i[ci] * p_j[cj])
                    p2_obs = float(_bvn_cdf(np.array(-taus_i[ci]), np.array(-taus_j[cj]), rho_obs_ij))
                    p2_mod = float(_bvn_cdf(np.array(-taus_i[ci]), np.array(-taus_j[cj]), rho_mod_ij))
                    c_obs += p2_obs - prod_marg
                    c_mod += p2_mod - prod_marg
            cov_obs_sum += 2.0 * c_obs
            cov_mod_sum += 2.0 * c_mod

    num = float(np.sum(var_true)) + cov_mod_sum
    denom = float(np.sum(var_obs)) + cov_obs_sum
    return float(num / max(1e-12, denom))


def cronbach_alpha(data: pd.DataFrame, corr: str = "pearson") -> pd.DataFrame:
    """
    Calcula Alfa de Cronbach bruto e padronizado (disponível apenas por chamada explícita).
    """
    df_clean = data.dropna()
    k = df_clean.shape[1]
    if corr == "polychoric":
        s_mat = polychoric(df_clean).corr.to_numpy(dtype=np.float64)
    else:
        s_mat = np.cov(df_clean.to_numpy(dtype=np.float64), rowvar=False, bias=False)
    r_mat = df_clean.corr().to_numpy(dtype=np.float64) if corr != "polychoric" else s_mat

    sum_var = float(np.sum(np.diag(s_mat)))
    tot_var = float(np.sum(s_mat))
    alpha_raw = (float(k) / float(k - 1)) * (1.0 - sum_var / max(1e-12, tot_var))

    mean_r = float(np.mean(r_mat[np.tril_indices(k, k=-1)]))
    alpha_std = (float(k) * mean_r) / (1.0 + (float(k) - 1.0) * mean_r)
    return pd.DataFrame([{"n_items": k, "alpha_raw": alpha_raw, "alpha_std": alpha_std}])


def htmt(
    data: pd.DataFrame,
    factors: dict[str, list[str]],
    corr: str = "auto",
    geometric: bool = True,
    absolute: bool = True,
) -> pd.DataFrame:
    """
    Calcula a matriz HTMT2 (Roemer et al., 2021, padrão) ou HTMT (Henseler et al., 2015) entre construtos,
    em paridade com semTools::htmt (absolute=True, htmt2=True por padrão).
    """
    df_clean = data.dropna()
    use_poly = corr == "polychoric" or (
        corr == "auto"
        and all(
            np.all(np.isclose(df_clean[c].to_numpy(), np.round(df_clean[c].to_numpy())))
            and len(np.unique(df_clean[c].to_numpy())) <= 7
            for c in df_clean.columns
        )
    )
    r_df = polychoric(df_clean).corr if use_poly else df_clean.corr(method="pearson")
    f_names = list(factors.keys())
    m = len(f_names)
    mat = np.eye(m, dtype=np.float64)

    def _agg(vals: np.ndarray) -> float:
        v = np.abs(vals) if absolute else vals
        if geometric:
            pos = np.maximum(v, 1e-12)
            return float(np.exp(np.mean(np.log(pos))))
        return float(np.mean(v))

    mono_means: dict[str, float] = {}
    for f in f_names:
        items = factors[f]
        sub = r_df.loc[items, items].to_numpy(dtype=np.float64)
        off = sub[np.tril_indices(len(items), k=-1)]
        mono_means[f] = _agg(off)

    for i in range(m):
        for j in range(i + 1, m):
            fi, fj = f_names[i], f_names[j]
            hetero = r_df.loc[factors[fi], factors[fj]].to_numpy(dtype=np.float64).ravel()
            h_mean = _agg(hetero)
            denom = np.sqrt(max(1e-12, mono_means[fi] * mono_means[fj]))
            val = h_mean / denom
            mat[i, j] = val
            mat[j, i] = val

    return pd.DataFrame(mat, index=f_names, columns=f_names)


@dataclass
class ReliabilityResult:
    constructs: pd.DataFrame
    htmt_matrix: pd.DataFrame | None
    htmt2_matrix: pd.DataFrame | None
    estimator: str
    ordered: bool

    def __repr__(self) -> str:
        out = f"ReliabilityResult(estimator='{self.estimator}', ordered={self.ordered})\n"
        out += self.constructs.round(4).to_string(index=False)
        if self.htmt_matrix is not None:
            out += "\n\nValidade Discriminante (HTMT):\n" + self.htmt_matrix.round(4).to_string()
        return out


def reliability(
    fit: AnvalaFit | EFAFit,
    data: pd.DataFrame | None = None,
    loading_cutoff: float = 0.30,
    include_alpha: bool = False,
) -> ReliabilityResult:
    """
    Calcula fidedignidade e validade baseada no modelo (Ômega de McDonald, Ômega categórico de
    Green & Yang, Confiabilidade Composta e AVE) diretamente de um ajuste CFA ou EFA.
    Alfa de Cronbach NÃO é calculado por padrão; passe include_alpha=True apenas se desejar.
    """
    if isinstance(fit, AnvalaFit):
        pe = fit.parameter_estimates()
        load_rows = pe[pe["op"] == "=~"]
        factor_map: dict[str, list[str]] = {}
        for _, row in load_rows.iterrows():
            factor_map.setdefault(str(row["lhs"]), []).append(str(row["rhs"]))

        est_name = fit.estimator
        is_ordered = fit.poly_result is not None
        r_poly_df = fit.polychoric() if is_ordered else None
        poly_obj = fit.poly_result
        var_rows = pe[(pe["op"] == "~~") & (pe["lhs"] == pe["rhs"])].set_index("lhs")["est"]

        rows = []
        for f_name, items in factor_map.items():
            sub_pe = load_rows[(load_rows["lhs"] == f_name) & (load_rows["rhs"].isin(items))]
            unstd_l = sub_pe["est"].to_numpy(dtype=np.float64)
            std_l = sub_pe["std_all"].to_numpy(dtype=np.float64)
            psi_f = float(var_rows.loc[f_name]) if f_name in var_rows.index else 1.0
            theta_unstd = np.array([float(var_rows.loc[it]) for it in items], dtype=np.float64)
            theta_std = np.maximum(0.0, 1.0 - std_l ** 2)

            num_unstd = (float(np.sum(unstd_l)) ** 2) * psi_f
            denom_implied = num_unstd + float(np.sum(theta_unstd))
            if data is not None and not is_ordered:
                denom_obs = float(np.sum(np.cov(data[items].to_numpy(dtype=np.float64), rowvar=False, bias=True)))
            else:
                denom_obs = denom_implied

            omega_total = num_unstd / max(1e-12, denom_obs)
            omega_implied = num_unstd / max(1e-12, denom_implied)
            sum_l2_std = float(np.sum(std_l) ** 2)
            omega_std = sum_l2_std / max(1e-12, sum_l2_std + float(np.sum(theta_std)))
            
            num_ave = psi_f * float(np.sum(unstd_l ** 2))
            if data is not None and not is_ordered:
                obs_var_items = np.var(data[items].to_numpy(dtype=np.float64), axis=0, ddof=0)
                ave_val = num_ave / max(1e-12, float(np.sum(obs_var_items)))
            else:
                ave_val = num_ave / max(1e-12, float(np.sum((unstd_l ** 2) * psi_f + theta_unstd)))
            ave_implied = float(np.mean(std_l ** 2))

            row_dict: dict = {
                "construct": f_name,
                "n_items": len(items),
                "omega_total": omega_total,
                "omega_implied": omega_implied,
                "omega_std": omega_std,
                "cr": omega_implied,
                "ave": ave_val,
                "ave_implied": ave_implied,
            }

            if is_ordered and r_poly_df is not None and poly_obj is not None:
                item_indices = [list(r_poly_df.columns).index(it) for it in items]
                taus_sub = [poly_obj.thresholds_list[idx] for idx in item_indices]
                r_sub = r_poly_df.loc[items, items].to_numpy(dtype=np.float64)
                row_dict["omega_categorical"] = _green_yang_categorical_omega(std_l, taus_sub, r_sub)

            if include_alpha and data is not None:
                a_df = cronbach_alpha(data[items], corr="polychoric" if is_ordered else "pearson")
                row_dict["alpha_raw"] = float(a_df["alpha_raw"].iloc[0])
                row_dict["alpha_std"] = float(a_df["alpha_std"].iloc[0])

            rows.append(row_dict)

        htmt_df = htmt(data, factor_map, corr="polychoric" if is_ordered else "pearson", geometric=True) if (data is not None and len(factor_map) > 1) else None
        htmt1_df = htmt(data, factor_map, corr="polychoric" if is_ordered else "pearson", geometric=False) if (data is not None and len(factor_map) > 1) else None

        return ReliabilityResult(
            constructs=pd.DataFrame(rows),
            htmt_matrix=htmt_df,
            htmt2_matrix=htmt1_df,
            estimator=est_name,
            ordered=is_ordered,
        )

    if isinstance(fit, EFAFit):
        # Fidedignidade baseada na solução rotacionada da EFA
        l_std = fit.loadings_std.to_numpy(dtype=np.float64)
        phi = fit.factor_corr.to_numpy(dtype=np.float64)
        u2 = fit.uniquenesses.to_numpy(dtype=np.float64)

        # Ômega total do instrumento completo na EFA: 1' L Phi L' 1 / (1' (L Phi L' + U2) 1)
        common_cov = l_std @ phi @ l_std.T
        total_cov = common_cov + np.diag(u2)
        omega_instrument = float(np.sum(common_cov)) / max(1e-12, float(np.sum(total_cov)))

        rows = []
        factor_map = {}
        for j, f_name in enumerate(fit.factor_names):
            salient_mask = np.abs(l_std[:, j]) >= loading_cutoff
            if not np.any(salient_mask):
                salient_mask = np.argmax(np.abs(l_std), axis=1) == j
            items_j = [fit.var_names[i] for i in np.where(salient_mask)[0]]
            factor_map[f_name] = items_j

            l_sub = l_std[salient_mask, j]
            u_sub = u2[salient_mask]
            omega_f = float(np.sum(l_sub) ** 2) / max(1e-12, float(np.sum(l_sub) ** 2) + float(np.sum(u_sub)))
            ave_f = float(np.mean(l_sub ** 2))

            row_dict = {
                "construct": f_name,
                "n_items": len(items_j),
                "omega_total": omega_f,
                "omega_scale_total": omega_instrument,
                "ave": ave_f,
            }
            if include_alpha and data is not None and len(items_j) >= 2:
                a_df = cronbach_alpha(data[items_j], corr="polychoric" if fit.ordered else "pearson")
                row_dict["alpha_raw"] = float(a_df["alpha_raw"].iloc[0])
            rows.append(row_dict)

        htmt_df = htmt(data, factor_map, corr="polychoric" if fit.ordered else "pearson") if (data is not None and len(factor_map) > 1 and all(len(v) >= 2 for v in factor_map.values())) else None
        htmt2_df = htmt(data, factor_map, corr="polychoric" if fit.ordered else "pearson", geometric=True) if htmt_df is not None else None

        return ReliabilityResult(
            constructs=pd.DataFrame(rows),
            htmt_matrix=htmt_df,
            htmt2_matrix=htmt2_df,
            estimator=fit.estimator,
            ordered=fit.ordered,
        )

    raise TypeError("O argumento 'fit' deve ser um objeto AnvalaFit (av.cfa) ou EFAFit (av.efa).")