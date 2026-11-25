from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import pandas as pd

from anvala.polychoric import polychoric


def _near_pd_corr(mat: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Projeta uma matriz simétrica não-positiva-definida na matriz de correlação PD mais próxima."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.all(eigvals > tol):
        d = np.sqrt(np.diag(sym))
        return sym / np.outer(d, d)
    eigvals_clipped = np.maximum(eigvals, tol)
    pd_mat = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(pd_mat))
    cor = pd_mat / np.outer(d, d)
    np.fill_diagonal(cor, 1.0)
    return 0.5 * (cor + cor.T)


def _resolve_corr_matrix(
    data: pd.DataFrame,
    corr: str = "auto",
) -> tuple[np.ndarray, list[str], int, list[np.ndarray] | None]:
    """Calcula a matriz de correlação (policórica por padrão para ordinais, Pearson para contínuas)."""
    df_clean = data.dropna()
    var_names = list(df_clean.columns)
    n_obs = df_clean.shape[0]

    use_poly = False
    if corr == "polychoric":
        use_poly = True
    elif corr == "auto":
        is_ord = True
        for col in var_names:
            vals = df_clean[col].to_numpy()
            if not np.all(np.isclose(vals, np.round(vals))) or len(np.unique(vals)) > 7:
                is_ord = False
                break
        use_poly = is_ord

    thresholds_list = None
    if use_poly:
        poly_res = polychoric(df_clean)
        r_mat = poly_res.corr.to_numpy(dtype=np.float64)
        thresholds_list = poly_res.thresholds_list
    elif corr == "spearman":
        r_mat = df_clean.corr(method="spearman").to_numpy(dtype=np.float64)
    else:
        r_mat = df_clean.corr(method="pearson").to_numpy(dtype=np.float64)

    r_mat = _near_pd_corr(r_mat, tol=1e-4)
    return r_mat, var_names, n_obs, thresholds_list


def _glasso_single(
    s_mat: np.ndarray,
    lam: float,
    w_init: np.ndarray | None = None,
    theta_init: np.ndarray | None = None,
    max_iter: int = 150,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Graphical Lasso rápido com atualização incremental de resíduos (Friedman et al., 2008).
    """
    p = s_mat.shape[0]
    w = s_mat.copy() if w_init is None else w_init.copy()
    theta = np.linalg.inv(w) if theta_init is None else theta_init.copy()

    if lam <= 0.0:
        theta = np.linalg.inv(s_mat)
        return s_mat.copy(), 0.5 * (theta + theta.T)

    idx_all = np.arange(p)
    not_j_list = [idx_all[idx_all != j] for j in range(p)]

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            not_j = not_j_list[j]
            w_11 = w[np.ix_(not_j, not_j)]
            s_12 = s_mat[not_j, j]
            diag_w11 = np.diag(w_11)

            # Warm start para beta usando a linha atual de theta
            t_jj = theta[j, j]
            if t_jj > 1e-10:
                beta = -theta[not_j, j] / t_jj
            else:
                beta = np.zeros(p - 1, dtype=np.float64)

            resid = s_12 - w_11 @ beta
            for _ in range(80):
                max_diff = 0.0
                for k in range(p - 1):
                    b_old = beta[k]
                    w_kk = diag_w11[k]
                    r_k = resid[k] + w_kk * b_old
                    if r_k > lam:
                        b_new = (r_k - lam) / w_kk
                    elif r_k < -lam:
                        b_new = (r_k + lam) / w_kk
                    else:
                        b_new = 0.0

                    delta = b_new - b_old
                    if delta != 0.0:
                        beta[k] = b_new
                        resid -= w_11[:, k] * delta
                        ad = abs(delta)
                        if ad > max_diff:
                            max_diff = ad
                if max_diff < tol * 0.1:
                    break

            w_12 = w_11 @ beta
            w[not_j, j] = w_12
            w[j, not_j] = w_12

            theta_22 = 1.0 / max(1e-10, float(w[j, j] - w_12 @ beta))
            theta_12 = -beta * theta_22
            theta[not_j, j] = theta_12
            theta[j, not_j] = theta_12
            theta[j, j] = theta_22

        if np.max(np.abs(w - w_old)) < tol:
            break

    theta = 0.5 * (theta + theta.T)
    return w, theta


@dataclass
class GlassoResult:
    partial_corr: pd.DataFrame
    precision: pd.DataFrame
    lambda_opt: float
    ebic: float
    gamma: float
    n_edges: int
    sparsity: float

    def __repr__(self) -> str:
        return (
            f"GlassoResult(lambda_opt={self.lambda_opt:.4f}, ebic={self.ebic:.2f}, "
            f"n_edges={self.n_edges}, sparsity={self.sparsity:.2%})\n"
            f"{self.partial_corr.round(4)}"
        )


def ebic_glasso(
    data: pd.DataFrame,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
) -> GlassoResult:
    """
    Estima a rede de correlações parciais esparsas via EBICglasso (paridade com qgraph / EGAnet).
    """
    s_mat, var_names, n_obs, _ = _resolve_corr_matrix(data, corr=corr)
    p = s_mat.shape[0]

    off_diag = np.abs(s_mat - np.diag(np.diag(s_mat)))
    lam_max = float(np.max(off_diag))
    if lam_max <= 1e-10:
        pcor_df = pd.DataFrame(np.zeros((p, p)), index=var_names, columns=var_names)
        prec_df = pd.DataFrame(np.eye(p), index=var_names, columns=var_names)
        return GlassoResult(pcor_df, prec_df, 0.0, np.nan, gamma, 0, 1.0)

    lam_min = lambda_min_ratio * lam_max
    lambdas = np.geomspace(lam_max, lam_min, n_lambda)

    best_ebic = np.inf
    best_theta = np.eye(p, dtype=np.float64)
    best_lam = float(lambdas[0])
    best_edges = 0

    w_warm = s_mat.copy()
    theta_warm = np.linalg.inv(s_mat)
    tril_r, tril_c = np.tril_indices(p, k=-1)
    max_possible_edges = len(tril_r)

    for lam in lambdas:
        w_warm, theta_warm = _glasso_single(s_mat, float(lam), w_init=w_warm, theta_init=theta_warm)
        sign, logdet = np.linalg.slogdet(theta_warm)
        if sign <= 0 or not np.isfinite(logdet):
            continue

        loglik = 0.5 * float(n_obs) * (float(logdet) - float(np.trace(s_mat @ theta_warm)))
        n_edges = int(np.sum(np.abs(theta_warm[tril_r, tril_c]) > 1e-7))
        ebic_val = -2.0 * loglik + n_edges * np.log(float(n_obs)) + 4.0 * gamma * n_edges * np.log(float(p))

        if ebic_val < best_ebic:
            best_ebic = float(ebic_val)
            best_theta = theta_warm.copy()
            best_lam = float(lam)
            best_edges = n_edges

    d_prec = np.sqrt(np.maximum(1e-12, np.diag(best_theta)))
    pcor = -best_theta / np.outer(d_prec, d_prec)
    np.fill_diagonal(pcor, 0.0)
    pcor[np.abs(best_theta) <= 1e-7] = 0.0
    pcor = 0.5 * (pcor + pcor.T)

    pcor_df = pd.DataFrame(pcor, index=var_names, columns=var_names)
    prec_df = pd.DataFrame(best_theta, index=var_names, columns=var_names)
    sparsity = 1.0 - (best_edges / max(1, max_possible_edges))

    return GlassoResult(
        partial_corr=pcor_df,
        precision=prec_df,
        lambda_opt=best_lam,
        ebic=best_ebic,
        gamma=gamma,
        n_edges=best_edges,
        sparsity=float(sparsity),
    )


def _modularity(w_abs: np.ndarray, labels: np.ndarray) -> float:
    k = np.sum(w_abs, axis=1)
    two_m = float(np.sum(k))
    if two_m <= 0.0:
        return 0.0
    same_comm = (labels[:, None] == labels[None, :]).astype(np.float64)
    b_mat = w_abs - np.outer(k, k) / two_m
    return float(np.sum(b_mat * same_comm) / two_m)


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    out = np.zeros_like(labels, dtype=np.int64)
    next_id = 1
    for i, val in enumerate(labels):
        iv = int(val)
        if iv not in mapping:
            mapping[iv] = next_id
            next_id += 1
        out[i] = mapping[iv]
    return out


def _detect_walktrap(pcor: np.ndarray, steps: int = 4) -> np.ndarray:
    """Algoritmo Walktrap (Pons & Latapy, 2005; igraph::cluster_walktrap)."""
    w = np.abs(pcor.copy())
    np.fill_diagonal(w, 0.0)
    p = w.shape[0]
    deg = np.sum(w, axis=1)
    if np.any(deg <= 1e-12):
        for idx in np.where(deg <= 1e-12)[0]:
            w[idx, idx] = 1.0
        deg = np.sum(w, axis=1)

    trans = w / deg[:, None]
    p_t = np.linalg.matrix_power(trans, steps)
    inv_deg = 1.0 / deg

    communities: dict[int, list[int]] = {i: [i] for i in range(p)}
    prob_vecs: dict[int, np.ndarray] = {i: p_t[i].copy() for i in range(p)}
    labels = np.arange(p, dtype=np.int64)

    best_q = _modularity(np.abs(pcor), labels)
    best_labels = labels.copy()

    next_comm_id = p
    while len(communities) > 1:
        comm_ids = list(communities.keys())
        best_delta_sigma = np.inf
        best_pair = None

        has_adjacent = False
        for idx_a in range(len(comm_ids)):
            ca = comm_ids[idx_a]
            nodes_a = communities[ca]
            for idx_b in range(idx_a + 1, len(comm_ids)):
                cb = comm_ids[idx_b]
                nodes_b = communities[cb]
                adj = np.any(w[np.ix_(nodes_a, nodes_b)] > 0)
                if adj:
                    has_adjacent = True
                elif has_adjacent:
                    continue

                diff = prob_vecs[ca] - prob_vecs[cb]
                r2 = float(np.sum((diff * diff) * inv_deg))
                na, nb = float(len(nodes_a)), float(len(nodes_b))
                delta_sigma = (na * nb / (na + nb)) * r2 / float(p)
                if delta_sigma < best_delta_sigma:
                    best_delta_sigma = delta_sigma
                    best_pair = (ca, cb)

        if best_pair is None:
            break

        ca, cb = best_pair
        nodes_a = communities.pop(ca)
        nodes_b = communities.pop(cb)
        pa = prob_vecs.pop(ca)
        pb = prob_vecs.pop(cb)

        na, nb = len(nodes_a), len(nodes_b)
        merged_nodes = nodes_a + nodes_b
        merged_prob = (na * pa + nb * pb) / float(na + nb)

        communities[next_comm_id] = merged_nodes
        prob_vecs[next_comm_id] = merged_prob
        labels[merged_nodes] = next_comm_id
        next_comm_id += 1

        q_curr = _modularity(np.abs(pcor), labels)
        if q_curr > best_q:
            best_q = q_curr
            best_labels = labels.copy()

    return _relabel_sequential(best_labels)


def _detect_louvain(pcor: np.ndarray, seed: int = 42) -> np.ndarray:
    """Algoritmo Louvain (Blondel et al., 2008)."""
    rng = np.random.default_rng(seed)
    w = np.abs(pcor.copy())
    np.fill_diagonal(w, 0.0)
    p = w.shape[0]
    k = np.sum(w, axis=1)
    two_m = float(np.sum(k))
    if two_m <= 1e-12:
        return np.ones(p, dtype=np.int64)

    labels = np.arange(p, dtype=np.int64)
    improved = True
    while improved:
        improved = False
        order = rng.permutation(p)
        for i in order:
            curr_c = labels[i]
            k_i = k[i]
            neighbor_comms = np.unique(labels)
            best_c = curr_c
            best_gain = 0.0

            mask_curr = (labels == curr_c)
            mask_curr[i] = False
            k_i_in_curr = float(np.sum(w[i, mask_curr]))
            tot_curr = float(np.sum(k[mask_curr]))

            for c in neighbor_comms:
                if c == curr_c:
                    continue
                mask_c = (labels == c)
                k_i_in = float(np.sum(w[i, mask_c]))
                tot_c = float(np.sum(k[mask_c]))
                delta_q = ((k_i_in - k_i_in_curr) - k_i * (tot_c - tot_curr) / two_m) / two_m
                if delta_q > best_gain + 1e-12:
                    best_gain = delta_q
                    best_c = c

            if best_c != curr_c:
                labels[i] = best_c
                improved = True

    return _relabel_sequential(labels)


def _leading_eigenvalue_unidimensional(r_mat: np.ndarray) -> bool:
    """Verifica unidimensionalidade via Leading Eigenvalue (Christensen et al., 2023)."""
    w = np.abs(r_mat.copy())
    np.fill_diagonal(w, 0.0)
    k = np.sum(w, axis=1)
    two_m = float(np.sum(k))
    if two_m <= 1e-12:
        return True
    b_mat = w - np.outer(k, k) / two_m
    eigvals, eigvecs = np.linalg.eigh(b_mat)
    lead_val = float(eigvals[-1])
    if lead_val <= 1e-8:
        return True
    lead_vec = eigvecs[:, -1]
    labels = np.where(lead_vec >= 0, 1, 2)
    return len(np.unique(labels)) == 1


def compute_tefi(corr_mat: np.ndarray, memberships: np.ndarray) -> float:
    """
    Calcula o Total Entropy Fit Index (TEFI — paridade exata com EGAnet::tefi).
    """
    r_abs = np.abs(corr_mat)
    log_base = np.log(2.718282)

    def _matrix_entropy(sub_r: np.ndarray) -> float:
        p_sub = sub_r.shape[0]
        if p_sub <= 1:
            return 0.0
        rho = sub_r / float(p_sub)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rho = np.where(rho > 0.0, np.log(rho) / log_base, 0.0)
        return float(-np.trace(rho @ log_rho))

    h_vn = _matrix_entropy(r_abs)
    unique_comms = np.unique(memberships)
    m_comms = len(unique_comms)
    h_subs = [_matrix_entropy(r_abs[np.ix_(memberships == c, memberships == c)]) for c in unique_comms]

    mean_h_sub = float(np.mean(h_subs))
    sum_h_sub = mean_h_sub * float(m_comms)
    mean_h_vn = mean_h_sub - h_vn
    tefi = mean_h_vn + ((h_vn - sum_h_sub) * np.sqrt(float(m_comms)))
    return float(tefi)


def compute_net_loads(
    pcor_df: pd.DataFrame,
    memberships: np.ndarray,
    method: str = "revised",
    scaling: float = 2.0,
) -> pd.DataFrame:
    """
    Calcula as cargas padronizadas de rede (net.loads — paridade com EGAnet >= 2.0.7).
    """
    a_mat = pcor_df.to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(a_mat, 0.0)
    var_names = list(pcor_df.columns)
    unique_comms = np.sort(np.unique(memberships))
    p = len(var_names)
    m = len(unique_comms)

    unstd = np.zeros((p, m), dtype=np.float64)
    signs = np.ones(p, dtype=np.float64)
    comm_sizes = np.array([int(np.sum(memberships == c)) for c in unique_comms], dtype=np.float64)

    for col_idx, c in enumerate(unique_comms):
        idx_c = np.where(memberships == c)[0]
        n_c = len(idx_c)
        sub_a = a_mat[np.ix_(idx_c, idx_c)].copy()

        # Orienta sinais internos positivamente (obtain_signs)
        sub_signs = np.ones(n_c, dtype=np.float64)
        for _ in range(n_c * 2):
            row_sums = np.sum(sub_a, axis=1)
            min_i = int(np.argmin(row_sums))
            if row_sums[min_i] < -1e-12:
                sub_a[min_i, :] = -sub_a[min_i, :]
                sub_a[:, min_i] = -sub_a[:, min_i]
                sub_signs[min_i] = -sub_signs[min_i]
            else:
                break
        signs[idx_c] = sub_signs

        if n_c > 1:
            unstd[idx_c, col_idx] = (np.sum(sub_a, axis=0) / float(n_c - 1)) * float(n_c)

    comm_sums = np.sum(np.abs(unstd), axis=0)

    if m > 1:
        neg_nodes = np.where(signs == -1.0)[0]
        for neg in neg_nodes:
            a_mat[neg, :] = -a_mat[neg, :]
            a_mat[:, neg] = -a_mat[:, neg]

        for col_c, c in enumerate(unique_comms):
            idx_c = np.where(memberships == c)[0]
            for col_cross, cross in enumerate(unique_comms):
                if c != cross:
                    idx_cross = np.where(memberships == cross)[0]
                    unstd[idx_c, col_cross] = np.sum(a_mat[np.ix_(idx_cross, idx_c)], axis=0)

    unstd = unstd * signs[:, None]

    std_loads = np.zeros_like(unstd)
    for col_idx in range(m):
        if method == "revised":
            denom = comm_sums[col_idx] ** (1.0 / np.log(scaling * max(2.0, comm_sizes[col_idx])))
        else:
            denom = np.sqrt(np.sum(np.abs(unstd[:, col_idx])))
        if denom > 1e-12:
            std_loads[:, col_idx] = unstd[:, col_idx] / denom

    col_names = [f"Dim_{int(c)}" for c in unique_comms]
    return pd.DataFrame(std_loads, index=var_names, columns=col_names)


def _wto_matrix(pcor: np.ndarray) -> np.ndarray:
    """Weighted Topological Overlap (wTO) assinado (Christensen et al., 2023)."""
    w = pcor.copy()
    np.fill_diagonal(w, 0.0)
    abs_w = np.abs(w)
    strength = np.sum(abs_w, axis=1)

    common_neighbors = w @ w
    np.fill_diagonal(common_neighbors, 0.0)
    num = common_neighbors + w

    min_k = np.minimum(strength[:, None], strength[None, :])
    denom = min_k + 1.0 - abs_w
    wto = num / np.maximum(denom, 1e-12)
    np.fill_diagonal(wto, 1.0)
    return 0.5 * (wto + wto.T)


@dataclass
class UVAResult:
    wto_matrix: pd.DataFrame
    redundant_pairs: pd.DataFrame
    keep: list[str]
    remove: list[str]
    reduced_data: pd.DataFrame
    cut_off: float

    def __repr__(self) -> str:
        return (
            f"UVAResult(cut_off={self.cut_off}, n_keep={len(self.keep)}, n_remove={len(self.remove)})\n"
            f"Sugeridos para remoção: {self.remove}"
        )


def uva(
    data: pd.DataFrame,
    cut_off: float = 0.25,
    gamma: float = 0.5,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
) -> UVAResult:
    """
    Unique Variable Analysis (UVA — Christensen, Garrido, & Golino, 2023).
    """
    df_curr = data.dropna().copy()
    initial_glasso = ebic_glasso(df_curr, gamma=gamma, lambda_min_ratio=lambda_min_ratio, corr=corr)
    initial_wto = _wto_matrix(initial_glasso.partial_corr.to_numpy())
    var_names_init = list(df_curr.columns)
    wto_df_init = pd.DataFrame(initial_wto, index=var_names_init, columns=var_names_init)

    tril_r, tril_c = np.tril_indices(len(var_names_init), k=-1)
    pair_rows = []
    for r, c in zip(tril_r, tril_c):
        val = float(initial_wto[r, c])
        if abs(val) > cut_off:
            pair_rows.append({"item_1": var_names_init[c], "item_2": var_names_init[r], "wto": val})
    redundant_df = (
        pd.DataFrame(pair_rows).sort_values("wto", key=np.abs, ascending=False).reset_index(drop=True)
        if pair_rows
        else pd.DataFrame(columns=["item_1", "item_2", "wto"])
    )

    removed_items: list[str] = []
    while df_curr.shape[1] > 2:
        g_step = ebic_glasso(df_curr, gamma=gamma, lambda_min_ratio=lambda_min_ratio, corr=corr)
        wto_step = _wto_matrix(g_step.partial_corr.to_numpy())
        cols_step = list(df_curr.columns)
        p_step = len(cols_step)

        off_wto = np.abs(wto_step - np.eye(p_step))
        max_wto = float(np.max(off_wto))
        if max_wto <= cut_off:
            break

        idx_i, idx_j = np.unravel_index(int(np.argmax(off_wto)), off_wto.shape)
        others = [u for u in range(p_step) if u not in (idx_i, idx_j)]
        if len(others) > 0:
            mean_i = float(np.mean(off_wto[idx_i, others]))
            mean_j = float(np.mean(off_wto[idx_j, others]))
        else:
            mean_i = float(np.mean(off_wto[idx_i, :]))
            mean_j = float(np.mean(off_wto[idx_j, :]))

        drop_col = cols_step[idx_i] if mean_i >= mean_j else cols_step[idx_j]
        removed_items.append(drop_col)
        df_curr = df_curr.drop(columns=[drop_col])

    keep_items = list(df_curr.columns)
    return UVAResult(
        wto_matrix=wto_df_init,
        redundant_pairs=redundant_df,
        keep=keep_items,
        remove=removed_items,
        reduced_data=df_curr,
        cut_off=cut_off,
    )


@dataclass
class EGAResult:
    n_dim: int
    communities: pd.DataFrame
    network: pd.DataFrame
    glasso: GlassoResult
    loadings: pd.DataFrame
    tefi: float
    algorithm: str

    def partial_corr(self) -> pd.DataFrame:
        """Retorna a matriz de correlações parciais esparsas estimada pelo EBICglasso."""
        return self.network

    def net_loads(self) -> pd.DataFrame:
        """Retorna as cargas padronizadas de rede (Christensen & Golino, 2021)."""
        return self.loadings

    def __repr__(self) -> str:
        return (
            f"EGAResult(n_dim={self.n_dim}, algorithm='{self.algorithm}', tefi={self.tefi:.4f})\n"
            f"{self.communities.to_string(index=False)}"
        )


def ega(
    data: pd.DataFrame,
    algorithm: str = "walktrap",
    gamma: float = 0.5,
    lambda_min_ratio: float = 0.1,
    corr: str = "auto",
    steps: int = 4,
    unidimensional_check: bool = True,
) -> EGAResult:
    """
    Exploratory Graph Analysis (EGA — paridade com EGAnet 2.4.1).
    Por padrão utiliza corr='auto' (correlação policórica para itens ordinais, Pearson para contínuos).
    """
    r_mat, var_names, _, _ = _resolve_corr_matrix(data, corr=corr)
    glasso_res = ebic_glasso(data, gamma=gamma, lambda_min_ratio=lambda_min_ratio, corr=corr)
    pcor_mat = glasso_res.partial_corr.to_numpy(dtype=np.float64)

    alg_lower = algorithm.lower()
    if unidimensional_check and _leading_eigenvalue_unidimensional(r_mat):
        memberships = np.ones(len(var_names), dtype=np.int64)
    elif alg_lower == "walktrap":
        memberships = _detect_walktrap(pcor_mat, steps=steps)
    elif alg_lower == "louvain":
        memberships = _detect_louvain(pcor_mat)
    else:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use 'walktrap' ou 'louvain'.")

    n_dim = int(len(np.unique(memberships)))
    comm_df = pd.DataFrame({"item": var_names, "dimension": memberships})
    loads_df = compute_net_loads(glasso_res.partial_corr, memberships)
    tefi_val = compute_tefi(r_mat, memberships)

    return EGAResult(
        n_dim=n_dim,
        communities=comm_df,
        network=glasso_res.partial_corr,
        glasso=glasso_res,
        loadings=loads_df,
        tefi=tefi_val,
        algorithm=alg_lower,
    )


@dataclass
class BootEGAResult:
    empirical: EGAResult
    n_boot: int
    boot_type: str
    summary: pd.DataFrame
    dimension_frequencies: pd.DataFrame
    dimension_stability: float
    item_stability: pd.DataFrame
    typical_graph: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"BootEGAResult(n_boot={self.n_boot}, type='{self.boot_type}', "
            f"dimension_stability={self.dimension_stability:.2%})\n"
            f"{self.item_stability.round(3)}"
        )


def _ebic_glasso_batch_torch(
    s_batch: "torch.Tensor",
    n_obs: int,
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.1,
) -> np.ndarray:
    """
    Resolve EBICglasso em lote 3D [B, p, p] simultaneamente (GPU CUDA ou CPU)
    via ADMM matricial vetorizado (Boyd et al., 2011), sem la?os internos por vari?vel.
    """
    import torch

    b_size, p, _ = s_batch.shape
    dev = s_batch.device
    dtype = s_batch.dtype

    eye_p = torch.eye(p, device=dev, dtype=dtype).unsqueeze(0)
    off_mask = 1.0 - eye_p
    tril_mask = torch.tril(torch.ones((p, p), device=dev, dtype=dtype), diagonal=-1).unsqueeze(0)

    lam_max_b = torch.amax(torch.abs(s_batch * off_mask), dim=(1, 2))
    lam_max_b = torch.clamp(lam_max_b, min=1e-6)
    lam_min_b = lambda_min_ratio * lam_max_b

    t_grid = torch.linspace(0.0, 1.0, n_lambda, device=dev, dtype=dtype).unsqueeze(1)
    lambdas_grid = lam_max_b.unsqueeze(0) * ((lam_min_b / lam_max_b).unsqueeze(0) ** t_grid)

    rho = 2.0
    inv_rho = 1.0 / rho
    four_over_rho = 4.0 / rho

    theta = eye_p.expand(b_size, p, p).clone()
    z = theta.clone()
    u = inv_rho * (eye_p - s_batch)
    s_scaled = inv_rho * s_batch

    best_ebic = torch.full((b_size,), float("inf"), device=dev, dtype=dtype)
    best_theta = theta.clone()
    best_z = z.clone()

    log_n = float(np.log(float(n_obs)))
    four_gamma_log_p = 4.0 * gamma * float(np.log(float(p)))

    for l_idx in range(n_lambda):
        thresh = (lambdas_grid[l_idx] * inv_rho).view(b_size, 1, 1)

        for _ in range(25):
            m = z - u - s_scaled
            m = 0.5 * (m + m.transpose(1, 2))
            eigvals, eigvecs = torch.linalg.eigh(m)
            d_new = 0.5 * (eigvals + torch.sqrt(eigvals * eigvals + four_over_rho))
            theta = torch.bmm(eigvecs * d_new.unsqueeze(1), eigvecs.transpose(1, 2))

            v = theta + u
            z_off = torch.sign(v) * torch.clamp(torch.abs(v) - thresh, min=0.0)
            z = z_off * off_mask + v * eye_p
            u = u + theta - z

        theta_sym = 0.5 * (theta + theta.transpose(1, 2))
        sign, logdet = torch.linalg.slogdet(theta_sym)
        tr_st = torch.sum(s_batch * theta_sym, dim=(1, 2))
        loglik = 0.5 * float(n_obs) * (logdet - tr_st)

        n_edges = torch.sum((torch.abs(z) > 1e-6).to(dtype) * tril_mask, dim=(1, 2))
        ebic_val = -2.0 * loglik + n_edges * log_n + n_edges * four_gamma_log_p
        valid = (sign > 0) & torch.isfinite(ebic_val) & (ebic_val < best_ebic)

        best_ebic = torch.where(valid, ebic_val, best_ebic)
        valid_3d = valid.view(b_size, 1, 1)
        best_theta = torch.where(valid_3d, theta_sym, best_theta)
        best_z = torch.where(valid_3d, z, best_z)

    d_prec = torch.sqrt(torch.clamp(torch.diagonal(best_theta, dim1=1, dim2=2), min=1e-12))
    pcor = -best_theta / (d_prec.unsqueeze(2) * d_prec.unsqueeze(1))
    pcor = pcor * off_mask
    pcor = torch.where(torch.abs(best_z) <= 1e-6, torch.zeros_like(pcor), pcor)
    pcor = 0.5 * (pcor + pcor.transpose(1, 2))
    return pcor.detach().cpu().numpy()

def boot_ega(
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
) -> BootEGAResult:
    """
    Bootstrap Exploratory Graph Analysis (bootEGA — Christensen & Golino, 2021)
    com aceleração GPU (CUDA) em lote para reamostragem e EBICglasso.
    """
    import torch

    df_clean = data.dropna()
    emp_ega = ega(
        df_clean,
        algorithm=algorithm,
        gamma=gamma,
        lambda_min_ratio=lambda_min_ratio,
        corr=corr,
        steps=steps,
    )
    r_mat, var_names, n_obs, thresholds_list = _resolve_corr_matrix(df_clean, corr=corr)
    p = len(var_names)

    if device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (device.lower() == "cuda") and torch.cuda.is_available()

    if use_cuda and corr.lower() != "polychoric":
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        # Gera amostras e matrizes de correlação em chunks na GPU para controlar VRAM
        chunk_size = min(n_boot, 256)
        pcor_chunks: list[np.ndarray] = []
        corr_chunks: list[np.ndarray] = []

        r_t = torch.tensor(r_mat, device=dev, dtype=torch.float64)
        l_chol = torch.linalg.cholesky(r_t + 1e-8 * torch.eye(p, device=dev, dtype=torch.float64))
        x_raw_t = torch.tensor(df_clean[var_names].to_numpy(dtype=np.float64), device=dev, dtype=torch.float64)

        for start_b in range(0, n_boot, chunk_size):
            b_cur = min(chunk_size, n_boot - start_b)
            if boot_type == "parametric":
                z_sim = torch.randn((b_cur, n_obs, p), generator=gen, device=dev, dtype=torch.float64) @ l_chol.T
                if thresholds_list is not None:
                    ord_sim = torch.zeros((b_cur, n_obs, p), device=dev, dtype=torch.float64)
                    for j, taus in enumerate(thresholds_list):
                        tau_t = torch.tensor(taus, device=dev, dtype=torch.float64)
                        ord_sim[:, :, j] = torch.searchsorted(tau_t, z_sim[:, :, j].contiguous()).to(torch.float64) + 1.0
                    x_b = ord_sim
                else:
                    x_b = z_sim
            elif boot_type == "nonparametric":
                idx_rows = torch.randint(0, n_obs, (b_cur, n_obs), generator=gen, device=dev)
                x_b = x_raw_t[idx_rows]
            else:
                raise ValueError("boot_type deve ser 'parametric' ou 'nonparametric'.")

            if corr.lower() == "spearman":
                # Converte colunas em postos (ranks) na GPU
                order1 = torch.argsort(x_b, dim=1)
                ranks = torch.argsort(order1, dim=1).to(torch.float64)
                x_b = ranks

            x_cent = x_b - torch.mean(x_b, dim=1, keepdim=True)
            cov_b = torch.bmm(x_cent.transpose(1, 2), x_cent) / max(1.0, float(n_obs - 1))
            sd_b = torch.sqrt(torch.clamp(torch.diagonal(cov_b, dim1=1, dim2=2), min=1e-12))
            corr_b = cov_b / (sd_b.unsqueeze(2) * sd_b.unsqueeze(1))
            corr_b = 0.5 * (corr_b + corr_b.transpose(1, 2))
            corr_b = corr_b + 1e-4 * torch.eye(p, device=dev, dtype=torch.float64).unsqueeze(0)
            sd_reg = torch.sqrt(torch.diagonal(corr_b, dim1=1, dim2=2))
            corr_b = corr_b / (sd_reg.unsqueeze(2) * sd_reg.unsqueeze(1))

            corr_chunks.append(corr_b.detach().cpu().numpy())
            pcor_chunk = _ebic_glasso_batch_torch(
                corr_b,
                n_obs=n_obs,
                gamma=gamma,
                n_lambda=100,
                lambda_min_ratio=lambda_min_ratio,
            )
            pcor_chunks.append(pcor_chunk)

        corr_nets = np.concatenate(corr_chunks, axis=0)  # [n_boot, p, p]
        boot_nets = np.concatenate(pcor_chunks, axis=0)  # [n_boot, p, p]

        alg_lower = algorithm.lower()
        if alg_lower not in ("walktrap", "louvain"):
            raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use 'walktrap' ou 'louvain'.")

        def _comm_one(b_idx: int) -> np.ndarray:
            if _leading_eigenvalue_unidimensional(corr_nets[b_idx]):
                return np.ones(p, dtype=np.int64)
            if alg_lower == "walktrap":
                return _detect_walktrap(boot_nets[b_idx], steps=steps)
            return _detect_louvain(boot_nets[b_idx])

        boot_comms = np.vstack([_comm_one(b) for b in range(n_boot)])
    else:
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**32 - 1, size=n_boot)

        def _run_one(b_seed: int) -> tuple[np.ndarray, np.ndarray]:
            local_rng = np.random.default_rng(int(b_seed))
            if boot_type == "parametric":
                z_sim = local_rng.multivariate_normal(np.zeros(p), r_mat, size=n_obs)
                if thresholds_list is not None:
                    ord_sim = np.zeros((n_obs, p), dtype=np.int64)
                    for j, taus in enumerate(thresholds_list):
                        ord_sim[:, j] = np.searchsorted(taus, z_sim[:, j]) + 1
                    df_b = pd.DataFrame(ord_sim, columns=var_names)
                else:
                    df_b = pd.DataFrame(z_sim, columns=var_names)
            elif boot_type == "nonparametric":
                idx_rows = local_rng.integers(0, n_obs, size=n_obs)
                df_b = df_clean.iloc[idx_rows].reset_index(drop=True)
            else:
                raise ValueError("boot_type deve ser 'parametric' ou 'nonparametric'.")

            res_b = ega(
                df_b,
                algorithm=algorithm,
                gamma=gamma,
                lambda_min_ratio=lambda_min_ratio,
                corr=corr,
                steps=steps,
            )
            return res_b.communities["dimension"].to_numpy(), res_b.network.to_numpy()

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                boot_outputs = list(pool.map(_run_one, seeds))
        else:
            boot_outputs = [_run_one(s) for s in seeds]

        boot_comms = np.vstack([out[0] for out in boot_outputs])
        boot_nets = np.stack([out[1] for out in boot_outputs], axis=0)

    n_dims_vec = np.array([len(np.unique(row)) for row in boot_comms])
    unique_k, counts_k = np.unique(n_dims_vec, return_counts=True)
    freq_df = pd.DataFrame({
        "n_dim": unique_k,
        "frequency": counts_k,
        "proportion": counts_k / float(n_boot),
    })

    dim_stab = float(np.mean(n_dims_vec == emp_ega.n_dim))
    med_dim = float(np.median(n_dims_vec))
    se_dim = float(np.std(n_dims_vec, ddof=1)) if n_boot > 1 else 0.0
    ci_low, ci_high = np.percentile(n_dims_vec, [2.5, 97.5])

    summary_df = pd.DataFrame([{
        "n_boot": n_boot,
        "empirical_dim": emp_ega.n_dim,
        "median_dim": med_dim,
        "se_dim": se_dim,
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "dimension_stability": dim_stab,
    }])

    emp_labels = emp_ega.communities["dimension"].to_numpy()
    emp_unique = np.unique(emp_labels)
    aligned_counts = np.zeros((p, len(emp_unique)), dtype=np.float64)

    for b in range(n_boot):
        b_labels = boot_comms[b]
        for b_comm in np.unique(b_labels):
            b_mask = (b_labels == b_comm)
            best_emp_col = 0
            best_jaccard = -1.0
            for col_idx, e_comm in enumerate(emp_unique):
                e_mask = (emp_labels == e_comm)
                inter = float(np.sum(b_mask & e_mask))
                union = float(np.sum(b_mask | e_mask))
                jacc = inter / union if union > 0 else 0.0
                if jacc > best_jaccard:
                    best_jaccard = jacc
                    best_emp_col = col_idx
            aligned_counts[b_mask, best_emp_col] += 1.0

    item_probs = aligned_counts / float(n_boot)
    emp_col_indices = [int(np.where(emp_unique == c)[0][0]) for c in emp_labels]
    replicability = np.array([item_probs[i, emp_col_indices[i]] for i in range(p)])

    item_stab_df = pd.DataFrame(
        item_probs,
        index=var_names,
        columns=[f"Dim_{int(c)}" for c in emp_unique],
    )
    item_stab_df.insert(0, "empirical_dim", emp_labels)
    item_stab_df["stability"] = replicability

    typical_df = pd.DataFrame(np.median(boot_nets, axis=0), index=var_names, columns=var_names)

    return BootEGAResult(
        empirical=emp_ega,
        n_boot=n_boot,
        boot_type=boot_type,
        summary=summary_df,
        dimension_frequencies=freq_df,
        dimension_stability=dim_stab,
        item_stability=item_stab_df,
        typical_graph=typical_df,
    )