import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# ==============================================================================
# Kernel Utilities
# ==============================================================================

def center_kernel(K):
    """
    Double-centers a kernel matrix by removing row, column, and grand means.
    Equivalent to applying the centering matrix H = I - (1/n)11^T on both sides.
    """
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


# ==============================================================================
# Projection Utilities (for InfoNCE)
# ==============================================================================

def rff_embedding(X, n_components=256, sigma=1.0, seed=42):
    """
    Random Fourier Features (RFF) embedding approximating the RBF kernel map.

    Maps X into a finite-dimensional space Z such that <Z_i, Z_j> approx k(x_i, x_j),
    where k is the RBF kernel with bandwidth sigma. This is the natural projection
    for InfoNCE when the underlying similarity measure is kernel-based (CKA/HSS).

    Parameters
    ----------
    X : ndarray of shape (n, d)
    n_components : int
        Dimensionality of the embedding space.
    sigma : float
        RBF bandwidth. Should match the sigma used in CKA for consistency.
    seed : int

    Returns
    -------
    Z : ndarray of shape (n, n_components)
    """
    rng = np.random.RandomState(seed)
    omega = rng.randn(X.shape[1], n_components) / sigma
    bias = rng.uniform(0, 2 * np.pi, n_components)
    return np.cos(X @ omega + bias) * np.sqrt(2 / n_components)


def pca_procrustes_embedding(X, Y, n_components=None):
    """
    Linear projection via PCA + Procrustes alignment into a shared space.

    Projects each modality to its top PCs then aligns them via Procrustes so
    both live in a common coordinate system. Used as an alternative projection
    for InfoNCE when a linear approach is preferred.

    Parameters
    ----------
    X : ndarray of shape (n, d_x)
    Y : ndarray of shape (n, d_y)
    n_components : int or None
        Number of PCs to retain. Defaults to min(d_x, d_y, n).

    Returns
    -------
    X_proj : ndarray of shape (n, k)
    Y_proj : ndarray of shape (n, k)
    """
    n = X.shape[0]
    k = n_components or min(X.shape[1], Y.shape[1], n)

    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    _, _, Vx = np.linalg.svd(X_c, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y_c, full_matrices=False)

    X_proj = X_c @ Vx[:k].T   # (n, k)
    Y_proj = Y_c @ Vy[:k].T   # (n, k)

    # Procrustes: find rotation R that best aligns X_proj onto Y_proj
    U, _, Vt = np.linalg.svd(X_proj.T @ Y_proj)
    R = U @ Vt
    X_proj = X_proj @ R

    return X_proj, Y_proj


# ==============================================================================
# Similarity Measures
# ==============================================================================

def cka_similarity(X, Y, sigma=1.0, **kwargs):
    """
    Centered Kernel Alignment (CKA) with RBF kernel.

    Normalized biased HSIC: HSIC_b(X,Y) / sqrt(HSIC_b(X,X) * HSIC_b(Y,Y)),
    where HSIC_b = (1/m^2) trace(KHLH) from Gretton et al.
    Zero iff X and Y are independent under the kernel; bounded in [0, 1].

    Parameters
    ----------
    X : ndarray of shape (n, d_x)
    Y : ndarray of shape (n, d_y)
    sigma : float
        RBF bandwidth.

    Returns
    -------
    float in [0, 1]
    """
    K_X = rbf_kernel(X, gamma=1 / (2 * sigma ** 2))
    K_Y = rbf_kernel(Y, gamma=1 / (2 * sigma ** 2))

    K_Xc = center_kernel(K_X)
    K_Yc = center_kernel(K_Y)

    m = K_Xc.shape[0]
    hsic_xy = np.sum(K_Xc * K_Yc) / m ** 2
    hsic_xx = np.sum(K_Xc * K_Xc) / m ** 2
    hsic_yy = np.sum(K_Yc * K_Yc) / m ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 1e-10 else 0.0


def dcor_similarity(X, Y, **kwargs):
    """
    Distance Correlation (dCor).

    Normalized distance covariance: dCor = dCov(X,Y) / sqrt(dCov(X,X) * dCov(Y,Y)).
    Zero iff X and Y are independent; bounded in [0, 1].
    Kernel-free analog of CKA: uses Euclidean distances rather than kernel matrices,
    with double-centering playing the role of kernel centering.

    Parameters
    ----------
    X : ndarray of shape (n, d_x)
    Y : ndarray of shape (n, d_y)

    Returns
    -------
    float in [0, 1]
    """
    n = X.shape[0]

    D_X = pairwise_distances(X, metric="euclidean")
    D_Y = pairwise_distances(Y, metric="euclidean")

    def double_center(D):
        return D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()

    A = double_center(D_X)
    B = double_center(D_Y)

    dcov2_xy = np.sum(A * B) / n ** 2
    dcov2_xx = np.sum(A * A) / n ** 2
    dcov2_yy = np.sum(B * B) / n ** 2

    denom = np.sqrt(dcov2_xx * dcov2_yy)
    # dcov2_xy can be slightly negative due to floating point; clip before sqrt
    return np.sqrt(max(dcov2_xy, 0.0) / denom) if denom > 1e-10 else 0.0


def infonce_similarity(X, Y, projection="rff", temperature=0.1, n_components=256, sigma=1.0, seed=42):
    """
    InfoNCE-based similarity between two modalities.

    Projects both modalities into a shared embedding space then computes a
    symmetrized InfoNCE score. Higher values indicate stronger cross-modal
    alignment. Two projection strategies:

      "rff"    : Random Fourier Features approximating the RBF kernel map.
                 Principled choice — consistent with CKA/HSS since it operates
                 in the same RKHS. Each modality gets its own independent RFF
                 projection (different seeds), so the contrastive signal reflects
                 genuine cross-modal structure rather than shared randomness.

      "linear" : PCA per modality followed by Procrustes rotation into a shared
                 coordinate system. No kernel assumption; purely linear.

    InfoNCE loss (symmetrized):
        L = -0.5 * [ mean_i log softmax(z_i · W / tau)_i
                   + mean_i log softmax(W_i · z / tau)_i ]

    Similarity = exp(-L), so independent modalities (L large) → score near 0.

    Parameters
    ----------
    X : ndarray of shape (n, d_x)
    Y : ndarray of shape (n, d_y)
    projection : {"rff", "linear"}
    temperature : float
        Softmax temperature tau. Smaller = sharper contrast.
    n_components : int
        Embedding dimensionality.
    sigma : float
        RBF bandwidth for RFF projection.
    seed : int
        Base random seed for RFF.

    Returns
    -------
    float in [0, 1]
    """
    if projection == "rff":
        Z_X = rff_embedding(X, n_components=n_components, sigma=sigma, seed=seed)
        Z_Y = rff_embedding(Y, n_components=n_components, sigma=sigma, seed=seed+1)
    elif projection == "linear":
        Z_X, Z_Y = pca_procrustes_embedding(X, Y, n_components=n_components)
    else:
        raise ValueError("projection must be 'rff' or 'linear'.")

    # L2-normalize rows so dot product = cosine similarity
    Z_X = Z_X / (np.linalg.norm(Z_X, axis=1, keepdims=True) + 1e-10)
    Z_Y = Z_Y / (np.linalg.norm(Z_Y, axis=1, keepdims=True) + 1e-10)

    # Logit matrix: logits[i, j] = z_x_i · z_y_j / tau
    logits = Z_X @ Z_Y.T / temperature

    def log_softmax_diag(M):
        """Numerically stable log-softmax along rows, evaluated at the diagonal."""
        M_stable = M - M.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(M_stable).sum(axis=1))
        return M_stable.diagonal() - log_sum

    loss_xy = -log_softmax_diag(logits).mean()
    loss_yx = -log_softmax_diag(logits.T).mean()
    loss = 0.5 * (loss_xy + loss_yx)

    return float(np.exp(-loss))


def hss_similarity(Y1, Y2, sigma=1.0, graph="knn", k=1, epsilon=0.1):
    """
    Hilbert-Schmidt Similarity (HSS): symmetrized KMAc.

    Measures how well the kernel similarity structure of one modality aligns
    with the geometric neighborhood graph of the other. Symmetrized by
    averaging KMAc(Y1->Y2) and KMAc(Y2->Y1).

    Parameters
    ----------
    Y1, Y2 : ndarray of shape (n, d)
    sigma : float
        RBF bandwidth.
    graph : {"knn", "epsilon"}
        Graph construction mode.
    k : int
        Neighbors for knn graph.
    epsilon : float
        Radius for epsilon graph.

    Returns
    -------
    float
    """
    kmac_1 = _compute_kmac(Y1, Y2, sigma=sigma, graph=graph, k=k, epsilon=epsilon)
    kmac_2 = _compute_kmac(Y2, Y1, sigma=sigma, graph=graph, k=k, epsilon=epsilon)
    return 0.5 * (kmac_1 + kmac_2)


# ==============================================================================
# KMAc (internal helper for HSS)
# ==============================================================================

def _compute_kmac(Y_ker, Y_gr, *, sigma=1.0, graph="knn", k=1, epsilon=0.1):
    """
    Kernel Measure of Association (KMAc): compares kernel similarity on Y_ker
    against the neighborhood graph structure on Y_gr.
    """
    if Y_ker.shape[0] != Y_gr.shape[0]:
        raise ValueError("Y_ker and Y_gr must have the same number of rows.")
    n = Y_ker.shape[0]
    if n < 2:
        return 0.0

    K_matrix = rbf_kernel(Y_ker, Y_ker, gamma=1.0 / (2.0 * sigma ** 2))
    adj_list = {i: set() for i in range(n)}

    if graph == "knn":
        k_eff = max(0, min(int(k), n - 1))
        if k_eff == 0:
            return 0.0
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(Y_gr)
        _, indices = nn.kneighbors(Y_gr)
        for i in range(n):
            for j in indices[i, 1:]:
                adj_list[i].add(int(j))
                adj_list[int(j)].add(i)
    elif graph == "epsilon":
        rad = NearestNeighbors(radius=epsilon).fit(Y_gr)
        for i, nbrs in enumerate(rad.radius_neighbors(Y_gr, return_distance=False)):
            for j in nbrs:
                if j != i:
                    adj_list[i].add(int(j))
                    adj_list[int(j)].add(i)
    else:
        raise ValueError("graph must be 'knn' or 'epsilon'.")

    degrees = np.array([len(adj_list[i]) for i in range(n)], dtype=int)
    degrees_safe = np.where(degrees == 0, 1, degrees)

    local_kernel_sums = [
        sum(K_matrix[i, j] for j in adj_list[i]) / degrees_safe[i]
        for i in range(n) if degrees[i] > 0
    ]

    if not local_kernel_sums:
        return 0.0

    first_term = float(np.mean(local_kernel_sums))
    second_term = (np.sum(K_matrix) - np.trace(K_matrix)) / (n * (n - 1))
    numerator = first_term - second_term

    total_offdiag = np.sum(K_matrix) - np.trace(K_matrix)
    kernel_variance = 1.0 - (total_offdiag / (n * (n - 1)))

    return numerator / kernel_variance if kernel_variance != 0 else 0.0


# ==============================================================================
# Similarity Registry
# ==============================================================================

SIMILARITY_FUNCTIONS = {
    "cka":            cka_similarity,
    "dcor":           dcor_similarity,
    "infonce_rff":    lambda X, Y, **kw: infonce_similarity(X, Y, projection="rff", **kw),
    "infonce_linear": lambda X, Y, **kw: infonce_similarity(X, Y, projection="linear", **kw),
    "hss":            hss_similarity,
}


# ==============================================================================
# ModalityClusterer
# ==============================================================================

class ModalityClusterer:
    """
    Computes pairwise similarity matrices and performs hierarchical clustering
    over a collection of aligned data modalities.

    Each modality is an ndarray of shape (n, d_k) where n is the shared sample
    count and d_k is the modality-specific feature dimension.

    Supported similarity metrics
    ----------------------------
    "cka"            : Centered Kernel Alignment (default). Normalized biased HSIC
                       with RBF kernel per Gretton et al.; bounded in [0, 1];
                       zero iff independent.
    "dcor"           : Distance Correlation. Normalized distance covariance;
                       bounded in [0, 1]; zero iff independent. Kernel-free analog of CKA.
    "infonce_rff"    : InfoNCE with Random Fourier Feature projection. Projects
                       modalities into the RBF RKHS approximation before computing
                       contrastive alignment. Principled complement to CKA.
    "infonce_linear" : InfoNCE with PCA + Procrustes projection. Linear alternative;
                       no kernel assumption.
    "hss"            : Hilbert-Schmidt Similarity. Symmetrized KMAc comparing kernel
                       similarity against geometric neighborhood structure.

    Parameters
    ----------
    modalities : list of ndarray
        List of m data matrices, each of shape (n, d_k).
    """

    def __init__(self, modalities):
        self.modalities = modalities
        self.similarity_matrix = None
        self.linkage_matrix = None

    def compute_similarity_matrix(self, similarity_metric="cka", **kwargs):
        """
        Compute the m×m pairwise similarity matrix between modalities.

        Parameters
        ----------
        similarity_metric : str
            One of: "cka" (default), "dcor", "infonce_rff", "infonce_linear", "hss".
        **kwargs
            Passed through to the chosen similarity function
            (e.g. sigma, temperature, n_components).

        Returns
        -------
        similarity_matrix : ndarray of shape (m, m)
        """
        if similarity_metric not in SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Unknown metric '{similarity_metric}'. "
                f"Choose from: {list(SIMILARITY_FUNCTIONS.keys())}"
            )

        fn = SIMILARITY_FUNCTIONS[similarity_metric]
        m = len(self.modalities)
        S = np.zeros((m, m))

        for i in range(m):
            S[i, i] = 1.0  # self-similarity is always 1
            for j in range(i + 1, m):
                try:
                    sim = fn(self.modalities[i], self.modalities[j], **kwargs)
                    S[i, j] = sim
                    S[j, i] = sim
                except Exception as e:
                    print(f"Warning: similarity({i},{j}) failed — {e}. Setting to 0.")

        self.similarity_matrix = S
        return self.similarity_matrix

    def cluster_modalities(
        self,
        similarity_metric="cka",
        num_clusters=None,
        threshold=None,
        method="average",
        auto_method="silhouette",
        gap_B=10,
        gap_K_max=10,
        gap_seed=42,
        gap_subsample=None,
        **kwargs
    ):
        """
        Perform hierarchical clustering on modalities.

        Converts the similarity matrix to distances via D = 1 - S, then runs
        scipy hierarchical clustering. Cluster count can be fixed, threshold-based,
        or auto-selected.

        Parameters
        ----------
        similarity_metric : str
            Metric to use if the matrix hasn't been computed yet.
            One of: "cka" (default), "dcor", "infonce_rff", "infonce_linear", "hss".
        num_clusters : int, optional
            Fixed number of clusters. Takes priority over threshold and auto_method.
        threshold : float, optional
            Distance threshold for cutting the dendrogram. Takes priority over auto_method.
        method : str
            Linkage method: "average", "complete", "ward", etc. Default "average".
        auto_method : str
            Auto-selection strategy when num_clusters and threshold are both None.
            "silhouette" (default) — maximise silhouette score over k in [2, K_max].
                                     Cannot select k=1.
            "gap"                  — Gap statistic (Tibshirani 2001) over k in [1, K_max].
                                     Can select k=1 (all modalities in one cluster).
        gap_B : int
            Number of reference permutations for the Gap statistic. Default 10.
        gap_K_max : int
            Maximum k to consider for auto-selection. Default 10.
        gap_seed : int
            Random seed for Gap statistic reference sampling. Default 42.
        gap_subsample : int or None
            If set, each reference run draws this many rows (without replacement)
            from the permuted modalities before recomputing similarities. Reduces
            reference cost from O(n²) to O(s²) per pair, where s = gap_subsample.
            The observed similarity matrix and clustering are always computed on
            the full data. Recommended range: 500–2000. Default None (use all rows).
        **kwargs
            Passed through to the similarity function.

        Returns
        -------
        cluster_labels : ndarray of shape (m,)
            Integer cluster labels (1-indexed, scipy convention).
        """
        if method == "ward":
            raise ValueError(
                "Ward linkage requires Euclidean distances and is not valid here. "
                "Distances are derived from 1 - similarity and are not guaranteed "
                "Euclidean. Use 'average', 'complete', or 'single' instead."
            )

        if self.similarity_matrix is None:
            self.compute_similarity_matrix(similarity_metric, **kwargs)

        S = np.clip(self.similarity_matrix, 0.0, 1.0)
        distance_matrix = 1.0 - S
        np.fill_diagonal(distance_matrix, 0.0)
        np.clip(distance_matrix, 0.0, None, out=distance_matrix)  # guard floating point negatives

        condensed = sch.distance.squareform(distance_matrix)
        self.linkage_matrix = sch.linkage(condensed, method=method)

        if num_clusters is not None:
            return sch.fcluster(self.linkage_matrix, num_clusters, criterion="maxclust")

        if threshold is not None:
            return sch.fcluster(self.linkage_matrix, threshold, criterion="distance")

        m = distance_matrix.shape[0]

        # ------------------------------------------------------------------ #
        # Auto-selection: silhouette (default) or Gap statistic              #
        # ------------------------------------------------------------------ #

        if np.allclose(distance_matrix, 0, atol=1e-3):
            print("Auto k-selection: all distances ≈ 0 → k=1")
            return np.ones(m, dtype=int)

        if auto_method == "gap":
            best_k = self._select_k_gap(
                distance_matrix, similarity_metric=similarity_metric,
                method=method, K_max=gap_K_max, B=gap_B, seed=gap_seed,
                subsample=gap_subsample, **kwargs
            )
            print(f"Gap statistic selected k={best_k}")
            return sch.fcluster(self.linkage_matrix, best_k, criterion="maxclust")

        # Default: silhouette (k >= 2 only)
        best_score, best_k = -1, 2
        for k in range(2, min(gap_K_max, m) + 1):
            labels = sch.fcluster(self.linkage_matrix, k, criterion="maxclust")
            try:
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                if score > best_score:
                    best_score, best_k = score, k
            except Exception:
                continue

        print(f"Silhouette selected k={best_k} (score={best_score:.4f})")
        return sch.fcluster(self.linkage_matrix, best_k, criterion="maxclust")

    def _select_k_gap(self, distance_matrix, similarity_metric="cka",
                      method="average", K_max=10, B=10, seed=42,
                      subsample=None, **sim_kwargs):
        """
        Gap statistic for hierarchical clustering on a precomputed distance matrix.

        Supports k=1 (unlike silhouette). Reference distribution follows the
        Tibshirani et al. (2001) principle directly: generate B null datasets that
        have the same marginal structure as the observed data but no cluster signal,
        then compare observed within-cluster dispersion against the null baseline.

        Here the "data" are the raw modality matrices in self.modalities. Independently
        permuting the rows of each modality destroys every source of cross-modal
        similarity (CKA, dCor, HSS all collapse to noise) while preserving each
        modality's own feature distribution exactly. The similarity matrix is then
        recomputed from scratch for each permuted dataset — the same pipeline used
        on the observed data — so the reference W_k is on the same scale as observed.

        K_max is capped at m-1 so eval_K = K_max+1 never exceeds m; requesting
        m+1 clusters from m points would make the boundary gap values degenerate.

        Selection rule: smallest k where Gap(k) >= Gap(k+1) - s_{k+1}.
        Falls back to argmax Gap over [1, K_max] if no such k exists.
        """
        rng = np.random.default_rng(seed)
        m = distance_matrix.shape[0]
        K_max = min(K_max, m - 1)
        if K_max < 1:
            return 1

        eval_K = K_max + 1
        fn = SIMILARITY_FUNCTIONS[similarity_metric]
        n_samples = self.modalities[0].shape[0]
        # subsample size for reference runs; clip to [1, n_samples]
        ref_n = n_samples if subsample is None else max(1, min(int(subsample), n_samples))

        def within_dispersion(D, labels):
            """W_k = sum_c (1/(2*n_c)) * sum_{i,j in c} D_{ij}"""
            W = 0.0
            for c in np.unique(labels):
                idx = np.where(labels == c)[0]
                n_c = len(idx)
                if n_c < 2:
                    continue
                sub = D[np.ix_(idx, idx)]
                W += sub.sum() / (2.0 * n_c)
            return W

        # When subsampling is active the reference similarities are computed on
        # ref_n rows. CKA and dCor both carry finite-sample bias, so observed
        # similarities (full n) and reference similarities (ref_n) would live on
        # different scales, biasing Gap downward. Fix: subsample the observed
        # modalities to the same ref_n once (fixed draw, not per-B) so both
        # sides of Gap(k) = E[log W_ref] - log W_obs use the same sample size.
        if ref_n < n_samples:
            obs_idx = rng.choice(n_samples, size=ref_n, replace=False)
            obs_sub = [mod[obs_idx] for mod in self.modalities]
            S_obs_sub = np.zeros((m, m))
            for i in range(m):
                S_obs_sub[i, i] = 1.0
                for j in range(i + 1, m):
                    try:
                        sim = fn(obs_sub[i], obs_sub[j], **sim_kwargs)
                        S_obs_sub[i, j] = sim
                        S_obs_sub[j, i] = sim
                    except Exception:
                        pass
            D_obs_gap = np.clip(1.0 - np.clip(S_obs_sub, 0.0, 1.0), 0.0, None)
            np.fill_diagonal(D_obs_gap, 0.0)
            linkage_obs_gap = sch.linkage(sch.distance.squareform(D_obs_gap), method=method)
        else:
            D_obs_gap = distance_matrix
            linkage_obs_gap = self.linkage_matrix

        # Observed W_k (up to K_max+1 so the 1-SE rule can check k=K_max)
        log_W_obs = np.zeros(eval_K)
        for k in range(1, eval_K + 1):
            labs = sch.fcluster(linkage_obs_gap, k, criterion="maxclust")
            log_W_obs[k - 1] = np.log(within_dispersion(D_obs_gap, labs) + 1e-10)

        # Reference W_k: independently permute each modality's rows, subsample
        # to ref_n rows (same size as the observed path above), recompute the
        # full similarity → distance pipeline, then cluster.
        log_W_ref = np.zeros((B, eval_K))
        for b in range(B):
            perm_indices = [rng.permutation(n_samples)[:ref_n] for _ in self.modalities]
            permuted = [
                mod[idx] for mod, idx in zip(self.modalities, perm_indices)
            ]
            S_ref = np.zeros((m, m))
            for i in range(m):
                S_ref[i, i] = 1.0
                for j in range(i + 1, m):
                    try:
                        sim = fn(permuted[i], permuted[j], **sim_kwargs)
                        S_ref[i, j] = sim
                        S_ref[j, i] = sim
                    except Exception:
                        pass
            D_ref = np.clip(1.0 - np.clip(S_ref, 0.0, 1.0), 0.0, None)
            np.fill_diagonal(D_ref, 0.0)
            cond_ref = sch.distance.squareform(D_ref)
            Z_ref = sch.linkage(cond_ref, method=method)
            for k in range(1, eval_K + 1):
                labs_ref = sch.fcluster(Z_ref, k, criterion="maxclust")
                log_W_ref[b, k - 1] = np.log(within_dispersion(D_ref, labs_ref) + 1e-10)

        gap = log_W_ref.mean(axis=0) - log_W_obs
        sdk = log_W_ref.std(axis=0, ddof=1) if B > 1 else np.zeros(eval_K)
        sk  = sdk * np.sqrt(1.0 + 1.0 / B)

        # gap[k-1]=Gap(k), gap[k]=Gap(k+1), sk[k]=s_{k+1}
        for k in range(1, K_max + 1):
            if gap[k - 1] >= gap[k] - sk[k]:
                return k
        return int(np.argmax(gap[:K_max])) + 1  # fallback: argmax over [1, K_max]

    def plot_dendrogram(self, labels=None, title="Modality Clustering Dendrogram"):
        """
        Plot the hierarchical clustering dendrogram.

        Parameters
        ----------
        labels : list of str, optional
            Modality names. Defaults to integer indices.
        title : str, optional
            Plot title.
        """
        if self.linkage_matrix is None:
            raise ValueError("Run cluster_modalities() before plotting.")

        plt.figure(figsize=(10, 5))
        sch.dendrogram(self.linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=12)
        plt.title(title)
        plt.xlabel("Modalities")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
