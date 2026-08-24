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

def cka_similarity(X, Y, sigma=1.0):
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


def dcor_similarity(X, Y):
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
        **kwargs
    ):
        """
        Perform hierarchical clustering on modalities.

        Converts the similarity matrix to distances via D = 1 - S, then runs
        scipy hierarchical clustering. Cluster count can be fixed, threshold-based,
        or auto-selected via silhouette score.

        Parameters
        ----------
        similarity_metric : str
            Metric to use if the matrix hasn't been computed yet.
            One of: "cka" (default), "dcor", "infonce_rff", "infonce_linear", "hss".
        num_clusters : int, optional
            Fixed number of clusters. Takes priority over threshold.
        threshold : float, optional
            Distance threshold for cutting the dendrogram.
        method : str
            Linkage method: "average", "complete", "ward", etc. Default "average".
        **kwargs
            Passed through to the similarity function.

        Returns
        -------
        cluster_labels : ndarray of shape (m,)
            Integer cluster labels (1-indexed, scipy convention).
        """
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

        # Auto-select k via silhouette score
        if np.allclose(distance_matrix, 0, atol=1e-3):
            return np.ones(distance_matrix.shape[0], dtype=int)

        best_score, best_k = -1, 2
        for k in range(2, min(10, distance_matrix.shape[0]) + 1):
            labels = sch.fcluster(self.linkage_matrix, k, criterion="maxclust")
            try:
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                if score > best_score:
                    best_score, best_k = score, k
            except Exception:
                continue

        return sch.fcluster(self.linkage_matrix, best_k, criterion="maxclust")

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