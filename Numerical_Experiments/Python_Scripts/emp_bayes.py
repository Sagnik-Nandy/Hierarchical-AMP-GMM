import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import scipy.linalg as sla
from scipy.linalg import block_diag
import importlib
import pandas as pd
from collections import namedtuple, defaultdict
from sklearn.mixture import GaussianMixture

pca_pack = importlib.import_module("pca_pack")
from pca_pack import MultiModalityPCA

class MultiModalityPCADiagnostics:
    """
    Performs normalization, singular value analysis, and principal component diagnostics for multimodal data.

    Methods
    -------
    normalize_obs(X_list, K_list):
        Normalize the noise level for each modality after regressing out top K PCs.
    normalize_pc(U_list):
        Normalize principal components for each modality.
    plot_pc(X_list, labels, K_list, nPCs=2, to_show=False, to_save=False):
        Plots singular values and PC distributions for each modality.
    """

    def normalize_obs(self, X_list, K_list):
        """
        Normalize the noise level of each modality after regressing out top K principal components.

        Parameters
        ----------
        X_list : list of ndarray
            List of m data matrices X_k of shape (n_samples, p_k).
        K_list : list of int
            List of m values specifying the number of principal components per modality.

        Returns
        -------
        X_normalized_list : list of ndarray
            List of normalized matrices.
        """
        if len(X_list) != len(K_list):
            raise ValueError("Mismatch: Number of modalities and number of K values must be the same.")

        X_normalized_list = []
        for k, (X_k, K_k) in enumerate(zip(X_list, K_list)):
            if K_k == 0:
                raise ValueError(f"Modality {k}: # Principal Components (K) cannot be zero.")

            n_features = X_k.shape[1]
            U, Lambda, Vh = np.linalg.svd(X_k, full_matrices=False)

            # Extract top-K components
            U_K = U[:, :K_k]
            Lambda_K = Lambda[:K_k]
            V_K = Vh[:K_k, :]

            # Compute residual matrix
            R = X_k - (U_K * Lambda_K) @ V_K
            tauSq = np.sum(R**2) / n_features

            # Normalize by estimated noise level
            X_normalized = X_k / np.sqrt(tauSq)
            X_normalized_list.append(X_normalized)

            print(f"[Modality {k}] Estimated noise std deviation: {np.sqrt(tauSq):.4f}")

        return X_normalized_list

    def normalize_pc(self, U_list):
        """
        Normalize the principal components for each modality.

        Parameters
        ----------
        U_list : list of ndarray
            List of principal component matrices U_k of shape (n_samples, r_k).

        Returns
        -------
        U_normalized_list : list of ndarray
            List of normalized PC matrices.
        """
        return [U_k / np.sqrt((U_k**2).sum(axis=0)) * np.sqrt(len(U_k)) for U_k in U_list]

    def plot_pc(self, X_list, labels, K_list, nPCs=2, to_show=False, to_save=False):
        """
        Plots singular values and PC distributions for each modality.

        Parameters
        ----------
        X_list : list of ndarray
            List of m data matrices X_k of shape (n_samples, p_k).
        labels : list of str
            List of modality names for labeling plots.
        K_list : list of int
            List of values specifying the number of principal components per modality.
        nPCs : int, optional
            Number of principal components to visualize (default is 2).
        to_show : bool, optional
            Whether to display the plots (default is False).
        to_save : bool, optional
            Whether to save the plots (default is False).
        """
        for k, (X_k, label, K_k) in enumerate(zip(X_list, labels, K_list)):
            print(f"\nPlotting PCA diagnostics for {label} (Modality {k})")

            U, S, Vh = np.linalg.svd(X_k, full_matrices=False)

            # Singular value spectrum
            plt.figure(figsize=(10, 3))
            plt.scatter(range(len(S)), S)
            plt.title(f'Singular Values - {label}')
            if to_save:
                plt.savefig(f'figures/singvals_{label}.png')
            if to_show:
                plt.show()
            plt.close()

            # PC distribution analysis
            for i in range(min(nPCs, K_k)):
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

                ax1.hist(U[:, i], bins=50)
                scipy.stats.probplot(U[:, i], plot=ax2)
                ax1.set_title(f'Left PC {i+1}')
                ax2.set_title(f'Q-Q Left PC {i+1}')

                ax3.hist(Vh[i, :], bins=50)
                scipy.stats.probplot(Vh[i, :], plot=ax4)
                ax3.set_title(f'Right PC {i+1}')
                ax4.set_title(f'Q-Q Right PC {i+1}')

                if to_save:
                    fig.savefig(f'figures/PC_{i}_{label}.png')
                if to_show:
                    plt.show()
                plt.close()

# A slimmed-down PCA pack for low-dimensional modalities
LowDimPcaPack = namedtuple("LowDimPCAPack", ["X", "U", "sample_aligns"])

class LowDimModalityLoadings:
    """
    Estimate loadings for low-dimensional modalities assuming X_k = L_k U_k + Z_k with Var(Z_k)=I.
    """

    def __init__(self, top_features=None):
        """
        Parameters
        ----------
        top_features : int or None
            Number of highly variable features to select per modality.
            If None, selects all features.
        """
        self.top_features = top_features
        self.indices_ = []       # list of index arrays of selected features
        self.means_ = []         # list of mean vectors for centering
        self.L_matrices_ = []    # list of estimated loading matrices
        self.pca_results = {}

    def fit(self, A_list):
        """
        Fit loading matrices for each low-dimensional modality.

        Parameters
        ----------
        A_list : list of ndarray, each of shape (n_samples, r_k)
            List of data matrices for each low-dimensional modality.

        Returns
        -------
        pca_results : dict
            Dictionary of PCA-like result objects for each low-dimensional modality.
        """
        # reset storage
        self.indices_.clear()
        self.means_.clear()
        self.L_matrices_.clear()
        self.pca_results = {}

        for A in A_list:
            p = A.shape[1]
            if self.top_features is None or self.top_features >= p:
                ind = np.arange(p)
            else:
                var_A = np.var(A, axis=0)
                ind = np.argpartition(var_A, -self.top_features)[-self.top_features:]
            self.indices_.append(ind)

            # 2) Subset and center
            A_sel = A[:, ind]
            mean_A = np.mean(A_sel, axis=0)
            A_sel_centered = A_sel - mean_A
            self.means_.append(mean_A)

            # 3) Compute empirical covariance
            cov_A = (1.0 / A_sel_centered.shape[0]) * (A_sel_centered.T @ A_sel_centered)

            # 4) Rough loading: subtract identity
            P = cov_A.shape[0]
            L_rough = cov_A - np.eye(P)

            # 5) Eigen-decompose and form loadings
            vals, vecs = np.linalg.eig(L_rough)
            print(f"Eigenvalues (before clipping): {vals}")
            vals_clipped = np.maximum(vals, 0)
            diag_sqrt = np.diag(np.sqrt(vals_clipped))
            L = vecs @ diag_sqrt

            self.L_matrices_.append(L)

            # 6) Build low-dimensional PCA-like pack
            U_k = A_sel_centered @ np.linalg.pinv(L)
            pca_pack = LowDimPcaPack(
                X=A_sel_centered,
                U=U_k,
                sample_aligns=L
            )
            self.pca_results[len(self.pca_results)] = pca_pack

        return self.pca_results

    def save_indices(self, file_paths):
        """
        Save selected feature indices for each modality to CSV files.

        Parameters
        ----------
        file_paths : list of str
            File paths to save each indices array.
        """
        if len(file_paths) != len(self.indices_):
            raise ValueError("Number of file paths must match number of modalities.")
        for ind, path in zip(self.indices_, file_paths):
            df = pd.DataFrame(ind, columns=["feature_index"])
            df.to_csv(path, index=False)

# =============================================================================
# Empirical Bayes Pipeline
# =============================================================================


def _symmetrize(A):
    return 0.5 * (A + A.T)

def _psd_clip(A, floor=1e-10):
    """Eigen-clip to enforce PSD with a small floor on eigenvalues."""
    A = _symmetrize(A)
    evals, evecs = np.linalg.eigh(A)
    evals_clipped = np.maximum(evals, floor)
    return (evecs * evals_clipped) @ evecs.T

def _solve_M_inv_A_M_invT(M, A):
    """
    Compute M^{-1} A M^{-T} without forming explicit inverses:
      1) Solve M X = A  for X
      2) Solve M Y = X^T for Y  => return Y^T
    Works for general square M (not necessarily SPD).
    """
    X = sla.solve(M, A, assume_a='gen', check_finite=False, overwrite_b=False)
    Y = sla.solve(M, X.T, assume_a='gen', check_finite=False, overwrite_b=False)
    return Y.T

def _as_blockdiag_or_matrix(obj):
    """Accept a list of blocks or a full matrix and return a full matrix."""
    if isinstance(obj, (list, tuple)):
        return block_diag(*obj)
    return obj


# =========================
# Parametric EB via GMM in observed space, pulled back to prior space
# =========================

class ParametricEB_GMM:
    """
    Empirical Bayes with a Gaussian Mixture prior inferred by fitting a GMM in observed space
    and pulling components back through X = M Z + eps,  eps ~ N(0, D).

    Parameters
    ----------
    n_components : int
        Maximum number of components if choose_comp=True; otherwise fixed K.
    max_iter : int
        Max EM iterations for sklearn GaussianMixture.
    covariance_mode : {'full', 'diagonal', 'isotropic-full', 'isotropic-pullback'}
        Parameterization of prior covariance S_k.
    choose_comp : bool
        If True, choose K by BIC over k in [1, n_components].
    reg_covar : float
        Regularizer added in sklearn GMM to improve stability.
    psd_floor : float
        Minimum eigenvalue used in PSD clipping.
    random_state : int or None
        RNG seed for reproducibility.
    """

    def __init__(self,
                 n_components=5,
                 max_iter=500,
                 covariance_mode='full',
                 choose_comp=False,
                 reg_covar=1e-6,
                 psd_floor=1e-10,
                 random_state=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_mode = covariance_mode
        self.choose_comp = choose_comp
        self.reg_covar = reg_covar
        self.psd_floor = psd_floor
        self.random_state = random_state

        # Learned (set in estimate_prior)
        self.M_bd = None
        self.D_bd = None
        self.gmm  = None

        self.m_prior   = None   # (K, P)
        self.cov_prior = None   # (K, P, P)
        self.weights   = None   # (K,)

    # ------------------------------------------------------------------
    # Core fitting
    # ------------------------------------------------------------------

    def estimate_prior(self, data, M_list_or_mat, D_list_or_mat):
        """
        Fit GMM in observed space and pull back means/covs to define the prior.

        Parameters
        ----------
        data : (N, P) array
        M_list_or_mat : list of (p_i x p_i) blocks OR a full (P x P) matrix
        D_list_or_mat : list of (p_i x p_i) blocks OR a full (P x P) matrix
        """
        M_bd = _as_blockdiag_or_matrix(M_list_or_mat).astype(np.float64)
        D_bd = _as_blockdiag_or_matrix(D_list_or_mat).astype(np.float64)

        if M_bd.shape[0] != M_bd.shape[1]:
            raise ValueError(f"M must be square, got {M_bd.shape}")
        if D_bd.shape[0] != D_bd.shape[1]:
            raise ValueError(f"D must be square, got {D_bd.shape}")
        if M_bd.shape != D_bd.shape:
            raise ValueError(f"M and D shapes must match; got {M_bd.shape} vs {D_bd.shape}")
        if data.shape[1] != M_bd.shape[0]:
            raise ValueError(f"data has P={data.shape[1]} but M,D are {M_bd.shape}")

        self.M_bd = M_bd
        self.D_bd = _psd_clip(D_bd, floor=self.psd_floor)
        P = self.M_bd.shape[0]

        # Fit GMM (optionally by BIC)
        if self.choose_comp:
            best_bic, best = np.inf, None
            for k in range(1, self.n_components + 1):
                cand = GaussianMixture(
                    n_components=k, covariance_type='full',
                    max_iter=self.max_iter, reg_covar=self.reg_covar,
                    random_state=self.random_state
                )
                cand.fit(data)
                bic = cand.bic(data)
                if bic < best_bic:
                    best_bic, best = bic, cand
            self.gmm = best
        else:
            self.gmm = GaussianMixture(
                n_components=self.n_components, covariance_type='full',
                max_iter=self.max_iter, reg_covar=self.reg_covar,
                random_state=self.random_state
            ).fit(data)

        K          = self.gmm.n_components
        means_obs  = self.gmm.means_        # (K, P)
        cov_obs    = self.gmm.covariances_  # (K, P, P)
        weights    = self.gmm.weights_      # (K,)

        m_prior   = np.zeros_like(means_obs)
        cov_prior = np.zeros((K, P, P), dtype=np.float64)

        # Use already-clipped D
        D_bd_clipped = self.D_bd
        M_bd_local   = self.M_bd

        for k in range(K):
            # Mean pullback:  mu_obs = M m  =>  m = solve(M, mu_obs)
            m_prior[k] = sla.solve(
                M_bd_local, means_obs[k],
                assume_a='gen', check_finite=False, overwrite_b=False
            )

            # Covariance pullback: cov_obs = M S M^T + D  =>  S = M^{-1}(cov_obs - D)M^{-T}
            # Symmetrize + PSD-clip once; reused by every covariance_mode branch below.
            cov_adj = _psd_clip(_symmetrize(cov_obs[k] - D_bd_clipped), floor=self.psd_floor)

            if self.covariance_mode == 'full':
                S_k = _psd_clip(_solve_M_inv_A_M_invT(M_bd_local, cov_adj), floor=self.psd_floor)

            elif self.covariance_mode == 'diagonal':
                S_full = _psd_clip(_solve_M_inv_A_M_invT(M_bd_local, cov_adj), floor=self.psd_floor)
                S_k = np.diag(np.maximum(np.diag(S_full), self.psd_floor))

            elif self.covariance_mode == 'isotropic-full':
                sigma2 = max(np.trace(cov_adj) / np.trace(M_bd_local @ M_bd_local.T), self.psd_floor)
                S_k = sigma2 * np.eye(P, dtype=np.float64)

            elif self.covariance_mode == 'isotropic-pullback':
                S_full = _psd_clip(_solve_M_inv_A_M_invT(M_bd_local, cov_adj), floor=self.psd_floor)
                sigma2 = max(np.trace(S_full) / P, self.psd_floor)
                S_k = sigma2 * np.eye(P, dtype=np.float64)

            else:
                raise ValueError(
                    "covariance_mode must be one of: "
                    "'full', 'diagonal', 'isotropic-full', 'isotropic-pullback'"
                )

            cov_prior[k] = _symmetrize(S_k)

        self.m_prior, self.cov_prior, self.weights = m_prior, cov_prior, weights
        return self.m_prior, self.cov_prior, self.weights

    # ------------------------------------------------------------------
    # Responsibilities
    # ------------------------------------------------------------------

    def _responsibilities(self, data, mu, cov):
        """
        W[i,k] = p(k | x_i) under the observed-space mixture induced by
        (m_prior, cov_prior, weights) pushed forward through X = M Z + eps.

        Parameters
        ----------
        data : (N, P)
        mu   : M matrix (list of blocks or full matrix)
        cov  : D matrix (list of blocks or full matrix)

        Returns
        -------
        W : (N, K) responsibility matrix
        """
        M_bd_local = _as_blockdiag_or_matrix(mu).astype(np.float64)
        D_bd_local = _psd_clip(
            _as_blockdiag_or_matrix(cov).astype(np.float64),
            floor=self.psd_floor
        )

        N, P = data.shape
        K    = self.m_prior.shape[0]

        logits = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            mu_k_obs  = self.m_prior[k] @ M_bd_local.T                                        # (P,)
            Sig_k_obs = _psd_clip(
                D_bd_local + M_bd_local @ self.cov_prior[k] @ M_bd_local.T,
                floor=self.psd_floor
            )                                                                                   # (P, P)

            inv_S = np.linalg.inv(Sig_k_obs)

            # slogdet returns (sign, logabsdet); only the log-determinant is needed here
            _, logdet = np.linalg.slogdet(Sig_k_obs)

            fsq = np.einsum("ij,ij->i", data @ inv_S, data) / 2   # (N,)  x^T S^{-1} x / 2
            zsq = (mu_k_obs @ inv_S @ mu_k_obs) / 2                # scalar  mu^T S^{-1} mu / 2
            fz  = data @ inv_S @ mu_k_obs                          # (N,)  x^T S^{-1} mu

            logits[:, k] = np.log(self.weights[k]) - 0.5 * logdet - fsq + fz - zsq

        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        logits  = np.clip(logits, -700, 700)
        W = np.exp(logits)
        W /= W.sum(axis=1, keepdims=True)
        return W   # (N, K)

    # ------------------------------------------------------------------
    # Denoiser  E[Z | x]
    # ------------------------------------------------------------------

    def denoise(self, data, mu, cov):
        """
        Posterior mean E[Z | x] using provided M (=mu) and D (=cov).

        Parameters
        ----------
        data : (N, P)
        mu   : M matrix
        cov  : D matrix

        Returns
        -------
        out : (N, P)  denoised latent estimates
        """
        M_bd_local = _as_blockdiag_or_matrix(mu).astype(np.float64)
        D_bd_local = _psd_clip(
            _as_blockdiag_or_matrix(cov).astype(np.float64),
            floor=self.psd_floor
        )

        N, P = data.shape
        K    = self.m_prior.shape[0]

        inv_D  = np.linalg.inv(D_bd_local)
        inv_C  = [np.linalg.pinv(_psd_clip(c, floor=self.psd_floor)) for c in self.cov_prior]

        W = self._responsibilities(data, mu=M_bd_local, cov=D_bd_local)   # (N, K)

        out = np.zeros((N, P), dtype=np.float64)
        for k in range(K):
            # Posterior precision:  A_k = M^T D^{-1} M + C_k^{-1}
            A_k   = _psd_clip(M_bd_local.T @ inv_D @ M_bd_local + inv_C[k], floor=self.psd_floor)
            Prec  = np.linalg.inv(A_k)   # (P, P) — symmetric

            # Posterior mean per component:
            #   E[Z | x, k] = Prec (M^T D^{-1} x + C_k^{-1} m_k)
            latent_mean = data @ inv_D @ M_bd_local @ Prec + self.m_prior[k] @ inv_C[k] @ Prec
            # shape: (N, P)

            out += W[:, [k]] * latent_mean

        return out

    # ------------------------------------------------------------------
    # Jacobian  d E[Z|x] / dx   shape (N, P, P)
    # ------------------------------------------------------------------

    def ddenoise(self, data, mu, cov):
        """
        Returns J[i,:,:] = Jacobian of the denoiser at x_i  (shape N x P x P),
        using the provided M (=mu) and D (=cov).

        The Jacobian of the GMM posterior mean follows the Tweedie/Stein identity
        for mixtures:

            d eta(x)/dx = sum_k w_k(x) T_k
                        + sum_k w_k(x) eta_k(x) s_k(x)^T
                        - eta(x) [sum_k w_k(x) s_k(x)]^T

        where:
            T_k   = Prec_k M^T D^{-1}           (P x P, constant per component)
            eta_k = E[Z | x, k]                  (N x P, per-component posterior mean)
            s_k   = Sig_k^{-1} (x - M m_k)      (N x P, per-component score)
            eta   = sum_k w_k eta_k              (N x P, full posterior mean)

        Parameters
        ----------
        data : (N, P)
        mu   : M matrix
        cov  : D matrix

        Returns
        -------
        J : (N, P, P)  J[i, a, b] = d eta_a(x_i) / d x_b
        """
        M_bd_local = _as_blockdiag_or_matrix(mu).astype(np.float64)
        D_bd_local = _psd_clip(
            _as_blockdiag_or_matrix(cov).astype(np.float64),
            floor=self.psd_floor
        )

        N, P = data.shape
        K    = self.m_prior.shape[0]

        inv_D = np.linalg.inv(D_bd_local)
        inv_C = [np.linalg.pinv(_psd_clip(c, floor=self.psd_floor)) for c in self.cov_prior]

        W = self._responsibilities(data, mu=M_bd_local, cov=D_bd_local)   # (N, K)

        # Pre-compute per-component quantities
        T_k       = np.zeros((K, P, P), dtype=np.float64)   # constant linear term per k
        eta_k     = np.zeros((K, N, P), dtype=np.float64)   # per-component posterior mean
        score_k   = np.zeros((K, N, P), dtype=np.float64)   # per-component score s_k(x)

        for k in range(K):
            A_k  = _psd_clip(M_bd_local.T @ inv_D @ M_bd_local + inv_C[k], floor=self.psd_floor)
            Prec = np.linalg.inv(A_k)   # (P, P)

            # T_k = Prec M^T D^{-1}  — the "linear" gradient of E[Z|x,k] wrt x
            T_k[k] = Prec @ M_bd_local.T @ inv_D   # (P, P)

            # eta_k[k] = E[Z | x, k]  (N, P)
            eta_k[k] = data @ inv_D @ M_bd_local @ Prec + self.m_prior[k] @ inv_C[k] @ Prec

            # Observed-space component covariance and its inverse
            Sig_k_obs = _psd_clip(
                D_bd_local + M_bd_local @ self.cov_prior[k] @ M_bd_local.T,
                floor=self.psd_floor
            )
            inv_Sig_k = np.linalg.inv(Sig_k_obs)

            # s_k(x) = Sig_k^{-1} (x - M m_k)   (N, P)
            mu_k_obs    = self.m_prior[k] @ M_bd_local.T   # (P,)
            score_k[k]  = (data - mu_k_obs) @ inv_Sig_k    # (N, P)

        # Full posterior mean  eta(x) = sum_k w_k eta_k   (N, P)
        eta = np.einsum("kn,knp->np", W.T, eta_k)   # (N, P)

        # Mixture score  s(x) = sum_k w_k s_k   (N, P)
        mix_score = np.einsum("kn,knp->np", W.T, score_k)   # (N, P)

        # Assemble Jacobian using the mixture Stein identity:
        #   J = sum_k w_k T_k
        #     + sum_k w_k  eta_k ⊗ s_k      (outer: eta_k is "output", s_k is "input grad")
        #     - eta ⊗ mix_score
        #
        # Convention: J[i, a, b] = d eta_a / d x_b
        #   => outer product is eta[:, :, None] * score[:, None, :]  i.e. (N, P_out, P_in)

        J = np.zeros((N, P, P), dtype=np.float64)

        # Term 1: weighted sum of constant linear terms  sum_k w_k T_k
        # T_k is (P, P), weight is (N,) — broadcast over samples
        J += np.einsum("nk,kab->nab", W, T_k)   # (N, P, P)

        # Term 2: sum_k w_k  eta_k(x) ⊗ s_k(x)
        for k in range(K):
            # eta_k[k]: (N, P),  score_k[k]: (N, P)
            # outer[i, a, b] = eta_k[k, i, a] * score_k[k, i, b]
            outer = eta_k[k][:, :, None] * score_k[k][:, None, :]   # (N, P, P)
            J += W[:, k][:, None, None] * outer

        # Term 3: - eta(x) ⊗ mix_score(x)
        J -= eta[:, :, None] * mix_score[:, None, :]   # (N, P, P)

        return J

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_denoisers(self):
        """
        Return fitted prior parameters and denoising functions.

        Returns
        -------
        eb_info : dict with keys:
            'prior'    : tuple (m_prior, cov_prior, weights)
            'denoise'  : callable (data, M, D) -> (N, P) denoised output
            'ddenoise' : callable (data, M, D) -> (N, P, P) Jacobian
        """
        return {
            "prior":    (self.m_prior, self.cov_prior, self.weights),
            "denoise":  lambda data, M, D: self.denoise(data, M, D),
            "ddenoise": lambda data, M, D: self.ddenoise(data, M, D),
        }


# =========================
# Cluster Parametric EB (GMM per cluster) + modality slicers
# =========================

class ClusterParametricEBPipeline:
    """
    Fits a ParametricEB_GMM per cluster over concatenated modalities in the cluster,
    then exposes cluster-level and modality-level denoisers/Jacobians.

    Attributes (after fit)
    ----------------------
    cluster_models   : dict[int -> ParametricEB_GMM]
    cluster_data     : dict[int -> (N, P_cluster) ndarray]
    cluster_M        : dict[int -> (P_cluster, P_cluster) ndarray]
    cluster_D        : dict[int -> (P_cluster, P_cluster) ndarray]
    modality_slices  : dict[int -> (cluster_id, start, end)]
    cluster_priors   : dict[int -> dict]  prior parameters per cluster
    cluster_denoisers: dict[int -> dict]  functional denoisers per cluster
    modality_denoisers: dict[int -> dict] functional denoisers per modality
    """

    def __init__(self,
                 covariance_mode='full',
                 choose_comp=False,
                 n_components=5,
                 max_iter=500,
                 reg_covar=1e-6,
                 psd_floor=1e-10,
                 random_state=0):
        self.covariance_mode = covariance_mode
        self.choose_comp     = choose_comp
        self.n_components    = n_components
        self.max_iter        = max_iter
        self.reg_covar       = reg_covar
        self.psd_floor       = psd_floor
        self.random_state    = random_state

        self.cluster_models    = {}
        self.cluster_data      = {}
        self.cluster_M         = {}
        self.cluster_D         = {}
        self.modality_slices   = {}
        self.cluster_denoisers  = {}
        self.modality_denoisers = {}
        self.cluster_priors     = {}

    # ------------------------------------------------------------------

    def _aggregate_by_cluster(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        """
        Concatenate modalities within each cluster and track per-modality slices.
        """
        cluster_data = defaultdict(list)
        cluster_M    = defaultdict(list)
        cluster_D    = defaultdict(list)

        for k, c in enumerate(cluster_labels):
            cluster_data[c].append(data_matrices[k].astype(np.float64))
            cluster_M[c].append(M_matrices[k].astype(np.float64))
            cluster_D[c].append(S_matrices[k].astype(np.float64))

        # Concatenate per cluster
        for c in cluster_data:
            Ns = [Xi.shape[0] for Xi in cluster_data[c]]
            if len(set(Ns)) != 1:
                raise ValueError(f"Sample size mismatch in cluster {c}: {Ns}")

            self.cluster_data[c] = np.concatenate(cluster_data[c], axis=1)
            self.cluster_M[c]    = block_diag(*cluster_M[c])
            self.cluster_D[c]    = block_diag(*cluster_D[c])

        # Build per-modality slices (start, end) within the cluster block
        running_offsets = defaultdict(int)
        for k, c in enumerate(cluster_labels):
            p_k   = data_matrices[k].shape[1]
            start = running_offsets[c]
            end   = start + p_k
            self.modality_slices[k] = (c, start, end)
            running_offsets[c]      = end

    # ------------------------------------------------------------------

    def fit(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        """
        Fit one ParametricEB_GMM per cluster.

        Parameters
        ----------
        data_matrices  : list of (N, p_k) arrays, one per modality
        M_matrices     : list of (p_k, p_k) arrays, one per modality
        S_matrices     : list of (p_k, p_k) noise covariance arrays, one per modality
        cluster_labels : list of int, cluster assignment per modality

        Returns
        -------
        self
        """
        if not (len(data_matrices) == len(M_matrices) == len(S_matrices) == len(cluster_labels)):
            raise ValueError("Length mismatch among inputs.")

        self.cluster_labels = cluster_labels
        self._aggregate_by_cluster(data_matrices, M_matrices, S_matrices, cluster_labels)

        for c in sorted(self.cluster_data.keys()):
            Xc = self.cluster_data[c]
            Mc = self.cluster_M[c]
            Dc = self.cluster_D[c]

            eb = ParametricEB_GMM(
                n_components=self.n_components,
                max_iter=self.max_iter,
                covariance_mode=self.covariance_mode,
                choose_comp=self.choose_comp,
                reg_covar=self.reg_covar,
                psd_floor=self.psd_floor,
                random_state=self.random_state,
            )
            eb.estimate_prior(Xc, Mc, Dc)
            eb.M_bd = Mc
            eb.D_bd = Dc

            self.cluster_models[c]    = eb
            self.cluster_denoisers[c] = eb.get_denoisers()

            # Store prior parameters
            gmm      = eb.gmm
            m_prior, cov_prior, weights = eb.m_prior, eb.cov_prior, eb.weights

            self.cluster_priors[c] = {
                "prior":    (m_prior.copy(), cov_prior.copy(), weights.copy()),
                "observed": {
                    "means":   np.array(gmm.means_,       copy=True),
                    "covs":    np.array(gmm.covariances_, copy=True),
                    "weights": np.array(gmm.weights_,     copy=True),
                },
                "gmm": gmm,
            }

        # Build modality-level functional closures
        for k, (c, start, end) in self.modality_slices.items():
            denoise_c  = self.cluster_denoisers[c]["denoise"]
            ddenoise_c = self.cluster_denoisers[c]["ddenoise"]

            def make_mod_denoise(denoise_c, start, end):
                def _f(Xc_full, M, D):
                    """Cluster-level denoiser, sliced to modality k's columns."""
                    return denoise_c(Xc_full, M, D)[:, start:end]
                return _f

            def make_mod_ddenoise(ddenoise_c, start, end):
                def _g(Xc_full, M, D):
                    """
                    Cluster-level Jacobian, diagonal block for modality k.

                    Returns J[:, start:end, start:end] — the sensitivity of
                    modality k's denoised output to its own input columns.
                    Cross-modality blocks (off-diagonal) are not returned here;
                    access the cluster-level ddenoise directly if needed.
                    """
                    Jc = ddenoise_c(Xc_full, M, D)   # (N, P_cluster, P_cluster)
                    return Jc[:, start:end, start:end]
                return _g

            self.modality_denoisers[k] = {
                "denoise":  make_mod_denoise(denoise_c, start, end),
                "ddenoise": make_mod_ddenoise(ddenoise_c, start, end),
            }

        return self