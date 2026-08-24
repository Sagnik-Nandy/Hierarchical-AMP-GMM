import numpy as np
import scipy as sp
import scipy.linalg as sla
from scipy.linalg import block_diag
from collections import defaultdict
from sklearn.mixture import GaussianMixture

print("[emp_bayes_data] LOADED", flush=True)
__version__ = "dev-2025-09-28-1"

# =========================
# Utilities (numerical hygiene)
# =========================
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

def _as_blockdiag_or_matrix(obj, print_diagonals=False):
    """
    Accept a list of blocks or a full matrix and return a full matrix.
    If 'obj' is a list/tuple of matrices, print the diagonal entries of each.
    """
    if isinstance(obj, (list, tuple)):
        for i, mat in enumerate(obj):
            arr = np.asarray(mat)
            diag = np.diag(arr)
            if print_diagonals:
                print(f"Block {i}: shape={arr.shape}, diag={np.round(diag, 6)}", flush=True)
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
    covariance_mode : {'full','diagonal','isotropic'}
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
        self.gmm = None

        self.m_prior = None      # (K, P)
        self.cov_prior = None    # (K, P, P)
        self.weights = None      # (K,)

    # ---- Core fitting ----
    def estimate_prior(self, data, M_list_or_mat, D_list_or_mat):
        """
        Fit GMM in observed space and pull back means/covs to define the prior.

        Parameters
        ----------
        data : (N, P) array
        M_list_or_mat : list of (p_i x p_i) blocks OR a full (P x P) matrix
        D_list_or_mat : list of (p_i x p_i) blocks OR a full (P x P) matrix
        """
        # Accept list-of-blocks or full matrices
        M_bd = _as_blockdiag_or_matrix(M_list_or_mat).astype(np.float64)
        D_bd = _as_blockdiag_or_matrix(D_list_or_mat).astype(np.float64)

        # Basic shape checks
        if M_bd.shape[0] != M_bd.shape[1]:
            raise ValueError(f"M must be square, got {M_bd.shape}")
        if D_bd.shape[0] != D_bd.shape[1]:
            raise ValueError(f"D must be square, got {D_bd.shape}")
        if M_bd.shape != D_bd.shape:
            raise ValueError(f"M and D shapes must match; got {M_bd.shape} vs {D_bd.shape}")
        if data.shape[1] != M_bd.shape[0]:
            raise ValueError(f"data has P={data.shape[1]} but M,D are {M_bd.shape}")

        # Store (with light hygiene on D)
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

        K = self.gmm.n_components
        means_obs = self.gmm.means_
        cov_obs   = self.gmm.covariances_
        weights   = self.gmm.weights_

        # Pull back to prior space
        m_prior   = np.zeros_like(means_obs)       # (K, P)
        cov_prior = np.zeros((K, P, P), dtype=np.float64)

        D_bd = _psd_clip(self.D_bd, floor=self.psd_floor)
        M_bd = self.M_bd

        for k in range(K):
            # Mean:  mu_obs = M m  => m = solve(M, mu_obs)
            m_prior[k] = sla.solve(M_bd, means_obs[k], assume_a='gen',
                                   check_finite=False, overwrite_b=False)

            # Covariance: cov_obs = M S M^T + D  => S = M^{-1}(cov_obs - D)M^{-T}
            cov_adj = _symmetrize(cov_obs[k] - D_bd)
            cov_adj = _psd_clip(cov_adj, floor=self.psd_floor)

            if self.covariance_mode == 'full':
                S_k = _solve_M_inv_A_M_invT(M_bd, cov_adj)
                S_k = _psd_clip(S_k, floor=self.psd_floor)
            elif self.covariance_mode == 'diagonal':
                S_full = _psd_clip(_solve_M_inv_A_M_invT(M_bd, cov_adj), floor=self.psd_floor)
                S_k = np.diag(np.clip(np.diag(S_full), self.psd_floor, np.inf))
            elif self.covariance_mode == 'isotropic-full':
                sigma2  = max(np.trace(cov_adj) / np.trace(M_bd @ M_bd.T), self.psd_floor)
                S_k     = sigma2* np.eye(P, dtype=np.float64)
            elif self.covariance_mode == 'isotropic-pullback':
                cov_adj = cov_obs[k] - D_bd
                S_full = _psd_clip(_solve_M_inv_A_M_invT(M_bd, cov_adj), floor=self.psd_floor)
                sigma2 = max(np.trace(S_full) / P, self.psd_floor)
                S_k = sigma2 * np.eye(P, dtype=np.float64)
            else:
                raise ValueError("covariance_mode must be in {'full','diagonal','isotopic-full','isotropic-pullback'}")

            cov_prior[k] = _symmetrize(S_k)

        self.m_prior, self.cov_prior, self.weights = m_prior, cov_prior, weights
        return self.m_prior, self.cov_prior, self.weights

    def _responsibilities(self, data, mu, cov):
        """
        W[i,k] = p(k | x_i) under the observed-space mixture induced by
        (m_prior, cov_prior, weights) and the *provided* M (=mu) and D (=cov).
        """
        M_bd_local = _as_blockdiag_or_matrix(mu).astype(np.float64)
        D_bd_local = _psd_clip(_as_blockdiag_or_matrix(cov).astype(np.float64), floor=self.psd_floor)

        N, P = data.shape
        K = self.m_prior.shape[0]

        logits = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            mu_k_obs = self.m_prior[k] @ M_bd_local.T                        # (P,)
            Sig_k_obs = D_bd_local + M_bd_local @ self.cov_prior[k] @ M_bd_local.T
            Sig_k_obs = _psd_clip(Sig_k_obs, floor=self.psd_floor)        # (P,P)
            inv_S = np.linalg.inv(Sig_k_obs)

            _, logdet = np.linalg.slogdet(Sig_k_obs)

            # (optional) faster/safer: use Cholesky
            # L = np.linalg.cholesky(Sig_k_obs)
            # solve v = L \ data.T ; then inv_S*data via triangular solves, etc.

            fsq = np.einsum("ij,ij->i", data @ inv_S, data) / 2           # (N,)
            zsq = (mu_k_obs @ inv_S @ mu_k_obs) / 2                       # scalar
            fz  = data @ inv_S @ mu_k_obs                                  # (N,)

            logits[:, k] = np.log(self.weights[k]) - fsq + fz - zsq - 0.5 * logdet

        # numerically-stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        logits = np.clip(logits, -700, 700)
        W = np.exp(logits)
        W /= W.sum(axis=1, keepdims=True)
        return W  # (N, K)

    def denoise(self, data, mu, cov):
        """
        Posterior mean E[X | x] using *provided* M (=mu) and D (=cov).
        Returns (N, P).
        """
        M_bd_local = _as_blockdiag_or_matrix(mu).astype(np.float64)
        D_bd_local = _psd_clip(_as_blockdiag_or_matrix(cov).astype(np.float64), floor=self.psd_floor)

        N, P = data.shape
        K = self.m_prior.shape[0]

        inv_D = np.linalg.inv(D_bd_local)   # (optional) swap with Cholesky solves
        inv_C = [np.linalg.pinv(_psd_clip(c, floor=self.psd_floor)) for c in self.cov_prior]

        # responsibilities must use the same local M,D
        W = self._responsibilities(data, mu=M_bd_local, cov=D_bd_local)  # (N, K)

        out = np.zeros((N, P), dtype=np.float64)
        for k in range(K):
            # A = M^T D^{-1} M + C^{-1}
            A = M_bd_local.T @ inv_D @ M_bd_local + inv_C[k]
            A = _psd_clip(A, floor=self.psd_floor)
            Prec = np.linalg.inv(A)

            # latent mean: E[Z|x,k] = Prec (M^T D^{-1} x + C^{-1} m_k)
            # shapes: (N,P) = (N,P)(P,P)(P,P) + (P,)(P,P) -> broadcast over N
            latent_mean = (data @ inv_D.T @ M_bd_local + self.m_prior[k] @ inv_C[k].T) @ Prec.T  # (N,P)

            out += W[:, [k]] * latent_mean
        return out


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
        T_k     = np.zeros((K, P, P), dtype=np.float64)   # constant linear term per k
        eta_k   = np.zeros((K, N, P), dtype=np.float64)   # per-component posterior mean
        score_k = np.zeros((K, N, P), dtype=np.float64)   # per-component score s_k(x)

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
            mu_k_obs   = self.m_prior[k] @ M_bd_local.T   # (P,)
            score_k[k] = (data - mu_k_obs) @ inv_Sig_k    # (N, P)

        # Full posterior mean  eta(x) = sum_k w_k eta_k   (N, P)
        eta = np.einsum("kn,knp->np", W.T, eta_k)

        # Mixture score  s(x) = sum_k w_k s_k   (N, P)
        mix_score = np.einsum("kn,knp->np", W.T, score_k)

        # Assemble Jacobian using the mixture Stein identity:
        #   J = sum_k w_k T_k
        #     + sum_k w_k  eta_k ⊗ s_k      (outer: eta_k is "output", s_k is "input grad")
        #     - eta ⊗ mix_score
        #
        # Convention: J[i, a, b] = d eta_a / d x_b
        #   => outer product is eta[:, :, None] * score[:, None, :]  i.e. (N, P_out, P_in)

        J = np.zeros((N, P, P), dtype=np.float64)

        # Term 1: sum_k w_k T_k
        J += np.einsum("nk,kab->nab", W, T_k)   # (N, P, P)

        # Term 2: sum_k w_k  eta_k(x) ⊗ s_k(x)
        for k in range(K):
            outer = eta_k[k][:, :, None] * score_k[k][:, None, :]   # (N, P, P)
            J += W[:, k][:, None, None] * outer

        # Term 3: - eta(x) ⊗ mix_score(x)
        J -= eta[:, :, None] * mix_score[:, None, :]   # (N, P, P)

        return J

    def get_denoisers(self):
        """
        Return fitted prior parameters and denoising functions for use in other scripts.

        Returns
        -------
        eb_info : dict
            Dictionary with the following keys:
              - 'prior': tuple (m_prior, cov_prior, weights)
              - 'denoise': callable function (data, mu, cov) -> denoised output
              - 'ddenoise': callable function (data, mu, cov) -> derivative tensor
        """
        return {
            "prior": (self.m_prior, self.cov_prior, self.weights),
            "denoise": lambda data, M, D: self.denoise(data, M, D),
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
    cluster_models : dict[int -> ParametricEB_GMM]
        Trained EB model per cluster.
    cluster_data   : dict[int -> (N, P_cluster) ndarray]
        Concatenated observed data per cluster.
    cluster_M      : dict[int -> (P_cluster, P_cluster) ndarray]
    cluster_D      : dict[int -> (P_cluster, P_cluster) ndarray]
    modality_slices: dict[int -> (cluster_id, start, end)]
        For each modality index k, the cluster it belongs to and its slice within that cluster block.
    """

    def __init__(self, covariance_mode='full', choose_comp=False,
                 n_components=5, max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0):
        self.covariance_mode = covariance_mode
        self.choose_comp = choose_comp
        self.n_components = n_components
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.psd_floor = psd_floor
        self.random_state = random_state

        self.cluster_models = {}
        self.cluster_data = {}
        self.cluster_M = {}
        self.cluster_D = {}
        self.modality_slices = {}
        self.cluster_denoisers  = {}
        self.modality_denoisers = {}
        # NEW: store priors
        self.cluster_priors     = {}  # c -> {"prior": (m_prior, cov_prior, weights),
                                  #       "observed": {"means": ..., "covs": ..., "weights": ...},
                                  #       "gmm": GaussianMixture}

    def _aggregate_by_cluster(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        cluster_data = defaultdict(list)
        cluster_M = defaultdict(list)
        cluster_D = defaultdict(list)

        self.cluster_labels = cluster_labels

        # Build cluster concatenations and track per-modality slices
        offset_in_cluster = defaultdict(int)
        for k, c in enumerate(cluster_labels):
            Xk = data_matrices[k].astype(np.float64)
            Mk = M_matrices[k].astype(np.float64)
            Sk = S_matrices[k].astype(np.float64)

            cluster_data[c].append(Xk)
            cluster_M[c].append(Mk)
            cluster_D[c].append(Sk)

        # Concatenate per cluster, sanity checks
        for c in cluster_data:
            Ns = [Xi.shape[0] for Xi in cluster_data[c]]
            if len(set(Ns)) != 1:
                raise ValueError(f"Sample size mismatch in cluster {c}: {Ns}")

            self.cluster_data[c] = np.concatenate(cluster_data[c], axis=1)
            self.cluster_M[c] = sp.linalg.block_diag(*cluster_M[c])
            self.cluster_D[c] = sp.linalg.block_diag(*cluster_D[c])

        # Build slices for each modality into its cluster
        running_offsets = defaultdict(int)
        for k, c in enumerate(cluster_labels):
            p_k = data_matrices[k].shape[1]
            start = running_offsets[c]
            end = start + p_k
            self.modality_slices[k] = (c, start, end)
            running_offsets[c] = end

    def fit(self, data_matrices, M_matrices, S_matrices, cluster_labels):
        
        if not (len(data_matrices) == len(M_matrices) == len(S_matrices) == len(cluster_labels)):
            raise ValueError("Length mismatch among inputs.")

        # Aggregate per cluster (training time only)
        self._aggregate_by_cluster(data_matrices, M_matrices, S_matrices, cluster_labels)

        for c in sorted(self.cluster_data.keys()):
            Xc_train = self.cluster_data[c]  # used only to learn the prior
            Mc       = self.cluster_M[c]
            Dc       = self.cluster_D[c]

            eb = ParametricEB_GMM(
                n_components=self.n_components,
                max_iter=self.max_iter,
                covariance_mode=self.covariance_mode,
                choose_comp=self.choose_comp,
                reg_covar=self.reg_covar,
                psd_floor=self.psd_floor,
                random_state=self.random_state,
            )

            # Fit the GMM-based EB prior
            eb.estimate_prior(Xc_train, Mc, Dc)
            eb.M_bd, eb.D_bd = Mc, Dc  # ensure identity

            self.cluster_models[c] = eb

            # Store functional GMM denoisers
            self.cluster_denoisers[c] = eb.get_denoisers()

            # --- NEW: store priors for this cluster ---
            m_prior, cov_prior, weights = eb.m_prior, eb.cov_prior, eb.weights           # prior space
            gmm = eb.gmm                                                                   # observed-space sklearn object
            means_obs = np.array(gmm.means_, copy=True)
            covs_obs  = np.array(gmm.covariances_, copy=True)
            w_obs     = np.array(gmm.weights_, copy=True)

            self.cluster_priors[c] = {
                "prior":   (m_prior.copy(), cov_prior.copy(), weights.copy()),
                "observed": {"means": means_obs, "covs": covs_obs, "weights": w_obs},
                "gmm":     gmm,  # keep reference to the fitted sklearn GaussianMixture
            }

        # Build modality-level functional closures
        for k, (c, start, end) in self.modality_slices.items():
            denoise_c  = self.cluster_denoisers[c]["denoise"]
            ddenoise_c = self.cluster_denoisers[c]["ddenoise"]

            def make_mod_denoise(denoise_c, start, end):
                def _f(Xc_full, M, D):
                    """
                    Apply cluster-level denoiser to full cluster data,
                    then extract slice [start:end] for this modality.
                    """
                    Xc_hat = denoise_c(Xc_full, M, D)
                    return Xc_hat[:, start:end]
                return _f

            def make_mod_ddenoise(ddenoise_c, start, end):
                def _g(Xc_full, M, D):
                    """
                    Apply cluster-level Jacobian, then extract modality block.
                    """
                    Jc = ddenoise_c(Xc_full, M, D)  # (N, P_cluster, P_cluster)
                    return Jc[:, start:end, start:end]
                return _g

            self.modality_denoisers[k] = {
                "denoise":  make_mod_denoise(denoise_c, start, end),
                "ddenoise": make_mod_ddenoise(ddenoise_c, start, end),
            }
        
        return self


# Helper to split a block diagonal back to a list (for API parity)
def _split_block_diagonal(B, tol=1e-12):
    """
    Heuristic splitter: if B is block-diagonal of square blocks, recover a list of blocks.
    Assumes exact block-diagonal construction (no cross-block entries beyond tol).
    If unsure, just return [B].
    """
    P = B.shape[0]
    # Detect zero off-diagonal bands; crude but works when blocks were built by block_diag
    # We scan rows to find contiguous non-zero diagonals -> block sizes
    row_nonzero = np.abs(B).sum(axis=1) > tol
    # This won't detect block boundaries reliably for all patterns; safest fallback:
    # return [B]. For a true block list you can store the original sizes.
    return [B]  # robust fallback (we already pass Mc/Dc directly to eb after fitting)

