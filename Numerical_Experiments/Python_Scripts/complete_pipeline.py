import numpy as np
import importlib
from scipy.linalg import block_diag
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Dict, Any, Optional

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# Dynamically import required modules
pca_pack     = importlib.import_module("pca_pack")
emp_bayes    = importlib.import_module("emp_bayes")
amp          = importlib.import_module("amp")
preprocessing = importlib.import_module("preprocessing")

from emp_bayes import ClusterParametricEBPipeline


# =============================================================================
# PC normalization helpers
# =============================================================================

def extract_normalized_U(pca_model, X_list):
    """
    Extract and normalize left singular vectors for each modality.
    Each U_k column is rescaled to have L2 norm sqrt(n).
    """
    normalized_U_list = []
    for k in range(len(X_list)):
        U_k = pca_model.pca_results[k].U
        n   = U_k.shape[0]
        normalized_U_list.append(U_k / np.sqrt((U_k**2).sum(axis=0)) * np.sqrt(n))
    return normalized_U_list

def extract_normalized_V(pca_model, X_list):
    """
    Extract and normalize right singular vectors for each modality.
    Each V_k column is rescaled to have L2 norm sqrt(p_k).
    """
    normalized_V_list = []
    for k in range(len(X_list)):
        V_k = pca_model.pca_results[k].V
        p   = V_k.shape[0]
        normalized_V_list.append(V_k / np.sqrt((V_k**2).sum(axis=0)) * np.sqrt(p))
    return normalized_V_list


# =============================================================================
# GMM posterior utilities
# =============================================================================

def _psd_clip(A, floor=1e-10):
    """Symmetrize and eigen-clip A to enforce PSD with a floor on eigenvalues."""
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, floor)
    return (evecs * evals) @ evecs.T

def gmm_responsibilities(X, M, S, m_prior, cov_prior, weights):
    """
    Compute responsibilities W[i,k] = p(k | x_i) under the observed-space GMM
    induced by the prior (m_prior, cov_prior, weights) pushed through x = M z + eps.

    Parameters
    ----------
    X        : (N, p)
    M        : (p, d)
    S        : (p, p)  noise covariance
    m_prior  : (K, d)
    cov_prior: (K, d, d)
    weights  : (K,)

    Returns
    -------
    W : (N, K)
    """
    N  = X.shape[0]
    K  = m_prior.shape[0]
    logits = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        mu_k   = M @ m_prior[k]                            # (p,)
        Sig_k  = _psd_clip(S + M @ cov_prior[k] @ M.T)    # (p, p)
        _, logdet = np.linalg.slogdet(Sig_k)
        inv_S  = np.linalg.inv(Sig_k)
        diff   = X - mu_k                                  # (N, p)
        quad   = np.einsum("ni,ij,nj->n", diff, inv_S, diff)
        logits[:, k] = np.log(weights[k] + 1e-32) - 0.5 * (logdet + quad)

    logits -= logits.max(axis=1, keepdims=True)
    W = np.exp(np.clip(logits, -700, 700))
    W /= W.sum(axis=1, keepdims=True)
    return W

def gmm_linear_gaussian_posterior(X, M, S, m_prior, cov_prior, weights):
    """
    Compute the GMM posterior p(z | x_i) for all samples under the linear
    Gaussian model:  x | z ~ N(M z, S),  z ~ sum_k w_k N(m_k, C_k).

    Uses the Kalman/Schur identity — no inversion of C_k or S directly.

    Returns
    -------
    R   : (N, K)       responsibilities p(k | x_i)
    MU  : (N, K, d)    posterior means E[z | x_i, k]
    SIG : (K, d, d)    posterior covariances Cov[z | x, k]  (shared across i)
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    N, p = X.shape
    K    = m_prior.shape[0]
    d    = M.shape[1]

    # Responsibilities in observed space
    R = gmm_responsibilities(X, M, S, m_prior, cov_prior, weights)   # (N, K)

    MU  = np.zeros((N, K, d), dtype=np.float64)
    SIG = np.zeros((K, d, d), dtype=np.float64)

    for k in range(K):
        m_k = np.asarray(m_prior[k],   dtype=np.float64)
        C_k = np.asarray(cov_prior[k], dtype=np.float64)

        # Observed-space covariance and its inverse (unavoidable here)
        Sig_obs     = _psd_clip(S + M @ C_k @ M.T)
        Sig_obs_inv = np.linalg.inv(Sig_obs)

        # Kalman gain:  K_k = C_k M^T (S + M C_k M^T)^{-1}
        K_gain = C_k @ M.T @ Sig_obs_inv              # (d, p)

        # Posterior covariance (independent of x_i)
        SIG[k] = _psd_clip(C_k - K_gain @ M @ C_k)   # (d, d)

        # Posterior means for all samples
        innovation  = X - (M @ m_k)                   # (N, p)
        MU[:, k, :] = m_k + innovation @ K_gain.T     # (N, d)

    return R, MU, SIG

def _build_posterior_gmm(weights, means, covs, random_state=None):
    """
    Construct a sklearn GaussianMixture for a single sample's posterior p(z | x_i).
    Parameters are set directly (no fit call).

    Parameters
    ----------
    weights : (K,)
    means   : (K, d)
    covs    : (K, d, d)
    """
    gmm = GaussianMixture(
        n_components=len(weights), covariance_type='full',
        random_state=random_state
    )
    gmm.weights_     = np.asarray(weights, dtype=np.float64)
    gmm.means_       = np.asarray(means,   dtype=np.float64)
    gmm.covariances_ = np.asarray(covs,    dtype=np.float64)
    try:
        gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, 'full')
    except Exception:
        pass
    gmm.converged_ = True
    return gmm

def _mc_sample_posteriors(R, MU, SIG, mc_samples, rng=None):
    """
    Draw MC samples from the per-sample GMM posteriors efficiently using numpy.
    Avoids rebuilding sklearn objects — samples directly from (R, MU, SIG).

    Parameters
    ----------
    R   : (N, K)
    MU  : (N, K, d)
    SIG : (K, d, d)
    mc_samples : int

    Returns
    -------
    Z : (N, mc_samples, d)   samples z ~ p(z | x_i) for each sample i
    """
    if rng is None:
        rng = np.random.default_rng()

    N, K, d = MU.shape
    Z = np.zeros((N, mc_samples, d), dtype=np.float64)

    # Precompute Cholesky factors for each component
    chols = np.zeros((K, d, d), dtype=np.float64)
    for k in range(K):
        chols[k] = np.linalg.cholesky(_psd_clip(SIG[k]))

    for i in range(N):
        # Draw component assignments from responsibilities
        comp = rng.choice(K, size=mc_samples, p=R[i])
        for s in range(mc_samples):
            k = comp[s]
            eps = rng.standard_normal(d)
            Z[i, s] = MU[i, k] + chols[k] @ eps

    return Z


# =============================================================================
# Neural network architectures
# =============================================================================

class TwoLayerNN(nn.Module):
    """Simple two-layer MLP: input_dim -> hidden_dim -> 1."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Posterior-aware U reconstruction
# =============================================================================

def _reconstruct_U_hat(X_list, V_dict, D_dict, n):
    """
    Reconstruct raw U_hat_k for each modality by projecting X_k onto the
    trained V direction:  U_hat_k = X_k A_k (A_k^T A_k)^{-1}
    where A_k = (1/n) V_k D_k.

    Parameters
    ----------
    X_list : list of (n_test, p_k)
    V_dict : dict[k -> (p_k, r_k)]  denoised V from AMP
    D_dict : dict[k -> (r_k, r_k)]  diagonal signal matrices
    n      : int  training sample size used to define A_k

    Returns
    -------
    U_hat : dict[k -> (n_test, r_k)]
    """
    U_hat = {}
    for k, X_k in enumerate(X_list):
        V_k     = V_dict[k]                       # (p_k, r_k)
        D_k     = D_dict[k]                       # (r_k, r_k)
        A_k     = (1.0 / n) * V_k @ D_k           # (p_k, r_k)
        AtA_inv = np.linalg.inv(A_k.T @ A_k)
        U_hat[k] = X_k @ A_k @ AtA_inv            # (n_test, r_k)
    return U_hat

def _reconstruct_U_hat_onsager(F_dict, V_dict, D_dict, n):
    """
    Onsager-debiased training covariate:
        U_hat_k = F_k (Sigma_L_k)^{-1} D_k^{-1}
    where F_k is the raw (pre-denoising) AMP field for modality k
    (amp_results["U_non_denoised"][k][:, :, -1]) and
    Sigma_L_k = (1/n) V_k^T V_k.

    Unlike _reconstruct_U_hat, this does not re-project the training X_k —
    it reuses the AMP field, which is already independent of the loading
    direction's in-sample noise (see DAIF_training_projection_fixes.pdf, Sec. 2).

    Parameters
    ----------
    F_dict : dict[k -> (n, r_k)]  raw AMP field per modality
    V_dict : dict[k -> (p_k, r_k)]  denoised V from AMP
    D_dict : dict[k -> (r_k, r_k)]  diagonal signal matrices
    n      : int  AMP training sample size

    Returns
    -------
    U_hat : dict[k -> (n, r_k)]
    """
    U_hat = {}
    for k, F_k in F_dict.items():
        V_k         = V_dict[k]
        D_k         = D_dict[k]
        Sigma_L_k   = (V_k.T @ V_k) / n
        Sigma_L_inv = np.linalg.inv(Sigma_L_k)
        D_inv       = np.linalg.inv(D_k)
        U_hat[k]    = F_k @ Sigma_L_inv @ D_inv
    return U_hat


def _denoise_U_hat(U_hat, cluster_model_u, V_dict, D_dict, n):
    """
    Apply the cluster EB denoiser to the raw U_hat estimates.

    For each cluster c, stacks U_hat across member modalities, builds the
    effective noise covariance S_cluster, calls the EB denoiser, then splits
    results back to per-modality arrays.

    Parameters
    ----------
    U_hat         : dict[k -> (n_test, r_k)]
    cluster_model_u : ClusterParametricEBPipeline
    V_dict        : dict[k -> (p_k, r_k)]
    D_dict        : dict[k -> (r_k, r_k)]
    n             : int  training sample size

    Returns
    -------
    U_denoised : dict[k -> (n_test, r_k)]
    """
    cluster_labels_u  = cluster_model_u.cluster_labels
    cluster_denoisers = cluster_model_u.cluster_denoisers
    m = len(U_hat)

    U_denoised = {}
    for cid in np.unique(cluster_labels_u):
        mods     = [k for k in range(m) if cluster_labels_u[k] == cid]
        uden     = cluster_denoisers[cid]["denoise"]
        U_concat = np.hstack([U_hat[k] for k in mods])    # (n_test, d_c)

        # Effective noise covariance S_cluster for this cluster:
        #   S_k = D_k^{-1} Sigma_k^{-1} D_k^{-1}
        #   where Sigma_k = (1/n) V_k^T V_k  (empirical second moment of denoised V)
        S_blocks = []
        for k in mods:
            V_k       = V_dict[k]
            D_k       = D_dict[k]
            D_inv     = np.linalg.inv(D_k)
            Sigma_k   = (1.0 / n) * (V_k.T @ V_k)
            S_blocks.append(D_inv @ np.linalg.inv(Sigma_k) @ D_inv)
        S_cluster = block_diag(*S_blocks)

        # Identity M: latent and observed spaces coincide after U reconstruction
        M_cluster      = np.eye(U_concat.shape[1])
        U_den_concat   = uden(U_concat, M_cluster, S_cluster)

        # Split denoised block back to per-modality arrays
        widths = [U_hat[k].shape[1] for k in mods]
        starts = np.cumsum([0] + widths[:-1])
        for k, s, w in zip(mods, starts, widths):
            U_denoised[k] = U_den_concat[:, s:s+w]

    return U_denoised

def _compute_cluster_posteriors(U_hat, cluster_model_u, V_dict, D_dict, n):
    """
    Compute GMM posterior parameters (R, MU, SIG) for each cluster.

    For cluster c with modalities {k}, the posterior is:
      p(z | U_hat_c) where U_hat_c = stack of U_hat_k,
      with M = I (identity), S = S_cluster.

    Returns
    -------
    cluster_posteriors : dict[c -> {"R": (N,K), "MU": (N,K,d), "SIG": (K,d,d), "mods": list}]
    """
    cluster_labels_u = cluster_model_u.cluster_labels
    m = len(U_hat)

    cluster_posteriors = {}
    for c in np.unique(cluster_labels_u):
        mods = [k for k in range(m) if cluster_labels_u[k] == c]
        Uc   = np.hstack([U_hat[k] for k in mods])    # (N, d_c)
        d_c  = Uc.shape[1]

        # Noise covariance for this cluster
        S_blocks = []
        for k in mods:
            V_k     = V_dict[k]
            D_k     = D_dict[k]
            D_inv   = np.linalg.inv(D_k)
            Sigma_k = (1.0 / n) * (V_k.T @ V_k)
            S_blocks.append(D_inv @ np.linalg.inv(Sigma_k) @ D_inv)
        S_cluster = block_diag(*S_blocks)

        # GMM prior for this cluster
        if hasattr(cluster_model_u, "cluster_priors") and c in cluster_model_u.cluster_priors:
            m_prior, cov_prior, weights = cluster_model_u.cluster_priors[c]["prior"]
        else:
            eb = cluster_model_u.cluster_models[c]
            m_prior, cov_prior, weights = eb.m_prior, eb.cov_prior, eb.weights

        R, MU, SIG = gmm_linear_gaussian_posterior(
            Uc, np.eye(d_c), S_cluster, m_prior, cov_prior, weights
        )
        cluster_posteriors[c] = {"R": R, "MU": MU, "SIG": SIG, "mods": mods, "d": d_c}

    return cluster_posteriors


# =============================================================================
# Posterior-aware predictors
# =============================================================================

class PosteriorAwarePredictor:
    """
    Posterior-aware predictor that jointly trains across all modality clusters.

    Training philosophy
    -------------------
    Each architecture uses a different strategy to incorporate posterior
    uncertainty p(z | U_hat_i) at training time:

      "linear" / "logistic":
        Use the posterior mean E[Z | U_hat_i] = sum_k r_ik * mu_ik as features.
        For linear regression this is exact; for logistic it's a good approximation.
        Fast, no MC needed.

      "random_forest" / "xgboost":
        Data augmentation: draw S MC samples per training point from p(z | U_hat_i),
        replicate the label y_i for each. Fit on the augmented dataset.
        The model sees the full spread of posterior uncertainty implicitly.

      "neural_net":
        Monte Carlo gradient: at each epoch, draw S samples per point, evaluate
        the network, average outputs, compute loss against y_i, backpropagate.
        Posterior parameters (R, MU, SIG) are precomputed once; only sampling
        happens inside the loop (fast numpy, no sklearn overhead).

    Test-time inference
    -------------------
    All architectures: compute posterior parameters from test U_hat,
    then predict using the same strategy (posterior mean for linear/logistic,
    MC average for the others).
    """

    SUPPORTED = ("linear", "logistic", "random_forest", "neural_net", "xgboost")

    def __init__(self, architecture="linear", task="regression",
                 mc_samples=8, epochs=2000, lr=1e-3, hidden_dim=32,
                 device="cpu", seed=None, **arch_kwargs):
        """
        Parameters
        ----------
        architecture : str
            One of "linear", "logistic", "random_forest", "neural_net", "xgboost".
        task : str
            "regression" or "classification".
        mc_samples : int
            MC draws per sample (used by random_forest, xgboost, neural_net).
        epochs : int
            Training epochs (neural_net only).
        lr : float
            Learning rate (neural_net only).
        hidden_dim : int
            Hidden layer width (neural_net only).
        device : str
            "cpu" or "cuda" (neural_net only).
        seed : int or None
        arch_kwargs : dict
            Additional kwargs forwarded to the underlying sklearn/xgb constructor.
        """
        if architecture not in self.SUPPORTED:
            raise ValueError(f"architecture must be one of {self.SUPPORTED}")
        if architecture == "linear" and task != "regression":
            raise ValueError("architecture='linear' requires task='regression'. Use 'logistic' for classification.")
        if architecture == "logistic" and task != "classification":
            raise ValueError("architecture='logistic' requires task='classification'. Use 'linear' for regression.")
        self.architecture = architecture
        self.task         = task
        self.mc_samples   = mc_samples
        self.epochs       = epochs
        self.lr           = lr
        self.hidden_dim   = hidden_dim
        self.device       = device
        self.seed         = seed
        self.arch_kwargs  = arch_kwargs

        self._models        = {}    # dict[cluster_id -> fitted neural net per cluster]
        self._cluster_order = []    # ordered cluster ids, fixed at fit time
        self._joint_model   = None  # single joint model (linear/logistic/rf/xgb)
        self._is_fitted     = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, cluster_posteriors, y):
        """
        Train a joint predictor using precomputed posterior parameters.

        For neural_net: per-cluster TwoLayerNN models trained jointly — a single
        loss on the sum of cluster outputs so each expert learns a partial contribution.
        For linear/logistic/rf/xgb: a single joint model on features concatenated
        across all clusters.

        Parameters
        ----------
        cluster_posteriors : dict[c -> {"R": (N,K), "MU": (N,K,d), "SIG": (K,d,d), "d": int}]
            Output of _compute_cluster_posteriors.
        y : (N,) array of training labels/targets.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        rng = np.random.default_rng(self.seed)
        self._cluster_order = sorted(cluster_posteriors.keys())
        arch = self.architecture

        if arch in ("linear", "logistic"):
            # Single model on posterior means concatenated across clusters
            Z_parts = [self._posterior_mean(cluster_posteriors[c]["R"],
                                            cluster_posteriors[c]["MU"])
                       for c in self._cluster_order]
            Z_all = np.hstack(Z_parts)   # (N, sum_c d_c)
            if arch == "linear":
                model = LinearRegression(**self.arch_kwargs)
            else:
                model = LogisticRegression(max_iter=1000, **self.arch_kwargs)
            model.fit(Z_all, y)
            self._joint_model = model

        elif arch in ("random_forest", "xgboost"):
            # Single model on MC-augmented features concatenated across clusters
            S = self.mc_samples
            Z_aug_parts = []
            for c in self._cluster_order:
                post = cluster_posteriors[c]
                Z_c, _ = self._augment(post["R"], post["MU"], post["SIG"], y, rng)
                Z_aug_parts.append(Z_c)            # each (N*S, d_c)
            Z_aug_all = np.hstack(Z_aug_parts)     # (N*S, sum_c d_c)
            y_aug = np.repeat(y, S)
            if arch == "random_forest":
                if self.task == "regression":
                    model = RandomForestRegressor(random_state=self.seed, **self.arch_kwargs)
                else:
                    model = RandomForestClassifier(random_state=self.seed, **self.arch_kwargs)
            else:
                if not _XGBOOST_AVAILABLE:
                    raise ImportError("xgboost is not installed.")
                if self.task == "regression":
                    model = xgb.XGBRegressor(random_state=self.seed, **self.arch_kwargs)
                else:
                    model = xgb.XGBClassifier(random_state=self.seed, **self.arch_kwargs)
            model.fit(Z_aug_all, y_aug)
            self._joint_model = model

        elif arch == "neural_net":
            self._models = self._fit_joint_neural_net(cluster_posteriors, y)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self._is_fitted = True
        return self

    def predict(self, cluster_posteriors):
        """
        Predict using the joint model trained by fit().

        For neural_net: sum of per-cluster network outputs (mirrors joint training).
        For linear/logistic/rf/xgb: single joint model on concatenated cluster features.

        Parameters
        ----------
        cluster_posteriors : dict[c -> {"R", "MU", "SIG", "d"}]

        Returns
        -------
        y_pred : (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        rng = np.random.default_rng(self.seed)
        N   = next(iter(cluster_posteriors.values()))["R"].shape[0]
        arch = self.architecture

        if arch in ("linear", "logistic"):
            Z_parts = [self._posterior_mean(cluster_posteriors[c]["R"],
                                            cluster_posteriors[c]["MU"])
                       for c in self._cluster_order]
            Z_all = np.hstack(Z_parts)
            if arch == "linear":
                return self._joint_model.predict(Z_all).ravel()
            else:
                if hasattr(self._joint_model, "predict_proba"):
                    proba = self._joint_model.predict_proba(Z_all)
                    return proba[:, 1] if proba.shape[1] == 2 else proba.argmax(axis=1).astype(float)
                return self._joint_model.predict(Z_all).ravel().astype(float)

        elif arch in ("random_forest", "xgboost"):
            S = self.mc_samples
            Z_parts = []
            for c in self._cluster_order:
                post = cluster_posteriors[c]
                Z_c = _mc_sample_posteriors(post["R"], post["MU"], post["SIG"], S, rng)  # (N, S, d_c)
                Z_parts.append(Z_c.reshape(N * S, -1))
            Z_all = np.hstack(Z_parts)   # (N*S, sum_c d_c)
            if self.task == "classification":
                # Average class probabilities over MC draws, then return prob or label
                proba_flat = self._joint_model.predict_proba(Z_all)         # (N*S, n_classes)
                proba_avg  = proba_flat.reshape(N, S, -1).mean(axis=1)      # (N, n_classes)
                return proba_avg[:, 1] if proba_avg.shape[1] == 2 else proba_avg.argmax(axis=1).astype(float)
            else:
                preds = self._joint_model.predict(Z_all).reshape(N, S)
                return preds.mean(axis=1)

        elif arch == "neural_net":
            return self._predict_joint_neural_net(cluster_posteriors)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

    # ------------------------------------------------------------------
    # Architecture-specific implementations
    # ------------------------------------------------------------------

    def _posterior_mean(self, R, MU):
        """
        Compute the posterior mean E[Z | x_i] = sum_k r_ik mu_ik.
        Shape: (N, d).
        """
        # R: (N, K),  MU: (N, K, d)
        return np.einsum("nk,nkd->nd", R, MU)


    def _augment(self, R, MU, SIG, y, rng):
        """
        Draw mc_samples from each sample's posterior and replicate labels.
        Returns Z_aug (N*S, d) and y_aug (N*S,).
        """
        Z   = _mc_sample_posteriors(R, MU, SIG, self.mc_samples, rng)  # (N, S, d)
        N, S, d = Z.shape
        Z_aug = Z.reshape(N * S, d)
        y_aug = np.repeat(y, S)
        return Z_aug, y_aug

    def _fit_joint_neural_net(self, cluster_posteriors, y):
        """
        Joint training of per-cluster TwoLayerNN models.
        Loss is computed on the sum of all cluster outputs so each expert learns
        a partial contribution (mirrors Hierarchical_AMP_GMM training loop).
        """
        device        = self.device
        cluster_order = self._cluster_order
        models     = {c: TwoLayerNN(cluster_posteriors[c]["d"], self.hidden_dim).to(device)
                      for c in cluster_order}
        optimizers = {c: torch.optim.Adam(m.parameters(), lr=self.lr)
                      for c, m in models.items()}
        loss_fn = nn.MSELoss() if self.task == "regression" else nn.BCEWithLogitsLoss()
        y_t     = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
        rng     = np.random.default_rng(self.seed)

        for ep in range(self.epochs):
            for opt in optimizers.values():
                opt.zero_grad()

            E_sum = torch.zeros_like(y_t)

            for c in cluster_order:
                post = cluster_posteriors[c]
                R, MU, SIG = post["R"], post["MU"], post["SIG"]
                Z = _mc_sample_posteriors(R, MU, SIG, self.mc_samples, rng)  # (N, S, d_c)
                N, S, d_c = Z.shape
                Z_t   = torch.as_tensor(Z.reshape(N * S, d_c), dtype=torch.float32, device=device)
                f_out = models[c](Z_t).reshape(N, S, 1).mean(dim=1)          # (N, 1)
                E_sum = E_sum + f_out

            loss = loss_fn(E_sum, y_t)
            loss.backward()
            for opt in optimizers.values():
                opt.step()

            if ep % 200 == 0:
                print(f"  [NeuralNet] epoch {ep:4d} | loss {loss.item():.6f}", flush=True)

        return models

    def _predict_joint_neural_net(self, cluster_posteriors):
        """Sum of per-cluster MC-averaged neural net outputs (mirrors joint training)."""
        device = self.device
        rng    = np.random.default_rng(self.seed)
        N      = next(iter(cluster_posteriors.values()))["R"].shape[0]
        y_sum  = np.zeros(N, dtype=np.float64)

        for c in self._cluster_order:
            post = cluster_posteriors[c]
            R, MU, SIG = post["R"], post["MU"], post["SIG"]
            Z = _mc_sample_posteriors(R, MU, SIG, self.mc_samples, rng)  # (N, S, d_c)
            N_s, S, d_c = Z.shape
            Z_t = torch.as_tensor(Z.reshape(N_s * S, d_c), dtype=torch.float32, device=device)
            model = self._models[c].eval()
            with torch.no_grad():
                f_out = model(Z_t).reshape(N_s, S).mean(dim=1).cpu().numpy()
            y_sum += f_out

        return y_sum


# =============================================================================
# Shared regression fitting and prediction (used by all pipeline classes)
# =============================================================================

def _fit_predictor(pipeline, X_list, y, architecture="linear", task="regression",
                   mc_samples=8, epochs=2000, lr=1e-3, hidden_dim=32,
                   device="cpu", seed=None, projection_mode="onsager", **arch_kwargs):
    """
    Fit a PosteriorAwarePredictor on training data using AMP results already
    stored in pipeline.amp_results.

    Computes U_hat and cluster posteriors, then trains the predictor.
    Stores the fitted predictor and cluster posteriors in pipeline for reuse.

    Parameters
    ----------
    pipeline      : fitted pipeline object with amp_results populated
    X_list        : list of (n_head, p_k)  data matrices to build the head's
                    training covariates from. Ignored when
                    projection_mode == "onsager" (the AMP field is reused
                    instead); required otherwise.
    y             : (n_head,)  training labels/targets
    architecture  : str  one of "linear", "logistic", "random_forest", "neural_net", "xgboost"
    task          : str  "regression" or "classification"
    mc_samples    : int
    epochs, lr, hidden_dim, device, seed : neural_net hyperparameters
    projection_mode : str  "onsager" (default) uses the Onsager-debiased AMP
                    field for the training covariates (see
                    _reconstruct_U_hat_onsager); "ols" reconstructs via
                    ordinary in-sample/out-of-sample projection of X_list
                    (see _reconstruct_U_hat) — used for the in_sample_ols
                    and sample_split pipeline.projection_mode settings.
    **arch_kwargs : forwarded to underlying sklearn/xgb model

    Returns
    -------
    predictor : fitted PosteriorAwarePredictor
    """
    amp_results     = pipeline.amp_results
    V_dict          = {k: amp_results["V_denoised"][k][:, :, -1]
                       for k in amp_results["V_denoised"]}
    D_dict          = amp_results["signal_diag_dict"]
    cluster_model_u = pipeline.cluster_model_u
    n_amp           = amp_results["_n_amp"]   # sample size the AMP loadings were fit on

    if projection_mode == "onsager":
        F_dict = {k: amp_results["U_non_denoised"][k][:, :, -1]
                  for k in amp_results["U_non_denoised"]}
        U_hat = _reconstruct_U_hat_onsager(F_dict, V_dict, D_dict, n_amp)
    elif projection_mode == "ols":
        U_hat = _reconstruct_U_hat(X_list, V_dict, D_dict, n_amp)
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode!r}")

    cluster_posteriors = _compute_cluster_posteriors(U_hat, cluster_model_u, V_dict, D_dict, n_amp)

    predictor = PosteriorAwarePredictor(
        architecture=architecture, task=task,
        mc_samples=mc_samples, epochs=epochs, lr=lr, hidden_dim=hidden_dim,
        device=device, seed=seed, **arch_kwargs
    ).fit(cluster_posteriors, y)

    # Store on pipeline for predict_from_test_data. Test-time projection is
    # always the independent OLS projection (Sec. 2.3 of the fixes note),
    # scaled by the AMP sample size regardless of projection_mode.
    pipeline.predictor              = predictor
    pipeline.train_cluster_posteriors = cluster_posteriors
    pipeline.amp_results["_V_dict_last"] = V_dict   # snapshot of last V iterate
    pipeline.amp_results["_n_train"]     = n_amp

    return predictor


def predict_from_test_data(pipeline, X_test, architecture=None, mc_samples=None,
                           device="cpu", seed=None):
    """
    Generate predictions on test data using the fitted predictor stored on pipeline.

    Parameters
    ----------
    pipeline     : fitted pipeline with pipeline.predictor populated by _fit_predictor
    X_test       : list of (n_test, p_k)
    architecture : ignored (uses pipeline.predictor.architecture); kept for API compatibility
    mc_samples   : override MC samples at test time (uses training value if None)
    device       : "cpu" or "cuda"
    seed         : random seed for MC draws

    Returns
    -------
    U_denoised_dict : dict[k -> (n_test, r_k)]
    y_pred          : (n_test,)
    """
    if not hasattr(pipeline, "predictor") or pipeline.predictor is None:
        raise RuntimeError("No predictor found. Call _fit_predictor (via denoise_amp) first.")

    predictor       = pipeline.predictor
    amp_results     = pipeline.amp_results
    V_dict          = amp_results["_V_dict_last"]
    D_dict          = amp_results["signal_diag_dict"]
    cluster_model_u = pipeline.cluster_model_u
    n_train         = amp_results["_n_train"]

    if mc_samples is not None:
        predictor.mc_samples = mc_samples

    # Reconstruct and denoise U_hat for test data
    U_hat           = _reconstruct_U_hat(X_test, V_dict, D_dict, n_train)
    U_denoised_dict = _denoise_U_hat(U_hat, cluster_model_u, V_dict, D_dict, n_train)

    # Compute test posteriors and predict
    cluster_posteriors = _compute_cluster_posteriors(U_hat, cluster_model_u, V_dict, D_dict, n_train)
    y_pred             = predictor.predict(cluster_posteriors)

    return U_denoised_dict, y_pred


# =============================================================================
# Shared denoise_amp core (extracted to avoid duplication across pipeline classes)
# =============================================================================

def _run_denoise_amp_core(pipeline, X_list, K_list, cluster_labels_U,
                          amp_iters, muteu, mutev, preprocess, verbose=True):
    """
    Core AMP denoising logic shared by all three pipeline classes.
    Fits PCA, builds EB models, runs AMP. Stores results on pipeline.

    Parameters
    ----------
    pipeline         : pipeline object to store results on
    X_list           : list of (n, p_k)
    K_list           : list of int  number of PCs per modality
    cluster_labels_U : array of int  cluster assignment per modality
    amp_iters        : int
    muteu, mutev     : bool
    preprocess       : bool  if True, normalize noise before PCA
    verbose          : bool  if False, suppress prints (simulation mode)
    """
    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Optional noise normalization
    if preprocess:
        diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
        X_list = diagnostic_tool.normalize_obs(X_list, K_list)

    # -------------------------------------------------------------------
    # Training-covariate projection mode. See DAIF_training_projection_fixes.pdf.
    #   "onsager"       (default) — Onsager-debiased AMP field for training
    #                    covariates; all subjects used for both AMP and the head.
    #   "sample_split"  — hold out holdout_frac of subjects before AMP runs;
    #                    those subjects never enter representation learning,
    #                    and are OLS-projected onto the AMP-fit loadings to
    #                    train the head.
    #   "in_sample_ols" — legacy behavior: OLS-project the same subjects that
    #                    were used to fit the AMP loadings (in-sample; kept
    #                    only for comparison/backward compatibility).
    # -------------------------------------------------------------------
    projection_mode = getattr(pipeline, "projection_mode", "onsager")
    if projection_mode not in ("onsager", "sample_split", "in_sample_ols"):
        raise ValueError(f"Unknown pipeline.projection_mode: {projection_mode!r}")

    y_train_full = getattr(pipeline, "y_train", None)
    X_head, y_head = None, None   # held-out data used to fit the head (sample_split only)

    if projection_mode == "sample_split" and y_train_full is not None:
        # If the caller already computed a split (e.g. to reuse the same
        # partition across multiple pipeline instances/strategies within a
        # trial), reuse it verbatim instead of drawing a new one.
        idx_A = getattr(pipeline, "_split_idx_A", None)
        idx_B = getattr(pipeline, "_split_idx_B", None)
        if idx_A is None or idx_B is None:
            holdout_frac = getattr(pipeline, "holdout_frac", 0.3)
            n_full = X_list[0].shape[0]
            rng    = np.random.default_rng(getattr(pipeline, "seed", None))
            perm   = rng.permutation(n_full)
            n_B    = int(round(holdout_frac * n_full))
            idx_B  = perm[:n_B]
            idx_A  = perm[n_B:]

        X_head = [X_k[idx_B] for X_k in X_list]
        y_head = y_train_full[idx_B]

        X_list = [X_k[idx_A] for X_k in X_list]   # AMP only ever sees split A
        pipeline._split_idx_A = idx_A
        pipeline._split_idx_B = idx_B
        _print(f"  Sample-split mode: {len(idx_A)} subjects -> AMP/representation "
               f"learning, {len(idx_B)} subjects -> predictor head (never seen by AMP)")

    # Fit PCA
    pipeline.pca_model = pca_pack.MultiModalityPCA()
    pipeline.pca_model.fit(X_list, K_list, plot_residual=False)

    # Normalized PCs for EB fitting
    U_norm = extract_normalized_U(pipeline.pca_model, X_list)
    V_norm = extract_normalized_V(pipeline.pca_model, X_list)

    # State evolution initial parameters from PCA alignment coefficients
    M_v = [np.diag(pipeline.pca_model.pca_results[k].feature_aligns)        for k in range(len(X_list))]
    S_v = [np.diag(1 - pipeline.pca_model.pca_results[k].feature_aligns**2) for k in range(len(X_list))]
    M_u = [np.diag(pipeline.pca_model.pca_results[k].sample_aligns)         for k in range(len(X_list))]
    S_u = [np.diag(1 - pipeline.pca_model.pca_results[k].sample_aligns**2)  for k in range(len(X_list))]

    # V clusters: one per modality (independent EB prior per modality)
    cluster_labels_V = np.arange(len(X_list))

    # Build EB cluster models for U and V
    pipeline.cluster_model_u = ClusterParametricEBPipeline(
        covariance_mode='full', choose_comp=False, n_components=5,
        max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
    ).fit(U_norm, M_u, S_u, cluster_labels_U)

    pipeline.cluster_model_v = ClusterParametricEBPipeline(
        covariance_mode='full', choose_comp=False, n_components=5,
        max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
    ).fit(V_norm, M_v, S_v, cluster_labels_V)

    # Run AMP (on split A only, under sample_split mode)
    pipeline.amp_results = amp.ebamp_multimodal(
        pipeline.pca_model, pipeline.cluster_model_v, pipeline.cluster_model_u,
        amp_iters=amp_iters, muteu=muteu, mutev=mutev
    )
    pipeline.amp_results["_n_amp"] = X_list[0].shape[0]

    # Fit predictor if training labels are available
    if projection_mode == "sample_split":
        if y_head is not None:
            architecture = getattr(pipeline, "architecture", "linear")
            task         = getattr(pipeline, "task", "regression")
            _print(f"  Fitting {architecture} predictor on held-out split "
                   f"(OLS projection onto split-A loadings)...")
            _fit_predictor(
                pipeline, X_head, y_head,
                architecture=architecture, task=task,
                mc_samples=getattr(pipeline, "mc_samples", 8),
                epochs=getattr(pipeline, "epochs", 2000),
                lr=getattr(pipeline, "lr", 1e-3),
                hidden_dim=getattr(pipeline, "hidden_dim", 32),
                device=getattr(pipeline, "device", "cpu"),
                seed=getattr(pipeline, "seed", None),
                projection_mode="ols",
                **getattr(pipeline, "arch_kwargs", {}),
            )
    elif y_train_full is not None:
        architecture = getattr(pipeline, "architecture", "linear")
        task         = getattr(pipeline, "task", "regression")
        _print(f"  Fitting {architecture} predictor ({projection_mode})...")
        _fit_predictor(
            pipeline, X_list, y_train_full,
            architecture=architecture, task=task,
            mc_samples=getattr(pipeline, "mc_samples", 8),
            epochs=getattr(pipeline, "epochs", 2000),
            lr=getattr(pipeline, "lr", 1e-3),
            hidden_dim=getattr(pipeline, "hidden_dim", 32),
            device=getattr(pipeline, "device", "cpu"),
            seed=getattr(pipeline, "seed", None),
            projection_mode=("onsager" if projection_mode == "onsager" else "ols"),
            **getattr(pipeline, "arch_kwargs", {}),
        )

    return pipeline.amp_results


# =============================================================================
# Pipeline classes
# =============================================================================

class MultimodalPCAPipeline:
    """
    Full multimodal PCA + AMP + EB denoising pipeline.
    Cluster labels for U are provided externally by the user.

    Attributes
    ----------
    y_train      : (n,) array  set before calling denoise_amp to enable predictor training
    architecture : str  predictor architecture (default "linear")
    task         : str  "regression" or "classification"
    mc_samples   : int  MC draws for posterior-aware training (default 8)
    projection_mode : str  training-covariate construction (default "onsager"):
                     "onsager"       — Onsager-debiased AMP field (recommended fix)
                     "sample_split"  — hold out holdout_frac subjects before AMP;
                                       project them via OLS onto the AMP-fit loadings
                     "in_sample_ols" — legacy in-sample OLS projection
                     See DAIF_training_projection_fixes.pdf.
    holdout_frac : float  fraction of subjects held out for the head under
                   "sample_split" (default 0.3)
    """

    def __init__(self):
        self.pca_model      = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results    = None
        self.predictor      = None
        self.y_train        = None
        self.architecture   = "linear"
        self.task           = "regression"
        self.mc_samples     = 8
        self.epochs         = 2000
        self.lr             = 1e-3
        self.hidden_dim     = 32
        self.device         = "cpu"
        self.seed           = None
        self.projection_mode = "onsager"
        self.holdout_frac    = 0.3

    def denoise_amp(self, X_list, K_list, cluster_labels_U,
                    amp_iters=10, muteu=False, mutev=False, preprocess=False):
        """
        Run PCA, build EB models, run AMP, optionally fit predictor.

        Parameters
        ----------
        X_list           : list of (n, p_k) data matrices
        K_list           : list of int  number of PCs per modality
        cluster_labels_U : array of int  cluster assignment per modality
        amp_iters        : int
        muteu, mutev     : bool  if True use identity denoiser in U/V direction
        preprocess       : bool  if True normalize noise before PCA
        """
        return _run_denoise_amp_core(
            self, X_list, K_list, cluster_labels_U,
            amp_iters, muteu, mutev, preprocess, verbose=True
        )


class MultimodalPCAPipelineClustering:
    """
    Same as MultimodalPCAPipeline but automatically computes cluster labels
    from the normalized U matrices using ModalityClusterer.

    Note: modality-cluster labels are computed from a preliminary PCA fit on
    the full X_list before projection_mode == "sample_split" partitions the
    data, so cluster assignment (not the AMP loadings or the head) sees all
    subjects. See projection_mode on MultimodalPCAPipeline for the training-
    covariate switch.
    """

    def __init__(self):
        self.pca_model      = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results    = None
        self.predictor      = None
        self.y_train        = None
        self.architecture   = "linear"
        self.task           = "regression"
        self.mc_samples     = 8
        self.epochs         = 2000
        self.lr             = 1e-3
        self.hidden_dim     = 32
        self.device         = "cpu"
        self.seed           = None
        self.projection_mode = "onsager"
        self.holdout_frac    = 0.3

    def denoise_amp(
        self, X_list, K_list,
        cluster_labels_U=None, compute_clusters=True, num_clusters=None,
        threshold=None, amp_iters=10, muteu=False, mutev=False,
        preprocess=False, similarity_method="hss", **sim_kwargs
    ):
        """
        Run PCA, optionally compute clusters, build EB models, run AMP.

        Parameters
        ----------
        X_list           : list of (n, p_k)
        K_list           : list of int
        cluster_labels_U : array of int or None  (used if compute_clusters=False)
        compute_clusters : bool  if True, compute labels automatically
        num_clusters     : int or None  passed to ModalityClusterer
        threshold        : float or None  passed to ModalityClusterer
        similarity_method: str  similarity metric for clustering (default "hss")
        sim_kwargs       : extra keyword args forwarded to the similarity function (e.g. epsilon, sigma)
        amp_iters, muteu, mutev, preprocess : as in MultimodalPCAPipeline
        """
        if preprocess:
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_list = diagnostic_tool.normalize_obs(X_list, K_list)

        # Fit PCA first so we can normalize U for clustering
        self.pca_model = pca_pack.MultiModalityPCA()
        self.pca_model.fit(X_list, K_list, plot_residual=False)

        if compute_clusters:
            from hierarchical_clustering_modalities import ModalityClusterer
            U_norm = extract_normalized_U(self.pca_model, X_list)
            clusterer_obj = ModalityClusterer(U_norm)
            _ = clusterer_obj.compute_similarity_matrix(similarity_method, **sim_kwargs)
            cluster_labels_U = clusterer_obj.cluster_modalities(
                similarity_method, num_clusters=num_clusters, threshold=threshold
            )
        elif cluster_labels_U is None:
            raise ValueError("Either set compute_clusters=True or provide cluster_labels_U.")

        # Delegate to shared core (PCA already fitted above — reuse by passing preprocess=False)
        return _run_denoise_amp_core(
            self, X_list, K_list, cluster_labels_U,
            amp_iters, muteu, mutev, preprocess=False, verbose=True
        )


class MultimodalPCAPipelineSimulation:
    """
    Silent version of MultimodalPCAPipeline for simulation studies.
    Cluster labels are provided externally. No prints during AMP.
    """

    def __init__(self):
        self.pca_model      = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results    = None
        self.predictor      = None
        self.y_train        = None
        self.architecture   = "linear"
        self.task           = "regression"
        self.mc_samples     = 8
        self.epochs         = 2000
        self.lr             = 1e-3
        self.hidden_dim     = 32
        self.device         = "cpu"
        self.seed           = None
        self.projection_mode = "onsager"
        self.holdout_frac    = 0.3

    def denoise_amp(self, X_list, K_list, cluster_labels_U,
                    amp_iters=10, muteu=False, mutev=False, preprocess=False):
        """Same as MultimodalPCAPipeline.denoise_amp but suppresses all prints."""
        return _run_denoise_amp_core(
            self, X_list, K_list, cluster_labels_U,
            amp_iters, muteu, mutev, preprocess, verbose=False
        )