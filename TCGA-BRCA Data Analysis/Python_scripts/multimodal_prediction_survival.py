"""
multimodal_prediction_survival.py
==================================
Stripped-down version of multimodal_prediction_unified.py that retains only what
the BRCA survival notebooks need (orchamp_brca_survival_{early,intermediate,late}_fusion.ipynb):
  - HD+LD PCA -> clustering -> cluster-wise EB denoising -> AMP
  - Linear Cox survival head trained via GMM posterior Monte-Carlo sampling
  - Test-time survival (log-risk) prediction

Removed entirely (not used by any survival notebook):
  - train_nonlinear_regressor / train_nonlinear_classifier
  - SimpleDeepRegressor / SimpleDeepClassifier / _mlp_layers
  - _cosine_with_warmup / _maybe_class_weights / _sharpen_rows /
    _softmax_with_temperature / _temp_scale_probs (nonlinear-training-only utils)
  - PosteriorAwarePredictor, _fit_predictor, predict_from_test_data
  - predict_from_test_data_all (regression), predict_celltypes_from_test_data_all
  - _mc_sample_posteriors, _build_per_modality_MS, _denoise_U_hat_hd_ld,
    _compute_cluster_posteriors_hd_ld, _psd_clip (posterior-aware-predictor only)
  - extract_normalized_U / extract_normalized_V (public helpers, unused internally)
  - Regression / classification branches of _post_amp_supervised, and the
    _post_amp_supervised auto-hook itself — see note on run_amp() below
  - All classification/regression-only sklearn & xgboost imports

Note on training the Cox head
------------------------------
In multimodal_prediction_unified.py, run_amp() tries to auto-train a supervised
head via self._post_amp_supervised(...) whenever y_train or y_surv_train is set.
But _post_amp_supervised() starts with `if self.y_train is None: return`, and the
survival notebooks only ever set y_surv_train — so that auto-hook is a silent
no-op for every survival run today. The notebooks work because they call
train_linear_cox_survival(...) explicitly right after run_amp(...). This file
keeps that behavior explicit (see run_amp()'s docstring) instead of carrying the
dead auto-hook forward.

Training-covariate projection: onsager (Onsager-debiased AMP field; see
DAIF_training_projection_fixes.pdf, Sec. 2). train_linear_cox_survival's
training U_hat is reconstructed from the raw pre-denoising AMP field
(amp_results["U_non_denoised"]) via _reconstruct_Uhat_onsager instead of
re-projecting the training X_hd in-sample (the legacy behavior kept in
tcga_brca_final_ols). Test-time reconstruction in
predict_survival_from_test_data_all is unchanged — it still uses the
independent OLS projection (_reconstruct_Uhat_from_VD) since test subjects
were never seen by AMP.

Public API (identical signatures to multimodal_prediction_unified.py):
  MultimodalClusterAllUPipeline       — pipeline class
  train_linear_cox_survival           — fit the Cox head
  predict_survival_from_test_data_all — test-time log-risk prediction
"""

import numpy as np
import importlib
from scipy.linalg import block_diag
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.mixture import GaussianMixture

from emp_bayes_data import ClusterParametricEBPipeline

# ---------------------------
# Project modules (loaded via importlib for robustness)
# ---------------------------
pca_pack      = importlib.import_module("pca_pack")
amp           = importlib.import_module("amp_data_pipeline")
preprocessing = importlib.import_module("preprocessing")


# =============================================================================
# Helpers
# =============================================================================

def _jitter_psd(A, eps=1e-8):
    A = np.asarray(A, dtype=np.float64)
    return A + eps * np.eye(A.shape[0], dtype=np.float64)


def _reconstruct_Uhat_from_VD(X_list, V_dict, D_dict, n):
    """
    Back-project U^ to test/train data using {V, D} from AMP.

    U^ = X * A * (A^T A)^+ where A = (1/n) * V * D

    Parameters
    ----------
    X_list : list of ndarray
        Data blocks (n_samples_test, p_k).
    V_dict : dict[int -> ndarray]
        Denoised loadings per view, shape (p_k, r_k).
    D_dict : dict[int -> ndarray]
        Signal diagonal per view, shape (r_k, r_k).
    n : int
        Training sample size used during AMP (for the 1/n scaling).

    Returns
    -------
    dict[int -> ndarray]
        Map view index -> U_hat (n_test, r_k).
    """
    U_hat = {}
    for k, Xk in enumerate(X_list):
        V_k = V_dict[k]
        D_k = D_dict[k]
        A_k = (1.0 / n) * V_k @ D_k
        AtA_inv = np.linalg.pinv(A_k.T @ A_k)
        U_hat[k] = Xk @ A_k @ AtA_inv
    return U_hat


def _reconstruct_Uhat_onsager(F_dict, V_dict, D_dict, n):
    """
    Onsager-debiased training covariate (see DAIF_training_projection_fixes.pdf, Sec. 2):
        U_hat_k = F_k (Sigma_L_k)^{-1} D_k^{-1}
    where F_k is the raw (pre-denoising) AMP field for HD modality k
    (amp_results["U_non_denoised"][k]) and Sigma_L_k = (1/n) V_k^T V_k.

    Unlike _reconstruct_Uhat_from_VD, this does not re-project the training
    X_k — it reuses the AMP field, which is already independent of the
    loading direction's in-sample noise. Training-covariates only; test-time
    reconstruction still uses _reconstruct_Uhat_from_VD (independent OLS
    projection of held-out data).

    Parameters
    ----------
    F_dict : dict[int -> ndarray]
        Raw (pre-denoising) AMP field per HD modality, shape (n, r_k).
    V_dict : dict[int -> ndarray]
        Denoised loadings per view, shape (p_k, r_k).
    D_dict : dict[int -> ndarray]
        Signal diagonal per view, shape (r_k, r_k).
    n : int
        AMP training sample size.

    Returns
    -------
    dict[int -> ndarray]
        Map view index -> U_hat (n, r_k).
    """
    U_hat = {}
    for k, F_k in F_dict.items():
        V_k = V_dict[k]
        D_k = D_dict[k]
        Sigma_L_k = (V_k.T @ V_k) / n
        Sigma_L_inv = np.linalg.inv(Sigma_L_k)
        D_inv = np.linalg.inv(D_k)
        U_hat[k] = F_k @ Sigma_L_inv @ D_inv
    return U_hat


def _transform_lowdim_to_Uhat(pca_model_low, X_ld_list):
    """
    Prepare LD blocks as raw observations for EB denoising.

    The LD model is X̃_j = L_j U_j + Z_j, Z_j ~ N(0, I).
    The EB denoiser takes the raw observation X̃_j as input with M=L̂_j, S=I,
    so we return X̃_j (feature-selected and centered) without pre-inverting L.

    Parameters
    ----------
    pca_model_low : object or None
        Low-dim model with `indices_`, `means_` lists.
    X_ld_list : list[ndarray] or None
        Test-time LD blocks.

    Returns
    -------
    list[ndarray]
        Each X̃_j has shape (n_samples, p_j) — the raw centered observation.
    """
    if pca_model_low is None or X_ld_list is None:
        return []
    U_hat_ld = []
    for j, A in enumerate(X_ld_list):
        ind = pca_model_low.indices_[j]
        mean_A = pca_model_low.means_[j]
        A_sel = A[:, ind]
        U_hat_ld.append(A_sel - mean_A)
    return U_hat_ld


# ==== GMM posterior utilities (robust, no inv(Ck)/inv(S)) =====================

def gmm_responsibilities(X, M, S, m_prior, cov_prior, weights, eps=1e-12):
    """
    Responsibilities r_i(k)  w_k · N(x_i; M m_k, S + M C_k M^T).
    We compute (per i,k): log w_k - 0.5[ log|£_k| + (x_i-¼_k)^T £_k^{-1} (x_i-¼_k) ].
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    m_prior  = np.asarray(m_prior, dtype=np.float64)
    cov_prior = np.asarray(cov_prior, dtype=np.float64)
    weights  = np.asarray(weights, dtype=np.float64)

    N, p = X.shape
    K, d = m_prior.shape[0], M.shape[1]

    logits = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        mu_k   = M @ m_prior[k]                     # (p,)
        Sigma_k = _jitter_psd(S + M @ cov_prior[k] @ M.T)
        sign, logabs = np.linalg.slogdet(Sigma_k)
        inv_S = np.linalg.inv(Sigma_k)
        diff  = X - mu_k
        quad  = np.einsum("ni,ij,nj->n", diff, inv_S, diff)   # (N,)
        logits[:, k] = np.log(weights[k] + eps) - 0.5 * (logabs + quad)  # omit constant p log 2pi

    logits -= logits.max(axis=1, keepdims=True)
    logits = np.clip(logits, -700, 700)
    R = np.exp(logits)
    R /= R.sum(axis=1, keepdims=True)
    return R  # (N, K)


def gmm_linear_gaussian_posterior(X, M, S, m_prior, cov_prior, weights):
    """
    Posterior for GMM prior and linear-Gaussian x|z (no inv(Ck)/inv(S)).
      £_{k|x}   = C_k - C_k M^T (S + M C_k M^T)^{-1} M C_k
      ¼_{k|x_i} = m_k + C_k M^T (S + M C_k M^T)^{-1} (x_i - M m_k)
    Returns
    -------
    R  : (N,K) responsibilities
    MU : (N,K,d) posterior means
    SIG: (K,d,d) posterior covariances (independent of i)
    """
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    m_prior   = np.asarray(m_prior, dtype=np.float64)
    cov_prior = np.asarray(cov_prior, dtype=np.float64)
    weights   = np.asarray(weights, dtype=np.float64)

    N, p = X.shape
    K, d = m_prior.shape[0], M.shape[1]

    R = gmm_responsibilities(X, M, S, m_prior, cov_prior, weights)  # (N,K)

    MU  = np.zeros((N, K, d), dtype=np.float64)
    SIG = np.zeros((K, d, d),   dtype=np.float64)

    for k in range(K):
        m_k = m_prior[k]                 # (d,)
        C_k = cov_prior[k]               # (d,d)

        Sigma_obs_k = _jitter_psd(S + M @ C_k @ M.T)          # (p,p)
        Sigma_obs_k_inv = np.linalg.inv(Sigma_obs_k)          # (p,p)

        # Kalman gain: K_k = C_k M^T (S + M C_k M^T)^{-1}
        K_k = C_k @ M.T @ Sigma_obs_k_inv                     # (d,p)

        # Posterior covariance (independent of i)
        SIG[k] = C_k - K_k @ M @ C_k                          # (d,d)

        # Posterior means for all samples
        innovation = X - (M @ m_k)                            # (N,p)
        MU[:, k, :] = m_k + innovation @ K_k.T                # (N,d)

    return R, MU, SIG


def _build_posterior_gmm(weights, means, covs, random_state=None):
    """
    Build an sklearn GaussianMixture representing p(z | x_i) from:
      weights: (K,), means: (K,d), covs: (K,d,d), covariance_type='full'
    """
    K, d = means.shape
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=random_state)
    gmm.weights_ = np.asarray(weights, dtype=np.float64)
    gmm.means_   = np.asarray(means, dtype=np.float64)
    gmm.covariances_ = np.asarray(covs, dtype=np.float64)
    return gmm


# =============================================================================
# Linear Cox survival with GMM posterior MC sampling
# =============================================================================

def _cox_partial_loss(log_hazard: torch.Tensor,
                      time: torch.Tensor,
                      event: torch.Tensor) -> torch.Tensor:
    """
    Breslow-style partial log-likelihood (negated, for minimization).

    Sort samples by descending survival time; log-cumsum-exp gives log of the
    risk-set denominator at each event time.  Normalized by number of events.
    """
    order = torch.argsort(time, descending=True)
    lh = log_hazard[order]
    ev = event[order]
    log_cumsum = torch.logcumsumexp(lh, dim=0)
    n_events = ev.sum().clamp_min(1.0)
    return -((lh - log_cumsum) * ev).sum() / n_events


def train_linear_cox_survival(
    pipeline, X_hd, X_ld, event, time,
    *,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    mc_samples: int = 16,
    seed: int = 42,
    print_every: int = 100,
):
    """
    Cluster-wise linear Cox model with GMM posterior MC sampling.

    For each epoch, draw mc_samples joint samples Z^(m) = (Z^(m)_1,...,Z^(m)_n)
    from the per-sample posterior GMMs, compute Cox partial log-likelihood on
    each draw, and optimize the mean over draws.  The MC averaging is necessary
    even for a linear β because the log-sum-exp risk-set denominator makes
    E[Cox(β'Z)|Û] ≠ Cox(β'E[Z|Û]).

    Risk scores: log_risk_i = Σ_c β_c' E[Z_{i,c}|Û_i]  (at prediction time).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    V_dict = pipeline.amp_results["V_denoised"]
    D_dict = pipeline.amp_results["signal_diag_dict"]
    F_dict = pipeline.amp_results["U_non_denoised"]
    cluster_u = pipeline.cluster_model_u
    labels_all = cluster_u.cluster_labels

    n_train = X_hd[0].shape[0]
    m_hd = len(X_hd)
    m_ld = len(X_ld) if X_ld is not None else 0

    U_hat_hd = _reconstruct_Uhat_onsager(F_dict, V_dict, D_dict, n_train)
    U_hat_ld_list = _transform_lowdim_to_Uhat(getattr(pipeline, "pca_model_low", None), X_ld)

    M_ld_list = []
    if m_ld:
        for j in range(m_ld):
            L = pipeline.pca_model_low.pca_results[j].sample_aligns
            M_ld_list.append(np.diag(L) if L.ndim == 1 else L)

    # Per-cluster GMM posterior params
    cluster_inputs = {}
    for c in np.unique(labels_all):
        mods_hd = [k for k in range(m_hd) if labels_all[k] == c]
        mods_ld = [j for j in range(m_ld) if labels_all[m_hd + j] == c]
        U_parts = []
        if mods_hd:
            U_parts.extend([U_hat_hd[k] for k in mods_hd])
        if mods_ld:
            U_parts.extend([U_hat_ld_list[j] for j in mods_ld])
        if not U_parts:
            continue
        Uc = np.hstack(U_parts)  # (n, d_c)
        d_c = Uc.shape[1]

        M_blocks, S_blocks = [], []
        for k in mods_hd:
            M_blocks.append(np.eye(U_hat_hd[k].shape[1]))
            V_k = V_dict[k]
            D_k = D_dict[k]
            D_inv = np.linalg.inv(D_k)
            Sigma_k = (1.0 / n_train) * (V_k.T @ V_k)
            S_blocks.append(D_inv @ np.linalg.pinv(Sigma_k) @ D_inv)
        for j in mods_ld:
            M_blocks.append(M_ld_list[j])
            S_blocks.append(np.eye(M_ld_list[j].shape[0]))

        M_c = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
        S_c = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))

        if hasattr(cluster_u, "cluster_priors_gmm") and c in cluster_u.cluster_priors_gmm:
            m_prior  = cluster_u.cluster_priors_gmm[c]["m_prior"]
            cov_prior = cluster_u.cluster_priors_gmm[c]["cov_prior"]
            weights  = cluster_u.cluster_priors_gmm[c]["weights"]
        else:
            eb = cluster_u.cluster_models[c]
            m_prior, cov_prior, weights = eb.m_prior, eb.cov_prior, eb.weights

        R_np, MU_np, SIG_np = gmm_linear_gaussian_posterior(
            Uc, M_c, S_c, m_prior, cov_prior, weights)
        cluster_inputs[c] = {
            "R_np": R_np, "MU_np": MU_np, "SIG_np": SIG_np, "d": d_c,
        }

    # One linear layer β_c : R^{d_c} → R per cluster (no bias)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster_models = {
        c: nn.Linear(inp["d"], 1, bias=False).to(device)
        for c, inp in cluster_inputs.items()
    }
    opts = {
        c: AdamW(mdl.parameters(), lr=lr, weight_decay=weight_decay)
        for c, mdl in cluster_models.items()
    }

    event_t = torch.as_tensor(event, dtype=torch.float32, device=device)
    time_t  = torch.as_tensor(time,  dtype=torch.float32, device=device)
    n = int(event_t.shape[0])

    for ep in range(epochs):
        for mdl in cluster_models.values():
            mdl.train()
            for p in mdl.parameters():
                p.requires_grad_(True)
        for opt in opts.values():
            opt.zero_grad()

        # Pre-sample mc_samples posterior draws for each (sample, cluster)
        # cluster_epoch_Z[c]: shape (n, mc_samples, d_c), fixed tensors
        cluster_epoch_Z = {}
        for c, inp in cluster_inputs.items():
            R_np, MU_np, SIG_np = inp["R_np"], inp["MU_np"], inp["SIG_np"]
            sample_list = []
            for i in range(n):
                gmm_i = _build_posterior_gmm(
                    weights=R_np[i], means=MU_np[i], covs=SIG_np,
                    random_state=None,          # fresh randomness each epoch
                )
                Zi, _ = gmm_i.sample(mc_samples)   # (mc_samples, d_c)
                sample_list.append(Zi)
            # stack → (n, mc_samples, d_c), convert to tensor (no grad)
            Z_c = np.stack(sample_list, axis=0)
            cluster_epoch_Z[c] = torch.tensor(Z_c, dtype=torch.float32, device=device)

        # Average Cox loss over mc_samples joint draws
        total_loss = torch.zeros(1, device=device)
        for m in range(mc_samples):
            log_h_m = torch.zeros(n, device=device)
            for c in cluster_inputs:
                Zm = cluster_epoch_Z[c][:, m, :]          # (n, d_c)
                log_h_m = log_h_m + cluster_models[c](Zm).squeeze(1)
            total_loss = total_loss + _cox_partial_loss(log_h_m, time_t, event_t) / mc_samples

        total_loss.backward()

        for c, opt in opts.items():
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(cluster_models[c].parameters(), grad_clip)
            opt.step()

        if print_every and (ep % print_every == 0):
            print(f"[LinearCox-GMM] Ep {ep:4d} | Cox NLL {total_loss.item():.6f}", flush=True)

    # Final training-set log-risk scores (MC average of posterior means)
    with torch.no_grad():
        for mdl in cluster_models.values():
            mdl.eval()
        log_risk = torch.zeros(n, device=device)
        for c, inp in cluster_inputs.items():
            R_np, MU_np, SIG_np = inp["R_np"], inp["MU_np"], inp["SIG_np"]
            rows = []
            for i in range(n):
                gmm_i = _build_posterior_gmm(
                    weights=R_np[i], means=MU_np[i], covs=SIG_np,
                    random_state=seed + 77777 + i,
                )
                Zi, _ = gmm_i.sample(mc_samples)
                Zi_t  = torch.tensor(Zi, dtype=torch.float32, device=device)
                rows.append(cluster_models[c](Zi_t).mean(dim=0, keepdim=True))   # (1,1)
            E_c = torch.cat(rows, dim=0).squeeze(1)   # (n,)
            log_risk = log_risk + E_c

    pipeline.cluster_models_cox = cluster_models
    pipeline.log_risk_train = log_risk.cpu().numpy()
    pipeline.amp_results["cox_model_linear"]  = cluster_models
    pipeline.amp_results["log_risk_train"]    = pipeline.log_risk_train
    return cluster_models, pipeline.log_risk_train


# =============================================================================
# Fallback wrappers for extract_normalized_U/V (HD)
# =============================================================================

try:
    from complete_pipeline import extract_normalized_U as _extract_normalized_U
    from complete_pipeline import extract_normalized_V as _extract_normalized_V
except Exception:
    _extract_normalized_U = None
    _extract_normalized_V = None


def _extract_normalized_U_hd(pca_model_hd, X_list_hd):
    if _extract_normalized_U is not None:
        return _extract_normalized_U(pca_model_hd, X_list_hd)
    U_list = []
    for k in range(len(X_list_hd)):
        pr = pca_model_hd.pca_results[k]
        for attr in ["U_norm", "U_normalized", "U"]:
            if hasattr(pr, attr):
                U = getattr(pr, attr)
                n = U.shape[0]
                U_list.append(U / np.sqrt((U**2).sum(axis=0)) * np.sqrt(n))
                break
        else:
            raise RuntimeError("No extract_normalized_U helper found and could not infer U from pca_results.")
    return U_list


def _extract_normalized_V_hd(pca_model_hd, X_list_hd):
    if _extract_normalized_V is not None:
        return _extract_normalized_V(pca_model_hd, X_list_hd)
    V_list = []
    for k in range(len(X_list_hd)):
        pr = pca_model_hd.pca_results[k]
        for attr in ["V_norm", "V_normalized", "V"]:
            if hasattr(pr, attr):
                V = getattr(pr, attr)
                p = V.shape[0]
                V_list.append(V / np.sqrt((V**2).sum(axis=0)) * np.sqrt(p))
                break
        else:
            raise RuntimeError("No extract_normalized_V helper found and could not infer V from pca_results.")
    return V_list


# =============================================================================
# Pipeline base — survival-only state
# =============================================================================

class _PipelineBase:
    """Base class holding common state shared by the survival pipeline."""
    def __init__(self):
        self.pca_model = None
        self.pca_model_low = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results = None
        self.task = "survival"            # kept only for predict_survival_from_test_data_all's check
        self.y_surv_train = None          # (event_arr, time_arr); informational — see train_linear_cox_survival
        self.m_hd = None
        self.m_ld = None


# =============================================================================
# Unified Pipeline (HD+LD clustering + AMP) — survival only
# =============================================================================

class MultimodalClusterAllUPipeline(_PipelineBase):
    """
    Full pipeline:
      1) PCA for HD; optional LowDim loadings for LD
      2) Cluster ALL modalities (HD + LD) using normalized U
      3) Build ClusterParametricEBPipeline for U over ALL (HD + LD)
      4) Build ClusterParametricEBPipeline for V over HD only (per modality)
      5) Run AMP; call train_linear_cox_survival(...) explicitly afterward
         to fit the Cox head (see the survival notebooks for the usage pattern)
    """
    def __init__(self):
        super().__init__()
        self.pca_hd = None
        self.pca_ld = None
        self.cluster_labels_U_all = None  # length m_hd + m_ld

    def fit_pca_highdim(self, X_list_hd, K_list_hd, preprocess=False):
        if preprocess:
            print("Preprocessing HD modalities (normalizing variance)...", flush=True)
            diag = preprocessing.MultiModalityPCADiagnostics()
            X_list_hd = diag.normalize_obs(X_list_hd, K_list_hd)

        print("Fitting PCA for HD modalities...", flush=True)
        self.pca_hd = pca_pack.MultiModalityPCA()
        self.pca_hd.fit(X_list_hd, K_list_hd, plot_residual=False)

        self.m_hd = len(X_list_hd)
        self.pca_model = self.pca_hd
        return self

    def fit_lowdim(self, A_list_ld=None, top_features=None):
        if A_list_ld is None:
            self.pca_ld = None
            self.m_ld = 0
            self.pca_model_low = None
            return self
        print("Fitting LowDimModalityLoadings for LD modalities...", flush=True)
        ld = preprocessing.LowDimModalityLoadings(top_features=top_features)
        ld.fit(A_list_ld)
        self.pca_ld = ld
        self.m_ld = len(A_list_ld)
        self.pca_model_low = self.pca_ld
        return self

    def cluster_all_modalities(self, X_list_hd, method="cka", num_clusters=None, threshold=None,
                               auto_method="silhouette", gap_B=10, gap_K_max=10, gap_seed=42):
        if self.pca_hd is None:
            raise RuntimeError("Call fit_pca_highdim first.")

        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)

        U_norm_ld = []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                U_ld = self.pca_ld.pca_results[j].X
                U_norm_ld.append(U_ld)

        U_all = U_norm_hd + U_norm_ld
        m_hd = len(U_norm_hd)
        m_ld = len(U_norm_ld)

        if len(U_all) == 0:
            raise ValueError("No modalities found to cluster.")

        print("Clustering modalities (HD + LD) ...", flush=True)
        from hierarchical_clustering_modalities import ModalityClusterer
        cl = ModalityClusterer(U_all)
        _ = cl.compute_similarity_matrix(method, epsilon=0.1, sigma=1.0)
        labels = np.asarray(cl.cluster_modalities(
            method, num_clusters=num_clusters, threshold=threshold,
            auto_method=auto_method, gap_B=gap_B, gap_K_max=gap_K_max, gap_seed=gap_seed
        ))

        if labels.shape[0] != (m_hd + m_ld):
            raise ValueError(f"cluster_labels_U length mismatch: got {labels.shape[0]}, expected {m_hd + m_ld}")

        self.cluster_labels_U_all = labels
        print(f"Cluster labels (HD first, then LD): {labels}", flush=True)
        return labels

    def build_cluster_models(self, X_list_hd, print_priors=True):
        if self.pca_hd is None:
            raise RuntimeError("PCA (HD) must be done first.")
        if self.cluster_labels_U_all is None:
            raise RuntimeError("Run cluster_all_modalities first.")

        m_hd = len(X_list_hd)
        m_ld = len(self.pca_ld.pca_results) if self.pca_ld is not None else 0

        # U over ALL
        # Clip alignments away from 0: a PC with sample_align=0 lies exactly on
        # the BBP threshold (no detectable signal). M_bd becomes singular there.
        # Reduce K_hd in the notebook if many PCs are clipped.
        _EPS = 1e-2
        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)
        M_u_hd, S_u_hd = [], []
        for k in range(m_hd):
            sa = self.pca_hd.pca_results[k].sample_aligns
            n_zero = int((sa < _EPS).sum())
            if n_zero:
                print(f"  [build_cluster_models] Modality {k}: {n_zero}/{len(sa)} "
                      f"sample_aligns < {_EPS} (below BBP threshold) — clipping. "
                      f"Consider reducing K.", flush=True)
            sa = np.clip(sa, _EPS, 1.0)
            M_u_hd.append(np.diag(sa))
            S_u_hd.append(np.diag(np.clip(1 - sa**2, _EPS, 1.0)))

        U_norm_ld, M_u_ld, S_u_ld = [], [], []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                pr_ld = self.pca_ld.pca_results[j]
                U_ld = pr_ld.X
                sa = pr_ld.sample_aligns
                U_norm_ld.append(U_ld)
                if sa.ndim == 1:
                    M_u_ld.append(np.diag(sa))
                    L = sa.shape[0]
                else:
                    M_u_ld.append(sa)
                    L = sa.shape[0]
                S_u_ld.append(np.eye(L))  # LD S = I

        U_all = U_norm_hd + U_norm_ld
        M_all = M_u_hd + M_u_ld
        S_all = S_u_hd + S_u_ld

        print("Fitting cluster-wise EB model for U (HD + LD) ...", flush=True)
        self.cluster_model_u = ClusterParametricEBPipeline(
            covariance_mode='full', choose_comp=True, n_components=int(np.cbrt(U_all[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(U_all, M_all, S_all, self.cluster_labels_U_all)

        if print_priors and getattr(self.cluster_model_u, "cluster_priors_gmm", None):
            for c, dct in self.cluster_model_u.cluster_priors_gmm.items():
                print(f"\n=== cluster_model_u (GMM) : Cluster {c} ===", flush=True)
                print(f"means shape: {dct['m_prior'].shape} | covs: {dct['cov_prior'].shape} | K: {len(dct['weights'])}", flush=True)

        # V over HD only
        V_norm_hd = _extract_normalized_V_hd(self.pca_hd, X_list_hd)
        M_v_hd, S_v_hd = [], []
        for k in range(m_hd):
            fa = self.pca_hd.pca_results[k].feature_aligns
            n_zero = int((fa < _EPS).sum())
            if n_zero:
                print(f"  [build_cluster_models] Modality {k}: {n_zero}/{len(fa)} "
                      f"feature_aligns < {_EPS} — clipping.", flush=True)
            fa = np.clip(fa, _EPS, 1.0)
            M_v_hd.append(np.diag(fa))
            S_v_hd.append(np.diag(np.clip(1 - fa**2, _EPS, 1.0)))
        labels_V = np.arange(m_hd)

        print("Fitting per-modality EB model for V (HD only) ...", flush=True)
        self.cluster_model_v = ClusterParametricEBPipeline(
            covariance_mode='full', choose_comp=True, n_components=int(np.cbrt(V_norm_hd[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(V_norm_hd, M_v_hd, S_v_hd, labels_V)

        self.pca_model = self.pca_hd
        self.pca_model_low = self.pca_ld
        return self

    def run_amp(self, X_list_hd, amp_iters=20, muteu=False, mutev=False):
        """
        Run AMP denoising only. Unlike multimodal_prediction_unified.py, this does
        NOT attempt to auto-train a supervised head afterward — in the unified
        pipeline that auto-hook is a no-op for survival tasks anyway (see the
        module docstring). Call train_linear_cox_survival(pipeline, ...)
        explicitly afterward, as the survival notebooks do.
        """
        if self.pca_hd is None or self.cluster_model_u is None or self.cluster_model_v is None:
            raise RuntimeError("PCA and cluster models must be prepared before AMP.")

        print("Running AMP denoising (HD + LD) ...", flush=True)
        self.amp_results = amp.ebamp_cluster_U_all_modalities(
            pca_model_hd=self.pca_hd,
            cluster_model_u=self.cluster_model_u,
            cluster_model_v=self.cluster_model_v,
            pca_model_ld=self.pca_ld,
            amp_iters=amp_iters,
            muteu=muteu,
            mutev=mutev,
        )
        self.amp_results["cluster_model_u"] = self.cluster_model_u
        return self.amp_results


# =============================================================================
# Prediction
# =============================================================================

def predict_survival_from_test_data_all(
    pipeline, X_test_hd, X_test_ld=None, mc_samples: int = 50, seed: int = 1234
):
    """
    SURVIVAL prediction using the linear Cox model trained with GMM posterior MC.

    Returns (U_denoised_hd, U_denoised_ld, log_risk) where log_risk[i] is the
    predicted log-hazard for test sample i (higher → higher risk).
    """
    if pipeline.amp_results is None:
        raise RuntimeError("Run pipeline.run_amp(...) first.")
    if pipeline.task != "survival":
        raise RuntimeError("Pipeline task is not 'survival'.")
    if "cox_model_linear" not in pipeline.amp_results:
        raise RuntimeError("No Cox model found; run train_linear_cox_survival(...) first.")

    V_dict = pipeline.amp_results["V_denoised"]
    D_dict = pipeline.amp_results["signal_diag_dict"]
    cluster_model_u = pipeline.amp_results["cluster_model_u"]
    labels_all = cluster_model_u.cluster_labels
    cluster_denoisers = cluster_model_u.cluster_denoisers
    cluster_models = pipeline.amp_results["cox_model_linear"]

    m_hd = pipeline.m_hd
    m_ld = pipeline.m_ld
    n_train = list(pipeline.pca_model.pca_results.values())[0].U.shape[0]
    n_test  = X_test_hd[0].shape[0]

    U_hat_hd = _reconstruct_Uhat_from_VD(X_test_hd, V_dict, D_dict, n_train)
    U_hat_ld_list = _transform_lowdim_to_Uhat(getattr(pipeline, "pca_model_low", None), X_test_ld)

    # Denoise test U per cluster (same as regression path)
    U_denoised_hd, U_denoised_ld = {}, {}
    for cid in np.unique(labels_all):
        mods_hd = [k for k in range(m_hd) if labels_all[k] == cid]
        mods_ld = [j for j in range(m_ld) if labels_all[m_hd + j] == cid]
        if not mods_hd and not mods_ld:
            continue
        U_parts = []
        if mods_hd: U_parts += [U_hat_hd[k] for k in mods_hd]
        if mods_ld: U_parts += [U_hat_ld_list[j] for j in mods_ld]
        U_concat = np.hstack(U_parts)

        M_blocks, S_blocks = [], []
        for k in mods_hd:
            M_blocks.append(np.eye(U_hat_hd[k].shape[1]))
            V_k = V_dict[k]; D_k = D_dict[k]
            D_inv = np.linalg.inv(D_k)
            Sigma_k = (1.0 / n_train) * (V_k.T @ V_k)
            S_blocks.append(D_inv @ np.linalg.pinv(Sigma_k) @ D_inv)
        for j in mods_ld:
            L = pipeline.pca_model_low.pca_results[j].sample_aligns
            M_blocks.append(np.diag(L) if L.ndim == 1 else L)
            S_blocks.append(np.eye(L.shape[0]))
        M_c = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
        S_c = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))
        U_den = cluster_denoisers[cid]["denoise"](U_concat, M_c, S_c)

        widths = ([U_hat_hd[k].shape[1] for k in mods_hd] +
                  [U_hat_ld_list[j].shape[1] for j in mods_ld])
        edges = np.cumsum([0] + widths)
        cursor = 0
        for k in mods_hd:
            s, e = edges[cursor], edges[cursor + 1]; cursor += 1
            U_denoised_hd[k] = U_den[:, s:e]
        for j in mods_ld:
            s, e = edges[cursor], edges[cursor + 1]; cursor += 1
            U_denoised_ld[j] = U_den[:, s:e]

    # Compute log-risk via posterior GMM MC (same MC averaging as training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_risk = torch.zeros(n_test, device=device)

    with torch.no_grad():
        for mdl in cluster_models.values():
            mdl.eval()

        for cid in np.unique(labels_all):
            mods_hd = [k for k in range(m_hd) if labels_all[k] == cid]
            mods_ld = [j for j in range(m_ld) if labels_all[m_hd + j] == cid]
            if not mods_hd and not mods_ld:
                continue
            if cid not in cluster_models:
                continue

            Uc = np.hstack(
                ([U_hat_hd[k] for k in mods_hd] if mods_hd else []) +
                ([U_hat_ld_list[j] for j in mods_ld] if mods_ld else [])
            )

            M_blocks, S_blocks = [], []
            for k in mods_hd:
                M_blocks.append(np.eye(U_hat_hd[k].shape[1]))
                V_k = V_dict[k]; D_k = D_dict[k]
                D_inv = np.linalg.inv(D_k)
                Sigma_k = (1.0 / n_train) * (V_k.T @ V_k)
                S_blocks.append(D_inv @ np.linalg.pinv(Sigma_k) @ D_inv)
            for j in mods_ld:
                L = pipeline.pca_model_low.pca_results[j].sample_aligns
                M_blocks.append(np.diag(L) if L.ndim == 1 else L)
                S_blocks.append(np.eye(L.shape[0]))
            M_c = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
            S_c = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))

            if hasattr(cluster_model_u, "cluster_priors_gmm") and cid in cluster_model_u.cluster_priors_gmm:
                m_prior  = cluster_model_u.cluster_priors_gmm[cid]["m_prior"]
                cov_prior = cluster_model_u.cluster_priors_gmm[cid]["cov_prior"]
                weights  = cluster_model_u.cluster_priors_gmm[cid]["weights"]
            else:
                eb = cluster_model_u.cluster_models[cid]
                m_prior, cov_prior, weights = eb.m_prior, eb.cov_prior, eb.weights

            R_np, MU_np, SIG_np = gmm_linear_gaussian_posterior(
                Uc, M_c, S_c, m_prior, cov_prior, weights)

            model = cluster_models[cid]
            E_c = torch.zeros(n_test, device=device)
            for i in range(n_test):
                gmm_i = _build_posterior_gmm(
                    weights=R_np[i], means=MU_np[i], covs=SIG_np,
                    random_state=seed + i,
                )
                Zi, _ = gmm_i.sample(mc_samples)
                Zi_t  = torch.tensor(Zi, dtype=torch.float32, device=device)
                E_c[i] = model(Zi_t).mean(dim=0).squeeze()
            log_risk = log_risk + E_c

    return U_denoised_hd, U_denoised_ld, log_risk.cpu().numpy()
