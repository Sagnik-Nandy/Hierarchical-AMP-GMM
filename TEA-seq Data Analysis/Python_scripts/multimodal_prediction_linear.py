"""
multimodal_prediction_linear.py
================================
Stripped-down version of multimodal_prediction_unified.py that retains only the
linear regression head (beta_hat = pinv(U_all) @ y_train).

Removed entirely:
  - train_nonlinear_regressor / train_nonlinear_classifier
  - SimpleDeepRegressor / SimpleDeepClassifier / _mlp_layers
  - gmm_responsibilities / gmm_linear_gaussian_posterior / _build_posterior_gmm
  - PosteriorAwarePredictor, _fit_predictor, predict_from_test_data
  - predict_celltypes_from_test_data_all
  - _mc_sample_posteriors, _build_per_modality_MS, _denoise_U_hat_hd_ld,
    _compute_cluster_posteriors_hd_ld
  - All Torch imports and utilities
  - Classification branch in _post_amp_supervised

Training-covariate projection: onsager (Onsager-debiased AMP field; see
DAIF_training_projection_fixes.pdf, Sec. 2). _post_amp_supervised's training
U_hat is reconstructed from the raw pre-denoising AMP field
(amp_results["U_non_denoised"]) via _reconstruct_Uhat_onsager instead of
re-projecting the training X_hd in-sample. Test-time reconstruction in
predict_from_test_data_all is unchanged — it still uses the independent OLS
projection (_reconstruct_Uhat_from_VD) since test subjects were never seen
by AMP.

Public API (identical signatures to multimodal_prediction_unified.py):
  MultimodalClusterAllUPipeline   — pipeline class
  predict_from_test_data_all      — test-time prediction (linear head only)
"""

import numpy as np
import importlib
from scipy.linalg import block_diag

from emp_bayes_data import ClusterParametricEBPipeline

# ---------------------------
# Project modules
# ---------------------------
pca_pack       = importlib.import_module("pca_pack")
emp_bayes      = importlib.import_module("emp_bayes_data")
amp            = importlib.import_module("amp_data_pipeline")
preprocessing  = importlib.import_module("preprocessing")


# =============================================================================
# Helpers
# =============================================================================

def extract_normalized_U(pca_model, X_list):
    """
    Compute column-wise L2-normalized (and n-rescaled) left factors U for each view.

    Parameters
    ----------
    pca_model : object with pca_results[k].U per modality k
    X_list : list of ndarray — only used for length

    Returns
    -------
    list of ndarray, each (n_samples, r_k) with columns normalized to norm n
    """
    out = []
    for k in range(len(X_list)):
        U = pca_model.pca_results[k].U
        n = U.shape[0]
        out.append(U / np.sqrt((U**2).sum(axis=0)) * np.sqrt(n))
    return out


def extract_normalized_V(pca_model, X_list):
    """
    Compute column-wise L2-normalized (and p-rescaled) right factors V for each view.

    Parameters
    ----------
    pca_model : object with pca_results[k].V per modality k
    X_list : list of ndarray — only used for length

    Returns
    -------
    list of ndarray, each (p_k, r_k) with columns normalized to norm p_k
    """
    out = []
    for k in range(len(X_list)):
        V = pca_model.pca_results[k].V
        p = V.shape[0]
        out.append(V / np.sqrt((V**2).sum(axis=0)) * np.sqrt(p))
    return out


def _psd_clip(A, floor=1e-10):
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, floor)
    return (evecs * evals) @ evecs.T


def _jitter_psd(A, eps=1e-8):
    A = np.asarray(A, dtype=np.float64)
    return A + eps * np.eye(A.shape[0], dtype=np.float64)


def _reconstruct_Uhat_from_VD(X_list, V_dict, D_dict, n):
    """
    Back-project U^ to test/train data using {V, D} from AMP.

    U^ = X A (A^T A)^+  where A = (1/n) V D

    Parameters
    ----------
    X_list : list of ndarray (n_samples_test, p_k)
    V_dict : dict[int -> ndarray]  denoised loadings per view (p_k, r_k)
    D_dict : dict[int -> ndarray]  signal diagonal per view  (r_k, r_k)
    n      : int  training sample size used during AMP

    Returns
    -------
    dict[int -> ndarray]  U_hat (n_test, r_k) per view index
    """
    U_hat = {}
    for k, Xk in enumerate(X_list):
        V_k   = V_dict[k]
        D_k   = D_dict[k]
        A_k   = (1.0 / n) * V_k @ D_k
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
    F_dict : dict[int -> ndarray]  raw (pre-denoising) AMP field per HD modality (n, r_k)
    V_dict : dict[int -> ndarray]  denoised loadings per view (p_k, r_k)
    D_dict : dict[int -> ndarray]  signal diagonal per view  (r_k, r_k)
    n      : int  AMP training sample size

    Returns
    -------
    dict[int -> ndarray]  U_hat (n, r_k) per view index
    """
    U_hat = {}
    for k, F_k in F_dict.items():
        V_k = V_dict[k]
        D_k = D_dict[k]
        Sigma_L_k   = (V_k.T @ V_k) / n
        Sigma_L_inv = np.linalg.inv(Sigma_L_k)
        D_inv       = np.linalg.inv(D_k)
        U_hat[k]    = F_k @ Sigma_L_inv @ D_inv
    return U_hat


def _transform_lowdim_to_Uhat(pca_model_low, X_ld_list):
    """
    Prepare LD blocks as raw observations for EB denoising.

    Returns feature-selected and centred X_ld blocks; shape (n_samples, p_j).

    Parameters
    ----------
    pca_model_low : object or None
    X_ld_list     : list[ndarray] or None

    Returns
    -------
    list[ndarray]
    """
    if pca_model_low is None or X_ld_list is None:
        return []
    U_hat_ld = []
    for j, A in enumerate(X_ld_list):
        ind    = pca_model_low.indices_[j]
        mean_A = pca_model_low.means_[j]
        U_hat_ld.append(A[:, ind] - mean_A)
    return U_hat_ld


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
            raise RuntimeError(
                "No extract_normalized_U helper found and could not infer U from pca_results."
            )
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
            raise RuntimeError(
                "No extract_normalized_V helper found and could not infer V from pca_results."
            )
    return V_list


# =============================================================================
# Pipeline base — linear regression head only
# =============================================================================

class _PipelineBase:
    """Base class holding common state and the post-AMP linear regression hook."""

    def __init__(self):
        self.pca_model      = None
        self.pca_model_low  = None
        self.cluster_model_u = None
        self.cluster_model_v = None
        self.amp_results    = None
        self.y_train        = None
        self.task           = "regression"
        self.relation       = "linear"
        self.m_hd           = None
        self.m_ld           = None

    def _post_amp_supervised(self, X_hd, X_ld=None):
        """
        After AMP, denoise per-cluster U (HD + LD) and fit the linear regression head:
            beta = pinv(U_all) @ y_train
        Result stored in amp_results["beta_hat"].
        """
        if self.y_train is None:
            return

        m_hd = len(X_hd)
        m_ld = len(X_ld) if X_ld is not None else 0

        V_dict           = self.amp_results["V_denoised"]
        D_dict           = self.amp_results["signal_diag_dict"]
        F_dict           = self.amp_results["U_non_denoised"]
        cluster_model_u  = self.cluster_model_u
        labels_all       = cluster_model_u.cluster_labels
        cluster_denoisers = cluster_model_u.cluster_denoisers

        n_train = X_hd[0].shape[0]

        U_hat_hd      = _reconstruct_Uhat_onsager(F_dict, V_dict, D_dict, n_train)
        # X_ld here is already pca_model_low.pca_results[j].X (selected + centered
        # by LowDimModalityLoadings.fit()) -- _transform_lowdim_to_Uhat expects RAW
        # input (that's what predict_from_test_data_all passes it at test time), so
        # calling it again here would double-center the training LD covariates.
        # Use the already-transformed values directly instead.
        U_hat_ld_list = (
            [np.asarray(A, dtype=np.float64) for A in X_ld] if X_ld is not None else []
        )

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
                V_k    = V_dict[k]
                D_k    = D_dict[k]
                D_inv  = np.linalg.inv(D_k)
                Sigma_k = (1.0 / n_train) * (V_k.T @ V_k)
                S_blocks.append(D_inv @ np.linalg.pinv(Sigma_k) @ D_inv)
            for j in mods_ld:
                L = self.pca_model_low.pca_results[j].sample_aligns
                M_blocks.append(np.diag(L) if L.ndim == 1 else L)
                S_blocks.append(np.eye(L.shape[0]))

            M_c   = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
            S_c   = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))
            U_den = cluster_denoisers[cid]["denoise"](U_concat, M_c, S_c)

            widths = ([U_hat_hd[k].shape[1] for k in mods_hd] +
                      [U_hat_ld_list[j].shape[1] for j in mods_ld])
            edges  = np.cumsum([0] + widths)
            cursor = 0
            for k in mods_hd:
                s, e = edges[cursor], edges[cursor + 1]; cursor += 1
                U_denoised_hd[k] = U_den[:, s:e]
            for j in mods_ld:
                s, e = edges[cursor], edges[cursor + 1]; cursor += 1
                U_denoised_ld[j] = U_den[:, s:e]

        parts = [U_denoised_hd[k] for k in sorted(U_denoised_hd.keys())]
        parts += [U_denoised_ld[j] for j in sorted(U_denoised_ld.keys())]
        U_all = np.hstack(parts) if parts else None
        if U_all is None:
            raise RuntimeError("No denoised features assembled for supervised head.")

        beta = np.linalg.pinv(U_all) @ self.y_train
        self.amp_results["beta_hat"] = beta


# =============================================================================
# Unified Pipeline (HD + LD clustering + AMP)
# =============================================================================

class MultimodalClusterAllUPipeline(_PipelineBase):
    """
    Full pipeline:
      1) PCA for HD; optional LowDim loadings for LD
      2) Cluster ALL modalities (HD + LD) using normalized U
      3) Build ClusterParametricEBPipeline for U over ALL (HD + LD)
      4) Build ClusterParametricEBPipeline for V over HD only (per modality)
      5) Run AMP and fit the linear regression head (beta_hat)
    """

    def __init__(self):
        super().__init__()
        self.pca_hd              = None
        self.pca_ld              = None
        self.cluster_labels_U_all = None
        self.nsupp_ratio         = 1.0
        self.num_epoch           = 1000
        self.lr                  = 1e-3
        self.hidden_dim          = 256

    def fit_pca_highdim(self, X_list_hd, K_list_hd, preprocess=False):
        if preprocess:
            print("Preprocessing HD modalities (normalizing variance)...", flush=True)
            diag = preprocessing.MultiModalityPCADiagnostics()
            X_list_hd = diag.normalize_obs(X_list_hd, K_list_hd)

        print("Fitting PCA for HD modalities...", flush=True)
        self.pca_hd = pca_pack.MultiModalityPCA()
        self.pca_hd.fit(X_list_hd, K_list_hd, plot_residual=False)

        self.m_hd        = len(X_list_hd)
        self.pca_model   = self.pca_hd
        return self

    def fit_lowdim(self, A_list_ld=None, top_features=None):
        if A_list_ld is None:
            self.pca_ld        = None
            self.m_ld          = 0
            self.pca_model_low = None
            return self
        print("Fitting LowDimModalityLoadings for LD modalities...", flush=True)
        ld = preprocessing.LowDimModalityLoadings(top_features=top_features)
        ld.fit(A_list_ld)
        self.pca_ld        = ld
        self.m_ld          = len(A_list_ld)
        self.pca_model_low = self.pca_ld
        return self

    def cluster_all_modalities(self, X_list_hd, similarity_metric="cka",
                               num_clusters=None, threshold=None,
                               linkage_method="average",
                               auto_method="silhouette",
                               gap_B=10, gap_K_max=10,
                               gap_seed=42, gap_subsample=None):
        if self.pca_hd is None:
            raise RuntimeError("Call fit_pca_highdim first.")

        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)

        U_norm_ld = []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                U_norm_ld.append(self.pca_ld.pca_results[j].X)

        U_all = U_norm_hd + U_norm_ld
        m_hd  = len(U_norm_hd)
        m_ld  = len(U_norm_ld)

        if len(U_all) == 0:
            raise ValueError("No modalities found to cluster.")

        print("Clustering modalities (HD + LD) ...", flush=True)
        from hierarchical_clustering_modalities import ModalityClusterer
        cl = ModalityClusterer(U_all)
        cl.compute_similarity_matrix(similarity_metric, epsilon=0.1, sigma=1.0)
        labels = np.asarray(cl.cluster_modalities(
            similarity_metric=similarity_metric,
            num_clusters=num_clusters,
            threshold=threshold,
            method=linkage_method,
            auto_method=auto_method,
            gap_B=gap_B,
            gap_K_max=gap_K_max,
            gap_seed=gap_seed,
            gap_subsample=gap_subsample,
        ))

        if labels.shape[0] != (m_hd + m_ld):
            raise ValueError(
                f"cluster_labels_U length mismatch: got {labels.shape[0]}, "
                f"expected {m_hd + m_ld}"
            )

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

        # U over ALL modalities
        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)
        M_u_hd    = [np.diag(self.pca_hd.pca_results[k].sample_aligns)  for k in range(m_hd)]
        S_u_hd    = [np.diag(1 - self.pca_hd.pca_results[k].sample_aligns**2) for k in range(m_hd)]

        U_norm_ld, M_u_ld, S_u_ld = [], [], []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                pr_ld = self.pca_ld.pca_results[j]
                U_norm_ld.append(pr_ld.X)
                sa = pr_ld.sample_aligns
                L  = sa.shape[0]
                M_u_ld.append(np.diag(sa) if sa.ndim == 1 else sa)
                S_u_ld.append(np.eye(L))

        U_all = U_norm_hd + U_norm_ld
        M_all = M_u_hd   + M_u_ld
        S_all = S_u_hd   + S_u_ld

        print("Fitting cluster-wise EB model for U (HD + LD) ...", flush=True)
        self.cluster_model_u = ClusterParametricEBPipeline(
            covariance_mode='full', choose_comp=True,
            n_components=int(np.cbrt(U_all[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(U_all, M_all, S_all, self.cluster_labels_U_all)

        if print_priors and getattr(self.cluster_model_u, "cluster_priors_gmm", None):
            for c, dct in self.cluster_model_u.cluster_priors_gmm.items():
                print(f"\n=== cluster_model_u (GMM) : Cluster {c} ===", flush=True)
                print(
                    f"means shape: {dct['m_prior'].shape} | "
                    f"covs: {dct['cov_prior'].shape} | "
                    f"K: {len(dct['weights'])}",
                    flush=True,
                )

        # V over HD only
        V_norm_hd = _extract_normalized_V_hd(self.pca_hd, X_list_hd)
        M_v_hd    = [np.diag(self.pca_hd.pca_results[k].feature_aligns) for k in range(m_hd)]
        S_v_hd    = [np.diag(1 - self.pca_hd.pca_results[k].feature_aligns**2) for k in range(m_hd)]
        labels_V  = np.arange(m_hd)

        print("Fitting per-modality EB model for V (HD only) ...", flush=True)
        self.cluster_model_v = ClusterParametricEBPipeline(
            covariance_mode='full', choose_comp=True,
            n_components=int(np.cbrt(V_norm_hd[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(V_norm_hd, M_v_hd, S_v_hd, labels_V)

        self.pca_model     = self.pca_hd
        self.pca_model_low = self.pca_ld
        return self

    def run_amp(self, X_list_hd, amp_iters=20, muteu=False, mutev=False):
        if self.pca_hd is None or self.cluster_model_u is None or self.cluster_model_v is None:
            raise RuntimeError("PCA and cluster models must be prepared before AMP.")

        print("Running AMP denoising (HD + LD) ...", flush=True)
        self.amp_results = amp.ebamp_cluster_U_all_modalities(
            pca_model_hd    = self.pca_hd,
            cluster_model_u = self.cluster_model_u,
            cluster_model_v = self.cluster_model_v,
            pca_model_ld    = self.pca_ld,
            amp_iters       = amp_iters,
            muteu           = muteu,
            mutev           = mutev,
        )
        self.amp_results["cluster_model_u"] = self.cluster_model_u

        if self.y_train is not None:
            print("Training supervised head (HD + LD) ...", flush=True)
            X_list_ld = (
                None if self.pca_ld is None
                else [p.X for p in self.pca_ld.pca_results.values()]
            )
            self._post_amp_supervised(X_list_hd, X_list_ld)

        return self.amp_results


# =============================================================================
# Prediction
# =============================================================================

def predict_from_test_data_all(pipeline, X_test_hd, X_test_ld=None):
    """
    Linear regression prediction on HD + LD test data.

    Reconstructs U_hat for each modality, applies cluster-wise EB denoising,
    concatenates the denoised factors, and applies the stored beta_hat:
        y_pred = U_all @ beta_hat

    Parameters
    ----------
    pipeline   : fitted MultimodalClusterAllUPipeline with amp_results["beta_hat"]
    X_test_hd  : list[(n_test, p_k)]  HD test blocks
    X_test_ld  : list[(n_test, q_j)] or None  LD test blocks

    Returns
    -------
    U_denoised_hd : dict[k -> (n_test, r_k)]
    U_denoised_ld : dict[j -> (n_test, r_j)]
    y_pred        : (n_test,) ndarray
    """
    if pipeline.amp_results is None:
        raise RuntimeError("Run pipeline.run_amp(...) first.")
    if pipeline.task != "regression":
        raise RuntimeError("Pipeline task is not 'regression'.")
    if "beta_hat" not in pipeline.amp_results:
        raise RuntimeError(
            "beta_hat not found in amp_results. "
            "Ensure pipe.relation == 'linear' before run_amp."
        )

    V_dict           = pipeline.amp_results["V_denoised"]
    D_dict           = pipeline.amp_results["signal_diag_dict"]
    cluster_model_u  = pipeline.amp_results["cluster_model_u"]
    labels_all       = cluster_model_u.cluster_labels
    cluster_denoisers = cluster_model_u.cluster_denoisers

    m_hd    = pipeline.m_hd
    m_ld    = pipeline.m_ld
    n_train = list(pipeline.pca_model.pca_results.values())[0].U.shape[0]

    U_hat_hd      = _reconstruct_Uhat_from_VD(X_test_hd, V_dict, D_dict, n_train)
    U_hat_ld_list = _transform_lowdim_to_Uhat(
        getattr(pipeline, "pca_model_low", None), X_test_ld
    )

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
            V_k    = V_dict[k]
            D_k    = D_dict[k]
            D_inv  = np.linalg.inv(D_k)
            Sigma_k = (1.0 / n_train) * (V_k.T @ V_k)
            S_blocks.append(D_inv @ np.linalg.pinv(Sigma_k) @ D_inv)
        for j in mods_ld:
            L = pipeline.pca_model_low.pca_results[j].sample_aligns
            M_blocks.append(np.diag(L) if L.ndim == 1 else L)
            S_blocks.append(np.eye(L.shape[0]))

        M_c   = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
        S_c   = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))
        U_den = cluster_denoisers[cid]["denoise"](U_concat, M_c, S_c)

        widths = ([U_hat_hd[k].shape[1] for k in mods_hd] +
                  [U_hat_ld_list[j].shape[1] for j in mods_ld])
        edges  = np.cumsum([0] + widths)
        cursor = 0
        for k in mods_hd:
            s, e = edges[cursor], edges[cursor + 1]; cursor += 1
            U_denoised_hd[k] = U_den[:, s:e]
        for j in mods_ld:
            s, e = edges[cursor], edges[cursor + 1]; cursor += 1
            U_denoised_ld[j] = U_den[:, s:e]

    U_all  = np.hstack(
        [U_denoised_hd[k] for k in sorted(U_denoised_hd)] +
        [U_denoised_ld[j] for j in sorted(U_denoised_ld)]
    )
    y_pred = U_all @ pipeline.amp_results["beta_hat"]
    return U_denoised_hd, U_denoised_ld, y_pred
