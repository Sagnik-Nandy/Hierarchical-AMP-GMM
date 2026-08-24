# pipeline_cluster_lowdim_all_in_u.py
import numpy as np
import importlib
import warnings
from scipy.linalg import block_diag
import typing
from typing import Dict, Any

# --- project modules (expected to exist) ---
pca_pack = importlib.import_module("pca_pack")
from pca_pack import MultiModalityPCA
from emp_bayes_data import ClusterParametricEBPipeline 

preprocessing = importlib.import_module("preprocessing")
from preprocessing import MultiModalityPCADiagnostics, LowDimModalityLoadings

# optional clustering helper
try:
    from hierarchical_clustering_modalities import ModalityClusterer
except ImportError:
    ModalityClusterer = None


# -------------------------------
# Helpers: normalized PCs from PCA
# -------------------------------
def _normalize_cols_to_sqrt_n(M):
    return M / np.sqrt((M**2).sum(axis=0)) * np.sqrt(M.shape[0])

def _extract_normalized_U_hd(pca_model, X_list_hd):
    out = []
    for k in range(len(X_list_hd)):
        U = pca_model.pca_results[k].U
        out.append(_normalize_cols_to_sqrt_n(U))
    return out

def _extract_normalized_V_hd(pca_model, X_list_hd):
    out = []
    for k in range(len(X_list_hd)):
        V = pca_model.pca_results[k].V
        p = V.shape[0]
        out.append(V / np.sqrt((V**2).sum(axis=0)) * np.sqrt(p))
    return out


# ----------------------------------------------------------------------
# AMP with clustering over HD + LD in U (cluster_model_u covers both)
# ----------------------------------------------------------------------


def ebamp_cluster_U_all_modalities(
    pca_model_hd: "MultiModalityPCA",
    cluster_model_u: "ClusterEmpiricalBayes",    # built over HD + LD (in that order)
    cluster_model_v: "ClusterEmpiricalBayes",    # built over HD only, one cluster per modality
    pca_model_ld: "LowDimModalityLoadings" = None,
    amp_iters: int = 20,
    muteu: bool = False,
    mutev: bool = False,
) -> Dict[str, Any]:
    """
    AMP where:
      - U-step is clustered over ALL modalities (HD + LD) using cluster_model_u.
      - V-step is per HD modality using cluster_model_v (LD has no V).

    Indexing convention:
      HD modalities: indices 0..m_hd-1
      LD modalities (if any): indices m_hd..(m_hd+m_ld-1)

    This version stores ONLY the latest iterates (2-D arrays). No 3-D stacks/history.
    """

    # --- unpack HD ---
    X_hd = [pca_model_hd.pca_results[k].X for k in pca_model_hd.pca_results.keys()]   # each (n, p_k)
    m_hd = len(X_hd)
    n = X_hd[0].shape[0]
    gammas = [X.shape[1] / n for X in X_hd]
    print(f"The shapes of hd modality are : {[X.shape[1]  for X in X_hd]}")
    signal_diag_dict = {k: np.diag(pca_model_hd.pca_results[k].signals) for k in range(m_hd)}

    # --- LD (optional) ---
    ld_present = pca_model_ld is not None
    if ld_present:
        X_ld = [pca_model_ld.pca_results[j].X for j in pca_model_ld.pca_results.keys()]  # each (n, L_j)
        m_ld = len(X_ld)
    else:
        X_ld, m_ld = [], 0

    # --- storage for current AMP iterates (2-D only) ---
    U_raw_hd, U_den_hd = {}, {}   # (n, r_k)
    V_raw_hd, V_den_hd = {}, {}   # (p_k, r_k)
    mu_u_hd, Sig_u_hd, mu_v_hd, Sig_v_hd = {}, {}, {}, {}
    bbar_u_hd = {}  # (r_k, r_k), used for residual in V-update
    U_den_ld = {} if ld_present else None

    # --- initialize HD blocks ---
    for k in range(m_hd):
        pca_k = pca_model_hd.pca_results[k]
        pk = X_hd[k].shape[1]
        U0 = pca_k.U                                # (n, r_k)
        V0 = pca_k.V                                # (p_k, r_k)
        # normalize like AMP init
        f0 = U0 * np.sqrt(n)     # (n, r_k)
        g0 = V0 * np.sqrt(pk)    # (p_k, r_k)

        # priors as (r_k, r_k)
        mu_v_hd[k] = np.diag(pca_k.feature_aligns)
        Sig_v_hd[k] = np.diag(1 - pca_k.feature_aligns**2)
        mu_u_hd[k]  = np.diag(pca_k.sample_aligns)
        Sig_u_hd[k] = np.diag(1 - pca_k.sample_aligns**2)

        # current iterates
        U_raw_hd[k] = f0
        V_raw_hd[k] = g0
        U_den_hd[k] = f0 @ np.sqrt(Sig_v_hd[k])     # (n, r_k)
        V_den_hd[k] = g0                            # (p_k, r_k)

    # --- initialize LD blocks (U only) ---
    if ld_present:
        for j in range(m_ld):
            ld_pca_pack = pca_model_ld.pca_results[j]
            U_ld0 = ld_pca_pack.U                                          # (n, L_j)
            U_den_ld[j] = U_ld0
            # For LD: M = L_j, S = I
            mu_u = ld_pca_pack.sample_aligns    
            mu_u = mu_u.astype(float)
            mu_dict_j = mu_u
            sigma_sq_j = np.eye(U_ld0.shape[1])
            # stash on the same dicts as HD U-priors for convenience
            # (we’ll not iterate over these in Step 1, only in U-step)
            # We'll keep them separate dictionaries:
            # they will be assembled via lists below.
            # pass  # just to indicate init above; values read directly from pca_model_ld inside the loop

    # --- cluster labels ---
    labels_u_all = cluster_model_u.cluster_labels  # length = m_hd + m_ld
    labels_v_hd = cluster_model_v.cluster_labels   # typically 0..m_hd-1

    # --- AMP iterations ---
    for t in range(amp_iters):
        print(f"AMP iteration {t+1}/{amp_iters}", flush=True)

        # ---- Step 1: V denoising per HD modality ----
        print(" V-step (HD modalities only)", flush=True)
        for k in range(m_hd):
            gamma = gammas[k]
            vden = cluster_model_v.cluster_denoisers[labels_v_hd[k]]  # per-modality EB denoiser

            g_k = V_raw_hd[k]              # (p_k, r_k)
            u_k = U_den_hd[k]              # (n, r_k)
            mu_k, S_k = mu_v_hd[k], Sig_v_hd[k]   # (r_k, r_k)

            if not mutev:
                v_k = vden["denoise"](g_k, mu_k, S_k)   # (p_k, r_k)
                V_den_hd[k] = v_k
                b_vec = gamma * np.mean(vden["ddenoise"](g_k, mu_k, S_k), axis=0)  # (r_k,)
                Sig_bar = (v_k.T @ v_k) / n
                # signals is (r_k,), broadcasting over columns
                mu_bar = Sig_bar * pca_model_hd.pca_results[k].signals
            else:
                mu_inv = np.linalg.pinv(mu_k)
                v_k = g_k @ mu_inv.T
                V_den_hd[k] = v_k
                # simple surrogate per-column Onsager (diagonal)
                b_vec   = mu_inv * gamma
                mu_bar  = np.diag(pca_model_hd.pca_results[k].signals)
                Sig_bar = np.eye(v_k.shape[1]) + mu_inv @ S_k @ mu_inv.T

            # AMP residual for next U raw (apply Onsager per column)
            U_raw_hd[k] = X_hd[k].dot(v_k) - u_k.dot(b_vec.T)

            # update U priors for this modality
            mu_u_hd[k]  = mu_bar
            Sig_u_hd[k] = Sig_bar
            print(f"  Modality {k}: mean signal sum = {np.mean(mu_u_hd[k]):.3f}", flush=True)

        # ---- Step 2: U denoising per cluster (HD + LD in the same cluster) ----
        unique_clusters = np.unique(labels_u_all)
        for c in unique_clusters:
            # HD indices in cluster c
            mods_hd = [k for k in range(m_hd) if labels_u_all[k] == c]
            # LD indices in c (offset by m_hd)
            mods_ld = [j for j in range(m_ld) if ld_present and labels_u_all[m_hd + j] == c] if ld_present else []

            # concat f, M, S
            f_blocks = [U_raw_hd[k] for k in mods_hd]                  # (n, r_k)
            M_blocks = [mu_u_hd[k] for k in mods_hd]                   # (r_k, r_k)
            S_blocks = [Sig_u_hd[k] for k in mods_hd]                  # (r_k, r_k)

            if ld_present and mods_ld:
                # For LD, read directly
                f_blocks += [pca_model_ld.pca_results[j].X for j in mods_ld]              # (n, L_j)
                M_blocks += [pca_model_ld.pca_results[j].sample_aligns
                             for j in mods_ld]                                            # (L_j, L_j)
                S_blocks += [np.eye(pca_model_ld.pca_results[j].U.shape[1]) for j in mods_ld]  # (L_j, L_j)

            f_cluster = np.hstack(f_blocks)                     # (n, R_tot)
            M_cluster = block_diag(*M_blocks) if M_blocks else np.zeros((0, 0))
            S_cluster = block_diag(*S_blocks) if S_blocks else np.zeros((0, 0))
            print(f"Peinting shapeof S: {S_cluster.shape}", flush=True)
            print(f"Peinting shapeof M: {M_cluster.shape}", flush=True)

            uden = cluster_model_u.cluster_denoisers[c]
            
            print("Denoising U in cluster", c, "with modalities HD", mods_hd, "LD", mods_ld, flush=True)
            if not muteu:
                u_cluster = uden["denoise"](f_cluster, M_cluster, S_cluster)                # (n, R_tot)
                bbar_cluster = np.mean(uden["ddenoise"](f_cluster, M_cluster, S_cluster), axis=0)  # (R_tot, R_tot)
                Sig_out = (u_cluster.T @ u_cluster) / n                                     # (R_tot, R_tot)

                sig_blocks = (np.concatenate([pca_model_hd.pca_results[k].signals for k in mods_hd])
                              if mods_hd else np.array([], dtype=float))
                if ld_present and mods_ld:
                    sig_blocks_ld = np.concatenate([np.zeros(pca_model_ld.pca_results[j].U.shape[1]) for j in mods_ld])
                    full_signals = (np.concatenate([sig_blocks, sig_blocks_ld])
                                    if sig_blocks.size else sig_blocks_ld)
                else:
                    full_signals = sig_blocks
                # column-wise scaling (broadcasting across rows)
                mu_out = Sig_out * full_signals if full_signals.size else Sig_out
                print("Sum of signals in this cluster:", mu_out.mean())
            else:
                M_inv = np.linalg.pinv(M_cluster) if M_cluster.size else M_cluster
                u_cluster = f_cluster @ M_inv.T if M_cluster.size else f_cluster
                bbar_cluster = M_inv if M_cluster.size else np.zeros((f_cluster.shape[1], f_cluster.shape[1]))
                Sig_out = (np.eye(u_cluster.shape[1]) + M_inv @ S_cluster @ M_inv.T) if M_cluster.size else np.eye(u_cluster.shape[1])
                sig_blocks = (np.concatenate([pca_model_hd.pca_results[k].signals for k in mods_hd])
                              if mods_hd else np.array([], dtype=float))
                if ld_present and mods_ld:
                    ones_ld = [np.ones(pca_model_ld.pca_results[j].U.shape[1]) for j in mods_ld]
                    sig_blocks = np.concatenate([sig_blocks] + ones_ld) if sig_blocks.size else np.concatenate(ones_ld)
                full_signals = sig_blocks
                mu_out = np.diag(full_signals) if full_signals.size else Sig_out

            # split back to HD and LD segments (overwrite 2-D arrays)
            widths_hd = [U_raw_hd[k].shape[1] for k in mods_hd]
            widths_ld = ([pca_model_ld.pca_results[j].U.shape[1] for j in mods_ld] if ld_present else [])
            widths = widths_hd + widths_ld
            starts = np.cumsum([0] + widths[:-1])
            ends   = np.cumsum(widths)

            # HD slices
            for (k, s, e) in zip(mods_hd, starts[:len(mods_hd)], ends[:len(mods_hd)]):
                print(f"start and end are: {s} and {e}")
                U_den_hd[k] = u_cluster[:, s:e]                  # (n, r_k)
                bbar_u_hd[k] = bbar_cluster[s:e, s:e]            # (r_k, r_k)
                mu_v_hd[k] = mu_out[s:e, s:e]                    # (r_k, r_k)
                Sig_v_hd[k] = Sig_out[s:e, s:e]                  # (r_k, r_k)
                # next raw V for modality k
                V_raw_hd[k] = np.transpose(X_hd[k]).dot(u_cluster[:, s:e]) - V_den_hd[k].dot(bbar_cluster[s:e, s:e].T)  # (p_k, r_k)

            # LD slices
            if ld_present:
                offs = len(mods_hd)
                for idx, j in enumerate(mods_ld):
                    s, e = starts[offs + idx], ends[offs + idx]
                    print(f"start and end are: {s} and {e}")
                    U_den_ld[j] = u_cluster[:, s:e]               # (n, L_j)

    # assemble result (current iterates only)
    out = {
        "U_non_denoised": U_raw_hd,      # dict k -> (n, r_k)
        "U_denoised": U_den_hd,          # dict k -> (n, r_k)
        "V_non_denoised": V_raw_hd,      # dict k -> (p_k, r_k)
        "V_denoised": V_den_hd,          # dict k -> (p_k, r_k)
        "signal_diag_dict": signal_diag_dict,
        "cluster_model_u": cluster_model_u,
        "cluster_model_v": cluster_model_v,
    }
    if ld_present:
        out["U_ld_denoised"] = U_den_ld  # dict j -> (n, L_j)
    return out


# ---------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------
class MultimodalClusterAllUPipeline:
    """
    - PCA for HD; optional LowDim loadings for LD
    - Cluster ALL modalities (HD + LD) using normalized U
    - Build ClusterEmpiricalBayes for U over ALL modalities (HD + LD)
    - Build ClusterEmpiricalBayes for V over HD only (one cluster per modality)
    - Run AMP that uses those models
    """

    def __init__(self):
        self.pca_hd = None
        self.pca_ld = None
        self.cluster_labels_U_all = None  # length m_hd + m_ld
        self.cluster_model_u = None       # built over HD+LD
        self.cluster_model_v = None       # HD-only (per-modality)
        self.amp_results = None
        self.nsupp_ratio = 1.0

    def fit_pca_highdim(self, X_list_hd, K_list_hd, preprocess=False):
        if preprocess:
            diag = MultiModalityPCADiagnostics()
            X_list_hd = diag.normalize_obs(X_list_hd, K_list_hd)
        self.pca_hd = MultiModalityPCA()
        self.pca_hd.fit(X_list_hd, K_list_hd, plot_residual=False)
        return self

    def fit_lowdim(self, A_list_ld=None):
        if A_list_ld is None:
            self.pca_ld = None
        else:
            ld = LowDimModalityLoadings()
            ld.fit(A_list_ld)
            self.pca_ld = ld
        return self

    def cluster_all_modalities(self, X_list_hd, method="hss", num_clusters=None, threshold=None):
        """
        Cluster using normalized U over HD+LD. Requires pca_hd fitted; pca_ld optional.
        Returns labels of length m_hd + m_ld (LD appended after HD).
        """
        if self.pca_hd is None:
            raise RuntimeError("Call fit_pca_highdim first.")

        # HD normalized U
        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)

        # LD normalized U (from low-dim pack) if present
        U_norm_ld = []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                U_ld = self.pca_ld.pca_results[j].X
                U_norm_ld.append(U_ld)

        U_all = U_norm_hd + U_norm_ld

        if ModalityClusterer is None:
            # fallback: trivial labels
            m_all = len(U_all)
            if num_clusters is None or num_clusters == 1:
                labels = np.zeros(m_all, dtype=int)
            else:
                labels = np.arange(m_all) % num_clusters
        else:
            cl = ModalityClusterer(U_all)
            _ = cl.compute_similarity_matrix(method, epsilon=0.1, sigma=1.0)
            labels = cl.cluster_modalities(method, num_clusters=num_clusters, threshold=threshold)

        self.cluster_labels_U_all = np.asarray(labels)
        return self.cluster_labels_U_all
    
    def build_cluster_models(self, X_list_hd, print_priors=True):
        """
        Build:
        - cluster_model_u over HD+LD
        - cluster_model_v over HD only (per modality)

        Parameters
        ----------
        X_list_hd : list of ndarray
            High-dimensional modality data.
        print_priors : bool, default=False
            If True, print clusterwise priors (support Z and weights pi) after estimation.
        """
        if self.pca_hd is None:
            raise RuntimeError("PCA (HD) must be done first.")
        if self.cluster_labels_U_all is None:
            raise RuntimeError("Run cluster_all_modalities first.")

        m_hd = len(X_list_hd)
        m_ld = len(self.pca_ld.pca_results) if self.pca_ld is not None else 0

        # --- assemble U data for EB over ALL ---
        U_norm_hd = _extract_normalized_U_hd(self.pca_hd, X_list_hd)
        M_u_hd = [np.diag(self.pca_hd.pca_results[k].sample_aligns) for k in range(m_hd)]
        S_u_hd = [np.diag(1 - self.pca_hd.pca_results[k].sample_aligns**2) for k in range(m_hd)]

        U_norm_ld, M_u_ld, S_u_ld = [], [], []
        if self.pca_ld is not None:
            for j in self.pca_ld.pca_results.keys():
                U_ld = self.pca_ld.pca_results[j].X
                U_norm_ld.append(U_ld)
                sa = self.pca_ld.pca_results[j].sample_aligns
                M_u_ld.append(np.diag(sa) if sa.ndim == 1 else sa)
                L_j = sa.shape[0] if sa.ndim == 1 else sa.shape[0]
                S_u_ld.append(np.eye(L_j))

        U_all = U_norm_hd + U_norm_ld
        M_all = M_u_hd + M_u_ld
        S_all = S_u_hd + S_u_ld

        # Build cluster EB for U over ALL
        self.cluster_model_u = ClusterParametricEBPipeline(
            covariance_mode='isotropic-full', choose_comp=True, n_components=int(np.cbrt(U_all[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(U_all, M_all, S_all, self.cluster_labels_U_all)

        if print_priors and getattr(self.cluster_model_u, "cluster_priors_gmm", None):
            for c, dct in self.cluster_model_u.cluster_priors_gmm.items():
                print(f"\n=== cluster_model_u (GMM) : Cluster {c} ===", flush=True)
                print(f"means shape: {dct['m_prior'].shape} | covs: {dct['cov_prior'].shape} | K: {len(dct['weights'])}", flush=True)

        # Build cluster EB for V over HD only (labels = 0..m_hd-1)
        V_norm_hd = _extract_normalized_V_hd(self.pca_hd, X_list_hd)
        M_v_hd = [np.diag(self.pca_hd.pca_results[k].feature_aligns) for k in range(m_hd)]
        S_v_hd = [np.diag(1 - self.pca_hd.pca_results[k].feature_aligns**2) for k in range(m_hd)]
        labels_V = np.arange(len(X_list_hd))
        self.cluster_model_v = ClusterParametricEBPipeline(
            covariance_mode='isotropic-full', choose_comp=True, n_components=int(np.cbrt(V_norm_hd[0].shape[0])),
            max_iter=500, reg_covar=1e-6, psd_floor=1e-10, random_state=0
        ).fit(V_norm_hd, M_v_hd, S_v_hd, labels_V)

        return self


    def run_amp(self, amp_iters=20, muteu=False, mutev=False):
        if self.pca_hd is None or self.cluster_model_u is None or self.cluster_model_v is None:
            raise RuntimeError("PCA and cluster models must be prepared before AMP.")
        self.amp_results = ebamp_cluster_U_all_modalities(
            pca_model_hd=self.pca_hd,
            cluster_model_u=self.cluster_model_u,     # HD+LD labels inside
            cluster_model_v=self.cluster_model_v,     # HD-only per-modality V
            pca_model_ld=self.pca_ld,
            amp_iters=amp_iters,
            muteu=muteu,
            mutev=mutev,
        )
        return self.amp_results
