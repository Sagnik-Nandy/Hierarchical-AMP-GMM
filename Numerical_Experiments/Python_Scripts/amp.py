import numpy as np
import importlib
from emp_bayes import ParametricEB_GMM, ClusterParametricEBPipeline
from pca_pack import MultiModalityPCA
from scipy.linalg import block_diag

def ebamp_multimodal(pca_model, cluster_model_v, cluster_model_u, amp_iters=10, muteu=False, mutev=False):
    """
    Multimodal Gaussian Bayes AMP with per-modality denoising for V and per-cluster denoising for U.

    Each AMP iteration alternates between two half-steps:
      1. V-step  : for each modality k, denoise the right singular vectors V_k using
                   the cluster-level EB denoiser associated with modality k, then
                   form the updated U-side observation f_k via the AMP residual update.
      2. U-step  : for each cluster, jointly denoise the stacked left singular vectors
                   U across all modalities in the cluster using the shared cluster-level
                   EB denoiser, then form the updated V-side observation g_k.

    State evolution quantities (mu, sigma_sq) track the effective signal and noise
    at each iteration and are passed to the EB denoisers as the M and D matrices.
    The Onsager correction terms (b_k, b_bar) debias the residual updates.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        PCA model containing PCA results for each modality. Each entry exposes:
          .X            — (n, p_k) centered data matrix
          .U, .V        — left/right singular vectors
          .signals      — (r_k,) singular values / signal strengths
          .sample_aligns, .feature_aligns — alignment coefficients
    cluster_model_v : ClusterParametricEBPipeline
        Fitted EB pipeline for V denoising. cluster_denoisers[c] gives the
        denoiser for the cluster c that modality k belongs to.
    cluster_model_u : ClusterParametricEBPipeline
        Fitted EB pipeline for U denoising. cluster_denoisers[c] gives the
        joint denoiser for all modalities in cluster c.
    amp_iters : int
        Number of AMP iterations.
    muteu : bool
        If True, replace the U denoiser with the identity map (no shrinkage).
    mutev : bool
        If True, replace the V denoiser with the identity map (no shrinkage).

    Returns
    -------
    dict with keys:
        "U_non_denoised"  : dict[k -> (n, r_k, T+1) array]  raw U iterates
        "U_denoised"      : dict[k -> (n, r_k, T+1) array]  denoised U iterates
        "V_non_denoised"  : dict[k -> (p_k, r_k, T+1) array] raw V iterates
        "V_denoised"      : dict[k -> (p_k, r_k, T+1) array] denoised V iterates
        "signal_diag_dict": dict[k -> (r_k, r_k) array]  diagonal signal matrices
        "cluster_model_u" : the U cluster model (passed through)
        "cluster_model_v" : the V cluster model (passed through)
    """

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Extract centered data matrices from PCA results; shape (n, p_k) each
    X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
    n_samples, _ = X_list[0].shape
    m = len(X_list)

    # Aspect ratio gamma_k = p_k / n for each modality — appears in Onsager corrections
    gamma_list = [X.shape[1] / n_samples for X in X_list]

    # Storage for raw (non-denoised) and denoised iterates, stacked along axis=2
    # Each entry grows as np.dstack across iterations: shape (n or p_k, r_k, t)
    U_dict, V_dict         = {}, {}   # raw AMP observations f_k, g_k
    U_dict_denois, V_dict_denois = {}, {}   # denoised outputs u_k, v_k

    # State evolution parameters passed to EB denoisers at each iteration:
    #   mu_dict_v[k]      — effective signal matrix M for V denoising  (r_k, r_k)
    #   sigma_sq_dict_v[k]— effective noise matrix D for V denoising   (r_k, r_k)
    #   mu_dict_u[k]      — effective signal matrix M for U denoising  (r_k, r_k)
    #   sigma_sq_dict_u[k]— effective noise matrix D for U denoising   (r_k, r_k)
    mu_dict_u, sigma_sq_dict_u = {}, {}
    mu_dict_v, sigma_sq_dict_v = {}, {}

    # Onsager correction matrices for U direction, keyed by modality
    b_bar_dict_u = {}

    # Diagonal signal matrices diag(signals_k), stored for output
    signal_diag_dict = {k: np.diag(pca_model.pca_results[k].signals) for k in range(m)}

    # -------------------------------------------------------------------------
    # Initialization  (t = 0)
    # -------------------------------------------------------------------------
    for k in range(m):
        pca_k = pca_model.pca_results[k]
        U_init, V_init = pca_k.U, pca_k.V

        # Normalize initial U, V to have unit columns scaled by sqrt(dimension)
        # so that (1/n) f_k^T f_k ≈ I  and  (1/p_k) g_k^T g_k ≈ I
        f_k = U_init / np.linalg.norm(U_init, axis=0) * np.sqrt(n_samples)
        g_k = V_init / np.linalg.norm(V_init, axis=0) * np.sqrt(X_list[k].shape[1])

        # Initial state evolution: alignment coefficients from PCA give the
        # signal (mu) and residual noise (sigma_sq = 1 - mu^2) at t=0
        mu_dict_v[k]       = np.diag(pca_k.feature_aligns)
        sigma_sq_dict_v[k] = np.diag(1 - pca_k.feature_aligns**2)
        mu_dict_u[k]       = np.diag(pca_k.sample_aligns)
        sigma_sq_dict_u[k] = np.diag(1 - pca_k.sample_aligns**2)

        # Store initial iterates (t=0 slice)
        U_dict[k]       = f_k[:, :, np.newaxis]
        V_dict[k]       = g_k[:, :, np.newaxis]
        V_dict_denois[k] = g_k[:, :, np.newaxis]

        # Initial denoised U: pre-whiten f_k by sqrt(sigma_sq_v) to account for
        # the noise level before the first V denoising step
        U_dict_denois[k] = (f_k @ np.sqrt(sigma_sq_dict_v[k]))[:, :, np.newaxis]

    # Cluster assignment for each modality — used to route to the right denoiser
    cluster_labels_v = cluster_model_v.cluster_labels   # which cluster each modality k belongs to (V side)
    cluster_labels_u = cluster_model_u.cluster_labels   # which cluster each modality k belongs to (U side)

    # -------------------------------------------------------------------------
    # AMP iterations
    # -------------------------------------------------------------------------
    for t in range(amp_iters):

        # =====================================================================
        # Step 1: V denoising — per modality
        #
        # Given the current U-side denoised estimate u_k and the V-side
        # observation g_k, apply the EB denoiser to get a cleaner v_k,
        # then compute the Onsager-corrected U-side observation f_k for
        # the next U denoising step.
        # =====================================================================
        for k in range(m):

            gamma_k  = gamma_list[k]  # p_k / n aspect ratio for this modality

            # Fetch the cluster-level V denoiser for modality k
            vdenoiser = cluster_model_v.cluster_denoisers[cluster_labels_v[k]]

            g_k       = V_dict[k][:, :, -1]         # current V observation  (p_k, r_k)
            mu_k      = mu_dict_v[k]                 # current signal matrix  (r_k, r_k)
            sigma_sq_k = sigma_sq_dict_v[k]          # current noise matrix   (r_k, r_k)
            u_k       = U_dict_denois[k][:, :, -1]  # latest denoised U      (n, r_k)

            if not mutev:
                # --- EB denoiser branch ---

                # Denoise g_k: v_k = E[V | g_k, mu_k, sigma_sq_k]
                v_k = vdenoiser["denoise"](g_k, mu_k, sigma_sq_k)
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))

                # Onsager correction: b_k = gamma_k * E[d(denoiser)/dg_k]
                # averaged over samples; shape (r_k, r_k)
                b_k = gamma_k * np.mean(vdenoiser["ddenoise"](g_k, mu_k, sigma_sq_k), axis=0)

                # State evolution update for U side:
                #   sigma_bar_sq = (1/n) v_k^T v_k  — empirical second moment of denoised V
                #   mu_bar       = sigma_bar_sq * signals_k  — effective signal after denoising
                sigma_bar_sq = v_k.T @ v_k / n_samples
                mu_bar       = sigma_bar_sq * pca_model.pca_results[k].signals

            else:
                # --- Identity denoiser branch (mutev=True): no shrinkage ---
                # v_k = mu_k^{-1} g_k  (invert the effective channel)
                mu_inv = np.linalg.pinv(mu_k)
                v_k    = g_k @ mu_inv.T
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))

                # Onsager correction for identity map: b_k = mu_k^{-1} * gamma_k
                b_k = mu_inv * gamma_k

                # State evolution for identity denoiser — no gamma_k scaling
                # (mu_bar and sigma_bar_sq must be on the same scale as the EB branch)
                mu_bar       = np.diag(pca_model.pca_results[k].signals)
                sigma_bar_sq = np.eye(v_k.shape[1]) + mu_inv @ sigma_sq_k @ mu_inv.T

            # AMP residual update for U observation:
            #   f_k = X_k v_k - u_k b_k^T   (Onsager-corrected projection onto U)
            f_k = X_list[k] @ v_k - u_k @ b_k.T
            U_dict[k] = np.dstack((U_dict[k], f_k[:, :, np.newaxis]))

            # Pass updated state evolution to U denoising step
            mu_dict_u[k]       = mu_bar
            sigma_sq_dict_u[k] = sigma_bar_sq

        # =====================================================================
        # Step 2: U denoising — per cluster
        #
        # For each cluster, stack the U observations across member modalities,
        # apply the joint cluster-level EB denoiser, then split results back
        # and compute the Onsager-corrected V-side observation g_k.
        # =====================================================================
        for cluster_k in np.unique(cluster_labels_u):

            # Modality indices belonging to this cluster
            cluster_modalities = [k for k in range(m) if cluster_labels_u[k] == cluster_k]

            # Concatenated signal strengths across modalities in this cluster
            signals_list = [pca_model.pca_results[k].signals for k in cluster_modalities]
            mu_signals   = np.concatenate(signals_list)   # (sum_k r_k,)

            # Fetch the shared cluster-level U denoiser
            udenoiser = cluster_model_u.cluster_denoisers[cluster_k]

            # Stack U observations horizontally: (n, sum_k r_k)
            f_cluster  = np.hstack([U_dict[k][:, :, -1] for k in cluster_modalities])
            split_sizes = [U_dict[k].shape[1] for k in cluster_modalities]  # r_k per modality

            # Build block-diagonal M and D from per-modality state evolution
            # M_cluster = diag(mu_k_1, mu_k_2, ...)     shape (P_c, P_c)
            # S_cluster = diag(sigma_sq_k_1, ...)        shape (P_c, P_c)
            M_cluster = block_diag(*[mu_dict_u[k] for k in cluster_modalities])
            S_cluster = block_diag(*[sigma_sq_dict_u[k] for k in cluster_modalities])

            if not muteu:
                # --- EB denoiser branch ---

                # Jointly denoise stacked U: u_cluster = E[U | f_cluster, M, S]
                u_cluster = udenoiser["denoise"](f_cluster, M_cluster, S_cluster)

                # Onsager correction: average Jacobian over samples; shape (P_c, P_c)
                b_bar_cluster = np.mean(udenoiser["ddenoise"](f_cluster, M_cluster, S_cluster), axis=0)

                # State evolution update for V side:
                #   sigma_sq = (1/n) u_cluster^T u_cluster — empirical second moment
                #   mu       = sigma_sq * mu_signals        — effective signal
                sigma_sq = u_cluster.T @ u_cluster / n_samples
                mu       = sigma_sq * mu_signals

            else:
                # --- Identity denoiser branch (muteu=True): no shrinkage ---
                # u_cluster = M_cluster^{-1} f_cluster
                mu_bar_inv = np.linalg.pinv(M_cluster)
                u_cluster  = f_cluster @ mu_bar_inv.T

                # Onsager correction for identity map
                b_bar_cluster = mu_bar_inv

                # State evolution for identity denoiser
                mu       = np.diag(np.concatenate(signals_list))
                sigma_sq = np.eye(u_cluster.shape[1]) + mu_bar_inv @ S_cluster @ mu_bar_inv.T

            # Precompute split boundaries for slicing the cluster block
            start_indices = np.cumsum([0] + split_sizes[:-1])
            end_indices   = np.cumsum(split_sizes)

            # Distribute denoised u_cluster back to per-modality U_dict_denois
            for k, start, end in zip(cluster_modalities, start_indices, end_indices):
                U_dict_denois[k] = np.dstack((U_dict_denois[k], u_cluster[:, start:end, np.newaxis]))

            # Extract per-modality blocks from cluster-level quantities:
            #   b_bar_dict_u[k] — diagonal block of Onsager correction for modality k
            #   mu_dict_v[k]    — updated signal matrix passed to V denoising
            #   sigma_sq_dict_v[k] — updated noise matrix passed to V denoising
            for k, start, end in zip(cluster_modalities, start_indices, end_indices):
                b_bar_dict_u[k]    = b_bar_cluster[start:end, start:end]
                mu_dict_v[k]       = mu[start:end, start:end]
                sigma_sq_dict_v[k] = sigma_sq[start:end, start:end]

            # AMP residual update for V observation:
            #   g_k = X_k^T u_k - v_k b_bar_k^T   (Onsager-corrected projection onto V)
            for k in cluster_modalities:
                g_k = (  X_list[k].T @ U_dict_denois[k][:, :, -1]
                        - V_dict_denois[k][:, :, -1] @ b_bar_dict_u[k].T )
                V_dict[k] = np.dstack((V_dict[k], g_k[:, :, np.newaxis]))

    # -------------------------------------------------------------------------
    # Return all iterates and models
    # -------------------------------------------------------------------------
    return {
        "U_non_denoised":  U_dict,           # raw f_k iterates per modality
        "U_denoised":      U_dict_denois,    # denoised u_k iterates per modality
        "V_non_denoised":  V_dict,           # raw g_k iterates per modality
        "V_denoised":      V_dict_denois,    # denoised v_k iterates per modality
        "signal_diag_dict": signal_diag_dict,
        "cluster_model_u": cluster_model_u,
        "cluster_model_v": cluster_model_v,
    }