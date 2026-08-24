"""
pred_err_vary_n_onsager.py
---------------------------
Prediction error experiment — linear response, varies n.
Onsager-debiased counterpart of pred_err_vary_n.py.

Response: y = hstack(U_true) @ beta + noise   (linear)
Predictor architecture: linear

Training-covariate projection: onsager (Onsager-debiased AMP field;
see DAIF_training_projection_fixes.pdf, Sec. 2). Pinned explicitly even
though it is the pipeline default, for clarity and future-proofing.

Usage
-----
    python pred_err_vary_n_onsager.py <n> <seed>

SLURM array
-----------
    #SBATCH --array=0-249   # 5 n_values x 50 seeds = 250 tasks
"""

import sys
import os
import numpy as np
import pandas as pd
import importlib


def load_modules():
    print(">>> Loading modules...", flush=True)
    sys.path.append('./Python_Scripts')
    global pipeline
    amp           = importlib.import_module("amp");           print("    loaded: amp", flush=True)
    pca_pack      = importlib.import_module("pca_pack");      print("    loaded: pca_pack", flush=True)
    preprocessing = importlib.import_module("preprocessing"); print("    loaded: preprocessing", flush=True)
    emp_bayes     = importlib.import_module("emp_bayes");     print("    loaded: emp_bayes", flush=True)
    hierarchical  = importlib.import_module("hierarchical_clustering_modalities"); print("    loaded: hierarchical_clustering_modalities", flush=True)
    pipeline      = importlib.import_module("complete_pipeline"); print("    loaded: complete_pipeline", flush=True)

    importlib.reload(amp)
    importlib.reload(pca_pack)
    importlib.reload(preprocessing)
    importlib.reload(emp_bayes)
    importlib.reload(hierarchical)
    importlib.reload(pipeline)
    print(">>> All modules loaded.", flush=True)

    global MultimodalPCAPipeline, MultimodalPCAPipelineClustering
    MultimodalPCAPipeline           = pipeline.MultimodalPCAPipelineSimulation
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClustering


# =============================================================================
# Data generating process
# =============================================================================

def generate_rademacher(shape):
    return np.random.choice([-1, 1], size=shape)

def generate_three_point(shape):
    vals  = np.array([-np.sqrt(2), 0.0, np.sqrt(2)])
    probs = np.array([0.25, 0.5, 0.25])
    return np.random.choice(vals, size=shape, p=probs)

def generate_nonlinear_rademacher(shape):
    W   = np.random.choice([-1, 1], size=shape).astype(float)
    eps = np.random.normal(0, 0.1, size=shape)
    F   = np.sign(W) * np.abs(W + eps) ** (1.0 / 3.0)
    F   = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)
    return F

def generate_gaussian_mixture_V(shape):
    signs = np.random.choice([-1, 1], size=shape)
    V     = signs + 0.5 * np.random.randn(*shape)
    return V / np.sqrt(1.25)

def generate_laplace_V(shape):
    return np.random.laplace(loc=0.0, scale=1.0 / np.sqrt(2), size=shape)

def row_standardize(M):
    M = M - M.mean(axis=1, keepdims=True)
    M = M / (M.std(axis=1, keepdims=True) + 1e-12)
    return M

def generate_data(n, p_list, r_list):
    U1 = generate_rademacher((n, r_list[0]))
    U2 = np.hstack([U1, generate_three_point((n, r_list[1] - r_list[0]))])
    U3 = generate_nonlinear_rademacher((n, r_list[2]))
    U_true = [U1, U2, U3]

    V1 = generate_rademacher((p_list[0], r_list[0]))
    V2 = generate_gaussian_mixture_V((p_list[1], r_list[1]))
    V3 = generate_laplace_V((p_list[2], r_list[2]))

    D1 = np.diag([5 * (i+1) for i in range(r_list[0])])
    D2 = np.diag([5 * (i+1) for i in range(r_list[1])])
    D3 = np.diag([5 * (i+1) for i in range(r_list[2])])

    Z1 = np.random.randn(n, p_list[0]) / np.sqrt(n)
    Z2 = np.random.randn(n, p_list[1]) / np.sqrt(n)
    Z3 = np.random.randn(n, p_list[2]) / np.sqrt(n)

    X1 = (1/n) * U1 @ D1 @ V1.T + Z1
    X2 = (1/n) * U2 @ D2 @ V2.T + Z2
    X3 = (1/n) * U3 @ D3 @ V3.T + Z3

    return [X1, X2, X3], U_true, [V1, V2, V3], [D1, D2, D3]


def reconstruction_error(U_est, U_true):
    P_est  = U_est  @ U_est.T
    P_true = U_true @ U_true.T
    return np.linalg.norm(P_est - P_true, 'fro')**2 / (U_true.shape[0]**2)


# =============================================================================
# Single trial
# =============================================================================

def run_single_trial(n, p_list, r_list, amp_iters, sigma=0.1):
    print(f"    Generating data (n={n}, p_list={p_list}, r_list={r_list})...", flush=True)
    X_list, U_true, V_list, D_list = generate_data(n, p_list, r_list)
    K_list = r_list

    # Linear response: y = hstack(U_true) @ beta + noise
    beta    = generate_rademacher(sum(r_list))
    y_train = np.hstack(U_true) @ beta + sigma * np.random.randn(n)

    # Test data — fresh U and Z, same V and D
    n_test  = n // 10
    U_test1 = generate_rademacher((n_test, r_list[0]))
    U_test2 = np.hstack([U_test1, generate_three_point((n_test, r_list[1] - r_list[0]))])
    U_test3 = generate_nonlinear_rademacher((n_test, r_list[2]))
    U_test  = [U_test1, U_test2, U_test3]
    X_test  = [(1/n) * U_test[i] @ D_list[i] @ V_list[i].T
               + np.random.randn(n_test, p_list[i]) / np.sqrt(n)
               for i in range(3)]
    y_test  = np.hstack(U_test) @ beta + sigma * np.random.randn(n_test)

    print(f"    Data generated. X shapes: {[X.shape for X in X_list]}", flush=True)

    results = {}

    # ------------------------------------------------------------------
    # AMP: clustered (CKA)
    # ------------------------------------------------------------------
    print("    AMP clustered (CKA), architecture=linear...", flush=True)
    pipe_cluster = MultimodalPCAPipelineClustering()
    pipe_cluster.y_train      = y_train
    pipe_cluster.architecture = "linear"
    pipe_cluster.projection_mode = "onsager"
    pipe_cluster.arch_kwargs  = {"fit_intercept": False}
    U_cluster = pipe_cluster.denoise_amp(
        X_list, K_list, compute_clusters=True, num_clusters=2,
        amp_iters=amp_iters, similarity_method="cka", sigma=1.0
    )["U_denoised"]
    _, y_pred_cluster = pipeline.predict_from_test_data(pipe_cluster, X_test)
    results["clustered"] = {
        "recon":      [reconstruction_error(U_cluster[i][:, :, -1], U_true[i]) for i in range(3)],
        "pred_error": float(np.mean((y_test - y_pred_cluster) ** 2)),
    }
    print("    AMP clustered done.", flush=True)

    # ------------------------------------------------------------------
    # AMP: same cluster
    # ------------------------------------------------------------------
    print("    AMP same cluster, architecture=linear...", flush=True)
    pipe_same = MultimodalPCAPipeline()
    pipe_same.y_train      = y_train
    pipe_same.architecture = "linear"
    pipe_same.projection_mode = "onsager"
    pipe_same.arch_kwargs  = {"fit_intercept": False}
    U_same = pipe_same.denoise_amp(
        X_list, K_list, cluster_labels_U=np.array([0, 0, 0]),
        amp_iters=amp_iters
    )["U_denoised"]
    _, y_pred_same = pipeline.predict_from_test_data(pipe_same, X_test)
    results["same_cluster"] = {
        "recon":      [reconstruction_error(U_same[i][:, :, -1], U_true[i]) for i in range(3)],
        "pred_error": float(np.mean((y_test - y_pred_same) ** 2)),
    }
    print("    AMP same cluster done.", flush=True)

    # ------------------------------------------------------------------
    # AMP: distinct clusters
    # ------------------------------------------------------------------
    print("    AMP distinct clusters, architecture=linear...", flush=True)
    pipe_distinct = MultimodalPCAPipeline()
    pipe_distinct.y_train      = y_train
    pipe_distinct.architecture = "linear"
    pipe_distinct.projection_mode = "onsager"
    pipe_distinct.arch_kwargs  = {"fit_intercept": False}
    U_distinct = pipe_distinct.denoise_amp(
        X_list, K_list, cluster_labels_U=np.array([0, 1, 2]),
        amp_iters=amp_iters
    )["U_denoised"]
    _, y_pred_distinct = pipeline.predict_from_test_data(pipe_distinct, X_test)
    results["distinct_clusters"] = {
        "recon":      [reconstruction_error(U_distinct[i][:, :, -1], U_true[i]) for i in range(3)],
        "pred_error": float(np.mean((y_test - y_pred_distinct) ** 2)),
    }
    print("    AMP distinct clusters done.", flush=True)

    return results


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python pred_err_vary_n_onsager.py <n> <seed>", flush=True)
        sys.exit(1)

    n    = int(sys.argv[1])
    seed = int(sys.argv[2])
    print(f">>> Worker started: n={n}, seed={seed}", flush=True)

    np.random.seed(seed)
    load_modules()

    gamma_list = [0.25, 0.8, 3.0]
    r_list     = [2, 3, 2]
    p_list     = [int(g * n) for g in gamma_list]

    print(f">>> n={n}, p_list={p_list}, r_list={r_list}", flush=True)
    print(f">>> Starting trial (n={n}, seed={seed}, amp_iters=10)...", flush=True)

    result = run_single_trial(n=n, p_list=p_list, r_list=r_list, amp_iters=10)

    print(f">>> Trial complete.", flush=True)

    os.makedirs("Results/pred_err_vary_n_onsager", exist_ok=True)
    rows = []
    for strategy, vals in result.items():
        rows.append({
            "n":          n,
            "seed":       seed,
            "strategy":   strategy,
            "modality_1": vals["recon"][0],
            "modality_2": vals["recon"][1],
            "modality_3": vals["recon"][2],
            "pred_error": vals["pred_error"],
        })

    df = pd.DataFrame(rows)
    out_path = f"Results/pred_err_vary_n_onsager/partial_result_{n}_{seed}.csv"
    df.to_csv(out_path, index=False)
    print(f">>> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
