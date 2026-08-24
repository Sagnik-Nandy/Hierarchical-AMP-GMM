"""
vary_n.py
---------
Runs a SINGLE trial for a given (n, seed) combination.
Designed to be launched as one node in an HPC job array.

Usage
-----
    python vary_n.py <n> <seed>

Example SLURM array job
-----------------------
    #SBATCH --array=0-249          # 5 n_values x 50 trials = 250 jobs
    python vary_n.py \
        $(python -c "ns=[3000,3500,4000,4500,5000]; t=50; i=$SLURM_ARRAY_TASK_ID; print(ns[i//t])") \
        $SLURM_ARRAY_TASK_ID
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
    print(">>> All modules loaded and reloaded.", flush=True)

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

    return [X1, X2, X3], U_true


# =============================================================================
# Reconstruction error
# =============================================================================

def reconstruction_error(U_est, U_true):
    P_est  = U_est  @ U_est.T
    P_true = U_true @ U_true.T
    return np.linalg.norm(P_est - P_true, 'fro')**2 / (U_true.shape[0]**2)


# =============================================================================
# Single trial
# =============================================================================

def run_single_trial(n, p_list, r_list, amp_iters):
    """
    Run one trial and return per-modality reconstruction errors for each
    of the three clustering strategies.
    """
    print(f"    Generating data (n={n}, p_list={p_list}, r_list={r_list})...", flush=True)
    X_list, U_true = generate_data(n, p_list, r_list)
    print(f"    Data generated. X shapes: {[X.shape for X in X_list]}", flush=True)
    K_list = r_list

    # Automatic clustering with CKA
    print("    Running strategy 1/3: clustered (CKA)...", flush=True)
    pipe_cluster = MultimodalPCAPipelineClustering()
    U_cluster = pipe_cluster.denoise_amp(
        X_list, K_list,
        compute_clusters=True, num_clusters=2,
        amp_iters=amp_iters, similarity_method="cka"
    )["U_denoised"]
    labels = pipe_cluster.cluster_model_u.cluster_labels
    cluster_correct = int(labels[0] == labels[1] and labels[0] != labels[2])
    print(f"    Strategy 1/3 done. cluster_labels={labels}, correct={cluster_correct}", flush=True)

    # All modalities in one cluster
    print("    Running strategy 2/3: same cluster...", flush=True)
    pipe_same = MultimodalPCAPipeline()
    U_same = pipe_same.denoise_amp(
        X_list, K_list,
        cluster_labels_U=np.array([0, 0, 0]),
        amp_iters=amp_iters
    )["U_denoised"]
    print("    Strategy 2/3 done.", flush=True)

    # Each modality in its own cluster
    print("    Running strategy 3/3: distinct clusters...", flush=True)
    pipe_distinct = MultimodalPCAPipeline()
    U_distinct = pipe_distinct.denoise_amp(
        X_list, K_list,
        cluster_labels_U=np.array([0, 1, 2]),
        amp_iters=amp_iters
    )["U_denoised"]
    print("    Strategy 3/3 done.", flush=True)

    return {
        "clustered":         [reconstruction_error(U_cluster[i][:, :, -1],  U_true[i]) for i in range(3)],
        "same_cluster":      [reconstruction_error(U_same[i][:, :, -1],     U_true[i]) for i in range(3)],
        "distinct_clusters": [reconstruction_error(U_distinct[i][:, :, -1], U_true[i]) for i in range(3)],
        "cluster_correct":   cluster_correct,
    }


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python cluster_ccoef_n_worker.py <n> <seed>", flush=True)
        sys.exit(1)

    n    = int(sys.argv[1])
    seed = int(sys.argv[2])

    print(f">>> Worker started: n={n}, seed={seed}", flush=True)

    # Set seed for full reproducibility of this trial
    np.random.seed(seed)

    load_modules()

    gamma_list = [0.25, 0.8, 3.0]
    r_list     = [2, 3, 2]
    p_list     = [int(g * n) for g in gamma_list]

    print(f">>> p_list={p_list}, r_list={r_list}", flush=True)

    print(f">>> Starting trial (n={n}, seed={seed}, p_list={p_list}, r_list={r_list}, amp_iters=10)...", flush=True)
    result = run_single_trial(n=n, p_list=p_list, r_list=r_list, amp_iters=10)

    print(f">>> Trial complete. Result: {result}", flush=True)

    # Save partial result — one row per clustering strategy, includes seed for SE computation
    os.makedirs("Results/vary_n", exist_ok=True)
    rows = []
    for strategy, errors in result.items():
        if strategy == "cluster_correct":
            continue
        rows.append({
            "n":              n,
            "seed":           seed,
            "trial":          seed,
            "strategy":       strategy,
            "modality_1":     errors[0],
            "modality_2":     errors[1],
            "modality_3":     errors[2],
            "cluster_correct": result["cluster_correct"] if strategy == "clustered" else None,
        })

    df = pd.DataFrame(rows)
    out_path = f"Results/vary_n/partial_result_{n}_{seed}.csv"
    df.to_csv(out_path, index=False)
    print(f">>> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()