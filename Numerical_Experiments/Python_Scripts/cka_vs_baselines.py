
"""
cka_vs_baselines.py
--------------------
Runs a SINGLE trial for a given (n, seed), comparing:
  - AMP pipeline with CKA clustering (DAIF-CKA)
  - Baseline multi-view methods: AJIVE, MCCA, GCCA, MFA, HPCA

Varies n in [2000, 2500, 3000], same DGP as vary_n.py.

Usage
-----
    python cka_vs_baselines.py <n> <seed>

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
    print(">>> AMP modules loaded.", flush=True)

    global MultimodalPCAPipelineClustering
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClustering

    global other
    other = importlib.import_module("other_multimodal")
    importlib.reload(other)
    print(">>> Baseline modules loaded.", flush=True)


# =============================================================================
# Data generating process  (aligned with vary_n.py / vary_meth.py)
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
    print(f"    Generating data (n={n}, p_list={p_list}, r_list={r_list})...", flush=True)
    X_list, U_true = generate_data(n, p_list, r_list)
    print(f"    Data generated. X shapes: {[X.shape for X in X_list]}", flush=True)
    K_list = r_list

    results = {}

    # joint_rank = r_list[0] = 2: U1 and U2 share the first 2 factors in the DGP
    joint_rank = r_list[0]

    # ------------------------------------------------------------------
    # AMP: clustered with CKA
    # ------------------------------------------------------------------
    print("    AMP clustered (CKA)...", flush=True)
    pipe_cluster = MultimodalPCAPipelineClustering()
    U_cluster = pipe_cluster.denoise_amp(
        X_list, K_list,
        compute_clusters=True, num_clusters=2,
        amp_iters=amp_iters, similarity_method="cka",
        sigma=1.0
    )["U_denoised"]
    results["amp_clustered_cka"] = [
        reconstruction_error(U_cluster[i][:, :, -1], U_true[i]) for i in range(3)
    ]
    print("    AMP clustered (CKA) done.", flush=True)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    for name, model in [
        ("ajive", other.AJIVEReconstructor(rank_list=r_list, joint_rank=joint_rank)),
        ("mcca",  other.MCCAJointIndividual(individual_ranks=r_list, joint_rank=joint_rank)),
        ("gcca",  other.GCCAJointIndividual(individual_ranks=r_list, joint_rank=joint_rank)),
        ("hpca",  other.HPCA(joint_rank=joint_rank, individual_ranks=r_list)),
    ]:
        print(f"    {name.upper()}...", flush=True)
        try:
            model.fit(X_list)
            results[name] = [
                reconstruction_error(model.U_denoised_[i], U_true[i]) for i in range(3)
            ]
        except Exception as e:
            print(f"    {name.upper()} failed: {e}", flush=True)
            results[name] = [np.nan, np.nan, np.nan]
        print(f"    {name.upper()} done.", flush=True)

    return results


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python cka_vs_baselines.py <n> <seed>", flush=True)
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

    os.makedirs("Results/cka_vs_baselines", exist_ok=True)
    rows = []
    for method_name, errors in result.items():
        rows.append({
            "n":          n,
            "seed":       seed,
            "method":     method_name,
            "modality_1": errors[0],
            "modality_2": errors[1],
            "modality_3": errors[2],
        })

    df = pd.DataFrame(rows)
    out_path = f"Results/cka_vs_baselines/partial_result_{n}_{seed}.csv"
    df.to_csv(out_path, index=False)
    print(f">>> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
