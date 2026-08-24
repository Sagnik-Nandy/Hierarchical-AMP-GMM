"""
pred_err_vs_baselines_onsager.py
-----------------------------------
Prediction error experiment — single-index nonlinear response, varies n.
Compares DAIF-CKA against multi-view baselines: AJIVE, MCCA, GCCA, MFA, HPCA.
Onsager-debiased counterpart of pred_err_vs_baselines.py.

Response: y = (U1@β₁ + U2@β₂)² + U3@β₃ + noise   (single-index / nonlinear)
Predictor: TwoLayerNN (input → Linear(32) → ReLU → Linear(1)),
           Adam, 2000 epochs, lr=1e-3  — same architecture for all methods.

DAIF-CKA: uses the full AMP posterior pipeline (_fit_predictor / predict_from_test_data).
Baselines: concatenated U_denoised across views → TwoLayerNN trained directly.
           Test features: X_test_i @ V_i @ inv(V_i^T V_i)  (OLS projection).

DAIF-CKA training-covariate projection: onsager (Onsager-debiased AMP field;
see DAIF_training_projection_fixes.pdf, Sec. 2). Pinned explicitly even
though it is the pipeline default, for clarity and future-proofing.
Baseline methods (AJIVE/MCCA/GCCA/MFA/HPCA) are unaffected — they use their
own OLS feature construction, independent of the DAIF pipeline.

Usage
-----
    python pred_err_vs_baselines_onsager.py <n> <seed>

SLURM array
-----------
    #SBATCH --array=0-149   # 3 n_values x 50 seeds = 150 tasks
"""

import sys
import os
import numpy as np
import pandas as pd
import importlib
import torch
import torch.nn as nn


# =============================================================================
# Module loading
# =============================================================================

def load_modules():
    print(">>> Loading modules...", flush=True)
    sys.path.append('./Python_Scripts')
    global pipeline, other, TwoLayerNN

    for mod_name in ["amp", "pca_pack", "preprocessing", "emp_bayes",
                     "hierarchical_clustering_modalities", "complete_pipeline"]:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        print(f"    loaded: {mod_name}", flush=True)

    pipeline = importlib.import_module("complete_pipeline")
    importlib.reload(pipeline)

    other = importlib.import_module("other_multimodal")
    importlib.reload(other)
    print(">>> All modules loaded.", flush=True)

    global MultimodalPCAPipelineClustering
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClustering
    TwoLayerNN = pipeline.TwoLayerNN


# =============================================================================
# Data generating process  (same as pred_err_vary_n_nl_linear.py)
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

    D1 = np.diag([5 * (i + 1) for i in range(r_list[0])])
    D2 = np.diag([5 * (i + 1) for i in range(r_list[1])])
    D3 = np.diag([5 * (i + 1) for i in range(r_list[2])])

    Z1 = np.random.randn(n, p_list[0]) / np.sqrt(n)
    Z2 = np.random.randn(n, p_list[1]) / np.sqrt(n)
    Z3 = np.random.randn(n, p_list[2]) / np.sqrt(n)

    X1 = (1 / n) * U1 @ D1 @ V1.T + Z1
    X2 = (1 / n) * U2 @ D2 @ V2.T + Z2
    X3 = (1 / n) * U3 @ D3 @ V3.T + Z3

    return [X1, X2, X3], U_true, [V1, V2, V3], [D1, D2, D3]


def nonlinear_response(U_list, beta, r_list, sigma, n):
    beta_1 = beta[:r_list[0]]
    beta_2 = beta[r_list[0]:r_list[0] + r_list[1]]
    beta_3 = beta[r_list[0] + r_list[1]:]
    nonlin  = (U_list[0] @ beta_1 + U_list[1] @ beta_2) ** 2
    lin     = U_list[2] @ beta_3
    return nonlin + lin + sigma * np.random.randn(n)


# =============================================================================
# Neural net training for baselines
# =============================================================================

def train_two_layer_nn(X_train_np, y_train_np, epochs=2000, lr=1e-3, hidden_dim=32,
                       device="cpu"):
    """
    Train a TwoLayerNN on (X_train_np, y_train_np) using Adam.
    Returns the fitted model in eval mode.
    """
    X = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)

    model = TwoLayerNN(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_two_layer_nn(model, X_test_np, device="cpu"):
    X = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(X).squeeze(1).cpu().numpy()


# =============================================================================
# Baseline feature construction
# =============================================================================

def baseline_train_features(model, X_train):
    """Project each training view onto learned V loadings via OLS, then concatenate."""
    parts = []
    for i, X_i in enumerate(X_train):
        V_i = model.V_denoised_[i]
        VtV = V_i.T @ V_i
        U_i = X_i @ V_i @ np.linalg.inv(VtV)
        parts.append(U_i)
    return np.hstack(parts)


def baseline_test_features(model, X_test):
    """
    Project each test view onto the learned V loadings via OLS:
        U_test_i = X_test_i @ V_i @ inv(V_i^T V_i)
    then concatenate across views.
    """
    parts = []
    for i, X_i in enumerate(X_test):
        V_i   = model.V_denoised_[i]                         # (p_i, r_i)
        VtV   = V_i.T @ V_i                                  # (r_i, r_i)
        U_i   = X_i @ V_i @ np.linalg.inv(VtV)              # (n_test, r_i)
        parts.append(U_i)
    return np.hstack(parts)


# =============================================================================
# Single trial
# =============================================================================

def run_single_trial(n, p_list, r_list, amp_iters, sigma=0.1):
    print(f"    Generating data (n={n}, p_list={p_list}, r_list={r_list})...", flush=True)
    X_list, U_true, V_list, D_list = generate_data(n, p_list, r_list)
    K_list = r_list

    beta    = generate_rademacher(sum(r_list))
    y_train = nonlinear_response(U_true, beta, r_list, sigma, n)

    # Test data — fresh U and Z, same V and D
    n_test  = n // 10
    U_test1 = generate_rademacher((n_test, r_list[0]))
    U_test2 = np.hstack([U_test1, generate_three_point((n_test, r_list[1] - r_list[0]))])
    U_test3 = generate_nonlinear_rademacher((n_test, r_list[2]))
    U_test  = [U_test1, U_test2, U_test3]
    X_test  = [(1 / n) * U_test[i] @ D_list[i] @ V_list[i].T
               + np.random.randn(n_test, p_list[i]) / np.sqrt(n)
               for i in range(3)]
    y_test  = nonlinear_response(U_test, beta, r_list, sigma, n_test)

    print(f"    Data generated. X shapes: {[X.shape for X in X_list]}", flush=True)

    results = {}

    # ------------------------------------------------------------------
    # DAIF-CKA
    # ------------------------------------------------------------------
    print("    DAIF-CKA...", flush=True)
    pipe = MultimodalPCAPipelineClustering()
    pipe.y_train      = y_train
    pipe.architecture = "neural_net"
    pipe.projection_mode = "onsager"

    pipe.denoise_amp(
        X_list, K_list,
        compute_clusters=True, num_clusters=2,
        similarity_method="cka", sigma=1.0,
        amp_iters=amp_iters,
    )
    _, y_pred = pipeline.predict_from_test_data(pipe, X_test)
    results["daif_cka"] = float(np.mean((y_test - y_pred) ** 2))
    print("    DAIF-CKA done.", flush=True)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    joint_rank = r_list[0]   # modalities 1 and 2 share the first 2 factors

    baseline_models = [
        ("ajive", other.AJIVEReconstructor(rank_list=r_list, joint_rank=joint_rank)),
        ("mcca",  other.MCCAJointIndividual(individual_ranks=r_list, joint_rank=joint_rank)),
        ("gcca",  other.GCCAJointIndividual(individual_ranks=r_list, joint_rank=joint_rank)),
        ("mfa",   other.MFAJointIndividual(individual_ranks=r_list, joint_rank=joint_rank)),
        ("hpca",  other.HPCA(joint_rank=joint_rank, individual_ranks=r_list)),
    ]

    for name, model in baseline_models:
        print(f"    {name.upper()}...", flush=True)
        try:
            model.fit(X_list)

            X_feat_train = baseline_train_features(model, X_list)   # (n, sum r)
            X_feat_test  = baseline_test_features(model, X_test)  # (n_test, sum r)

            nn_model = train_two_layer_nn(X_feat_train, y_train)
            y_pred   = predict_two_layer_nn(nn_model, X_feat_test)

            results[name] = float(np.mean((y_test - y_pred) ** 2))
        except Exception as e:
            print(f"    {name.upper()} failed: {e}", flush=True)
            results[name] = np.nan
        print(f"    {name.upper()} done.", flush=True)

    return results, y_test


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python pred_err_vs_baselines_onsager.py <n> <seed>", flush=True)
        sys.exit(1)

    n    = int(sys.argv[1])
    seed = int(sys.argv[2])
    print(f">>> Worker started: n={n}, seed={seed}", flush=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    load_modules()

    gamma_list = [0.25, 0.8, 3.0]
    r_list     = [2, 3, 2]
    p_list     = [int(g * n) for g in gamma_list]

    print(f">>> n={n}, p_list={p_list}, r_list={r_list}", flush=True)
    print(f">>> Starting trial (n={n}, seed={seed}, amp_iters=10)...", flush=True)

    results, _ = run_single_trial(n=n, p_list=p_list, r_list=r_list, amp_iters=10)

    print(">>> Trial complete.", flush=True)

    os.makedirs("Results/pred_err_vs_baselines_onsager", exist_ok=True)
    rows = []
    for method, pred_error in results.items():
        rows.append({
            "n":            n,
            "seed":         seed,
            "method":       method,
            "pred_error_nn": pred_error,
        })

    df = pd.DataFrame(rows)
    out_path = f"Results/pred_err_vs_baselines_onsager/partial_result_{n}_{seed}.csv"
    df.to_csv(out_path, index=False)
    print(f">>> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
