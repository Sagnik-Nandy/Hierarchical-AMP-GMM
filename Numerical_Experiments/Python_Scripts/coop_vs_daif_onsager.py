"""
coop_vs_daif_onsager.py
------------------------
Compares DAIF (clustered CKA) against Cooperative Lasso (Ding & Tibshirani 2021,
eq. 17) on a linear prediction task.
Onsager-debiased counterpart of coop_vs_daif.py.

Cooperative Lasso uses the raw modality matrices X_1,...,X_M directly (no
dimension reduction) and solves:

  J(theta) = (1/2)||y - sum_m X_m theta_m||^2
           + (rho/2) sum_{m<m'} ||X_m theta_m - X_{m'} theta_{m'}||^2
           + sum_m lambda_m ||theta_m||_1

by reformulating it as a standard lasso on an augmented (stacked) design matrix.
rho and lambda are selected jointly by a validation-set search.

DGP: same as pred_err_vary_n.py.
n values: 1000, 1500, 2000.

DAIF training-covariate projection: onsager (Onsager-debiased AMP field;
see DAIF_training_projection_fixes.pdf, Sec. 2). Pinned explicitly even
though it is the pipeline default, for clarity and future-proofing.

Usage
-----
    python coop_vs_daif_onsager.py <n> <seed>

SLURM array
-----------
    #SBATCH --array=0-149   # 3 n_values x 50 seeds = 150 tasks
"""

import sys
import os
import numpy as np
import pandas as pd
import importlib
from itertools import combinations
from sklearn.linear_model import lasso_path, Lasso
from sklearn.preprocessing import StandardScaler


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

    global MultimodalPCAPipelineClustering
    MultimodalPCAPipelineClustering = pipeline.MultimodalPCAPipelineClustering


# =============================================================================
# Data generating process  (aligned with pred_err_vary_n.py)
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


# =============================================================================
# Cooperative Lasso  (Ding & Tibshirani 2021, eq. 17)
# =============================================================================

class CooperativeLasso:
    """
    Cooperative regularized linear regression for M views.

    Minimises the normalised objective (all terms O(1)):

      J(theta) = (1/2n)||y - sum_m X_m theta_m||^2
               + (rho/2n) sum_{m<m'} ||X_m theta_m - X_{m'} theta_{m'}||^2
               + sum_m (lambda/p_m) ||theta_m||_1

    Reformulated as a standard lasso on an augmented design matrix via two
    normalisation steps applied inside _build_augmented:

      Row scaling   : multiply all rows by sqrt(n_tilde/n) so that sklearn's
                      internal 1/n_tilde division becomes 1/n.
      Column scaling: multiply block m by p_m, reparametrising gamma_m = theta_m/p_m,
                      so sklearn's uniform alpha yields alpha/p_m per modality.
                      theta is recovered as theta_m = p_m * gamma_m after fitting.

    LassoCV selects lambda; rho is chosen by a held-out validation set.

    rho is selected by a held-out validation split on the original n samples.

    Parameters
    ----------
    rho_grid   : array-like   grid of cooperation strengths
    val_frac   : float        fraction of training data reserved for rho selection
    cv         : int          inner LassoCV folds
    max_iter   : int          Lasso solver max iterations
    """

    def __init__(self, rho_grid=(0.0, 0.5, 2.0),
                 val_frac=0.2, max_iter=5000):
        self.rho_grid  = list(rho_grid)
        self.val_frac  = val_frac
        self.max_iter  = max_iter

        self.best_rho_    = None
        self.best_lambda_ = None
        self.scalers_     = None   # per-modality column scalers fitted on training data
        self.thetas_      = None   # list of per-modality coefficient vectors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmented(self, X_list_scaled, y, rho):
        """Stack prediction block + all pairwise agreement blocks, normalised.

        Row scale sqrt(n_tilde/n) makes the quadratic terms O(1/n).
        Column scale p_m makes the L1 term O(lambda/p_m) per modality.
        Coefficients returned by sklearn are gamma_m = theta_m / p_m;
        callers must recover theta_m = p_m * gamma_m before predicting.
        """
        M   = len(X_list_scaled)
        n   = X_list_scaled[0].shape[0]
        sr  = np.sqrt(rho)

        n_pairs   = M * (M - 1) // 2
        n_tilde   = n * (1 + n_pairs)
        row_scale = np.sqrt(n_tilde / n)   # corrects sklearn's 1/n_tilde → 1/n

        # Column scaling: multiply block m by p_m so L1 is lambda/p_m per modality
        p_list = [X.shape[1] for X in X_list_scaled]
        X_col  = [X * p for X, p in zip(X_list_scaled, p_list)]

        # Prediction block
        X_pred = np.hstack(X_col) * row_scale
        y_pred = y.copy() * row_scale

        # Agreement blocks: one block per pair (m, m')
        X_agree_blocks = []
        y_agree_blocks = []
        for m, mp in combinations(range(M), 2):
            row_block = np.zeros((n, sum(X.shape[1] for X in X_list_scaled)))
            col_start = 0
            for k, Xk in enumerate(X_col):
                col_end = col_start + Xk.shape[1]
                if k == m:
                    row_block[:, col_start:col_end] = sr * Xk
                elif k == mp:
                    row_block[:, col_start:col_end] = -sr * Xk
                col_start = col_end
            X_agree_blocks.append(row_block * row_scale)
            y_agree_blocks.append(np.zeros(n))

        if X_agree_blocks:
            X_tilde = np.vstack([X_pred] + X_agree_blocks)
            y_tilde = np.concatenate([y_pred] + y_agree_blocks)
        else:
            X_tilde = X_pred
            y_tilde = y_pred

        return X_tilde, y_tilde

    def _scale(self, X_list, fit=True):
        """Standardize columns of each modality. Fit scalers if fit=True."""
        if fit:
            self.scalers_ = []
            out = []
            for X in X_list:
                sc = StandardScaler(with_mean=True, with_std=True)
                out.append(sc.fit_transform(X))
                self.scalers_.append(sc)
        else:
            out = [sc.transform(X) for sc, X in zip(self.scalers_, X_list)]
        return out

    def _predict_raw(self, X_list_scaled, thetas):
        """y_hat = sum_m X_m @ theta_m"""
        return sum(X @ t for X, t in zip(X_list_scaled, thetas))

    def _split_theta(self, beta, X_list_scaled):
        """Split flat beta vector back into per-modality theta_m."""
        thetas, start = [], 0
        for X in X_list_scaled:
            end = start + X.shape[1]
            thetas.append(beta[start:end])
            start = end
        return thetas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_list, y):
        """
        Fit cooperative lasso on training data.

        Parameters
        ----------
        X_list : list of (n, p_m) arrays
        y      : (n,) array
        """
        n     = X_list[0].shape[0]
        n_val = max(1, int(n * self.val_frac))
        n_tr  = n - n_val

        # Fixed validation split used for both lambda and rho selection
        idx    = np.random.permutation(n)
        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:]

        X_tr_scaled = self._scale([X[tr_idx] for X in X_list], fit=True)
        X_va_scaled = self._scale([X[va_idx] for X in X_list], fit=False)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Step 1: fix lambda using the lasso path on the prediction block only (rho=0).
        # No cross-validation — evaluate each lambda on the held-out validation set.
        p_list       = [X.shape[1] for X in X_tr_scaled]
        X_col_tr     = [X * p for X, p in zip(X_tr_scaled, p_list)]
        X_col_va     = [X * p for X, p in zip(X_va_scaled, p_list)]
        X_pred_only  = np.hstack(X_col_tr)   # (n_tr, sum_p); rho=0 → row_scale=1

        print(f"      [CoopLasso] computing lasso path at rho=0 to fix lambda...", flush=True)
        alphas, coefs, _ = lasso_path(X_pred_only, y_tr,
                                       max_iter=self.max_iter)
        # coefs: (sum_p, n_alphas); alphas: descending

        X_va_pred = np.hstack(X_col_va)
        val_mses  = np.mean((y_va[:, None] - X_va_pred @ coefs) ** 2, axis=0)
        best_idx  = int(np.argmin(val_mses))
        lambda_fixed = float(alphas[best_idx])

        _rel_top = lambda_fixed / alphas[0]
        _rel_bot = lambda_fixed / alphas[-1]
        if _rel_top > 0.95:
            print(f"        [WARN] lambda*={lambda_fixed:.5e} at TOP of path "
                  f"(over-regularised?). Path=[{alphas[-1]:.5e}, {alphas[0]:.5e}]", flush=True)
        elif _rel_bot < 1.05:
            print(f"        [WARN] lambda*={lambda_fixed:.5e} at BOTTOM of path "
                  f"(under-regularised?). Path=[{alphas[-1]:.5e}, {alphas[0]:.5e}]", flush=True)
        print(f"      [CoopLasso] lambda_fixed={lambda_fixed:.5e}  "
              f"path=[{alphas[-1]:.5e}, {alphas[0]:.5e}]", flush=True)

        # Step 2: search over rho with fixed lambda, using the same validation split
        best_val_mse = np.inf
        best_rho     = self.rho_grid[0]

        print(f"      [CoopLasso] searching rho in {self.rho_grid}...", flush=True)
        for rho in self.rho_grid:
            X_tilde, y_tilde = self._build_augmented(X_tr_scaled, y_tr, rho)

            lasso = Lasso(alpha=lambda_fixed, max_iter=self.max_iter, fit_intercept=False)
            lasso.fit(X_tilde, y_tilde)

            gammas  = self._split_theta(lasso.coef_, X_tr_scaled)
            thetas  = [g * X.shape[1] for g, X in zip(gammas, X_tr_scaled)]
            y_hat_v = self._predict_raw(X_va_scaled, thetas)
            val_mse = float(np.mean((y_va - y_hat_v) ** 2))

            print(f"        rho={rho:.2f}  val_mse={val_mse:.6f}", flush=True)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_rho     = rho

        print(f"      [CoopLasso] best rho={best_rho}, lambda={lambda_fixed:.5e}, "
              f"val_mse={best_val_mse:.6f}", flush=True)

        # Refit on full training data with best (rho, lambda_fixed)
        X_full_scaled = self._scale(X_list, fit=True)
        X_tilde_full, y_tilde_full = self._build_augmented(X_full_scaled, y, best_rho)

        lasso = Lasso(alpha=lambda_fixed, max_iter=self.max_iter, fit_intercept=False)
        lasso.fit(X_tilde_full, y_tilde_full)

        self.best_rho_    = best_rho
        self.best_lambda_ = lambda_fixed
        gammas            = self._split_theta(lasso.coef_, X_full_scaled)
        self.thetas_      = [g * X.shape[1] for g, X in zip(gammas, X_full_scaled)]

        return self

    def predict(self, X_test_list):
        """
        Predict on test data.

        Parameters
        ----------
        X_test_list : list of (n_test, p_m) arrays  (same column order as training)
        """
        X_scaled = self._scale(X_test_list, fit=False)
        return self._predict_raw(X_scaled, self.thetas_)


# =============================================================================
# Single trial
# =============================================================================

def run_single_trial(n, p_list, r_list, amp_iters, sigma=0.1):
    print(f"    Generating data (n={n}, p_list={p_list}, r_list={r_list})...", flush=True)
    X_list, U_true, V_list, D_list = generate_data(n, p_list, r_list)
    K_list = r_list

    # Linear response
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
    # DAIF: clustered with CKA  (num_clusters=2)
    # ------------------------------------------------------------------
    print("    DAIF (clustered CKA, linear)...", flush=True)
    pipe = MultimodalPCAPipelineClustering()
    pipe.y_train      = y_train
    pipe.architecture = "linear"
    pipe.projection_mode = "onsager"
    pipe.arch_kwargs  = {"fit_intercept": False}
    pipe.denoise_amp(
        X_list, K_list,
        compute_clusters=True, num_clusters=2,
        amp_iters=amp_iters, similarity_method="cka", sigma=1.0
    )
    _, y_pred_daif = pipeline.predict_from_test_data(pipe, X_test)
    results["daif"] = {
        "pred_error": float(np.mean((y_test - y_pred_daif) ** 2)),
    }
    print("    DAIF done.", flush=True)

    # ------------------------------------------------------------------
    # Cooperative Lasso  (Ding & Tibshirani 2021)
    # ------------------------------------------------------------------
    print("    Cooperative Lasso...", flush=True)
    coop = CooperativeLasso(
        rho_grid=[0.0, 0.5, 2.0],
        val_frac=0.2, max_iter=5000
    )
    coop.fit(X_list, y_train)
    y_pred_coop = coop.predict(X_test)
    results["coop_lasso"] = {
        "pred_error":  float(np.mean((y_test - y_pred_coop) ** 2)),
        "best_rho":    float(coop.best_rho_),
        "best_lambda": float(coop.best_lambda_),
    }
    print(f"    Cooperative Lasso done. "
          f"(rho={coop.best_rho_}, lambda={coop.best_lambda_:.5f})", flush=True)

    return results


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python coop_vs_daif_onsager.py <n> <seed>", flush=True)
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

    print(f">>> Trial complete. Results: { {k: v['pred_error'] for k,v in result.items()} }",
          flush=True)

    os.makedirs("Results/coop_vs_daif_onsager", exist_ok=True)
    rows = []
    for method, vals in result.items():
        row = {"n": n, "seed": seed, "method": method,
               "pred_error": vals["pred_error"]}
        if "best_rho" in vals:
            row["best_rho"]    = vals["best_rho"]
            row["best_lambda"] = vals["best_lambda"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = f"Results/coop_vs_daif_onsager/partial_result_{n}_{seed}.csv"
    df.to_csv(out_path, index=False)
    print(f">>> Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
