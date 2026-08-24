# Hierarchical AMP-GMM

Numerical experiments for DAIF: a multi-modal denoising and prediction pipeline
built on Approximate Message Passing (AMP) with a per-cluster Gaussian-Mixture
empirical-Bayes prior. Modalities are grouped into clusters (by CKA similarity or
otherwise), jointly denoised via AMP, and the denoised representations feed a
downstream predictor. Compared here against multi-view baselines (AJIVE, MCCA,
GCCA, MFA, HPCA) and Cooperative Lasso (Ding & Tibshirani, 2021).

See `Numerical_Experiments/Python_Scripts/complete_pipeline.py` for the training-
covariate projection modes (`onsager`, `sample_split`, `in_sample_ols`) referenced
throughout the scripts below.

## Repository layout

```
Numerical_Experiments/
├── Python_Scripts/   AMP/EB pipeline modules + the experiment driver scripts
├── Slurm_Scripts/    One SLURM array-job launcher per experiment (mirrors Python_Scripts)
└── Notebooks/        Aggregates each experiment's per-(n, seed) CSVs into a
                       LaTeX table + plots
```

Each experiment follows the same chain: a Slurm array job launches many
`(n, seed)` trials of a Python script, each trial writes
`Results/<experiment>/partial_result_{n}_{seed}.csv`, and the matching notebook
globs that directory to produce the final table/figure.

| Notebook | Script | Slurm job |
|---|---|---|
| `cka_vs_baselines.ipynb` | `cka_vs_baselines.py` | `cka_vs_baselines/cka_vs_baselines.sh` |
| `coop_vs_daif_onsager.ipynb` | `coop_vs_daif_onsager.py` | `coop_vs_daif_onsager/coop_vs_daif_onsager.sh` |
| `pred_err_onsager.ipynb` | `pred_err_vary_n_onsager.py` + `pred_err_vary_n_nl_linear_onsager.py` | matching `.sh` in each script's own Slurm folder |
| `pred_err_vs_baselines_onsager.ipynb` | `pred_err_vs_baselines_onsager.py` | `pred_err_vs_baselines_onsager/pred_err_vs_baselines_onsager.sh` |
| `vary_n.ipynb` | `vary_n.py` | `vary_n/vary_n.sh` |

All six driver scripts import the shared pipeline modules also in
`Python_Scripts/`: `amp.py`, `pca_pack.py`, `preprocessing.py`, `emp_bayes.py`,
`hierarchical_clustering_modalities.py`, `complete_pipeline.py`, and
`other_multimodal.py` (baselines only).

## Requirements

Python ≥ 3.10 (tested with 3.10.18). Install with:

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `numpy`, `scipy` | Core linear algebra, AMP/EB computations |
| `pandas` | Loading/aggregating per-trial result CSVs |
| `scikit-learn` | `GaussianMixture` (EB prior), lasso path, PCA/clustering utilities, metrics |
| `torch` | Neural-net predictor head (`complete_pipeline.py`) |
| `matplotlib` | Result plots |
| `mvlearn` | Baseline methods: AJIVE, MCCA, GCCA |
| `prince` | Baseline method: MFA |
| `ipython`, `jupyter` | Running the aggregation notebooks (`IPython.display.Latex`) |
| `xgboost` *(optional)* | Only for the `"xgboost"` predictor architecture; imported in a `try/except`, so the rest of the pipeline works without it |

The Slurm scripts additionally expect a conda environment named
`multiview-regression` with the above installed (`conda activate
multiview-regression`).

**Note:** `complete_pipeline.py` imports a private sklearn function
(`sklearn.mixture._gaussian_mixture._compute_precision_cholesky`). This isn't
part of sklearn's public API and can change between versions — pin
`scikit-learn==1.7.2` (as in `requirements.txt`) if you hit an `ImportError`
there.

## Running an experiment

Each script takes `<n> <seed>` and writes one partial-result CSV:

```bash
python Numerical_Experiments/Python_Scripts/vary_n.py 3000 0
```

At scale, submit the matching Slurm array job (adjust the account/partition
and the hardcoded paths for your cluster):

```bash
sbatch Numerical_Experiments/Slurm_Scripts/vary_n/vary_n.sh
```

Once all array tasks finish, run the corresponding notebook in
`Numerical_Experiments/Notebooks/` to aggregate results into a LaTeX table and
plots.
