# TEA-seq Data Analysis

Applies the same AMP-GMM method (here called OrchAMP) to a real trimodal
single-cell dataset — TEA-seq PBMCs (RNA + ATAC + ADT/protein) — instead of
the synthetic data in `../Numerical_Experiments/`. Task: predict a held-out
ADT protein (CD45RA) from RNA + ATAC (high-dimensional modalities) and the
remaining ADT proteins (low-dimensional), benchmarked against JAFAR, MOFA+,
Multigrate, and Cooperative Lasso.

Scoped to one cell population, `split3_all_celltypes` (all cell types, no
subsetting) — see `Pre-processing/README.md` for the five other splits that
were considered upstream but aren't included in this repo. See the repo
root README for install requirements (Python + R packages needed for this
folder).

## Layout

```
TEA-seq Data Analysis/
├── Pre-processing/    Raw TEA-seq (RNA/ATAC/ADT) → cleaned per-modality
│                       matrices → regression target + train/test splits.
│                       7-step pipeline; see its README for the full order
│                       and what each step reads/writes.
├── Python_scripts/    OrchAMP's own pipeline modules (PCA, empirical-Bayes
│                       denoising, modality clustering) — the dependency
│                       chain used by the orchamp_*_fusion notebooks below.
│                       See its README for the module graph.
└── Notebooks/         Train/test notebooks for OrchAMP (3 fusion strategies)
                        + baselines (JAFAR, MOFA+, Multigrate, Cooperative
                        Lasso), plus a results-aggregation notebook. See its
                        README for the per-notebook breakdown and directory
                        expectations (data/, splits/, models/).
```

Run order across folders: everything in `Pre-processing/` first (produces
`data/` and `splits/`, siblings of these three folders), then anything in
`Notebooks/` (the `orchamp_*_fusion` notebooks there additionally import
`Python_scripts/`).
