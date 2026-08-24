# Hierarchical AMP-GMM

DAIF: a multi-modal denoising and prediction pipeline built on Approximate
Message Passing (AMP) with a per-cluster Gaussian-Mixture empirical-Bayes
prior. Modalities are grouped into clusters (by CKA similarity or otherwise),
jointly denoised via AMP, and the denoised representations feed a downstream
predictor. Compared against multi-view baselines (AJIVE, MCCA, GCCA, MFA,
HPCA), Cooperative Lasso (Ding & Tibshirani, 2021), JAFAR, MOFA+, and
Multigrate.

## Repository layout

| Folder | Contents |
|---|---|
| [`Numerical_Experiments/`](Numerical_Experiments/README.md) | Synthetic-data experiments: Slurm-launched AMP/EB trials aggregated into LaTeX tables and plots. |
| [`TEA-seq Data Analysis/`](TEA-seq%20Data%20Analysis/README.md) | The same method (OrchAMP) applied to a real trimodal single-cell dataset (TEA-seq PBMCs: RNA + ATAC + ADT), benchmarked against JAFAR, MOFA+, Multigrate, and Cooperative Lasso. |
| [`TCGA-BRCA Data Analysis/`](TCGA-BRCA%20Data%20Analysis/README.md) | OrchAMP applied to bulk TCGA breast cancer multi-omics (RNA-seq + methylation + CNV) for Cox survival prediction, benchmarked against MOFA+ and Multigrate. |

Each folder's README has the detailed layout, run order, and per-file
breakdown.

## Requirements

Two runtimes across the repo: Python for everything, plus R for
`TEA-seq Data Analysis/`'s original Seurat-based preprocessing and two of its
baselines (MOFA+, JAFAR), and for `TCGA-BRCA Data Analysis/`'s MOFA+
baseline.

### Python — `Numerical_Experiments/`

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

### Python — `TEA-seq Data Analysis/`

Not pinned in `requirements.txt` (separate environment from the above —
single-cell stack, no `torch`/`mvlearn`/`prince` needed here):

| Package | Used for |
|---|---|
| `numpy`, `scipy`, `pandas` | Core numerics and tabular data |
| `scikit-learn` | Metrics, `StandardScaler`/PCA/`TruncatedSVD`, `LinearRegression`, `GaussianMixture` (via `Python_scripts/`) |
| `anndata`, `scanpy`, `muon` | `.h5ad`/`.h5mu` single-cell data structures and I/O |
| `h5py` | Reading MOFA2's HDF5 export directly (`test_mofa.ipynb`) |
| `matplotlib` | PCA diagnostics, dendrograms, UMAP plots |
| `umap-learn` (imported as `umap`) | UMAP embeddings of denoised test features |
| `joblib` | Parallel grid search (Cooperative Lasso) and model persistence |
| `multigrate`, `scvi-tools` (imported as `scvi`) | The Multigrate baseline (VAE-based multi-omics integration) |

### R — `TEA-seq Data Analysis/`

| Package | Used for |
|---|---|
| `Seurat`, `SeuratDisk`, `SeuratData` | RNA/ATAC QC, clustering, UMAP, cell-type label transfer, `.h5ad` export |
| `rhdf5`, `Matrix` | Parsing the raw 10x-style `feature_matrix.h5` sparse matrix |
| `anndata` (R package), `reticulate` | Writing `.h5ad` files from R |
| `EnsDb.Hsapiens.v86` | Gene annotation reference |
| `leiden`, `nng`, `combinat` | Clustering utilities used in preprocessing |
| `tidyverse`, `dplyr`, `readr`, `glue` | General data wrangling |
| `ggplot2`, `cowplot`, `patchwork`, `RColorBrewer` | QC/UMAP plots |
| `Biobase` | Bioconductor infrastructure dependency (Seurat/SeuratDisk chain) |
| `MOFA2` | The MOFA+ baseline |
| `jafar`, `jsonlite`, `uwot` | The JAFAR baseline (model + metrics I/O + UMAP backend) |
| `rstudioapi` *(optional)* | RStudio-convenience working-directory detection in the JAFAR/MOFA+ scripts; guarded by `requireNamespace(...)`, so its absence doesn't break anything |

### Python — `TCGA-BRCA Data Analysis/`

Not pinned in `requirements.txt` (separate environment from
`Numerical_Experiments/`; overlaps substantially with the TEA-seq Python
environment above, minus the single-cell-specific packages):

| Package | Used for |
|---|---|
| `numpy`, `scipy`, `pandas` | Core numerics and tabular data |
| `scikit-learn` | `StandardScaler`, `TruncatedSVD`, metrics, `GaussianMixture` (via `Python_scripts/`) |
| `torch` | The Cox survival head (`multimodal_prediction_survival.py`), trained via `AdamW` |
| `matplotlib`, `seaborn` | Preprocessing diagnostics (scree plots, residual histograms) |
| `requests`, `tqdm` | Downloading raw data from UCSC Xena with a progress bar |
| `joblib` | Model/scaler persistence |
| `lifelines` | Concordance-index (C-index) computation for survival evaluation |
| `anndata`, `scanpy`, `h5py` | `.h5ad` structures and reading MOFA2's HDF5 export, mirroring the TEA-seq baselines |
| `multigrate`, `scvi-tools` (imported as `scvi`) | The Multigrate baseline |

### R — `TCGA-BRCA Data Analysis/`

Only needed for the MOFA+ baseline — no Seurat stack here (this
preprocessing pipeline is pure Python/pandas, not Seurat-based like TEA-seq):

| Package | Used for |
|---|---|
| `MOFA2` | The MOFA+ baseline |
| `survival` | Fitting the Cox proportional-hazards head on MOFA factors |
| `Matrix`, `reticulate` | Sparse matrices / Python bridge |
| `ggplot2` | Diagnostic plots |
| `jsonlite` | Metrics I/O |
| `rstudioapi` *(optional)* | RStudio-convenience working-directory detection; guarded by `requireNamespace(...)` |
