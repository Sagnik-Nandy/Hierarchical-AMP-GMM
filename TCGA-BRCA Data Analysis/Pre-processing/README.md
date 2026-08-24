# Pre-processing

Turns raw TCGA-BRCA multi-omics data (RNA-seq, CNV, DNA methylation,
survival) into aligned low-rank matrices, a low-dimensional CNV feature set,
and frozen train/test splits. Run in this order:

1. `tcga_brca_pipeline.py` **or** `tcga_brca_pipeline.ipynb` — see note below
2. `preprocess_cnv_brca.ipynb`
3. `generate_brca_survival_splits.ipynb`

## Note on step 1: two versions, not interchangeable

`tcga_brca_pipeline.py` and `tcga_brca_pipeline.ipynb` have diverged and are
both included here rather than picking one:

- **`tcga_brca_pipeline.py`** — the clean, portable version. Downloads RNA-seq,
  CNV, and methylation directly from public UCSC Xena URLs into a relative
  `tcga_brca_data/`, filters each modality by a variance **percentile**
  threshold (bottom `rna_var_pct`/`meth_var_pct`% dropped), and rank-selects
  the SVD by a variance-explained threshold (`variance_threshold=0.90`).
- **`tcga_brca_pipeline.ipynb`** — the notebook actually run to produce what's
  in `../matrices/`. Assumes the raw files are already downloaded, and
  instead of a percentile filter, keeps a **fixed top-N features** per
  modality (`TOP_FEATURES = {"rna": 2000, "cnv": 1000, "meth": 5000}`) before
  the same SVD rank-selection step.

If you're reproducing `../matrices/` as it exists in this repo, run the
notebook. If you want a portable starting point for a different feature
count or threshold, start from the script.

## What it does / produces

**1. `tcga_brca_pipeline.{py,ipynb}`** — for each modality:
- Downloads from UCSC Xena (script only; the notebook expects the four raw
  files already present — see Raw inputs below).
- Loads (Xena files are features×samples; transposed to samples×features).
- Restricts to primary-tumor samples (TCGA barcode position 14 = `01`) and
  intersects sample IDs across all three modalities.
- Per modality: drops high-missing features, imputes remainder (median /
  zero for CNV), removes low-variance features, and normalizes (Z-score for
  RNA/methylation; methylation additionally gets an M-value transform,
  `M = log2(beta / (1 - beta))`, before scaling).
- Rank-selects and computes a truncated SVD per modality: `X ≈ U D Vᵀ`,
  `Z = X - X_approx` the residual.
- Saves `../matrices/{RNA,CNV,Methylation}_{X_full,X_approx,U,D,V,Z_residual}.csv`
  and diagnostic plots (singular-value scree, residual-vs-Gaussian
  histograms).

**2. `preprocess_cnv_brca.ipynb`** — builds the CNV **low-dimensional (LD)**
modality (mirrors `separate_response.ipynb`'s role in the TEA-seq pipeline):
selects the top-5 CNV features by marginal variance from `CNV_X_full.csv`,
removes the top-3 signal PCs, estimates **per-feature** noise variance
`tau_j²` from the residual (chosen over a single global `tau²` because it's
what guarantees `cov(A_norm) - I` stays PSD under heteroscedastic noise),
and divides each column by `tau_j`. Verifies the result is PSD, then saves
`../matrices/CNV_X_ld_features.csv`.

**3. `generate_brca_survival_splits.ipynb`** — loads survival labels
(`OS`/`OS.time` from `BRCA_survival.tsv`), drops rows with missing or
non-positive survival time, intersects sample IDs across all three
modalities and the cleaned survival table, then does one stratified 80/20
train/test split (stratified on the event indicator). Saves
`../splits/brca_survival_{sample_ids,train_idx,test_idx}.csv`. Unlike the
TEA-seq preprocessing, there's only one population/split here — no per-split
scoping needed.

## Raw inputs (for the notebook path, or if you skip the script's download)

Place in a `data/` folder as a sibling of this one. `tcga_brca_pipeline.py`
downloads these automatically from UCSC Xena; the Dropbox links are a mirror
if you'd rather skip that:

| Filename | Source | Dropbox mirror |
|---|---|---|
| `BRCA_RNAseq.tsv.gz` | UCSC Xena, RSEM log2-normalized RNA-seq | [Dropbox](https://www.dropbox.com/scl/fi/2y8aygrlxcvkgf6qen8yz/BRCA_RNAseq.tsv.gz?rlkey=2gcylsxlilcxuvjxmfd16zsdw&st=7oa4m4s0&dl=0) |
| `BRCA_CNV.tsv.gz` | UCSC Xena, GISTIC2 thresholded copy number | [Dropbox](https://www.dropbox.com/scl/fi/jav0ojo3rt2bagrec41i6/BRCA_CNV.tsv.gz?rlkey=yval2esy8ubrekgwsf494mx68&st=akftx6gg&dl=0) |
| `BRCA_Methylation450.tsv.gz` | UCSC Xena, HumanMethylation450 beta values | [Dropbox](https://www.dropbox.com/scl/fi/jfhu52uzfbat4fmc11s7t/BRCA_Methylation450.tsv.gz?rlkey=dy9u9b2xl3izqscqn6p0v4f1v&st=bz5uz6z9&dl=0) |
| `BRCA_survival.tsv` | UCSC Xena, survival/clinical labels | [Dropbox](https://www.dropbox.com/scl/fi/80m85dtl04et621tvuoqg/BRCA_survival.tsv?rlkey=km7p381x7i4tgv0pfjqvvuero&st=iv5kr9va&dl=0) |

## Directory layout

```
TCGA-BRCA Data Analysis/
├── Pre-processing/    <- this folder
├── data/                <- raw Xena downloads (sibling of Pre-processing/)
├── matrices/             <- created by step 1 (+ CNV_X_ld_features.csv from step 2)
├── splits/               <- created by step 3
└── Notebooks/            <- consumes matrices/, splits/, data/ (survival labels)
```

## Pre-processed downloads

If you'd rather skip running the pipeline above, here are the
already-computed outputs (mirrors of this project's local `../matrices/` and
`../splits/`, not tracked in git — see `.gitignore`), shared as whole
folders:

| Folder | Contents | Download |
|---|---|---|
| `matrices/` | The 20 files from steps 1–2: per-modality `{RNA,CNV,Methylation}_{X_full,X_approx,U,D,V,Z_residual}.csv` plus `CNV_X_preprocessed.csv` and `CNV_X_ld_features.csv` | [Dropbox folder](https://www.dropbox.com/scl/fo/8f2y72ylaqn3yeze91v69/ADgVz2ySHb2UI7MHaS2Skpg?rlkey=eo52medy3d8igeutbkgp5dllo&st=64k2de5c&dl=0) |
| `splits/` | `brca_survival_{sample_ids,train_idx,test_idx}.csv` from step 3 | [Dropbox folder](https://www.dropbox.com/scl/fo/bsanzulhup10wdj4l24kt/ACqD3cEouwGq0HYhwdCeUUA?rlkey=7r8xm6qqkm9ntp4x8ii2y7vt0&st=nzodn59a&dl=0) |
