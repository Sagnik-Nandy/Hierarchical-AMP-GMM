# Pre-processing

Turns raw TEA-seq (RNA + ATAC + ADT) and a CITE-seq reference into cleaned,
matched-cell matrices for each modality, merges them into one `MuData`
object, then prepares the regression target and train/test/hyperparameter
splits consumed by `../Notebooks/`. Run the files in this folder in order:

1. `Pre-processing_tea_seq_data.R`
2. `adt_pre_processing.ipynb`
3. `pre_processing.ipynb`
4. `generate_cell_labels_meta.ipynb`
5. `separate_response.ipynb`
6. `generate_hyperparameter_splits.ipynb`
7. `generate_train_test_splits.ipynb`

This pipeline mirrors the TEA-seq pre-processing used in
[Nandy & Ma (2024), "Multimodal data integration and cross-modal querying
via orchestrated approximate message passing"](https://arxiv.org/abs/2407.19030)
— the same dataset is analyzed there. See the companion code at
[Sagnik-Nandy/OrchAMP](https://github.com/Sagnik-Nandy/OrchAMP).

The dataset itself is from
[Swanson et al. (2021), "Simultaneous trimodal single-cell measurement of
transcripts, epitopes, and chromatin accessibility using TEA-seq,"
*eLife* 10:e63632](https://pubmed.ncbi.nlm.nih.gov/33835024/), which
introduced the TEA-seq assay.

## Directory layout

```
TEA-seq Data Analysis/
├── Pre-processing/   <- this folder: scripts + raw inputs go here
├── data/              <- created by step 3 (sibling of Pre-processing/); step 5 adds data/response/
├── splits/            <- created by steps 6-7
└── Notebooks/         <- consumes data/ and splits/ (see ../Notebooks/README.md)
```

Before running, set `data/` up as a sibling of `Pre-processing/` — step 3
writes to `../data/`. `splits/` is created automatically by step 6.

## Raw inputs to place in this folder

Filenames are hardcoded in the scripts, so they must match **exactly**
(case-sensitive):

| Filename | What it is | Download |
|---|---|---|
| `feature_matrix.h5` | 10x-style combined feature-barcode matrix (RNA + ATAC) for the TEA-seq sample. Must contain `matrix/barcodes` and `matrix/features/{name,feature_type}`, with `feature_type` values `"Gene Expression"` and `"Peaks"`. | [Dropbox](https://www.dropbox.com/scl/fi/afw1hxewpcpw2t3vgvl6c/feature_matrix.h5?rlkey=hr0fy7xloa9dq0x9zbuzs85kt&st=rult3nj7&dl=0) |
| `adt_counts.csv` | Raw ADT/protein counts, one row per cell barcode. Column 1 = barcode ID; column 2 is dropped as metadata; remaining columns = protein marker counts. | [Dropbox](https://www.dropbox.com/scl/fi/km1nwfualgeenvm8hpr79/adt_counts.csv?rlkey=mvjs2w95a7o2eb4xnv09nwfgx&st=3kbmidtm&dl=0) |
| `pbmc_10k_v3.rds` | CITE-seq reference Seurat object used for cell-type label transfer onto the RNA cells. Optional to place manually — `Pre-processing_tea_seq_data.R` auto-downloads it if missing. | *(auto-downloaded)* |

Also update the `setwd(...)` line near the top of
`Pre-processing_tea_seq_data.R` to point at this folder on your machine.

## What each step produces

**1. `Pre-processing_tea_seq_data.R`** — QC-filters and clusters RNA/ATAC,
transfers cell-type labels from the reference, filters ADT by depth,
intersects the trimodal cell set, and writes (into this same folder):

- `cleaned_rna_reads_tea_seq.h5ad`, `cleaned_rna_counts_tea_seq.h5ad`
- `cleaned_atac_reads_tea_seq.h5ad`, `cleaned_atac_counts_tea_seq.h5ad`
- `cleaned_adt_tea_seq.csv`
- `cleaned_cell_labels_meta_tea_seq.csv`
- `final_cell_barcodes_tea_seq.csv`, `final_rna_features_tea_seq.csv`,
  `final_atac_features_tea_seq.csv`, `final_adt_features_tea_seq.csv`

**2. `adt_pre_processing.ipynb`** — reads `cleaned_adt_tea_seq.csv`, applies
CLR normalization + AMP-style noise-variance whitening, selects the top 40
most variable proteins, and overwrites:

- `cleaned_adt_normalized.csv`, `cleaned_adt_counts.csv`,
  `final_adt_features_tea_seq.csv` (now the 40 selected proteins)

**3. `pre_processing.ipynb`** — reads all `cleaned_*`/`final_*` files above,
packages counts + normalized layers per modality, merges into one `MuData`,
and writes to `../data/`:

- `multi.h5mu` (merged trimodal object)
- `rna.h5ad`, `atac.h5ad`, `adt.h5ad` (per-modality)

**4. `generate_cell_labels_meta.ipynb`** — regenerates
`../data/cleaned_cell_labels_meta_tea_seq.csv` directly from `rna.h5ad`'s
`celltype` obs column (with a barcode/order verification check), so the
metadata file can't drift out of sync with the AnnData files. Re-run whenever
`rna.h5ad` is regenerated.

**5. `separate_response.ipynb`** — extracts the regression target protein
(`CD45RA`, configurable) from `adt.h5ad`'s `norm` layer into
`../data/response/CD45RA.csv`, and writes `../data/adt_minus_CD45RA.h5ad`
(the remaining 39 proteins, target removed to prevent leakage) — this is the
LD modality the `orchamp_*_fusion` notebooks and baselines train against.

**6. `generate_hyperparameter_splits.ipynb`** — carves out a stratified 10%
hyperparameter-tuning subset (`split3_all_celltypes` only — see note below)
into `../splits/tea_split3_all_celltypes_hyper_idx.csv`.

**7. `generate_train_test_splits.ipynb`** — excludes the hyper-indices from
step 6, then does a stratified 80/20 train/test split into
`../splits/tea_split3_all_celltypes_{train,test}_idx.csv`.

**Note on split scope:** the source versions of steps 6–7 generate six named
splits, each restricting the cell population to a different subset before
the hyper/train/test partition:

| Split | Cell types included |
|---|---|
| `split1_tcells` | CD4 Memory, CD4 Naive, CD8 Naive, CD8 effector, Double negative T cell |
| `split2_cd4_cd8_only` | CD4 Memory, CD4 Naive, CD8 Naive, CD8 effector (T cells minus double-negatives) |
| `split3_all_celltypes` | No restriction — every cell type in the dataset (the split pushed here) |
| `split4_b_lineage` | B cell progenitor, pre-B cell |
| `split5_myeloid` | CD14+ Monocytes, CD16+ Monocytes, Dendritic cell, pDC, Platelets |
| `split6_mixed_b` | All of `split4_b_lineage`'s remaining (non-hyper) cells, plus a 50%-matched random sample of every other cell type |

Since only `split3_all_celltypes` is pushed to this repo (see
`../Notebooks/`), both notebooks here have been trimmed to generate
`split3_all_celltypes` only — the `split6_mixed_b` construction (a separate
function, only in `generate_train_test_splits.ipynb`) has been removed
entirely rather than trimmed, since it isn't a plain cell-type filter like
the other five.
