# Pre-processing

Turns raw TEA-seq (RNA + ATAC + ADT) and a CITE-seq reference into cleaned,
matched-cell matrices for each modality, then merges them into one `MuData`
object. Run the three files in this folder in order:

1. `Pre-processing_tea_seq_data.R`
2. `adt_pre_processing.ipynb`
3. `pre_processing.ipynb`

This pipeline mirrors the TEA-seq pre-processing used in
[Nandy & Ma (2024), "Multimodal data integration and cross-modal querying
via orchestrated approximate message passing"](https://arxiv.org/abs/2407.19030)
— the same dataset is analyzed there. See the companion code at
[Sagnik-Nandy/OrchAMP](https://github.com/Sagnik-Nandy/OrchAMP).

## Directory layout

```
TEA-seq Data Analysis/
├── Pre-processing/   <- this folder: scripts + raw inputs go here
└── data/              <- created by step 3 (sibling of Pre-processing/)
```

Before running, set `data/` up as a sibling of `Pre-processing/` — step 3
writes to `../data/`.

## Raw inputs to place in this folder

Filenames are hardcoded in the scripts, so they must match **exactly**
(case-sensitive):

| Filename | What it is |
|---|---|
| `feature_matrix.h5` | 10x-style combined feature-barcode matrix (RNA + ATAC) for the TEA-seq sample. Must contain `matrix/barcodes` and `matrix/features/{name,feature_type}`, with `feature_type` values `"Gene Expression"` and `"Peaks"`. |
| `adt_counts.csv` | Raw ADT/protein counts, one row per cell barcode. Column 1 = barcode ID; column 2 is dropped as metadata; remaining columns = protein marker counts. |
| `pbmc_10k_v3.rds` | CITE-seq reference Seurat object used for cell-type label transfer onto the RNA cells. Optional to place manually — `Pre-processing_tea_seq_data.R` auto-downloads it if missing. |

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
