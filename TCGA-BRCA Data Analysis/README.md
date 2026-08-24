# TCGA-BRCA Data Analysis

Applies the same AMP-GMM method (OrchAMP) to real bulk multi-omics data —
TCGA breast cancer (RNA-seq + DNA methylation + CNV) from UCSC Xena —
instead of synthetic data or the single-cell TEA-seq dataset. Task: Cox
proportional-hazards survival prediction (event = death, `OS.time` as
follow-up) from RNA-seq + methylation (high-dimensional modalities) and a
5-feature CNV subset (low-dimensional), benchmarked against MOFA+ and
Multigrate.

## Layout

```
TCGA-BRCA Data Analysis/
├── Pre-processing/    Raw Xena downloads (RNA-seq/CNV/methylation/survival)
│                       → aligned low-rank matrices → CNV low-dim features
│                       → frozen train/test split. See its README for the
│                       full run order (and a note on two diverged versions
│                       of the main pipeline step).
├── Python_scripts/    OrchAMP's own pipeline modules — shared, unmodified,
│                       with the TEA-seq Data Analysis/Python_scripts/ of
│                       the same name. See its README for the module graph.
└── Notebooks/         Train/test notebooks for OrchAMP (3 fusion
                        strategies) + baselines (MOFA+, Multigrate), plus a
                        results-aggregation notebook. See its README for the
                        per-notebook breakdown and directory expectations.
```

Run order across folders: everything in `Pre-processing/` first (produces
`data/`, `matrices/`, `splits/`, siblings of these three folders), then
anything in `Notebooks/` (the `orchamp_brca_survival_*` notebooks there
additionally import `Python_scripts/`).
