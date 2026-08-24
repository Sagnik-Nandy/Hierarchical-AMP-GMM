# Python_scripts

The AMP/EB pipeline modules used by the three `orchamp_brca_survival_*`
notebooks in `../Notebooks/` (the OrchAMP method itself, as opposed to the
baseline notebooks, which only use third-party packages). Loaded via
`sys.path.append("../Python_scripts")` from `Notebooks/`.

## Entry point and dependency graph

`multimodal_prediction_survival.py` is the public entry point
(`MultimodalClusterAllUPipeline`, `train_linear_cox_survival`,
`predict_survival_from_test_data_all`) — a stripped-down, Cox-survival-only
version of the (unpublished here) `multimodal_prediction_unified.py`.
Everything else in this folder is a dependency, pulled in via plain
`import`, `importlib.import_module(...)` at module load, or lazily inside a
method body — all required at runtime:

```
multimodal_prediction_survival.py       (entry point)
├── emp_bayes_data.py                    (hard, module-level)
├── pca_pack.py                          (hard, module-level)
├── amp_data_pipeline.py                 (hard, module-level)
│   ├── pca_pack.py
│   ├── emp_bayes_data.py
│   └── preprocessing.py                 (hard)
│       └── pca_pack.py
├── preprocessing.py                     (hard, module-level)
└── hierarchical_clustering_modalities.py (hard, but lazily imported —
                                            only inside the clustering method)
```

This is the same dependency graph as the TEA-seq
`Python_scripts/multimodal_prediction_linear.py` — these five files
(`emp_bayes_data.py`, `pca_pack.py`, `amp_data_pipeline.py`,
`preprocessing.py`, `hierarchical_clustering_modalities.py`) are shared,
unmodified, across both datasets.

## File roles

| File | Role |
|---|---|
| `multimodal_prediction_survival.py` | Pipeline entry point. PCA each modality, cluster modalities, denoise via AMP with a per-cluster empirical-Bayes GMM prior, then fit a linear Cox proportional-hazards head (`train_linear_cox_survival`) on Monte-Carlo samples from the GMM posterior. Training-covariate projection uses the Onsager-debiased AMP field; test-time uses the independent OLS projection. |
| `emp_bayes_data.py` | `ClusterParametricEBPipeline` — per-cluster Gaussian-Mixture empirical-Bayes prior fitting for the AMP denoising step. |
| `pca_pack.py` | `MultiModalityPCA` — per-modality PCA and residual-spectrum analysis; defines the `PcaPack` result type shared across the other modules. |
| `amp_data_pipeline.py` | Orchestrates PCA + empirical-Bayes + preprocessing (and clustering, if available) into the AMP pipeline; consumed by `multimodal_prediction_survival.py`. |
| `preprocessing.py` | `MultiModalityPCADiagnostics`, `LowDimModalityLoadings` — per-modality noise normalization and PC diagnostics ahead of AMP. |
| `hierarchical_clustering_modalities.py` | `ModalityClusterer` — clusters modalities by similarity (kernel-based; CKA-style, or Gap-statistic for auto-selecting cluster count) before joint denoising. |

`torch` is required here (unlike the TEA-seq entry point) — the Cox head is
trained via `torch`/`AdamW`.
