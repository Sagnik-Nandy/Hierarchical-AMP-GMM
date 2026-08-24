# Python_scripts

The AMP/EB pipeline modules used by `Notebooks/orchamp_early_fusion.ipynb`,
`orchamp_intermediate_fusion.ipynb`, and `orchamp_late_fusion.ipynb` (the
OrchAMP method itself, as opposed to the baseline notebooks, which only use
third-party packages). Loaded via `sys.path.append("../Python_scripts")` from
`Notebooks/`.

## Entry point and dependency graph

`multimodal_prediction_linear.py` is the public entry point (`MultimodalClusterAllUPipeline`,
`predict_from_test_data_all`). Everything else in this folder is a dependency,
some pulled in via plain `import`, some via `importlib.import_module(...)`
at module load, some lazily inside a method body — all are required at
runtime:

```
multimodal_prediction_linear.py         (entry point)
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

## File roles

| File | Role |
|---|---|
| `multimodal_prediction_linear.py` | Pipeline entry point. Stripped-down, linear-regression-head-only version of the (unpublished here) `multimodal_prediction_unified.py`: PCA each modality, denoise via AMP with a per-cluster empirical-Bayes GMM prior, fit `beta_hat = pinv(U_all) @ y_train`. Training-covariate projection uses the Onsager-debiased AMP field. |
| `emp_bayes_data.py` | `ClusterParametricEBPipeline` — per-cluster Gaussian-Mixture empirical-Bayes prior fitting for the AMP denoising step. |
| `pca_pack.py` | `MultiModalityPCA` — per-modality PCA and residual-spectrum analysis; defines the `PcaPack` result type shared across the other modules. |
| `amp_data_pipeline.py` | Orchestrates PCA + empirical-Bayes + preprocessing (and clustering, if available) into the AMP pipeline; consumed by `multimodal_prediction_linear.py`. |
| `preprocessing.py` | `MultiModalityPCADiagnostics`, `LowDimModalityLoadings` — per-modality noise normalization and PC diagnostics ahead of AMP. |
| `hierarchical_clustering_modalities.py` | `ModalityClusterer` — clusters modalities by similarity (kernel-based; CKA-style) before joint denoising. This is the "hierarchical" step in the project's AMP-GMM method. |

## Not included

`complete_pipeline.py`, `multimodal_prediction_unified.py`, and
`multimodal_unified_pipeline.py` also live in the source `Python_scripts/`
folder but aren't needed here: the first is only pulled in as an optional
`try/except`-guarded fast path (and is currently stale — it references
module names from before a rename, so the import always fails and the code
falls back to the built-in implementation); the other two aren't imported by
anything in this repo.
