#!/usr/bin/env Rscript

# ------------------------------------------------------------
# 1. Setup and Libraries
# ------------------------------------------------------------
suppressPackageStartupMessages({
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  library(Matrix)
  library(reticulate)
  library(jafar)   
  library(uwot)
  library(ggplot2)
  library(dplyr)
})

np      <- import("numpy", convert = TRUE)
anndata <- import("anndata", convert = FALSE)

# --- Helper: Python -> R matrix conversion ---
py_matrix_to_R <- function(py_mat) {
  if (inherits(py_mat, "scipy.sparse.spmatrix")) {
    as.matrix(py_to_r(py_mat$toarray()))
  } else {
    as.matrix(py_to_r(py_mat))
  }
}

# --- Load split indices ---
load_split_indices <- function(splits_dir, split_tag, which = c("train", "test")) {
  which <- match.arg(which)
  path  <- file.path(splits_dir, paste0(split_tag, "_", which, "_idx.csv"))
  df    <- tryCatch(read.csv(path, header = TRUE), error = function(e) read.csv(path, header = FALSE))
  as.integer(unlist(df)) + 1L 
}

# --- PCA & Scaling helpers ---
fit_pca_view    <- function(X, n_comp) prcomp(X, center = TRUE, scale. = FALSE, rank. = n_comp)
compute_scaling <- function(X) {
  mu <- colMeans(X); sds <- apply(X, 2, sd); sds[sds == 0] <- 1
  list(mean = mu, sd = sds)
}
apply_scaling   <- function(X, scaler) sweep(sweep(X, 2, scaler$mean, FUN = "-"), 2, scaler$sd, FUN = "/")

# ------------------------------------------------------------
# 2. Main Training Function
# ------------------------------------------------------------
train_jafar_regression_tea <- function(
    target_protein = "CD45RA",
    split_tag      = "tea_split3_all_celltypes",
    base_dir       = "../data",
    splits_dir     = "../splits",
    model_dir      = "../models",
    pca_dims       = list(rna = 100L, atac = 100L, adt = NULL),
    K0 = 30L, K0_m = c(10L, 10L, 5L),
    tMCMC = 2000L, tBurnIn = 1000L, tThin = 5L, seed = 1L
) {
  set.seed(seed)
  model_root <- file.path(model_dir, "tea", paste0(split_tag, "_jafar_", target_protein))
  dir.create(model_root, recursive = TRUE, showWarnings = FALSE)
  
  # 1. Load Data
  cat(sprintf("[JAFAR] Loading data for %s...\n", target_protein))
  rna_py  <- anndata$read_h5ad(file.path(base_dir, "rna.h5ad"))
  atac_py <- anndata$read_h5ad(file.path(base_dir, "atac.h5ad"))
  adt_py  <- anndata$read_h5ad(file.path(base_dir, paste0("adt_minus_", target_protein, ".h5ad")))
  y_df    <- read.csv(file.path(base_dir, "response", paste0(target_protein, ".csv")), row.names = 1)
  
  # 2. Extract Indices
  idx_tr <- load_split_indices(splits_dir, split_tag, which = "train")
  
  # 3. Subsetting & Conversion
  X_rna_tr  <- py_matrix_to_R(rna_py$layers[["norm"]])[idx_tr, ]
  X_atac_tr <- py_matrix_to_R(atac_py$layers[["norm"]])[idx_tr, ]
  X_adt_tr  <- py_matrix_to_R(adt_py$layers[["norm"]])[idx_tr, ]
  y_train   <- as.numeric(y_df[idx_tr, 1])
  
  # 4. PCA Preprocessing
  cat("[JAFAR] Preprocessing modalities...\n")
  pca_rna  <- fit_pca_view(X_rna_tr, pca_dims$rna)
  pca_atac <- fit_pca_view(X_atac_tr, pca_dims$atac)
  
  # 5. Scaling
  sc_rna  <- compute_scaling(pca_rna$x);  sc_atac <- compute_scaling(pca_atac$x); sc_adt  <- compute_scaling(X_adt_tr)
  X_m_train <- list(
    rna  = apply_scaling(pca_rna$x, sc_rna),
    atac = apply_scaling(pca_atac$x, sc_atac),
    adt  = apply_scaling(X_adt_tr, sc_adt)
  )
  
  # 6. Fit JAFAR (Note: package uses gibbs_jfr based on source code provided)
  cat(sprintf("\n[JAFAR] Fitting regression for %s (yBinary=FALSE)...\n", target_protein))
  fit_c <- gibbs_jfr(
    X_m             = X_m_train,
    y               = y_train,
    yBinary         = FALSE, 
    K0              = K0,
    tMCMC           = tMCMC,
    tBurnIn         = tBurnIn,
    tThin           = tThin,
    parallel        = TRUE,
    get_latent_vars = TRUE
  )
  
  # ------------------------------------------------------------
  # 7. Manual Reconstruction of Predictions for R-squared
  # ------------------------------------------------------------
  cat("[JAFAR] Calculating R-squared (Manual Reconstruction)...\n")
  
  # Extract components from fit_c
  mu_y_samples  <- fit_c$mu_y   # [tEff]
  Theta_samples <- fit_c$Theta  # [tEff x K]
  eta_samples   <- fit_c$eta    # [tEff x n x K]
  
  tEff    <- length(mu_y_samples)
  n_cells <- length(y_train)
  
  # Accumulate predictions across all MCMC samples
  y_hat_accum <- numeric(n_cells)
  
  for (i in seq_len(tEff)) {
    # Prediction = Intercept + Factors %*% Loadings
    # eta_samples[i,,] is [n x K], Theta_samples[i,] is [K]
    pred_iter <- mu_y_samples[i] + (eta_samples[i, , ] %*% Theta_samples[i, ])
    y_hat_accum <- y_hat_accum + as.vector(pred_iter)
  }
  
  # Average to get the posterior mean prediction
  y_pred_mean <- y_hat_accum / tEff
  
  # Final R-squared calculation
  train_r2 <- cor(y_train, y_pred_mean, use = "complete.obs")^2
  cat(sprintf("\n>>> Final %s Training R-squared: %.4f\n", target_protein, train_r2))
  
  # 8. Save
  saveRDS(fit_c, file.path(model_root, "jfr_reg_fit.rds"))
  saveRDS(list(pca_rna=pca_rna, pca_atac=pca_atac, sc_rna=sc_rna, sc_atac=sc_atac, sc_adt=sc_adt), 
          file.path(model_root, "jfr_prep.rds"))
  
  cat("[JAFAR] Pipeline Complete.\n")
}

# ------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------
train_jafar_regression_tea(target_protein = "CD45RA")