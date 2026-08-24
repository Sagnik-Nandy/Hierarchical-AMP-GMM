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
  library(MOFA2)
  library(uwot)
  library(ggplot2)
  library(dplyr)
})

# Python bridges
np        <- import("numpy")
anndata   <- import("anndata")
scipy     <- import("scipy", convert = FALSE)
sk_linear <- import("sklearn.linear_model")
joblib    <- import("joblib")

# ------------------------------------------------------------
# 2. Configuration & Paths (Dynamic)
# ------------------------------------------------------------
target_protein <- "CD45RA" # Simply change this to switch proteins
split_tag      <- "tea_split3_all_celltypes" 
base_dir       <- "../data"
splits_dir     <- "../splits"

# Dynamic Directory Creation
model_dir <- file.path("../models/tea", paste0(target_protein, "_regression"))
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 3. Helper Functions (Restored & Fixed)
# ------------------------------------------------------------
py_to_dgC <- function(mat_py) {
  if (is.matrix(mat_py) || is.numeric(mat_py)) return(Matrix::Matrix(mat_py, sparse = TRUE))
  if (inherits(mat_py, "dgCMatrix")) return(mat_py)
  if (reticulate::py_has_attr(mat_py, "format")) {
    fmt <- mat_py$format
    if (!(fmt %in% c("csr", "csc"))) mat_py <- mat_py$tocsr()
  } else { mat_py <- scipy$sparse$csr_matrix(mat_py) }
  
  Matrix::sparseMatrix(
    i = as.integer(mat_py$indices) + 1L,
    p = as.integer(mat_py$indptr),
    x = as.numeric(mat_py$data),
    dims = c(as.integer(mat_py$shape[[1]]), as.integer(mat_py$shape[[2]])),
    giveCsparse = TRUE
  )
}

joint_ridge_project_R <- function(X_by_view, W_by_view, lam = 1e-3) {
  views <- names(X_by_view); K <- ncol(W_by_view[[views[1]]]); n_samp <- ncol(X_by_view[[views[1]]])
  A <- lam * diag(K); B <- matrix(0, nrow = n_samp, ncol = K)
  for (v in views) {
    X <- X_by_view[[v]]; W <- W_by_view[[v]]
    B <- B + t(X) %*% W
    A <- A + t(W) %*% W
  }
  return(as.matrix(B %*% solve(A)))
}

# ------------------------------------------------------------
# 4. Load Data (Dynamic paths)
# ------------------------------------------------------------
cat(sprintf("[Pipeline] Loading RNA, ATAC, and ADT (minus %s)...\n", target_protein))

rna_py  <- anndata$read_h5ad(file.path(base_dir, "rna.h5ad"))
atac_py <- anndata$read_h5ad(file.path(base_dir, "atac.h5ad"))

# Dynamic file naming based on target_protein
adt_file <- file.path(base_dir, paste0("adt_minus_", target_protein, ".h5ad"))
adt_py   <- anndata$read_h5ad(adt_file)

# Dynamic response loading
response_file <- file.path(base_dir, "response", paste0(target_protein, ".csv"))
y_df <- read.csv(response_file, row.names = 1)

# Matrices (Cells x Features to Features x Cells)
rna_norm_R  <- t(py_to_dgC(rna_py$layers["norm"]))
atac_norm_R <- t(py_to_dgC(atac_py$layers["norm"]))
adt_norm_R  <- t(py_to_dgC(adt_py$layers["norm"]))

rownames(rna_norm_R)  <- as.character(np$array(rna_py$var_names))
colnames(rna_norm_R)  <- as.character(np$array(rna_py$obs_names))
rownames(atac_norm_R) <- as.character(np$array(atac_py$var_names))
colnames(atac_norm_R) <- as.character(np$array(atac_py$obs_names))
rownames(adt_norm_R)  <- as.character(np$array(adt_py$var_names))
colnames(adt_norm_R)  <- as.character(np$array(adt_py$obs_names))

# Subset to Train
train_idx_raw <- read.csv(file.path(splits_dir, paste0(split_tag, "_train_idx.csv")))[[1]]
train_idx     <- train_idx_raw + 1L 
train_cells   <- colnames(rna_norm_R)[train_idx]

# ------------------------------------------------------------
# 5. MOFA+ Feature Extraction
# ------------------------------------------------------------
views_train <- list(RNA = rna_norm_R[, train_cells], 
                    ATAC = atac_norm_R[, train_cells], 
                    ADT = adt_norm_R[, train_cells])

cat("\n[Check] Modality Matrix Dimensions:\n")
for (v in names(views_train)) cat(sprintf("  - %s: %d x %d\n", v, nrow(views_train[[v]]), ncol(views_train[[v]])))

mofa_obj <- create_mofa(views_train)
data_opts <- get_default_data_options(mofa_obj)
data_opts$scale_views <- FALSE
model_opts <- get_default_model_options(mofa_obj)
model_opts$num_factors <- 30 # Increased for better capture

mofa_hdf5_path <- file.path(model_dir, paste0(split_tag, "_mofa_model.hdf5"))
cat("[MOFA+] Training MOFA model (using basilisk env)...\n")
cat("         Will save HDF5 model to: ", mofa_hdf5_path, "\n")

mofa_trained <- run_mofa(prepare_mofa(mofa_obj, data_options = data_opts, model_options = model_opts), use_basilisk = TRUE, outfile = mofa_hdf5_path)

mofa_rds_path <- file.path(model_dir, paste0(split_tag, "_mofa_model.rds"))
saveRDS(mofa_trained, file = mofa_rds_path)


cat("[MOFA+] MOFA training DONE.\n")
cat("  - Model (HDF5): ", mofa_hdf5_path, "\n")
cat("  - Model (RDS):  ", mofa_rds_path, "\n")

# Factor Reconstruction
W_by_view <- get_weights(mofa_trained, factors = "all")
Z_train   <- joint_ridge_project_R(X_by_view = views_train, W_by_view = W_by_view, lam = 1e-3)

# ------------------------------------------------------------
# 6. Linear Regression (Dynamic Target)
# ------------------------------------------------------------
cat(sprintf("[Regression] Predicting %s levels...\n", target_protein))
y_train <- y_df[train_cells, 1]

reg_model <- sk_linear$LinearRegression()
reg_model$fit(Z_train, y_train)

r2_score <- reg_model$score(Z_train, y_train)
cat(sprintf("\n>>> Result: %s Training R-squared: %.4f\n", target_protein, r2_score))

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------
reg_name <- paste0(split_tag, "_", target_protein, "_linear_reg.pkl")
joblib$dump(reg_model, file.path(model_dir, reg_name))
cat("[Done] Pipeline finished.\n")