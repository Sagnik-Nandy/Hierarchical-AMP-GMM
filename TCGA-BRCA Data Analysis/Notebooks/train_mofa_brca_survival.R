#!/usr/bin/env Rscript

# ============================================================
# MOFA2 + Cox survival benchmark for TCGA-BRCA
# - Uses frozen BRCA survival train/test split
# - Trains MOFA on TRAIN samples only
# - Projects TRAIN/TEST samples through learned MOFA weights
# - Fits a Cox PH head on MOFA factors
# - Saves AMP-compatible survival benchmark outputs
# ============================================================

suppressPackageStartupMessages({
  cmd_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(cmd_file) == 1L && nzchar(cmd_file)) {
    setwd(dirname(normalizePath(cmd_file)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
    }
  }

  library(Matrix)
  library(MOFA2)
  library(survival)
  library(jsonlite)
  library(ggplot2)
  library(reticulate)
})

np <- import("numpy", convert = TRUE)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
matrices_dir <- "../matrices"
data_dir     <- "../data"
splits_dir   <- "../splits"
model_dir    <- "../models/mofa_survival"
results_dir  <- "../results/brca_survival/mofa"

SPLIT_TAG   <- "brca_survival"
NUM_FACTORS <- 30
RIDGE_LAM   <- 1e-3
SEED        <- 42

dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(SEED)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
read_index_csv_0based <- function(path, n_total) {
  df <- read.csv(path, check.names = FALSE)
  is_num <- vapply(df, is.numeric, logical(1))
  if (!any(is_num)) stop("No numeric index column in: ", path)

  idx <- as.integer(df[[which(is_num)[1]]])
  if (anyNA(idx)) stop("NA index found in: ", path)

  if (min(idx) == 0L && max(idx) == (n_total - 1L)) return(idx)
  if (min(idx) == 1L && max(idx) == n_total) return(idx - 1L)

  if (min(idx) < 0L || max(idx) >= n_total) {
    stop("Out-of-range indices in ", path, ": min=", min(idx),
         " max=", max(idx), " n_total=", n_total)
  }
  idx
}

joint_ridge_project_R <- function(X_by_view, W_by_view, lam = 1e-3) {
  views <- names(X_by_view)
  if (length(views) == 0L) stop("X_by_view is empty")

  K <- ncol(W_by_view[[views[1]]])
  n_samp <- ncol(X_by_view[[views[1]]])

  A <- lam * diag(K)
  B <- matrix(0, nrow = n_samp, ncol = K)

  for (v in views) {
    X <- X_by_view[[v]]
    W <- W_by_view[[v]]

    if (nrow(X) != nrow(W)) {
      stop(sprintf("[%s] Feature mismatch: X has %d rows, W has %d rows",
                   v, nrow(X), nrow(W)))
    }
    if (ncol(X) != n_samp) {
      stop(sprintf("[%s] Sample mismatch: X has %d columns, expected %d",
                   v, ncol(X), n_samp))
    }
    if (ncol(W) != K) {
      stop(sprintf("[%s] Factor mismatch: W has %d columns, expected %d",
                   v, ncol(W), K))
    }

    B <- B + t(X) %*% W
    A <- A + t(W) %*% W
  }

  Z <- as.matrix(B %*% solve(A))
  colnames(Z) <- paste0("F", seq_len(ncol(Z)))
  Z
}

cox_c_index <- function(time, event, log_risk) {
  # Match lifelines usage in existing notebooks:
  # concordance_index(time, -log_risk, event).
  as.numeric(survival::concordance(
    survival::Surv(time, event) ~ I(-log_risk)
  )$concordance)
}

save_npy <- function(path, x) {
  np$save(path, np$array(x))
}

# ------------------------------------------------------------
# Load frozen split and aligned data
# ------------------------------------------------------------
cat("[MOFA-SURV] Loading frozen split...\n")
sid_path <- file.path(splits_dir, paste0(SPLIT_TAG, "_sample_ids.csv"))
trn_path <- file.path(splits_dir, paste0(SPLIT_TAG, "_train_idx.csv"))
tst_path <- file.path(splits_dir, paste0(SPLIT_TAG, "_test_idx.csv"))

for (p in c(sid_path, trn_path, tst_path)) {
  if (!file.exists(p)) stop("Missing split file: ", p)
}

sample_ids <- read.csv(sid_path, check.names = FALSE)$sample_id
n_total <- length(sample_ids)
idx_tr0 <- read_index_csv_0based(trn_path, n_total)
idx_te0 <- read_index_csv_0based(tst_path, n_total)
idx_trR <- idx_tr0 + 1L
idx_teR <- idx_te0 + 1L

cat(sprintf("[MOFA-SURV] Samples: %d | train %d | test %d\n",
            n_total, length(idx_trR), length(idx_teR)))

cat("[MOFA-SURV] Loading matrices...\n")
rna_df  <- read.csv(file.path(matrices_dir, "RNA_X_full.csv"),
                    row.names = 1, check.names = FALSE)
meth_df <- read.csv(file.path(matrices_dir, "Methylation_X_full.csv"),
                    row.names = 1, check.names = FALSE)
cnv_df  <- read.csv(file.path(matrices_dir, "CNV_X_ld_features.csv"),
                    row.names = 1, check.names = FALSE)

surv_raw <- read.delim(file.path(data_dir, "BRCA_survival.tsv"),
                       row.names = 1, check.names = FALSE)
surv_df <- surv_raw[, c("OS", "OS.time")]
colnames(surv_df) <- c("event", "time")
surv_df$event <- as.numeric(surv_df$event)
surv_df$time  <- as.numeric(surv_df$time)
surv_df <- surv_df[stats::complete.cases(surv_df) & surv_df$time > 0, , drop = FALSE]

missing_mats <- setdiff(sample_ids, rownames(rna_df))
if (length(missing_mats)) stop("RNA matrix missing sample: ", missing_mats[1])
missing_mats <- setdiff(sample_ids, rownames(meth_df))
if (length(missing_mats)) stop("Methylation matrix missing sample: ", missing_mats[1])
missing_mats <- setdiff(sample_ids, rownames(cnv_df))
if (length(missing_mats)) stop("CNV matrix missing sample: ", missing_mats[1])
missing_surv <- setdiff(sample_ids, rownames(surv_df))
if (length(missing_surv)) stop("Survival table missing sample: ", missing_surv[1])

X_rna  <- as.matrix(rna_df[sample_ids, , drop = FALSE])
X_meth <- as.matrix(meth_df[sample_ids, , drop = FALSE])
X_cnv  <- as.matrix(cnv_df[sample_ids, , drop = FALSE])
event  <- as.numeric(surv_df[sample_ids, "event"])
time   <- as.numeric(surv_df[sample_ids, "time"])

if (anyNA(X_rna) || anyNA(X_meth) || anyNA(X_cnv)) stop("NA found in input matrices")
if (anyNA(event) || anyNA(time) || any(time <= 0)) stop("Invalid survival response")

train_ids <- sample_ids[idx_trR]
test_ids  <- sample_ids[idx_teR]

event_tr <- event[idx_trR]
time_tr  <- time[idx_trR]
event_te <- event[idx_teR]
time_te  <- time[idx_teR]

cat(sprintf("[MOFA-SURV] Train events: %d / %d | Test events: %d / %d\n",
            sum(event_tr), length(event_tr), sum(event_te), length(event_te)))

# MOFA2 expects features x samples for each view in this workflow.
views_train <- list(
  RNA = Matrix::Matrix(t(X_rna[idx_trR, , drop = FALSE]), sparse = TRUE),
  Methylation = Matrix::Matrix(t(X_meth[idx_trR, , drop = FALSE]), sparse = TRUE),
  CNV = Matrix::Matrix(t(X_cnv[idx_trR, , drop = FALSE]), sparse = TRUE)
)

views_test <- list(
  RNA = Matrix::Matrix(t(X_rna[idx_teR, , drop = FALSE]), sparse = TRUE),
  Methylation = Matrix::Matrix(t(X_meth[idx_teR, , drop = FALSE]), sparse = TRUE),
  CNV = Matrix::Matrix(t(X_cnv[idx_teR, , drop = FALSE]), sparse = TRUE)
)

for (v in names(views_train)) {
  colnames(views_train[[v]]) <- train_ids
  colnames(views_test[[v]]) <- test_ids
}
rownames(views_train$RNA) <- rownames(views_test$RNA) <- colnames(X_rna)
rownames(views_train$Methylation) <- rownames(views_test$Methylation) <- colnames(X_meth)
rownames(views_train$CNV) <- rownames(views_test$CNV) <- colnames(X_cnv)

cat("[MOFA-SURV] Training view dimensions:\n")
for (v in names(views_train)) {
  cat(sprintf("  - %s: %d features x %d samples\n",
              v, nrow(views_train[[v]]), ncol(views_train[[v]])))
}

# ------------------------------------------------------------
# Train MOFA2
# ------------------------------------------------------------
cat("[MOFA-SURV] Creating MOFA object...\n")
mofa_obj <- create_mofa(views_train)

data_opts <- get_default_data_options(mofa_obj)
data_opts$scale_views <- FALSE

model_opts <- get_default_model_options(mofa_obj)
model_opts$num_factors <- NUM_FACTORS

train_opts <- get_default_training_options(mofa_obj)
train_opts$seed <- SEED

mofa_hdf5_path <- file.path(model_dir, paste0(SPLIT_TAG, "_mofa_survival_model.hdf5"))
mofa_rds_path  <- file.path(model_dir, paste0(SPLIT_TAG, "_mofa_survival_model.rds"))
cox_rds_path   <- file.path(model_dir, paste0(SPLIT_TAG, "_mofa_coxph.rds"))
cox_coef_path  <- file.path(model_dir, paste0(SPLIT_TAG, "_mofa_cox_coefficients.csv"))

cat("[MOFA-SURV] Training MOFA2 model...\n")
cat("  HDF5: ", mofa_hdf5_path, "\n", sep = "")

mofa_trained <- run_mofa(
  prepare_mofa(
    mofa_obj,
    data_options = data_opts,
    model_options = model_opts,
    training_options = train_opts
  ),
  use_basilisk = TRUE,
  outfile = mofa_hdf5_path
)
saveRDS(mofa_trained, file = mofa_rds_path)

cat("[MOFA-SURV] MOFA training done.\n")

# ------------------------------------------------------------
# Project factors and fit Cox head
# ------------------------------------------------------------
cat("[MOFA-SURV] Projecting train/test samples into MOFA factors...\n")
W_by_view <- get_weights(mofa_trained, factors = "all")

Z_train <- joint_ridge_project_R(views_train, W_by_view, lam = RIDGE_LAM)
Z_test  <- joint_ridge_project_R(views_test,  W_by_view, lam = RIDGE_LAM)

cox_df <- data.frame(time = time_tr, event = event_tr, Z_train, check.names = FALSE)
cox_formula <- stats::as.formula(
  paste("survival::Surv(time, event) ~", paste(colnames(Z_train), collapse = " + "))
)

cat("[MOFA-SURV] Fitting Cox PH head on MOFA factors...\n")
cox_fit <- survival::coxph(cox_formula, data = cox_df, x = TRUE)
saveRDS(cox_fit, file = cox_rds_path)
cox_coef <- stats::coef(cox_fit)
write.csv(
  data.frame(factor = names(cox_coef), coef = as.numeric(cox_coef), check.names = FALSE),
  file = cox_coef_path,
  row.names = FALSE
)

log_risk_train <- as.numeric(stats::predict(cox_fit, newdata = as.data.frame(Z_train), type = "lp"))
log_risk_test  <- as.numeric(stats::predict(cox_fit, newdata = as.data.frame(Z_test),  type = "lp"))

c_train <- cox_c_index(time_tr, event_tr, log_risk_train)
c_test  <- cox_c_index(time_te, event_te, log_risk_test)

cat(sprintf("[MOFA-SURV] Train C-index: %.4f | Test C-index: %.4f\n", c_train, c_test))

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
cat("[MOFA-SURV] Saving outputs...\n")
save_npy(file.path(results_dir, "Z_train.npy"), Z_train)
save_npy(file.path(results_dir, "Z_test.npy"), Z_test)
save_npy(file.path(results_dir, "log_risk_train.npy"), log_risk_train)
save_npy(file.path(results_dir, "log_risk_test.npy"), log_risk_test)
save_npy(file.path(results_dir, "event_test.npy"), event_te)
save_npy(file.path(results_dir, "time_test.npy"), time_te)
save_npy(file.path(results_dir, "idx_test.npy"), as.integer(idx_te0))

pred_df <- data.frame(
  sample_id = test_ids,
  time = time_te,
  event = event_te,
  log_risk = log_risk_test,
  risk_group = ifelse(log_risk_test >= stats::median(log_risk_test), "High risk", "Low risk"),
  stringsAsFactors = FALSE
)
write.csv(pred_df, file.path(results_dir, "test_predictions.csv"), row.names = FALSE)

metrics <- list(
  c_index_train = unname(c_train),
  c_index_test = unname(c_test),
  n_train = length(idx_trR),
  n_test = length(idx_teR),
  n_events_train = unname(sum(event_tr)),
  n_events_test = unname(sum(event_te)),
  run = list(
    method = "MOFA2 + CoxPH",
    task = "survival",
    split_tag = SPLIT_TAG,
    num_factors = NUM_FACTORS,
    ridge_projection_lambda = RIDGE_LAM,
    seed = SEED,
    model_hdf5 = mofa_hdf5_path,
    model_rds = mofa_rds_path,
    cox_rds = cox_rds_path,
    cox_coefficients = cox_coef_path,
    views = names(views_train)
  ),
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

writeLines(
  jsonlite::toJSON(metrics, pretty = TRUE, auto_unbox = TRUE),
  con = file.path(results_dir, "metrics.json")
)

# Kaplan-Meier median-risk plot.
pdf(file.path(results_dir, "km_plot.pdf"), width = 6, height = 4)
km_fit <- survival::survfit(
  survival::Surv(time, event) ~ risk_group,
  data = pred_df
)
plot(
  km_fit,
  col = c("firebrick", "steelblue"),
  lwd = 2,
  xlab = "Days",
  ylab = "Survival probability",
  main = sprintf("MOFA2 + CoxPH - TCGA-BRCA (C-index=%.3f)", c_test)
)
legend("bottomleft", legend = levels(factor(pred_df$risk_group)),
       col = c("firebrick", "steelblue"), lwd = 2, bty = "n")
dev.off()

cat("[MOFA-SURV] Saved outputs under:\n  ", normalizePath(results_dir), "\n", sep = "")
cat("[MOFA-SURV] Done.\n")
