#!/usr/bin/env Rscript
# ============================================================
# JAFAR Regression TEST Pipeline (TEA-seq)
# - Uses saved training preprocessing (PCA + scalers)
# - Uses saved JAFAR/JFR MCMC fit
# - Predicts on TEST split using predict_y_raw (per MCMC draw)
# - Averages over draws -> posterior mean prediction
# - Saves .npy outputs + CSV + test_metrics.json
# ============================================================

suppressPackageStartupMessages({
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
    }
  }
})

suppressPackageStartupMessages({
  library(reticulate)
  library(Matrix)
  library(jafar)
  library(jsonlite)
})

np      <- import("numpy", convert = TRUE)
anndata <- import("anndata", convert = FALSE)


# response coefficients ----

jfr_coeff_y_t <- function(Xpred,nPred,M,p_m,K,Lambda_m,Theta,mu_y,
                          s2_inv_y,mu_m,s2_inv_m,rescale_pred=F){
  
  lin_pred <- rep(mu_y,nPred)
  
  Q_eta <- diag(1,K,K)
  r_eta_m <- coeff_m <- list()
  
  for(m in 1:M){
    
    mar_std_m = rep(1,p_m[m])
    if(rescale_pred){
      mar_std_m = sqrt(1/s2_inv_m[[m]] + rowSums(as.matrix(Lambda_m[[m]]^2)))
    }
    s2_m = s2_inv_m[[m]]*(mar_std_m^2)
    La_m = Lambda_m[[m]]/mar_std_m
    
    s2_La_m <- s2_m*La_m
    
    Q_eta <- Q_eta + crossprod(La_m,s2_La_m) 
    r_eta_m[[m]] <- t(s2_La_m)
  }
  
  L_eta    <- t(chol(Q_eta))
  
  for(m in 1:M){
    coeff_m[[m]] <- as.vector(Theta%*%backsolve(t(L_eta),forwardsolve(L_eta, r_eta_m[[m]])))
    
    if(!is.null(Xpred[[m]])){
      res_m <- Xpred[[m]]-rep(1,nPred)%x%t(mu_m[[m]])
      res_m[is.na(res_m)] <- 0
      
      lin_pred <- lin_pred + as.vector(res_m%*%coeff_m[[m]])
    }
  }
  
  return(list(lin_pred=lin_pred,coeff=coeff_m))
}

jafar_coeff_y_t <- function(Xpred,nPred,M,p_m,K,K_m,Lambda_m,Gamma_m,Theta,
                            Theta_m,mu_y,s2_inv_y,mu_m,s2_inv_m,rescale_pred=F){
  
  lin_pred <- rep(mu_y,nPred)
  coeff_m <- lapply(1:M, function(m) rep(0,p_m[m]))
  
  Q_eta <- diag(1,K,K)
  r_eta_m <- list()
  
  for(m in 1:M){
    
    mar_std_m = rep(1,p_m[m])
    if(rescale_pred){
      mar_std_m = sqrt(1/s2_inv_m[[m]] + rowSums(as.matrix(Lambda_m[[m]]^2)) + rowSums(as.matrix(Gamma_m[[m]]^2)))
    }
    Ga_m = Gamma_m[[m]]/mar_std_m
    s2_m = s2_inv_m[[m]]*(mar_std_m^2)
    La_m = Lambda_m[[m]]/mar_std_m
    
    s2_La_m <- s2_m*La_m
    s2_Ga_m <- s2_m*Ga_m
    GaT_s2_La_m = crossprod(s2_Ga_m,La_m)
    
    D_m_inv = chol(diag(1.,K_m[m],K_m[m])+crossprod(Ga_m,s2_Ga_m))
    D_GaT_s2_La_m = backsolve(D_m_inv,forwardsolve(t(D_m_inv),GaT_s2_La_m))
    D_Theta_m = backsolve(D_m_inv,forwardsolve(t(D_m_inv),Theta_m[[m]]))
    
    coeff_m[[m]] <- coeff_m[[m]] + as.vector(s2_Ga_m%*%D_Theta_m)
    
    Theta <- Theta - as.vector(Theta_m[[m]]%*%D_GaT_s2_La_m)
    
    Q_eta <- Q_eta + crossprod(La_m,s2_La_m) - crossprod(GaT_s2_La_m,D_GaT_s2_La_m)
    
    r_eta_m[[m]] <- t(s2_La_m - s2_Ga_m%*%D_GaT_s2_La_m)
  }
  
  L_eta    <- t(chol(Q_eta))
  LtL_Theta <- backsolve(t(L_eta),forwardsolve(L_eta,Theta))
  
  for(m in 1:M){
    coeff_m[[m]] <- coeff_m[[m]] + as.vector(LtL_Theta%*%r_eta_m[[m]])
    
    if(!is.null(Xpred[[m]])){
      res_m <- Xpred[[m]]-rep(1,nPred)%x%t(mu_m[[m]])
      res_m[is.na(res_m)] <- 0
      
      lin_pred <- lin_pred + as.vector(res_m%*%coeff_m[[m]])
    }
  }
  
  return(list(lin_pred=lin_pred,coeff=coeff_m))
}

#' Response predictions for \code{jafar} and \code{jfr}
#' 
#' @description Compute induced regression coefficients and predicted responses either in-sample or out-of-sample.
#'
#' @param Xpred A list of \code{M} features matrices, the m-th one of dimension \code{nPred x p_m[m]} or possibly missing (i.e. \code{X_m[[m]]=NULL}).
#' @param risMCMC Output of \code{\link{gibbs_jafar}} or \code{\link{gibbs_jfr}} containing posterior samples.
#' @param rescale_pred Rescale loadings when computing response predictions (logical, default: \code{FALSE}).
#'
#' @return A list containing posterior samples of the predicted responses (matrix \code{tEff x nPred}),
#'    and of the induced regression coefficients for each view (list of length \code{M}; m-th element: \code{tEff x p_m[m]}).
#'
#' @export
#'
predict_y <- function(Xpred,risMCMC,rescale_pred=FALSE){
  
  M = length(risMCMC$mu_m)
  p_m = sapply(risMCMC$mu_m,ncol)
  
  nPred = unlist(sapply(Xpred,nrow))[1]
  
  tMCMC = nrow(risMCMC$K_Lm_eff)
  iter_print <- max(1,tMCMC %/% 10)  
  
  mu_MC  <- matrix(NA,tMCMC,nPred)
  coeff_MC <- lapply(1:M, function(m) matrix(NA,tMCMC,p_m[m]))
  
  t_vals = 1:(risMCMC$hyper_param$tBurnIn%/%risMCMC$hyper_param$tThin)
  
  risMCMC$K = risMCMC$K[-t_vals]
  risMCMC$K_Gm = risMCMC$K_Gm[-t_vals,]
  
  print(" - Computing Response Predictions - ")
  print(sprintf(fmt = "%10s%3s%2s", "[",0,"%]"))
  
  for(t in 1:tMCMC){
    
    La_m_t <- lapply(risMCMC$Lambda_m, function(df) df[t,,1:risMCMC$K[t]])
    Theta_t   <- risMCMC$Theta[t,1:risMCMC$K[t]]
    
    mu_m_t <- lapply(risMCMC$mu_m, function(df) df[t,])
    s2_m_t <- lapply(risMCMC$s2_inv_m, function(df) df[t,])
    
    if(grepl('jafar',risMCMC$hyper_param$model)){
      Ga_m_t <- lapply(1:M, function(m) risMCMC$Gamma_m[[m]][t,,1:risMCMC$K_Gm[t,m]])
      Theta_m_t <- lapply(1:M, function(m) risMCMC$Theta_m[[m]][t,1:risMCMC$K_Gm[t,m]])
      
      pred_t <- jafar_coeff_y_t(Xpred,nPred,M,p_m,risMCMC$K[t],risMCMC$K_Gm[t,],
                                La_m_t,Ga_m_t,Theta_t,Theta_m_t,
                                risMCMC$mu_y[t],risMCMC$s2_inv[t],mu_m_t,s2_m_t,
                                rescale_pred=rescale_pred)   
    } else {
      pred_t <- jfr_coeff_y_t(Xpred,nPred,M,p_m,risMCMC$K[t],
                              La_m_t,Theta_t,
                              risMCMC$mu_y[t],risMCMC$s2_inv[t],mu_m_t,s2_m_t,
                              rescale_pred=rescale_pred)   
    }
    
    mu_MC[t,]  <- pred_t$lin_pred
    for(m in 1:M){
      coeff_MC[[m]][t,]  <- pred_t$coeff[[m]]
    }
    
    if(t %% iter_print == 0){
      print(sprintf(fmt = "%10s%3s%2s", "[",(t%/%iter_print)*10,"%]"))
    }
  }
  
  return(list(mean=mu_MC,coeff=coeff_MC))
  
}


# response predictions ----

jfr_pred_y_t <- function(Xpred,nPred,M,p_m,K,Lambda_m,Theta,mu_y,
                         s2_inv_y,mu_m,s2_inv_m,rescale_pred=F){
  
  lin_pred <- rep(mu_y,nPred)
  
  Q_eta <- diag(1,K,K)
  r_eta <- matrix(0,K,nPred)
  
  for(m in 1:M){
    
    mar_std_m = rep(1,p_m[m])
    if(rescale_pred){
      mar_std_m = sqrt(1/s2_inv_m[[m]] + rowSums(as.matrix(Lambda_m[[m]]^2)))
    }
    s2_m = s2_inv_m[[m]]*(mar_std_m^2)
    La_m = Lambda_m[[m]]/mar_std_m
    
    s2_La_m <- s2_m*La_m
    
    Q_eta <- Q_eta + crossprod(La_m,s2_La_m) 
    
    if(!is.null(Xpred[[m]])){
      
      res_m <- Xpred[[m]]-rep(1,nPred)%x%t(mu_m[[m]])
      res_m[is.na(res_m)] <- 0
      
      r_eta <- r_eta + t(res_m%*%s2_La_m)
    }
  }
  
  L_eta    <- t(chol(Q_eta))
  Lr_eta   <- backsolve(t(L_eta),forwardsolve(L_eta, r_eta))
  
  lin_pred <- lin_pred + as.vector(crossprod(Lr_eta,Theta))
  
  return(lin_pred)
}

jafar_pred_y_t <- function(Xpred,nPred,M,p_m,K,K_m,Lambda_m,Gamma_m,Theta,
                           Theta_m,mu_y,s2_inv_y,mu_m,s2_inv_m,rescale_pred=F){
  
  lin_pred <- rep(mu_y,nPred)
  
  Q_eta <- diag(1,K,K)
  r_eta <- matrix(0,K,nPred)
  
  for(m in 1:M){
    
    mar_std_m = rep(1,p_m[m])
    if(rescale_pred){
      mar_std_m = sqrt(1/s2_inv_m[[m]] + rowSums(as.matrix(Lambda_m[[m]]^2)) + rowSums(as.matrix(Gamma_m[[m]]^2)))
    }
    Ga_m = Gamma_m[[m]]/mar_std_m
    s2_m = s2_inv_m[[m]]*(mar_std_m^2)
    La_m = Lambda_m[[m]]/mar_std_m
    
    s2_La_m <- s2_m*La_m
    s2_Ga_m <- s2_m*Ga_m
    GaT_s2_La_m = crossprod(s2_Ga_m,La_m)
    
    D_m_inv = chol(diag(1.,K_m[m],K_m[m])+crossprod(Ga_m,s2_Ga_m))
    D_GaT_s2_La_m = backsolve(D_m_inv,forwardsolve(t(D_m_inv),GaT_s2_La_m))
    D_Theta_m = backsolve(D_m_inv,forwardsolve(t(D_m_inv),Theta_m[[m]]))
    
    Theta <- Theta - as.vector(Theta_m[[m]]%*%D_GaT_s2_La_m)
    
    Q_eta <- Q_eta + crossprod(La_m,s2_La_m) - crossprod(GaT_s2_La_m,D_GaT_s2_La_m)
    
    if(!is.null(Xpred[[m]])){
      
      res_m <- Xpred[[m]]-rep(1,nPred)%x%t(mu_m[[m]])
      res_m[is.na(res_m)] <- 0
      
      lin_pred <- lin_pred + as.vector(res_m%*%(s2_Ga_m%*%D_Theta_m))
      
      r_eta <- r_eta + t( res_m%*%(s2_La_m - s2_Ga_m%*%D_GaT_s2_La_m) )
    }
  }
  
  L_eta    <- t(chol(Q_eta))
  Lr_eta   <- backsolve(t(L_eta),forwardsolve(L_eta, r_eta))
  
  lin_pred <- lin_pred + as.vector(crossprod(Lr_eta,Theta))
  
  return(lin_pred)
}

predict_y_raw <- function(Xpred, risMCMC, rescale_pred = FALSE){
  
  # -------------------------
  # small helpers
  # -------------------------
  .nrow_safe <- function(x) {
    d <- dim(x)
    if (is.null(d)) return(NA_integer_)
    d[1]
  }
  
  .cap_int <- function(x, lo, hi) {
    x <- as.integer(x)
    x[x < lo] <- lo
    x[x > hi] <- hi
    x
  }
  
  # Slice Lambda_m[[m]]: [t, p_m, Kmax] -> matrix(p_m, Kt)
  .slice_lambda <- function(df, t, Kt) {
    d <- dim(df)
    if (length(d) != 3L) stop("Lambda_m element must be 3D (t × p × K). Got dim=", paste(d, collapse="x"))
    Kt <- min(Kt, d[3])
    # keep as matrix p × Kt
    out <- df[t, , seq_len(Kt), drop = FALSE]
    matrix(out, nrow = d[2], ncol = Kt)
  }
  
  # Slice Gamma_m[[m]]:
  # - if 2D: (t × p) -> interpret as 1 local factor; return matrix(p, 1)
  # - if 3D: (t × p × K) -> return matrix(p, Kgmt)
  .slice_gamma <- function(gm, t, Kgmt) {
    d <- dim(gm)
    if (is.null(d)) stop("Gamma_m element has NULL dim.")
    if (length(d) == 2L) {
      # only 1 factor available
      Kgmt <- 1L
      v <- gm[t, , drop = TRUE]
      return(matrix(v, nrow = d[2], ncol = 1L))
    }
    if (length(d) == 3L) {
      Kgmt <- min(Kgmt, d[3])
      out <- gm[t, , seq_len(Kgmt), drop = FALSE]
      return(matrix(out, nrow = d[2], ncol = Kgmt))
    }
    stop("Gamma_m element must be 2D or 3D. Got dim=", paste(d, collapse="x"))
  }
  
  # Slice Theta_m[[m]]:
  # - if vector len=t : interpret as 1 local factor; return numeric(1)
  # - if matrix t × K : return numeric(Kgmt)
  .slice_theta_m <- function(th, t, Kgmt) {
    d <- dim(th)
    if (is.null(d)) {
      # vector: only 1 factor
      return(as.numeric(th[t]))
    }
    if (length(d) == 2L) {
      Kgmt <- min(Kgmt, d[2])
      return(as.numeric(th[t, seq_len(Kgmt), drop = TRUE]))
    }
    stop("Theta_m element must be vector or 2D matrix. Got dim=", paste(d, collapse="x"))
  }
  
  # How many factors are actually available for Gamma/Theta_m at view m?
  .avail_local_K <- function(gm, th) {
    # Gamma
    dgm <- dim(gm)
    Kg <- if (is.null(dgm)) 0L else if (length(dgm) == 2L) 1L else dgm[3]
    # Theta_m
    dth <- dim(th)
    Kt <- if (is.null(dth)) 1L else dth[2]
    min(Kg, Kt)
  }
  
  # -------------------------
  # basic setup
  # -------------------------
  M  <- length(risMCMC$mu_m)
  p_m <- vapply(risMCMC$mu_m, ncol, integer(1))
  nPred <- unlist(lapply(Xpred, nrow))[1]
  
  tMCMC <- nrow(risMCMC$K_Lm_eff)
  iter_print <- max(1L, tMCMC %/% 10L)
  mu_MC <- matrix(NA_real_, tMCMC, nPred)
  
  # -------------------------
  # burn-in/thin alignment fix
  # (only drop if needed; then force lengths/rows to >= tMCMC)
  # -------------------------
  t_drop <- 0L
  if (!is.null(risMCMC$hyper_param$tBurnIn) && !is.null(risMCMC$hyper_param$tThin)) {
    t_drop <- as.integer(risMCMC$hyper_param$tBurnIn %/% risMCMC$hyper_param$tThin)
    if (!is.finite(t_drop) || t_drop < 0L) t_drop <- 0L
  }
  
  # K
  if (!is.null(risMCMC$K)) {
    if (length(risMCMC$K) > tMCMC) {
      # drop at most length(K)-tMCMC so we don't end up too short
      drop_eff <- min(t_drop, length(risMCMC$K) - tMCMC)
      if (drop_eff > 0L) risMCMC$K <- risMCMC$K[-seq_len(drop_eff)]
    }
    if (length(risMCMC$K) < tMCMC) {
      stop("After burn-in handling, length(K)=", length(risMCMC$K), " < tMCMC=", tMCMC,
           ". Your saved K does not align with K_Lm_eff/Lambda_m.")
    }
    risMCMC$K <- as.integer(risMCMC$K[seq_len(tMCMC)])
  } else {
    stop("risMCMC$K is missing.")
  }
  
  # K_Gm
  if (grepl("jafar", risMCMC$hyper_param$model)) {
    if (is.null(risMCMC$K_Gm)) stop("risMCMC$K_Gm is missing for a jafar model.")
    if (nrow(risMCMC$K_Gm) > tMCMC) {
      drop_eff <- min(t_drop, nrow(risMCMC$K_Gm) - tMCMC)
      if (drop_eff > 0L) risMCMC$K_Gm <- risMCMC$K_Gm[-seq_len(drop_eff), , drop = FALSE]
    }
    if (nrow(risMCMC$K_Gm) < tMCMC) {
      stop("After burn-in handling, nrow(K_Gm)=", nrow(risMCMC$K_Gm), " < tMCMC=", tMCMC,
           ". Your saved K_Gm does not align with K_Lm_eff/Lambda_m.")
    }
    risMCMC$K_Gm <- as.matrix(risMCMC$K_Gm[seq_len(tMCMC), , drop = FALSE])
    storage.mode(risMCMC$K_Gm) <- "integer"
  }
  
  # -------------------------
  # Precompute global K caps from what is stored
  # -------------------------
  K_lambda_cap <- min(vapply(risMCMC$Lambda_m, function(df) dim(df)[3], integer(1)))
  K_theta_cap  <- ncol(risMCMC$Theta)
  K_cap <- min(K_lambda_cap, K_theta_cap)
  
  # -------------------------
  # prediction loop
  # -------------------------
  print(" - Computing Response Predictions - ")
  print(sprintf(fmt = "%10s%3s%2s", "[", 0, "%]"))
  
  for (t in seq_len(tMCMC)) {
    
    # ---- make K[t] safe ----
    Kt <- as.integer(risMCMC$K[t])
    if (!is.finite(Kt)) Kt <- 1L
    Kt <- max(1L, min(Kt, K_cap))
    
    # ---- slice global factors ----
    La_m_t  <- lapply(risMCMC$Lambda_m, .slice_lambda, t = t, Kt = Kt)
    Theta_t <- as.numeric(risMCMC$Theta[t, seq_len(Kt), drop = TRUE])
    
    mu_m_t <- lapply(risMCMC$mu_m, function(df) as.numeric(df[t, ]))
    s2_m_t <- lapply(risMCMC$s2_inv_m, function(df) as.numeric(df[t, ]))
    
    if (grepl("jafar", risMCMC$hyper_param$model)) {
      
      # ---- make per-view local Ks safe (can vary by view and by iteration) ----
      K_Gm_t <- as.integer(risMCMC$K_Gm[t, ])
      if (length(K_Gm_t) != M) stop("K_Gm row has length ", length(K_Gm_t), " but M=", M)
      
      # cap each view's local K by what is *actually* stored for that view
      for (m in seq_len(M)) {
        avail <- .avail_local_K(risMCMC$Gamma_m[[m]], risMCMC$Theta_m[[m]])
        if (avail < 1L) avail <- 1L
        if (!is.finite(K_Gm_t[m]) || K_Gm_t[m] < 1L) K_Gm_t[m] <- 1L
        K_Gm_t[m] <- min(K_Gm_t[m], avail)
      }
      
      # ---- slice local factors ----
      Ga_m_t <- vector("list", M)
      Theta_m_t <- vector("list", M)
      for (m in seq_len(M)) {
        Ga_m_t[[m]] <- .slice_gamma(risMCMC$Gamma_m[[m]], t = t, Kgmt = K_Gm_t[m])
        Theta_m_t[[m]] <- .slice_theta_m(risMCMC$Theta_m[[m]], t = t, Kgmt = K_Gm_t[m])
      }
      
      pred_t <- jafar_pred_y_t(
        Xpred, nPred, M, p_m,
        Kt, K_Gm_t,
        La_m_t, Ga_m_t, Theta_t, Theta_m_t,
        risMCMC$mu_y[t], risMCMC$s2_inv[t],
        mu_m_t, s2_m_t,
        rescale_pred = rescale_pred
      )
      
    } else {
      
      pred_t <- jfr_pred_y_t(
        Xpred, nPred, M, p_m, Kt,
        La_m_t, Theta_t,
        risMCMC$mu_y[t], risMCMC$s2_inv[t],
        mu_m_t, s2_m_t,
        rescale_pred = rescale_pred
      )
    }
    
    mu_MC[t, ] <- pred_t
    
    if (t %% iter_print == 0L) {
      print(sprintf(fmt = "%10s%3s%2s", "[", (t %/% iter_print) * 10, "%]"))
    }
  }
  
  return(list(mean = mu_MC))
}


py_matrix_to_R <- function(py_mat) {
  if (inherits(py_mat, "scipy.sparse.spmatrix")) {
    as.matrix(py_to_r(py_mat$toarray()))
  } else {
    as.matrix(py_to_r(py_mat))
  }
}

read_index_csv_0based <- function(path, n_total = NULL) {
  df <- tryCatch(read.csv(path, header = TRUE),
                 error = function(e) read.csv(path, header = FALSE))
  is_num <- vapply(df, is.numeric, logical(1))
  if (!any(is_num)) stop("No numeric column found in ", path)
  idx <- as.integer(df[[ which(is_num)[1] ]])
  
  if (!is.null(n_total)) {
    # accept 0-based or 1-based
    if (min(idx) == 0L && max(idx) == (n_total - 1L)) return(idx)
    if (min(idx) == 1L && max(idx) == n_total) return(idx - 1L)
  }
  # default assume 0-based
  idx
}

apply_pca_generic <- function(X, pca_obj) {
  # Works for prcomp objects and list-like objects that have $center and $rotation
  if (is.null(pca_obj)) return(X)
  
  center <- NULL
  rotation <- NULL
  if (inherits(pca_obj, "prcomp")) {
    center <- pca_obj$center
    rotation <- pca_obj$rotation
  } else if (is.list(pca_obj) && !is.null(pca_obj$center) && !is.null(pca_obj$rotation)) {
    center <- pca_obj$center
    rotation <- pca_obj$rotation
  } else {
    stop("Unsupported PCA object type: ", paste(class(pca_obj), collapse=", "))
  }
  
  Xc <- sweep(X, 2, center, FUN = "-")
  Z  <- Xc %*% rotation
  Z
}

apply_scaling <- function(X, scaler) {
  # scaler is list(mean=..., sd=...)
  sweep(sweep(X, 2, scaler$mean, FUN = "-"), 2, scaler$sd, FUN = "/")
}

reg_metrics <- function(y_true, y_pred) {
  ok <- complete.cases(y_true, y_pred)
  y_true <- as.numeric(y_true[ok])
  y_pred <- as.numeric(y_pred[ok])
  
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae  <- mean(abs(y_true - y_pred))
  
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - ss_res / ss_tot
  
  pear <- cor(y_true, y_pred)
  spear <- cor(y_true, y_pred, method = "spearman")

  list(rmse = rmse, mae = mae, r2 = r2, pearson_r = pear, spearman_r = spear)
}


# ------------------------------------------------------------
# IMPORTANT: patch fit object so predict_y_raw won't break
# ------------------------------------------------------------
prepare_fit_for_predict_y_raw <- function(risMCMC) {
  if (is.null(risMCMC$K_Lm_eff)) stop("fit missing K_Lm_eff")
  tMCMC <- nrow(risMCMC$K_Lm_eff)
  
  # stop internal dropping
  if (is.null(risMCMC$hyper_param)) risMCMC$hyper_param <- list()
  risMCMC$hyper_param$tBurnIn <- 0L
  risMCMC$hyper_param$tThin   <- 1L
  
  # truncate K / K_Gm to tMCMC
  if (is.null(risMCMC$K)) stop("fit missing K")
  if (length(risMCMC$K) < tMCMC) stop("K too short vs tMCMC")
  risMCMC$K <- as.integer(risMCMC$K[seq_len(tMCMC)])
  
  # cap global K to what is stored (Lambda_m and Theta)
  K_lambda_cap <- min(vapply(risMCMC$Lambda_m, function(df) dim(df)[3], integer(1)))
  K_theta_cap  <- ncol(risMCMC$Theta)
  K_cap <- min(K_lambda_cap, K_theta_cap)
  risMCMC$K <- pmax(1L, pmin(risMCMC$K, K_cap))
  
  # if jafar: handle locals
  if (!is.null(risMCMC$hyper_param$model) && grepl("jafar", risMCMC$hyper_param$model)) {
    if (is.null(risMCMC$K_Gm)) stop("fit missing K_Gm")
    if (nrow(risMCMC$K_Gm) < tMCMC) stop("K_Gm too short vs tMCMC")
    risMCMC$K_Gm <- as.matrix(risMCMC$K_Gm[seq_len(tMCMC), , drop = FALSE])
    storage.mode(risMCMC$K_Gm) <- "integer"
    
    M <- length(risMCMC$Gamma_m)
    if (length(risMCMC$Theta_m) != M) stop("Theta_m and Gamma_m length mismatch")
    
    for (m in seq_len(M)) {
      gm <- risMCMC$Gamma_m[[m]]
      dgm <- dim(gm)
      if (is.null(dgm)) stop("Gamma_m[[",m,"]] dim NULL")
      
      # force Gamma_m to 3D: [t, p, K_m]
      if (length(dgm) == 2L) {
        risMCMC$Gamma_m[[m]] <- array(gm, dim = c(dgm[1], dgm[2], 1L))
        dgm <- dim(risMCMC$Gamma_m[[m]])
      }
      if (length(dgm) != 3L) stop("Gamma_m[[",m,"]] must be 3D")
      if (dgm[1] != tMCMC) stop("Gamma_m[[",m,"]] t mismatch with tMCMC")
      
      # force Theta_m to 2D: [t, K_m]
      th <- risMCMC$Theta_m[[m]]
      dth <- dim(th)
      if (is.null(dth)) {
        if (length(th) < tMCMC) stop("Theta_m[[",m,"]] vector too short")
        risMCMC$Theta_m[[m]] <- matrix(th[seq_len(tMCMC)], nrow = tMCMC, ncol = 1L)
        dth <- dim(risMCMC$Theta_m[[m]])
      }
      if (length(dth) != 2L) stop("Theta_m[[",m,"]] must be 2D")
      if (dth[1] != tMCMC) stop("Theta_m[[",m,"]] t mismatch with tMCMC")
      
      Km_cap <- min(dgm[3], dth[2])
      risMCMC$K_Gm[, m] <- pmax(1L, pmin(risMCMC$K_Gm[, m], Km_cap))
    }
  }
  
  risMCMC
}

# ============================================================
# MAIN
# ============================================================
test_jafar_regression_tea <- function(
    target_protein = "CD45RA",
    split_tag      = "tea_split3_all_celltypes",
    base_dir       = "../data",
    splits_dir     = "../splits",
    model_dir      = "../models",
    results_dir    = "../results"
) {
  # ---- load saved train-time objects ----
  model_root <- file.path(model_dir, "tea", paste0(split_tag, "_jafar_", target_protein))
  prep_path  <- file.path(model_root, "jfr_prep.rds")
  fit_path   <- file.path(model_root, "jfr_reg_fit.rds")
  
  if (!file.exists(prep_path)) stop("Missing prep file: ", prep_path)
  if (!file.exists(fit_path))  stop("Missing fit file: ",  fit_path)
  
  out_root <- file.path(results_dir, "tea", paste0(split_tag, "_jafar_", target_protein))
  dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
  
  cat("[JAFAR-TEST] Loading saved training preprocessing + fit...\n")
  prep  <- readRDS(prep_path)
  fit_c <- readRDS(fit_path)
  
  # what we use from prep:
  # - PCA objects: pca_rna, pca_atac  (and optionally pca_adt if you saved it)
  # - scalers: sc_rna, sc_atac, sc_adt
  pca_rna  <- prep$pca_rna
  pca_atac <- prep$pca_atac
  sc_rna   <- prep$sc_rna
  sc_atac  <- prep$sc_atac
  sc_adt   <- prep$sc_adt
  
  # ---- load full TEA-seq ----
  # ---- load full TEA-seq ----
  cat(sprintf("[JAFAR-TEST] Loading TEA-seq for %s...\n", target_protein))
  rna_py  <- anndata$read_h5ad(file.path(base_dir, "rna.h5ad"))
  atac_py <- anndata$read_h5ad(file.path(base_dir, "atac.h5ad"))
  adt_py  <- anndata$read_h5ad(file.path(base_dir, paste0("adt_minus_", target_protein, ".h5ad")))
  y_df    <- read.csv(file.path(base_dir, "response", paste0(target_protein, ".csv")), row.names = 1)
  
  # FIX: robust n_cells
  n_cells <- length(py_to_r(rna_py$obs_names$tolist()))
  
  # ---- test indices ----
  test_idx_path <- file.path(splits_dir, paste0(split_tag, "_test_idx.csv"))
  idx_te0 <- read_index_csv_0based(test_idx_path, n_total = n_cells)
  idx_teR <- idx_te0 + 1L
  cat("[JAFAR-TEST] #TEST cells:", length(idx_teR), "\n")
  
  # ---- extract test matrices (cells x features) ----
  X_rna_full  <- py_matrix_to_R(rna_py$layers[["norm"]])
  X_atac_full <- py_matrix_to_R(atac_py$layers[["norm"]])
  X_adt_full  <- py_matrix_to_R(adt_py$layers[["norm"]])
  
  X_rna_te  <- X_rna_full[idx_teR, , drop = FALSE]
  X_atac_te <- X_atac_full[idx_teR, , drop = FALSE]
  X_adt_te  <- X_adt_full[idx_teR, , drop = FALSE]
  
  cat("[JAFAR-TEST] TEST dims (cells x features):\n")
  cat("  RNA :", paste(dim(X_rna_te), collapse=" x "), "\n")
  cat("  ATAC:", paste(dim(X_atac_te), collapse=" x "), "\n")
  cat("  ADT :", paste(dim(X_adt_te), collapse=" x "), "\n")
  
  # ---- y_true aligned to rows (IMPORTANT: use cell IDs, not idx numeric) ----
  # safest: match by cell names
  cell_ids_all <- tryCatch(py_to_r(rna_py$obs_names$tolist()), error=function(e) NULL)
  if (!is.null(cell_ids_all)) {
    cell_ids_te <- cell_ids_all[idx_teR]
    y_true <- as.numeric(y_df[cell_ids_te, 1])
  } else {
    # fallback: positional (only if your CSV row order matches anndata)
    y_true <- as.numeric(y_df[idx_teR, 1])
    cell_ids_te <- as.character(idx_teR)
  }
  
  # ---- apply TRAIN PCA + scaling ----
  cat("[JAFAR-TEST] Applying TRAIN PCA (center/rotation) + TRAIN scalers...\n")
  Z_rna_te  <- apply_pca_generic(X_rna_te,  pca_rna)
  Z_atac_te <- apply_pca_generic(X_atac_te, pca_atac)
  
  # defensive: ensure we use the exact trained number of PCs
  K_rna  <- ncol(Z_rna_te)  # already equals ncol(rotation)
  K_atac <- ncol(Z_atac_te)
  
  X_m_test <- list(
    rna  = apply_scaling(Z_rna_te[,  seq_len(K_rna),  drop=FALSE], sc_rna),
    atac = apply_scaling(Z_atac_te[, seq_len(K_atac), drop=FALSE], sc_atac),
    adt  = apply_scaling(X_adt_te, sc_adt)
  )
  
  # ---- patch fit + predict using per-draw closed form ----
  cat("[JAFAR-TEST] Patching fit for predict_y_raw safety...\n")
  fit_c2 <- prepare_fit_for_predict_y_raw(fit_c)
  
  cat("[JAFAR-TEST] Predicting with predict_y_raw (per MCMC draw)...\n")
  pred <- predict_y_raw(Xpred = X_m_test, risMCMC = fit_c2, rescale_pred = FALSE)
  mu_MC <- pred$mean   # [tMCMC x n_test]
  if (is.null(dim(mu_MC)) || length(dim(mu_MC)) != 2) stop("predict_y_raw returned bad shape")
  
  # posterior mean prediction per cell
  y_pred <- colMeans(mu_MC, na.rm = TRUE)
  
  # ---- metrics + save ----
  met <- reg_metrics(y_true, y_pred)
  cat(sprintf("[JAFAR-TEST] RMSE=%.4f  MAE=%.4f  R2=%.4f  Pearson=%.4f  Spearman=%.4f\n",
              met$rmse, met$mae, met$r2, met$pearson_r, met$spearman_r))
  
  np$save(file.path(out_root, "y_true.npy"), y_true)
  np$save(file.path(out_root, "y_pred.npy"), y_pred)
  np$save(file.path(out_root, "idx_test.npy"), as.integer(idx_te0))
  
  pred_df <- data.frame(cell = cell_ids_te, y_true = y_true, y_pred = y_pred, stringsAsFactors = FALSE)
  write.csv(pred_df, file.path(out_root, "test_predictions.csv"), row.names = FALSE)
  
  report <- list(
    n_test = length(y_true),
    rmse = unname(met$rmse),
    mae  = unname(met$mae),
    r2   = unname(met$r2),
    pearson_r  = unname(met$pearson_r),
    spearman_r = unname(met$spearman_r),
    run = list(
      method = "JAFAR/JFR regression",
      target_protein = target_protein,
      split_tag = split_tag,
      used_saved_objects = list(
        prep_rds = prep_path,
        fit_rds  = fit_path,
        from_prep = c("pca_rna", "pca_atac", "sc_rna", "sc_atac", "sc_adt"),
        from_fit  = c("Lambda_m", "Theta", "mu_m", "s2_inv_m", "mu_y", "s2_inv", "K", "K_Lm_eff",
                      "Gamma_m/Theta_m/K_Gm if jafar model")
      ),
      fit_patch = list(tBurnIn = 0L, tThin = 1L, capped_K = TRUE, capped_K_Gm = TRUE)
    ),
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  )
  
  writeLines(jsonlite::toJSON(report, pretty = TRUE, auto_unbox = TRUE),
             con = file.path(out_root, "test_metrics.json"))
  
  cat("[JAFAR-TEST] Saved outputs under:\n  ", out_root, "\n")
  invisible(report)
}

# -----------------------------
# Run
# -----------------------------
test_jafar_regression_tea(
  target_protein = "CD45RA",
  split_tag      = "tea_split3_all_celltypes",
  base_dir       = "../data",
  splits_dir     = "../splits",
  model_dir      = "../models",
  results_dir    = "../results"
)