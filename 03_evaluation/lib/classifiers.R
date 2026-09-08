downstream_outcomes <- c("CV_404", "GU_582")
minimum_test_cases_per_subgroup <- 50
classification_threshold <- 0.5
analysis_seed <- 20260701

ridge_lambda <- 0.05
glmnet_standardize <- FALSE
glmnet_maxit <- 5000
glmnet_thresh <- 1e-04

random_forest_ntree <- 100
random_forest_nodesize <- 10
random_forest_mtry <- NULL

xgboost_nrounds <- 100
xgboost_max_depth <- 3
xgboost_eta <- 0.05
xgboost_subsample <- 0.80
xgboost_colsample_bytree <- 0.80
xgboost_nthread <- max(1, min(4, parallel::detectCores(logical = TRUE) - 1))
clip_probability <- function(prob) {
  prob <- as.numeric(prob)
  prob[is.na(prob)] <- mean(prob, na.rm = TRUE)
  prob[is.na(prob)] <- 0.5
  pmin(pmax(prob, 0), 1)
}

fit_fixed_real_model <- function(prediction_model, x_train, y_train) {
  y_train <- as.integer(y_train)
  if (length(unique(y_train[!is.na(y_train)])) < 2) {
    return(list(kind = "constant", probability = mean(y_train, na.rm = TRUE)))
  }

  if (prediction_model == "ridge_logistic") {
    fit <- glmnet::glmnet(
      x = Matrix::Matrix(x_train, sparse = TRUE),
      y = y_train,
      family = "binomial",
      alpha = 0,
      lambda = ridge_lambda,
      standardize = glmnet_standardize,
      maxit = glmnet_maxit,
      thresh = glmnet_thresh
    )
    return(list(kind = prediction_model, fit = fit))
  }

  if (prediction_model == "random_forest") {
    mtry_value <- random_forest_mtry
    if (is.null(mtry_value)) {
      mtry_value <- max(1, floor(sqrt(ncol(x_train))))
    }
    fit <- randomForest::randomForest(
      x = x_train,
      y = factor(y_train, levels = c(0, 1)),
      ntree = random_forest_ntree,
      mtry = mtry_value,
      nodesize = random_forest_nodesize
    )
    return(list(kind = prediction_model, fit = fit))
  }

  if (prediction_model == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(
      data = Matrix::Matrix(x_train, sparse = TRUE),
      label = y_train
    )
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = xgboost_max_depth,
      eta = xgboost_eta,
      subsample = xgboost_subsample,
      colsample_bytree = xgboost_colsample_bytree,
      min_child_weight = 1,
      lambda = 1,
      alpha = 0,
      nthread = xgboost_nthread
    )
    fit <- xgboost::xgb.train(
      params = params,
      data = dtrain,
      nrounds = xgboost_nrounds,
      verbose = 0
    )
    return(list(kind = prediction_model, fit = fit))
  }

  stop("Unknown prediction model: ", prediction_model)
}

predict_fixed_real_model <- function(fitted_model, x_new) {
  prob <- switch(
    fitted_model$kind,
    constant = rep(fitted_model$probability, nrow(x_new)),
    ridge_logistic = as.numeric(predict(
      fitted_model$fit,
      newx = Matrix::Matrix(x_new, sparse = TRUE),
      type = "response",
      s = ridge_lambda
    )),
    random_forest = {
      pred <- predict(fitted_model$fit, x_new, type = "prob")
      as.numeric(pred[, "1"])
    },
    xgboost = as.numeric(predict(
      fitted_model$fit,
      xgboost::xgb.DMatrix(data = Matrix::Matrix(x_new, sparse = TRUE))
    )),
    stop("Unknown fitted model kind: ", fitted_model$kind)
  )
  clip_probability(prob)
}

safe_auc <- function(y, score) {
  ok <- !is.na(y) & !is.na(score)
  y <- as.integer(y[ok])
  score <- as.numeric(score[ok])
  if (length(unique(y)) < 2 || length(score) == 0) {
    return(NA_real_)
  }
  roc_obj <- suppressMessages(
    pROC::roc(response = y, predictor = score, levels = c(0, 1), direction = "<", quiet = TRUE)
  )
  as.numeric(pROC::auc(roc_obj))
}

compute_binary_metrics <- function(y_true, prob, threshold = 0.5) {
  ok <- !is.na(y_true) & !is.na(prob)
  y_true <- as.integer(y_true[ok])
  prob <- as.numeric(prob[ok])
  if (length(y_true) == 0) {
    return(c(AUROC = NA_real_, ACC = NA_real_, TPR = NA_real_, FPR = NA_real_, PPR = NA_real_))
  }
  y_pred <- as.integer(prob >= threshold)
  acc <- mean(y_pred == y_true)
  tpr <- if (sum(y_true == 1) == 0) NA_real_ else mean(y_pred[y_true == 1] == 1)
  fpr <- if (sum(y_true == 0) == 0) NA_real_ else mean(y_pred[y_true == 0] == 1)
  ppr <- mean(y_pred == 1)
  c(AUROC = safe_auc(y_true, prob), ACC = acc, TPR = tpr, FPR = fpr, PPR = ppr)
}

get_prediction_phenotypes <- function(
  real_train,
  real_test,
  prevalence_threshold = 0.01,
  minimum_test_cases = 50,
  sensitive_columns = sensitive_vars
) {
  eval_info <- compute_eval_phenotypes(real_train, prevalence_threshold, sensitive_columns)
  prediction_phenotypes <- character(0)

  for (phenotype in eval_info$phenotypes) {
    y <- real_test[[phenotype]]
    overall_positive <- sum(y == 1, na.rm = TRUE)
    overall_negative <- sum(y == 0, na.rm = TRUE)
    subgroup_min_positive <- Inf
    subgroup_min_negative <- Inf

    for (attr in sensitive_columns) {
      for (group_value in c(0L, 1L)) {
        idx <- real_test[[attr]] == group_value
        idx[is.na(idx)] <- FALSE
        subgroup_min_positive <- min(subgroup_min_positive, sum(y[idx] == 1, na.rm = TRUE))
        subgroup_min_negative <- min(subgroup_min_negative, sum(y[idx] == 0, na.rm = TRUE))
      }
    }

    keep <- overall_positive >= minimum_test_cases &&
      overall_negative >= minimum_test_cases &&
      subgroup_min_positive >= minimum_test_cases &&
      subgroup_min_negative >= minimum_test_cases

    if (keep) {
      prediction_phenotypes <- c(prediction_phenotypes, phenotype)
    }
  }

  prediction_phenotypes
}


prediction_models <- c("ridge_logistic", "random_forest", "xgboost")
