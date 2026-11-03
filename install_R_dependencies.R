# Run once with: Rscript install_R_dependencies.R
# Package installation is separate from loading or evaluating the benchmark.
packages <- c("data.table", "dplyr", "tidyr", "stringr", "caret",
              "Matrix", "glmnet", "pROC", "randomForest", "xgboost", "Rcpp")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")

# The preprocessing code uses short_to_decimal() from the authors' icd package.
if (!requireNamespace("icd", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
  remotes::install_github("jackwasey/icd", upgrade = "never")
}
