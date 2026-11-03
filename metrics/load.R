# Source this file from any working directory to load the metric functions.
.metrics_source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
if (is.null(.metrics_source_file)) stop("Load with source('/path/to/metrics/load.R').")
.metrics_directory <- dirname(normalizePath(.metrics_source_file, mustWork = TRUE))
for (.metrics_file in c("common.R", "classifiers.R", "atpr.R", "nearest_neighbor.R",
                        "data_level.R", "downstream.R", "privacy.R")) {
  source(file.path(.metrics_directory, .metrics_file), local = environment())
}
rm(.metrics_source_file, .metrics_directory, .metrics_file)
