atpr_reference_groups <- c(0L, 1L)
metrics_adjuster_cal_degree <- 2L
metrics_adjuster_dr_degree <- 1L
metrics_adjuster_seed <- 20260701L
format_cli_number <- function(value) format(value, scientific = FALSE, trim = TRUE)
build_metrics_adjuster_command <- function() {
  python_path <- Sys.getenv("METRICS_ADJUSTER_PYTHON", unset = "")
  if (nzchar(python_path)) {
    return(list(
      command = python_path,
      prefix_args = c("-m", "metrics_adjuster.cli"),
      display = paste(shQuote(python_path), "-m metrics_adjuster.cli")
    ))
  }

  cli_path <- Sys.getenv("METRICS_ADJUSTER_BIN", unset = "metrics-adjuster")
  list(
    command = cli_path,
    prefix_args = character(0),
    display = shQuote(cli_path)
  )
}

run_external_command <- function(command, args) {
  output <- tryCatch(
    suppressWarnings(system2(command, args = args, stdout = TRUE, stderr = TRUE)),
    error = function(e) {
      structure(conditionMessage(e), status = 127L)
    }
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }
  list(status = as.integer(status), output = output)
}

ensure_metrics_adjuster_available <- function(command_spec) {
  result <- run_external_command(
    command_spec$command,
    c(command_spec$prefix_args, "--help")
  )
  if (result$status != 0L) {
    stop(
      "Cannot run metrics-adjuster via ", command_spec$display, ".\n",
      "Install the Python package in a Python 3.11+ environment:\n",
      "  python -m pip install metrics-adjuster\n",
      "Then set METRICS_ADJUSTER_BIN or METRICS_ADJUSTER_PYTHON if needed.\n\n",
      paste(result$output, collapse = "\n")
    )
  }
  invisible(TRUE)
}

make_metrics_adjuster_input <- function(y_test, prob, group_values) {
  frame <- data.frame(
    patient_id = seq_along(y_test),
    group = as.integer(group_values),
    outcome = as.integer(y_test),
    risk = clip_probability(prob),
    stringsAsFactors = FALSE
  )
  ok <- !is.na(frame$group) & !is.na(frame$outcome) & !is.na(frame$risk)
  frame[ok, , drop = FALSE]
}


run_metrics_adjuster_atpr <- function(
  input_frame,
  reference_group_value,
  run_label,
  command_spec = build_metrics_adjuster_command(),
  temporary_directory = tempdir()
) {
  run_dir <- tempfile(pattern = "atpr_", tmpdir = temporary_directory)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  Sys.chmod(run_dir, mode = "0700")
  on.exit(unlink(run_dir, recursive = TRUE, force = TRUE), add = TRUE)
  input_path <- file.path(run_dir, "input.csv")
  data.table::fwrite(input_frame, input_path)
  args <- c(command_spec$prefix_args, "run",
    "--input", shQuote(input_path), "--output-dir", shQuote(run_dir),
    "--group-col", "group", "--ref-group", as.character(reference_group_value),
    "--response-col", "outcome", "--risk-col", "risk", "--id-col", "patient_id",
    "--metrics", "aTPR", "--quantiles", shQuote(""),
    "--thresholds", format_cli_number(classification_threshold),
    "--cal-degree", as.character(metrics_adjuster_cal_degree),
    "--dr-degree", as.character(metrics_adjuster_dr_degree),
    "--seed", as.character(metrics_adjuster_seed))
  result <- run_external_command(command_spec$command, args)
  if (result$status != 0L) stop("metrics-adjuster failed for ", run_label, ":\n", paste(result$output, collapse = "\n"))
  atpr_path <- file.path(run_dir, "aTPR.csv")
  if (!file.exists(atpr_path)) stop("metrics-adjuster did not produce aTPR.csv for ", run_label)
  atpr <- data.table::fread(atpr_path, data.table = FALSE)
  data.table::setnames(atpr, "group", "group_value")
  atpr
}

compute_atpr_values_for_attribute <- function(
  y_test,
  prob,
  group_values,
  reference_group_value,
  run_label,
  command_spec = build_metrics_adjuster_command(),
  temporary_directory = tempdir()
) {
  input_frame <- make_metrics_adjuster_input(y_test = y_test, prob = prob, group_values = group_values)
  atpr_table <- run_metrics_adjuster_atpr(
    input_frame = input_frame,
    reference_group_value = reference_group_value,
    run_label = run_label,
    command_spec = command_spec,
    temporary_directory = temporary_directory
  )
  atpr_table$group_value <- as.integer(as.character(atpr_table$group_value))
  atpr_values <- setNames(rep(NA_real_, 2), c("0", "1"))
  for (group_value in c(0L, 1L)) {
    row <- atpr_table[atpr_table$group_value == group_value, , drop = FALSE]
    if (nrow(row) == 0) {
      next
    }
    atpr_values[[as.character(group_value)]] <- if (group_value == reference_group_value) {
      as.numeric(row$TPR[1])
    } else {
      as.numeric(row$aTPR[1])
    }
  }
  atpr_values
}
