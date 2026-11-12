source("subgroup_sensitivity.R")
# Invented IDs/attributes; no study data.
raw <- data.frame(subject_id = c(4, 2, 1, 3), age = c(90, 84, 30, 85),
                  ethnicity = c(NA, "BLACK/CAPE VERDEAN", "WHITE", " black/african american "))
x <- data.frame(subject_id = 1:4, age80 = c(0, 1, 1, 1), male = c(0, 1, 0, 1),
                white = c(1, 0, 0, 0), private = c(1, 1, 0, 0), P = c(1, 0, 1, 0))
age <- recode_sensitive_definition(x, raw, "age85")
black <- recode_sensitive_definition(x, raw, "black")
stopifnot(identical(age$age85, c(0L, 0L, 1L, 1L)),
          identical(black$black, c(0L, 1L, 1L, 0L)),
          identical(age$subject_id, x$subject_id), identical(black$P, x$P))
visits <- x[c(2, 1, 2, 3), ]
stopifnot(identical(recode_sensitive_definition(visits, raw, "black")$black, c(1L, 0L, 1L, 1L)))
prepared <- prepare_subgroup_sensitivity(x[1:2, ], x[3:4, ], raw, "P", "age85")
stopifnot(identical(prepared$evaluation_phenotypes, "P"),
          identical(prepared$sensitive_columns, c("age85", "male", "white", "private")))
stopifnot(inherits(try(recode_sensitive_definition(x, raw[-1, ], "black"), silent = TRUE), "try-error"),
          inherits(try(recode_sensitive_definition(x, rbind(raw, raw[1, ]), "black"), silent = TRUE), "try-error"),
          inherits(try(prepare_subgroup_sensitivity(x[1:2, ], x[2:4, ], raw, "P", "black"), silent = TRUE), "try-error"))
cat("PASS: ID-matched alternative coding, unknown ethnicity, fixed split/schema/phenotypes, HALO visits.\n")
