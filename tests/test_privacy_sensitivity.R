# Artificial fixtures only: independent exact search, sampling and aggregation.
source("metrics/load.R")
if (!exists("compute_matched_privacy_metrics")) source("metrics/privacy_sensitivity.R")
library(data.table)
set.seed(19)
q <- matrix(rbinom(6 * 5, 1, .4), nrow = 6)
s <- matrix(rbinom(8 * 5, 1, .4), nrow = 8)
s[2, ] <- s[1, ]
a <- matrix(rbinom(8 * 4, 1, .5), nrow = 8)
pool <- cbind(1L, as.integer(seq_len(8) <= 4), as.integer(seq_len(8) %% 2 == 0))
k <- .batch_privacy_kernel()(q, s, a, pool)
for (i in seq_len(nrow(q))) for (b in seq_len(ncol(pool))) {
  d <- rowSums(abs(sweep(s, 2, q[i, ], "-")))
  selected <- which(pool[, b] == 1L)
  ids <- selected[d[selected] == min(d[selected])]
  stopifnot(k$distance[i, b] == min(d[selected]), k$tie_count[i, b] == length(ids),
            k$nearest_index[i, b] == min(ids))
  for (j in seq_len(ncol(a))) stopifnot(k$tied_ones[[j]][i, b] == sum(a[ids, j]))
}
toy <- expand.grid(age80 = 0:1, male = 0:1, white = 0:1, private = 0:1)
toy[paste0("P", 1:4)] <- toy[sensitive_vars]
syn <- rbind(toy, toy[toy$age80 == 0, ])
real_test <- toy
real_test$P1 <- 1L - real_test$P1
reps <- list(seed_1 = syn, seed_2 = syn[c(seq_len(16), 17:20), ])
ans <- compute_matched_privacy_metrics(toy, real_test, reps, "fixture", n_resamples = 3L)
full <- compute_privacy_metrics(toy, real_test, reps, "fixture")
compare <- merge(ans$population_metrics[resample == 0L], full$population_metrics,
  by = c("method", "repetition", "sensitive_attribute", "group_value", "population"))
stopifnot(all(compare$MIR.x == compare$MIR.y), all(compare$AIR.x == compare$AIR.y))
# Reconstruct each draw independently and use the original per-pool API.
for (rep in names(reps)) for (attribute in sensitive_vars) for (b in 1:3) {
  synthetic <- reps[[rep]]
  m <- min(table(synthetic[[attribute]]))
  ids <- unlist(lapply(0:1, function(g) {
    candidates <- which(synthetic[[attribute]] == g)
    set.seed(stable_air_vote_seed("candidate_pool_size_v1", "fixture", rep, attribute, b, g),
      kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    sort(candidates[sample.int(length(candidates), m, replace = FALSE)])
  }))
  ordinary <- compute_privacy_metrics(toy, real_test, setNames(list(synthetic[ids, ]), rep), "fixture")
  expected <- ordinary$gaps[sensitive_attribute == attribute]
  actual <- ans$gaps_by_resample[repetition == rep & sensitive_attribute == attribute & resample == b]
  stopifnot(isTRUE(all.equal(actual$baMIRG, expected$baMIRG)),
            isTRUE(all.equal(actual$baAIRG, expected$baAIRG)))
}
for (attribute in sensitive_vars) {
  rep_values <- ans$gaps_by_resample[sensitive_attribute == attribute & resample > 0,
                                    .(value = mean(baMIRG)), by = repetition]$value
  summary <- ans$summary[sensitive_attribute == attribute & condition == "matched"]
  stopifnot(summary$n_repetitions == 2L, summary$baMIRG == mean(rep_values),
            summary$baMIRG_se == sd(rep_values) / sqrt(2))
}
all_zero <- toy
all_zero$private <- 0L
stopifnot(inherits(try(compute_matched_privacy_metrics(all_zero, toy, reps), silent = TRUE), "try-error"))
stopifnot(inherits(try(compute_matched_privacy_metrics(toy, toy, reps, n_resamples = 32), silent = TRUE), "try-error"))
undefined <- toy[toy$age80 == toy$white, ]
ordinary_undefined <- compute_privacy_metrics(undefined, undefined, undefined)
stopifnot(is.na(ordinary_undefined$population_metrics[
  population == "mem" & sensitive_attribute == "age80" & group_value == 0L, AIR]))
stopifnot(inherits(try(compute_matched_privacy_metrics(undefined, undefined, undefined),
                       silent = TRUE), "try-error"))
cat("PASS: batched Hamming/all-tie votes, identical full-pool metrics, independent draws, nested SE.\n")
