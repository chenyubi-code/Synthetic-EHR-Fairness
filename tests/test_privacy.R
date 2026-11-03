# Artificial examples only; no patient records or precomputed study results.
source("metrics/load.R")
library(data.table)
set.seed(97)
queries <- matrix(rbinom(7 * 6, 1, .4), nrow = 7)
candidates <- matrix(rbinom(9 * 6, 1, .4), nrow = 9)
candidates[2, ] <- candidates[1, ]
labels <- matrix(rbinom(9 * 4, 1, .5), nrow = 9)
answer <- nearest_hamming_tie_vote(queries, candidates, labels)
for (i in seq_len(nrow(queries))) {
  distances <- rowSums(abs(sweep(candidates, 2, queries[i, ], "-")))
  nearest <- which(distances == min(distances))
  stopifnot(answer$distance[i] == min(distances), answer$tie_count[i] == length(nearest),
            all(answer$tied_ones[i, ] == colSums(labels[nearest, , drop = FALSE])))
}
toy <- expand.grid(age80 = 0:1, male = 0:1, white = 0:1, private = 0:1)
toy[paste0("P", 1:4)] <- toy[sensitive_vars]
identity <- compute_privacy_metrics(toy, toy, list(r1 = toy))
stopifnot(all(identity$population_metrics$MIR == 0), all(identity$population_metrics$AIR == 1),
          all(identity$by_repetition$baMIR == 0), all(identity$by_repetition$baAIR == 0),
          all(identity$gaps$baMIRG == 0), all(identity$gaps$baAIRG == 0))
# Contrasts use population metrics, with opposite directions for distance and F1.
raw <- copy(identity$population_metrics)
raw[population == "mem", `:=`(MIR = 2, MIR_normalized = .5, AIR = .6)]
raw[population == "nonmem", `:=`(MIR = 3, MIR_normalized = .75, AIR = .4)]
adjusted <- adjust_privacy_exposure(raw[sample(.N)])
stopifnot(all(adjusted$baMIR == 1), all(abs(adjusted$baAIR - .2) < 1e-12))
raw[population == "nonmem", `:=`(MIR = 1, MIR_normalized = .25, AIR = .8)]
negative <- adjust_privacy_exposure(raw)
stopifnot(all(negative$baMIR == -1), all(abs(negative$baAIR + .2) < 1e-12))
missing <- compute_privacy_metrics(toy, toy, toy[toy$age80 == 0, ])
stopifnot(is.na(missing$by_repetition[sensitive_attribute == "age80" & group_value == 1, baMIR]),
          is.na(missing$gaps[sensitive_attribute == "age80", baMIRG]))
stopifnot(inherits(try(adjust_privacy_exposure(raw[-1]), silent = TRUE), "try-error"))
ties <- toy
ties[paste0("P", 1:4)] <- 0L
tie_first <- compute_privacy_metrics(ties, ties, ties)
tie_second <- compute_privacy_metrics(ties, ties, ties)
stopifnot(identical(tie_first$target_metrics, tie_second$target_metrics),
          all(tie_first$target_metrics$n_random_vote_ties > 0))
cat("PASS: exact Hamming distances/all nearest ties against brute force,\n",
    "F1 inference, zero exposure, signed member/nonmember contrasts, missing cells.\n")
