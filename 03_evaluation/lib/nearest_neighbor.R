f1_score_binary <- function(y_true, y_pred) {
  ok <- !is.na(y_true) & !is.na(y_pred)
  y_true <- as.integer(y_true[ok])
  y_pred <- as.integer(y_pred[ok])
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  denom <- 2 * tp + fp + fn
  if (denom == 0) {
    return(NA_real_)
  }
  (2 * tp) / denom
}

compile_nearest_hamming_tie_vote_cpp <- function() {
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    return(FALSE)
  }
  if (exists("nearest_hamming_tie_vote_cpp", mode = "function", envir = .GlobalEnv)) {
    return(TRUE)
  }
  Rcpp::cppFunction(
    code = '
      Rcpp::List nearest_hamming_tie_vote_cpp(
        Rcpp::NumericMatrix real_mat,
        Rcpp::NumericMatrix syn_mat,
        Rcpp::IntegerMatrix syn_sensitive
      ) {
        using namespace Rcpp;
        int n_real = real_mat.nrow();
        int n_syn = syn_mat.nrow();
        int p = real_mat.ncol();
        int q = syn_sensitive.ncol();

        NumericVector distances(n_real);
        IntegerVector nearest_index(n_real);
        IntegerVector tie_count(n_real);
        IntegerMatrix tied_ones(n_real, q);

        if (n_syn == 0) {
          for (int i = 0; i < n_real; ++i) {
            distances[i] = R_PosInf;
            nearest_index[i] = NA_INTEGER;
            tie_count[i] = 0;
          }
          return List::create(
            Named("distance") = distances,
            Named("nearest_index") = nearest_index,
            Named("tie_count") = tie_count,
            Named("tied_ones") = tied_ones
          );
        }

        std::vector< std::vector<int> > postings(p);
        std::vector<int> syn_nnz(n_syn, 0);
        std::vector<int> nnz_freq(p + 1, 0);
        std::vector< std::vector<int> > nnz_ones(p + 1, std::vector<int>(q, 0));

        for (int s = 0; s < n_syn; ++s) {
          for (int j = 0; j < p; ++j) {
            if (syn_mat(s, j) >= 0.5) {
              postings[j].push_back(s);
              syn_nnz[s] += 1;
            }
          }
          if (syn_nnz[s] >= 0 && syn_nnz[s] <= p) {
            nnz_freq[syn_nnz[s]] += 1;
            for (int a = 0; a < q; ++a) {
              if (syn_sensitive(s, a) == 1) {
                nnz_ones[syn_nnz[s]][a] += 1;
              }
            }
          }
        }

        std::vector<int> sorted_syn(n_syn);
        std::iota(sorted_syn.begin(), sorted_syn.end(), 0);
        std::sort(sorted_syn.begin(), sorted_syn.end(), [&](int a, int b) {
          if (syn_nnz[a] == syn_nnz[b]) return a < b;
          return syn_nnz[a] < syn_nnz[b];
        });

        std::vector<int> counts(n_syn, 0);
        std::vector<int> mark(n_syn, 0);
        std::vector<int> touched;
        int stamp = 0;

        for (int i = 0; i < n_real; ++i) {
          stamp += 1;
          if (stamp == INT_MAX) {
            std::fill(mark.begin(), mark.end(), 0);
            stamp = 1;
          }

          touched.clear();
          int real_nnz = 0;

          for (int j = 0; j < p; ++j) {
            if (real_mat(i, j) >= 0.5) {
              real_nnz += 1;
              const std::vector<int>& posting = postings[j];
              for (std::vector<int>::const_iterator it = posting.begin(); it != posting.end(); ++it) {
                int s = *it;
                if (mark[s] != stamp) {
                  mark[s] = stamp;
                  counts[s] = 1;
                  touched.push_back(s);
                } else {
                  counts[s] += 1;
                }
              }
            }
          }

          int best_dist = INT_MAX;
          int best_idx = -1;

          for (std::vector<int>::const_iterator it = touched.begin(); it != touched.end(); ++it) {
            int s = *it;
            int dist = real_nnz + syn_nnz[s] - 2 * counts[s];
            if (dist < best_dist || (dist == best_dist && s < best_idx)) {
              best_dist = dist;
              best_idx = s;
            }
          }

          for (std::vector<int>::const_iterator it = sorted_syn.begin(); it != sorted_syn.end(); ++it) {
            int s = *it;
            if (mark[s] != stamp) {
              int dist = real_nnz + syn_nnz[s];
              if (dist < best_dist || (dist == best_dist && (best_idx < 0 || s < best_idx))) {
                best_dist = dist;
                best_idx = s;
              }
              break;
            }
          }

          int ties = 0;
          std::vector<int> ones(q, 0);
          int req_nnz = best_dist - real_nnz;
          int touched_req = 0;
          std::vector<int> touched_req_ones(q, 0);

          for (std::vector<int>::const_iterator it = touched.begin(); it != touched.end(); ++it) {
            int s = *it;
            int dist = real_nnz + syn_nnz[s] - 2 * counts[s];
            if (dist == best_dist) {
              ties += 1;
              for (int a = 0; a < q; ++a) {
                if (syn_sensitive(s, a) == 1) {
                  ones[a] += 1;
                }
              }
            }
            if (req_nnz >= 0 && req_nnz <= p && syn_nnz[s] == req_nnz) {
              touched_req += 1;
              for (int a = 0; a < q; ++a) {
                if (syn_sensitive(s, a) == 1) {
                  touched_req_ones[a] += 1;
                }
              }
            }
          }

          if (req_nnz >= 0 && req_nnz <= p) {
            ties += nnz_freq[req_nnz] - touched_req;
            for (int a = 0; a < q; ++a) {
              ones[a] += nnz_ones[req_nnz][a] - touched_req_ones[a];
            }
          }

          distances[i] = best_dist;
          nearest_index[i] = best_idx + 1;
          tie_count[i] = ties;
          for (int a = 0; a < q; ++a) {
            tied_ones(i, a) = ones[a];
          }
        }

        return List::create(
          Named("distance") = distances,
          Named("nearest_index") = nearest_index,
          Named("tie_count") = tie_count,
          Named("tied_ones") = tied_ones
        );
      }
    ',
    includes = '
      #include <algorithm>
      #include <climits>
      #include <numeric>
    ',
    env = .GlobalEnv
  )
  TRUE
}

nearest_hamming_tie_vote <- function(real_mat, syn_mat, syn_sensitive, engine = "cpp") {
  engine <- match.arg(engine, c("cpp"))
  storage.mode(syn_sensitive) <- "integer"
  if (engine == "cpp" && compile_nearest_hamming_tie_vote_cpp()) {
    return(nearest_hamming_tie_vote_cpp(real_mat, syn_mat, syn_sensitive))
  }
  stop("Tie-aware nearest-neighbor voting requires the Rcpp package.")
}


air_vote_random_seed <- 20260706L
stable_air_vote_seed <- function(...) {
  key <- paste(..., collapse = "|")
  seed <- as.numeric(air_vote_random_seed)
  for (value in utf8ToInt(key)) {
    seed <- (seed * 131 + value) %% 2147483647
  }
  if (seed <= 0) {
    seed <- 1
  }
  as.integer(seed)
}
