// Exact minimum Hamming distances and all-nearest-neighbor label counts for
// several candidate pools. Each query/candidate distance is calculated once.
// Original candidate row order defines the representative nearest index.
#include <Rcpp.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List batch_nearest_hamming(Rcpp::NumericMatrix real_mat,
                               Rcpp::NumericMatrix syn_mat,
                               Rcpp::IntegerMatrix syn_sensitive,
                               Rcpp::IntegerMatrix pool_membership) {
  using namespace Rcpp;
  const int nr = real_mat.nrow(), ns = syn_mat.nrow(), p = real_mat.ncol();
  const int q = syn_sensitive.ncol(), np = pool_membership.ncol();
  if (syn_mat.ncol() != p || syn_sensitive.nrow() != ns ||
      pool_membership.nrow() != ns || p == 0 || ns == 0 ||
      q == 0 || np < 1 || np > 32)
    stop("Invalid dimensions, empty candidates, or pool count outside 1..32.");

  // Identical pools (notably the smaller subgroup's 21 full-pool columns)
  // share computation, then are expanded to the requested output columns.
  std::unordered_map<std::string, int> pool_map;
  std::vector<int> pool_id(np), representative;
  for (int b = 0; b < np; ++b) {
    std::string key(ns, '\0');
    int size = 0;
    for (int s = 0; s < ns; ++s) {
      int val = pool_membership(s, b);
      if (val != 0 && val != 1) stop("Pool membership must be binary.");
      key[s] = static_cast<char>(val);
      size += val;
    }
    if (size == 0) stop("Every candidate pool must be nonempty.");
    auto found = pool_map.find(key);
    if (found == pool_map.end()) {
      int k = representative.size();
      pool_map.emplace(key, k);
      representative.push_back(b);
      pool_id[b] = k;
    } else pool_id[b] = found->second;
  }
  const int nu = representative.size();
  const uint32_t all_pools = nu == 32 ? UINT32_MAX : ((uint32_t(1) << nu) - 1);
  std::vector<uint32_t> masks(ns, 0);
  for (int s = 0; s < ns; ++s)
    for (int k = 0; k < nu; ++k)
      if (pool_membership(s, representative[k])) masks[s] |= uint32_t(1) << k;

  std::vector<std::vector<int> > postings(p);
  std::vector<int> syn_nnz(ns, 0), labels(ns * q, 0);
  for (int s = 0; s < ns; ++s) {
    for (int j = 0; j < p; ++j) {
      double val = syn_mat(s, j);
      if (val != 0.0 && val != 1.0) stop("Synthetic features must be complete binary values.");
      if (val == 1.0) {
        postings[j].push_back(s);
        ++syn_nnz[s];
      }
    }
    for (int a = 0; a < q; ++a) {
      int val = syn_sensitive(s, a);
      if (val != 0 && val != 1) stop("Sensitive attributes must be complete binary values.");
      labels[s * q + a] = val;
    }
  }

  IntegerMatrix distance(nr, np), nearest_index(nr, np), tie_count(nr, np);
  List tied_ones(q);
  std::vector<IntegerMatrix> ones_matrices;
  for (int a = 0; a < q; ++a) {
    IntegerMatrix one(nr, np);
    tied_ones[a] = one;
    ones_matrices.push_back(one);
  }
  std::vector<int> overlap(ns, 0), next(ns, -1), head(p + 1, -1);
  std::vector<int> distances(nu), first(nu), ties(nu), ones(nu * q);
  std::vector<int> real_positive;
  std::unordered_map<std::string, int> query_cache;
  query_cache.reserve(nr);
  int unique_queries = 0;

  for (int i = 0; i < nr; ++i) {
    if ((i & 255) == 0) checkUserInterrupt();
    std::string query_key((p + 7) / 8, '\0');
    real_positive.clear();
    for (int j = 0; j < p; ++j) {
      double val = real_mat(i, j);
      if (val != 0.0 && val != 1.0) stop("Real features must be complete binary values.");
      if (val == 1.0) {
        real_positive.push_back(j);
        query_key[j / 8] = static_cast<char>(
          static_cast<unsigned char>(query_key[j / 8]) | (1u << (j % 8)));
      }
    }
    auto cached = query_cache.find(query_key);
    if (cached != query_cache.end()) {
      int prev = cached->second;
      for (int b = 0; b < np; ++b) {
        distance(i, b) = distance(prev, b);
        nearest_index(i, b) = nearest_index(prev, b);
        tie_count(i, b) = tie_count(prev, b);
        for (int a = 0; a < q; ++a) ones_matrices[a](i, b) = ones_matrices[a](prev, b);
      }
      continue;
    }
    query_cache.emplace(std::move(query_key), i);
    ++unique_queries;

    std::fill(overlap.begin(), overlap.end(), 0);
    for (int j : real_positive)
      for (int s : postings[j]) ++overlap[s];
    std::fill(head.begin(), head.end(), -1);
    const int real_nnz = real_positive.size();
    int min_distance = p;
    // Reverse insertion makes each linked distance bucket follow ascending
    // original synthetic row order, including duplicated records.
    for (int s = ns - 1; s >= 0; --s) {
      if (masks[s] == 0) continue;
      int d = real_nnz + syn_nnz[s] - 2 * overlap[s];
      next[s] = head[d];
      head[d] = s;
      if (d < min_distance) min_distance = d;
    }
    std::fill(first.begin(), first.end(), -1);
    std::fill(ties.begin(), ties.end(), 0);
    std::fill(ones.begin(), ones.end(), 0);
    uint32_t unresolved = all_pools;
    for (int d = min_distance; d <= p && unresolved; ++d) {
      uint32_t resolved_here = 0;
      for (int s = head[d]; s >= 0; s = next[s]) {
        uint32_t active = masks[s] & unresolved;
        resolved_here |= active;
        while (active) {
          const int k = __builtin_ctz(active);
          active &= active - 1;
          if (first[k] == -1) {
            first[k] = s + 1;
            distances[k] = d;
          }
          ++ties[k];
          for (int a = 0; a < q; ++a) ones[k * q + a] += labels[s * q + a];
        }
      }
      // Remove a pool only after ALL candidates at its minimum distance have
      // contributed; stopping at the first nearest row would corrupt AIR.
      unresolved &= ~resolved_here;
    }
    if (unresolved) stop("Internal error: a nonempty pool was not resolved.");
    for (int b = 0; b < np; ++b) {
      const int k = pool_id[b];
      distance(i, b) = distances[k];
      nearest_index(i, b) = first[k];
      tie_count(i, b) = ties[k];
      for (int a = 0; a < q; ++a) ones_matrices[a](i, b) = ones[k * q + a];
    }
  }
  return List::create(_["distance"] = distance,
                      _["nearest_index"] = nearest_index,
                      _["tie_count"] = tie_count,
                      _["tied_ones"] = tied_ones,
                      _["n_unique_queries"] = unique_queries,
                      _["n_unique_pools"] = nu);
}
