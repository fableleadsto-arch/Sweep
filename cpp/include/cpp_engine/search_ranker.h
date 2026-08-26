#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sweep {

struct RankedHit {
    std::string url;
    std::string title;
    std::string snippet;
    std::string engine;
    double score = 0.0;
};

// Tokenize text into lowercase words.
std::vector<std::string> tokenize(const std::string& text);

// Compute TF-IDF-like score between a query and document text.
double tfidf_score(const std::string& query, const std::string& text);

// Rank a list of search hits by relevance to query.
std::vector<RankedHit> rank_hits(
    const std::string& query,
    const std::vector<RankedHit>& hits,
    int limit = 10
);

// Deduplicate hits by URL (canonical form).
std::vector<RankedHit> dedup_hits(const std::vector<RankedHit>& hits);

}  // namespace sweep
