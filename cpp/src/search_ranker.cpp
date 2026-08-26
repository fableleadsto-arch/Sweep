#include "cpp_engine/search_ranker.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <sstream>

namespace sweep {

// ── TF-IDF scoring ──────────────────────────────────────────────────

double tfidf_score(const std::string& query, const std::string& text) {
    auto query_tokens = tokenize(query);
    auto text_tokens = tokenize(text);

    if (query_tokens.empty() || text_tokens.empty()) return 0.0;

    // Build term frequency for text
    std::unordered_map<std::string, int> tf;
    for (const auto& t : text_tokens) {
        tf[t]++;
    }

    double score = 0.0;
    int matched = 0;

    for (const auto& qt : query_tokens) {
        auto it = tf.find(qt);
        if (it != tf.end()) {
            // TF component
            double tf_val = static_cast<double>(it->second) / text_tokens.size();
            // Boost for exact match in title-like positions (first 100 chars)
            std::string lower_text;
            lower_text.reserve(std::min(text.size(), (size_t)200));
            for (size_t i = 0; i < std::min(text.size(), (size_t)200); i++) {
                lower_text += std::tolower(static_cast<unsigned char>(text[i]));
            }
            if (lower_text.find(qt) != std::string::npos) {
                tf_val *= 1.5;
            }
            score += tf_val;
            matched++;
        }
    }

    // Normalize by query length
    if (!query_tokens.empty()) {
        score = (score / query_tokens.size()) * 100.0;
    }

    // Boost for matching more query terms
    score += (static_cast<double>(matched) / query_tokens.size()) * 20.0;

    return std::min(100.0, score);
}

// ── Deduplication ────────────────────────────────────────────────────

static std::string canonical_url(const std::string& url) {
    // Strip fragment, trailing slash, common tracking params
    std::string result = url;
    size_t frag = result.find('#');
    if (frag != std::string::npos) result = result.substr(0, frag);

    // Remove utm_* params (simplified)
    size_t q = result.find('?');
    if (q != std::string::npos) {
        std::string path = result.substr(0, q);
        std::string params = result.substr(q + 1);
        std::string clean_params;
        std::istringstream ss(params);
        std::string param;
        while (std::getline(ss, param, '&')) {
            if (param.find("utm_") != 0 && param.find("ref=") != 0 && param.find("fbclid=") != 0) {
                if (!clean_params.empty()) clean_params += "&";
                clean_params += param;
            }
        }
        result = clean_params.empty() ? path : path + "?" + clean_params;
    }

    // Strip trailing slash
    while (result.length() > 1 && result.back() == '/') {
        result.pop_back();
    }

    return result;
}

std::vector<RankedHit> dedup_hits(const std::vector<RankedHit>& hits) {
    std::vector<RankedHit> result;
    std::unordered_set<std::string> seen;

    for (const auto& hit : hits) {
        std::string key = canonical_url(hit.url);
        if (seen.count(key)) continue;
        seen.insert(key);
        result.push_back(hit);
    }
    return result;
}

// ── Ranking ──────────────────────────────────────────────────────────

std::vector<RankedHit> rank_hits(
    const std::string& query,
    const std::vector<RankedHit>& hits,
    int limit
) {
    std::vector<RankedHit> ranked = hits;

    for (auto& hit : ranked) {
        std::string combined = hit.title + " " + hit.snippet;
        hit.score = tfidf_score(query, combined);
    }

    // Sort by score descending
    std::sort(ranked.begin(), ranked.end(),
        [](const RankedHit& a, const RankedHit& b) { return a.score > b.score; });

    // Deduplicate
    ranked = dedup_hits(ranked);

    // Limit
    if ((int)ranked.size() > limit) {
        ranked.resize(limit);
    }

    return ranked;
}

}  // namespace sweep
