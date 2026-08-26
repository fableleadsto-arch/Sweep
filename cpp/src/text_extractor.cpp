#include "cpp_engine/text_extractor.h"
#include "cpp_engine/regex_engine.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <unordered_set>

namespace sweep {

// ── Heuristic extraction ─────────────────────────────────────────────

ExtractedFields extract_heuristic(const std::string& text) {
    ExtractedFields fields;
    PatternSet patterns;

    fields.emails = patterns.find_emails(text);
    fields.phones = patterns.find_phones(text);
    fields.social_urls = patterns.find_social_urls(text);

    fields.fields_found = fields.emails.size() + fields.phones.size() + fields.social_urls.size();

    return fields;
}

// ── Relevance scoring ────────────────────────────────────────────────

static const std::unordered_set<std::string> STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "what", "when",
    "where", "about", "into", "over", "their", "they", "them", "there", "than",
    "then", "will", "would", "should", "could", "also", "were", "been", "being",
    "which", "while", "your", "our", "some", "more", "most", "other", "only",
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!word.empty() && word.length() > 2) {
                tokens.push_back(word);
            }
            word.clear();
        }
    }
    if (!word.empty() && word.length() > 2) {
        tokens.push_back(word);
    }
    return tokens;
}

double score_relevance(const std::string& text, const std::vector<std::string>& keywords) {
    if (keywords.empty()) return 0.5;

    std::string lower_text;
    lower_text.reserve(text.size());
    for (char c : text) {
        lower_text += std::tolower(static_cast<unsigned char>(c));
    }

    int hits = 0;
    for (const auto& kw : keywords) {
        if (lower_text.find(kw) != std::string::npos) {
            hits++;
        }
    }

    return std::min(1.0, static_cast<double>(hits) / std::max(keywords.size() * 0.3, 1.0));
}

// ── Sentence extraction ──────────────────────────────────────────────

struct ScoredSentence {
    std::string sentence;
    int index;
    double score;
};

std::vector<std::string> extract_top_sentences(
    const std::string& text,
    const std::vector<std::string>& keywords,
    int max_sentences,
    int max_chars
) {
    // Split into sentences
    std::vector<std::string> sentences;
    std::string current;
    for (char c : text) {
        current += c;
        if ((c == '.' || c == '!' || c == '?') && current.length() >= 40) {
            // Trim
            size_t start = current.find_first_not_of(" \t\n");
            if (start != std::string::npos) {
                sentences.push_back(current.substr(start));
            }
            current.clear();
        } else if (c == '\n' && current.length() >= 40) {
            size_t start = current.find_first_not_of(" \t\n");
            if (start != std::string::npos) {
                sentences.push_back(current.substr(start));
            }
            current.clear();
        }
    }
    if (!current.empty() && current.length() >= 40) {
        size_t start = current.find_first_not_of(" \t\n");
        if (start != std::string::npos) {
            sentences.push_back(current.substr(start));
        }
    }

    // Score each sentence
    std::vector<ScoredSentence> scored;
    for (int i = 0; i < (int)sentences.size() && i < max_sentences * 3; i++) {
        const auto& s = sentences[i];
        if ((int)s.length() > max_chars) continue;

        double score = score_relevance(s, keywords);

        // Bonus for numbers (concrete detail)
        bool has_numbers = false;
        for (char c : s) {
            if (std::isdigit(static_cast<unsigned char>(c))) { has_numbers = true; break; }
        }
        if (has_numbers) score += 0.5;

        // Bonus for length (more detail)
        if (s.length() >= 120) score += 0.25;

        // Bonus for early position (page leads)
        if (i < 4) score += 0.25;

        scored.push_back({s, i, score});
    }

    // Sort by score descending
    std::sort(scored.begin(), scored.end(),
        [](const ScoredSentence& a, const ScoredSentence& b) { return a.score > b.score; });

    // Take top N
    std::vector<std::string> result;
    for (int i = 0; i < (int)scored.size() && i < max_sentences; i++) {
        result.push_back(scored[i].sentence);
    }
    return result;
}

// ── Injection detection ──────────────────────────────────────────────

bool detect_injection(const std::string& text, std::vector<std::string>& signals) {
    PatternSet patterns;
    return patterns.has_injection_signals(text, signals);
}

}  // namespace sweep
