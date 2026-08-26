#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sweep {

struct ExtractedFields {
    std::unordered_map<std::string, std::string> data;
    std::vector<std::string> emails;
    std::vector<std::string> phones;
    std::vector<std::string> social_urls;
    int fields_found = 0;
};

// Extract emails, phones, social URLs from text using fast regex.
ExtractedFields extract_heuristic(const std::string& text);

// Score how relevant text is to a set of keywords (fast TF-IDF-like).
double score_relevance(const std::string& text, const std::vector<std::string>& keywords);

// Extract the best sentences from a page given an objective.
std::vector<std::string> extract_top_sentences(
    const std::string& text,
    const std::vector<std::string>& keywords,
    int max_sentences = 8,
    int max_chars = 320
);

// Detect prompt injection patterns in text.
bool detect_injection(const std::string& text, std::vector<std::string>& signals);

}  // namespace sweep
