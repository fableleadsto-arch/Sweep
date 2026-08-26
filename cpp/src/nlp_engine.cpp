#include "cpp_engine/nlp_engine.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <regex>

namespace sweep {

// ── Stopwords ────────────────────────────────────────────────────────

static const std::unordered_set<std::string> STOPWORDS = {
    "the","and","for","with","that","this","from","have","what","when",
    "where","about","into","over","their","they","them","there","than",
    "then","will","would","should","could","also","were","been","being",
    "which","while","your","our","some","more","most","other","only",
    "very","not","but","all","can","had","her","was","one","our","out",
    "are","has","his","how","its","may","new","now","old","see","way",
    "who","did","get","let","say","she","too","use","him","his","has",
    "two","way","who","boy","did","get","his","how","man","old","see",
};

// ── Tokenization ─────────────────────────────────────────────────────

std::vector<std::string> tokenize_words(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!word.empty()) { tokens.push_back(word); word.clear(); }
        }
    }
    if (!word.empty()) tokens.push_back(word);
    return tokens;
}

std::vector<std::string> tokenize_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    for (char c : text) {
        current += c;
        if ((c == '.' || c == '!' || c == '?') && current.length() > 10) {
            // Trim leading whitespace
            size_t start = current.find_first_not_of(" \t\n");
            if (start != std::string::npos) {
                sentences.push_back(current.substr(start));
            }
            current.clear();
        }
    }
    if (!current.empty() && current.length() > 10) {
        size_t start = current.find_first_not_of(" \t\n");
        if (start != std::string::npos) sentences.push_back(current.substr(start));
    }
    return sentences;
}

std::vector<Token> tokenize_full(const std::string& text) {
    std::vector<Token> tokens;
    auto words = tokenize_words(text);
    int pos = 0;
    for (const auto& w : words) {
        Token t;
        t.text = w;
        t.lemma = w;  // Simplified
        t.pos = "NOUN";  // Simplified
        t.start = text.find(w, pos);
        t.end = t.start + w.length();
        pos = t.end;
        tokens.push_back(t);
    }
    return tokens;
}

// ── Entity Extraction ────────────────────────────────────────────────

std::vector<Entity> extract_entities(const std::string& text) {
    std::vector<Entity> entities;

    // Email extraction
    auto emails = extract_emails(text);
    entities.insert(entities.end(), emails.begin(), emails.end());

    // Phone extraction
    auto phones = extract_phones(text);
    entities.insert(entities.end(), phones.begin(), phones.end());

    // URL extraction
    auto urls = extract_urls(text);
    entities.insert(entities.end(), urls.begin(), urls.end());

    // Date extraction
    auto dates = extract_dates(text);
    entities.insert(entities.end(), dates.begin(), dates.end());

    // Simple capitalized word extraction (likely names/orgs)
    std::regex name_re(R"(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))");
    std::smatch m;
    auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, name_re)) {
        Entity e;
        e.text = m[1].str();
        e.label = "PERSON";
        e.start = m.position();
        e.end = e.start + e.length();
        e.confidence = 0.6;
        entities.push_back(e);
        si = m.suffix().first;
        if (entities.size() > 50) break;
    }

    return entities;
}

std::vector<Entity> extract_emails(const std::string& text) {
    std::vector<Entity> entities;
    std::regex re(R"([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})");
    std::smatch m;
    auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, re)) {
        Entity e;
        e.text = m[0].str();
        e.label = "EMAIL";
        e.start = m.position();
        e.end = e.start + e.length();
        e.confidence = 0.95;
        entities.push_back(e);
        si = m.suffix().first;
        if (entities.size() >= 20) break;
    }
    return entities;
}

std::vector<Entity> extract_phones(const std::string& text) {
    std::vector<Entity> entities;
    std::regex re(R"(\+?\d[\d\s().-]{7,17}\d)");
    std::smatch m;
    auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, re)) {
        std::string phone = m[0].str();
        std::string digits;
        for (char c : phone) {
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '+') digits += c;
        }
        if (digits.length() >= 8 && digits.length() <= 18) {
            Entity e;
            e.text = phone;
            e.label = "PHONE";
            e.start = m.position();
            e.end = e.start + e.length();
            e.confidence = 0.8;
            entities.push_back(e);
        }
        si = m.suffix().first;
        if (entities.size() >= 10) break;
    }
    return entities;
}

std::vector<Entity> extract_urls(const std::string& text) {
    std::vector<Entity> entities;
    std::regex re(R"(https?://[^\s"'<>]+)");
    std::smatch m;
    auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, re)) {
        Entity e;
        e.text = m[0].str();
        e.label = "URL";
        e.start = m.position();
        e.end = e.start + e.length();
        e.confidence = 0.99;
        entities.push_back(e);
        si = m.suffix().first;
        if (entities.size() >= 30) break;
    }
    return entities;
}

std::vector<Entity> extract_dates(const std::string& text) {
    std::vector<Entity> entities;
    // Simple date patterns
    std::vector<std::regex> date_res = {
        std::regex(R"(\d{4}-\d{2}-\d{2})"),           // 2024-01-15
        std::regex(R"(\d{1,2}/\d{1,2}/\d{4})"),       // 01/15/2024
        std::regex(R"(\w+ \d{1,2},? \d{4})"),          // January 15, 2024
        std::regex(R"(\d{1,2} \w+ \d{4})"),             // 15 January 2024
    };

    for (const auto& re : date_res) {
        std::smatch m;
        auto si = text.cbegin();
        while (std::regex_search(si, text.cend(), m, re)) {
            Entity e;
            e.text = m[0].str();
            e.label = "DATE";
            e.start = m.position();
            e.end = e.start + e.length();
            e.confidence = 0.7;
            entities.push_back(e);
            si = m.suffix().first;
            if (entities.size() >= 10) break;
        }
    }
    return entities;
}

// ── Text Analysis ────────────────────────────────────────────────────

std::vector<std::string> extract_keywords(const std::string& text, int top_n) {
    auto words = tokenize_words(text);
    std::unordered_map<std::string, int> freq;
    for (const auto& w : words) {
        if (STOPWORDS.count(w) || w.length() < 3) continue;
        freq[w]++;
    }

    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_words(freq.begin(), freq.end());
    std::sort(sorted_words.begin(), sorted_words.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<std::string> keywords;
    for (int i = 0; i < std::min(top_n, static_cast<int>(sorted_words.size())); i++) {
        keywords.push_back(sorted_words[i].first);
    }
    return keywords;
}

double text_similarity(const std::string& a, const std::string& b) {
    auto words_a = tokenize_words(a);
    auto words_b = tokenize_words(b);

    std::unordered_set<std::string> set_a(words_a.begin(), words_a.end());
    std::unordered_set<std::string> set_b(words_b.begin(), words_b.end());

    int intersection = 0;
    for (const auto& w : set_a) {
        if (set_b.count(w)) intersection++;
    }

    int union_size = set_a.size() + set_b.size() - intersection;
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0;
}

std::string summarize(const std::string& text, int max_sentences) {
    auto sentences = tokenize_sentences(text);
    if ((int)sentences.size() <= max_sentences) return text;

    // Simple extractive summarization: take first N sentences
    std::string summary;
    for (int i = 0; i < max_sentences && i < (int)sentences.size(); i++) {
        summary += sentences[i] + " ";
    }
    return summary;
}

std::string sentiment(const std::string& text) {
    // Simple lexicon-based sentiment
    static const std::unordered_set<std::string> positive = {
        "good","great","excellent","amazing","wonderful","fantastic","love",
        "happy","best","beautiful","perfect","awesome","brilliant","outstanding",
        "superb","magnificent","terrific","pleasant","delightful","nice",
    };
    static const std::unordered_set<std::string> negative = {
        "bad","terrible","awful","horrible","worst","hate","ugly","poor",
        "disappointing","failed","broken","wrong","error","problem","issue",
        "difficult","hard","struggle","fail","crash","bug","slow","laggy",
    };

    auto words = tokenize_words(text);
    int pos = 0, neg = 0;
    for (const auto& w : words) {
        if (positive.count(w)) pos++;
        if (negative.count(w)) neg++;
    }

    if (pos > neg + 2) return "positive";
    if (neg > pos + 2) return "negative";
    return "neutral";
}

// ── Text Processing ──────────────────────────────────────────────────

std::string stem(const std::string& word) {
    // Simple suffix stripping
    std::string w = word;
    if (w.length() > 4) {
        if (w.substr(w.length() - 3) == "ing") w = w.substr(0, w.length() - 3);
        else if (w.substr(w.length() - 3) == "tion") w = w.substr(0, w.length() - 4);
        else if (w.substr(w.length() - 2) == "ed") w = w.substr(0, w.length() - 2);
        else if (w.substr(w.length() - 2) == "ly") w = w.substr(0, w.length() - 2);
        else if (w.substr(w.length() - 3) == "ies") w = w.substr(0, w.length() - 3) + "y";
        else if (w.back() == 's' && w.length() > 4) w.pop_back();
    }
    return w;
}

std::string lemmatize(const std::string& word) {
    // Simplified lemmatization (stem is close enough for most cases)
    return stem(word);
}

std::string to_lower(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string remove_stopwords(const std::string& text) {
    auto words = tokenize_words(text);
    std::string result;
    for (const auto& w : words) {
        if (!STOPWORDS.count(w) && w.length() >= 3) {
            if (!result.empty()) result += " ";
            result += w;
        }
    }
    return result;
}

std::string normalize_text(const std::string& text) {
    std::string result = to_lower(text);
    // Remove extra whitespace
    std::string out;
    bool prev_space = false;
    for (char c : result) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) out += ' ';
            prev_space = true;
        } else {
            out += c;
            prev_space = false;
        }
    }
    return out;
}

// ── TF-IDF ───────────────────────────────────────────────────────────

TFIDFResult compute_tfidf(const std::vector<std::string>& documents) {
    TFIDFResult result;

    // Build vocabulary
    std::unordered_map<std::string, int> vocab_map;
    for (const auto& doc : documents) {
        auto words = tokenize_words(doc);
        std::unordered_set<std::string> unique(words.begin(), words.end());
        for (const auto& w : unique) {
            if (STOPWORDS.count(w) || w.length() < 3) continue;
            if (!vocab_map.count(w)) {
                vocab_map[w] = result.vocabulary.size();
                result.vocabulary.push_back(w);
            }
        }
    }

    int n_docs = documents.size();
    int n_terms = result.vocabulary.size();

    // Compute IDF
    std::vector<double> idf(n_terms, 0);
    for (const auto& doc : documents) {
        auto words = tokenize_words(doc);
        std::unordered_set<std::string> unique(words.begin(), words.end());
        for (const auto& w : unique) {
            if (vocab_map.count(w)) idf[vocab_map[w]]++;
        }
    }
    for (auto& v : idf) {
        v = std::log((1.0 + n_docs) / (1.0 + v)) + 1.0;
    }

    // Compute TF-IDF matrix
    result.matrix.resize(n_docs, std::vector<double>(n_terms, 0));
    for (int i = 0; i < n_docs; i++) {
        auto words = tokenize_words(documents[i]);
        std::unordered_map<std::string, int> tf;
        for (const auto& w : words) tf[w]++;

        for (const auto& [word, count] : tf) {
            if (vocab_map.count(word)) {
                int j = vocab_map[word];
                double tf_val = static_cast<double>(count) / words.size();
                result.matrix[i][j] = tf_val * idf[j];
            }
        }
    }

    return result;
}

std::vector<double> vectorize_text(const std::string& text, const std::vector<std::string>& vocabulary) {
    auto words = tokenize_words(text);
    std::unordered_map<std::string, int> tf;
    for (const auto& w : words) tf[w]++;

    std::vector<double> vec(vocabulary.size(), 0);
    for (size_t i = 0; i < vocabulary.size(); i++) {
        if (tf.count(vocabulary[i])) {
            vec[i] = static_cast<double>(tf[vocabulary[i]]) / words.size();
        }
    }
    return vec;
}

}  // namespace sweep
