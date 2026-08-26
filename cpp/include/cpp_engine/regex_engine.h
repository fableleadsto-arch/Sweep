#pragma once

#include <string>
#include <vector>
#include <regex>

namespace sweep {

struct RegexPattern {
    std::string name;
    std::regex pattern;
};

// Built-in patterns for common extraction tasks.
class PatternSet {
public:
    PatternSet();

    // Email extraction
    std::vector<std::string> find_emails(const std::string& text) const;

    // Phone extraction
    std::vector<std::string> find_phones(const std::string& text) const;

    // URL extraction
    std::vector<std::string> find_urls(const std::string& text) const;

    // Social media URLs
    std::vector<std::string> find_social_urls(const std::string& text) const;

    // Injection detection patterns
    bool has_injection_signals(const std::string& text, std::vector<std::string>& signals) const;

    // CAPTCHA/block detection
    bool looks_blocked(int status, const std::string& text) const;

private:
    std::regex email_re_;
    std::regex phone_re_;
    std::regex url_re_;
    std::regex social_re_;
    std::vector<std::regex> injection_res_;
    std::vector<std::string> captcha_markers_;
};

}  // namespace sweep
