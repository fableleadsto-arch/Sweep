#include "cpp_engine/html_parser.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <regex>
#include <unordered_set>

namespace sweep {

// ── Entity decoding ──────────────────────────────────────────────────

static std::string decode_entities(const std::string& input) {
    std::string out = input;
    auto replace_all = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = out.find(from, pos)) != std::string::npos) {
            out.replace(pos, from.length(), to);
            pos += to.length();
        }
    };
    replace_all("&nbsp;", " ");
    replace_all("&amp;", "&");
    replace_all("&quot;", "\"");
    replace_all("&lt;", "<");
    replace_all("&gt;", ">");
    replace_all("&mdash;", "\xe2\x80\x94");
    replace_all("&ndash;", "\xe2\x80\x93");
    replace_all("&hellip;", "\xe2\x80\xa6");
    replace_all("&#039;", "'");
    replace_all("&apos;", "'");
    return out;
}

// ── Tag stripping ────────────────────────────────────────────────────

static std::string strip_noise(const std::string& html) {
    static const std::vector<std::pair<std::string, std::string>> patterns = {
        {"<!--", "-->"},
        {"<script", "</script>"},
        {"<style", "</style>"},
        {"<noscript", "</noscript>"},
        {"<svg", "</svg>"},
        {"<iframe", "</iframe>"},
        {"<form", "</form>"},
    };

    std::string out = html;
    for (const auto& [open, close] : patterns) {
        size_t pos = 0;
        while ((pos = out.find(open, pos)) != std::string::npos) {
            size_t end = out.find(close, pos);
            if (end != std::string::npos) {
                end += close.length();
            } else {
                end = pos + open.length();
            }
            out.erase(pos, end - pos);
        }
    }
    return out;
}

std::string html_to_text(const std::string& html) {
    std::string cleaned = strip_noise(html);
    // Strip tags
    std::string result;
    result.reserve(cleaned.size());
    bool in_tag = false;
    for (char c : cleaned) {
        if (c == '<') { in_tag = true; result += ' '; }
        else if (c == '>') { in_tag = false; }
        else if (!in_tag) { result += c; }
    }
    // Collapse whitespace
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
    return decode_entities(trim_copy(out));
}

// ── Helpers ──────────────────────────────────────────────────────────

static std::string trim_copy(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::string get_attr(const std::string& tag, const std::string& attr) {
    std::regex re(attr + R"(=["']([^"']*)["'])", std::regex::icase);
    std::smatch m;
    if (std::regex_search(tag, m, re) && m.size() > 1) {
        return m[1].str();
    }
    return "";
}

static std::string strip_tags(const std::string& s) {
    std::string out;
    bool in_tag = false;
    for (char c : s) {
        if (c == '<') in_tag = true;
        else if (c == '>') in_tag = false;
        else if (!in_tag) out += c;
    }
    return decode_entities(trim_copy(out));
}

// ── Main region detection ────────────────────────────────────────────

std::string extract_main_region(const std::string& html) {
    // Try article, main, or common content divs
    static const std::vector<std::regex> patterns = {
        std::regex(R"(<article[^>]*>([\s\S]*?)</article>)", std::regex::icase),
        std::regex(R"(<main[^>]*>([\s\S]*?)</main>)", std::regex::icase),
    };

    std::string best;
    size_t best_len = 0;

    for (const auto& re : patterns) {
        std::smatch m;
        std::string::const_iterator search_start(html.cbegin());
        while (std::regex_search(search_start, html.cend(), m, re)) {
            std::string text = html_to_text(m[1].str());
            if (text.length() > best_len) {
                best_len = text.length();
                best = m[1].str();
            }
            search_start = m.suffix().first;
        }
    }

    // Also try common content containers
    static std::regex content_re(
        R"(<div[^>]+(?:id|class)="[^"]*(?:post|content|entry|story|body)[^"]*"[^>]*>([\s\S]*?)</div>)",
        std::regex::icase
    );
    std::smatch m;
    std::string::const_iterator si(html.cbegin());
    while (std::regex_search(si, html.cend(), m, content_re)) {
        std::string text = html_to_text(m[1].str());
        if (text.length() > best_len) {
            best_len = text.length();
            best = m[1].str();
        }
        si = m.suffix().first;
    }

    if (!best.empty()) return best;

    // Fallback: body
    static std::regex body_re(R"(<body[^>]*>([\s\S]*?)</body>)", std::regex::icase);
    if (std::regex_search(html, m, body_re)) {
        return m[1].str();
    }
    return html;
}

// ── Metadata extraction ──────────────────────────────────────────────

PageMeta extract_meta(const std::string& html) {
    PageMeta meta;

    // Title
    static std::regex title_re(R"(<title[^>]*>(.*?)</title>)", std::regex::icase);
    std::smatch m;
    if (std::regex_search(html, m, title_re) && m.size() > 1) {
        meta.title = strip_tags(m[1].str());
    }

    // Meta tags
    static std::regex meta_re(
        R"(<meta\s+(?:name|property)="([^"]*)"\s+content="([^"]*)")",
        std::regex::icase
    );
    auto si = html.cbegin();
    while (std::regex_search(si, html.cend(), m, meta_re)) {
        std::string name = m[1].str();
        std::string content = m[2].str();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name == "description") meta.description = content;
        else if (name == "og:site_name" || name == "site-name") meta.site_name = content;
        else if (name == "author") meta.author = content;
        else if (name == "lang") meta.lang = content;
        si = m.suffix().first;
    }

    // HTML lang
    if (meta.lang.empty()) {
        static std::regex lang_re(R"(<html[^>]+lang="([^"]*)")", std::regex::icase);
        if (std::regex_search(html, m, lang_re) && m.size() > 1) {
            meta.lang = m[1].str();
        }
    }

    // Canonical
    static std::regex canon_re(R"(<link[^>]+rel="canonical"[^>]+href="([^"]*)")", std::regex::icase);
    if (std::regex_search(html, m, canon_re) && m.size() > 1) {
        meta.canonical = m[1].str();
    }

    return meta;
}

// ── Link extraction ──────────────────────────────────────────────────

std::vector<Link> extract_links(const std::string& html, const std::string& base_url) {
    std::vector<Link> links;
    std::unordered_set<std::string> seen;

    static std::regex link_re(R"(<a[^>]+href="([^"]*)"[^>]*>([\s\S]*?)</a>)", std::regex::icase);
    std::smatch m;
    auto si = html.cbegin();

    // Link intent patterns
    static const std::vector<std::pair<std::regex, std::string>> intents = {
        {std::regex(R"(\bpricing\b)", std::regex::icase), "pricing"},
        {std::regex(R"(\bdocs?\b|documentation|api\s*reference)", std::regex::icase), "documentation"},
        {std::regex(R"(\bfaq\b|frequently\s+asked)", std::regex::icase), "faq"},
        {std::regex(R"(\babout\b|our\s+story)", std::regex::icase), "about"},
        {std::regex(R"(\bcontact\b|reach\s+us|support\b)", std::regex::icase), "contact"},
        {std::regex(R"(\bgithub\.com\b|source\s+code)", std::regex::icase), "github"},
        {std::regex(R"(\bblog\b|news\b|articles?\b)", std::regex::icase), "blog"},
        {std::regex(R"(\blogin\b|sign\s*in\b)", std::regex::icase), "login"},
        {std::regex(R"(\bsign\s*up\b|register|get\s+started)", std::regex::icase), "signup"},
        {std::regex(R"(\bfeatures?\b|capabilities\b)", std::regex::icase), "features"},
    };

    while (std::regex_search(si, html.cend(), m, link_re)) {
        std::string href = m[1].str();
        std::string text = strip_tags(m[2].str());

        if (href.empty() || href[0] == '#' || href.find("javascript:") == 0) {
            si = m.suffix().first;
            continue;
        }

        // Resolve relative URLs (simplified)
        if (href.find("http") != 0) {
            // Simple resolution against base_url
            if (href[0] == '/') {
                // Extract scheme + host from base_url
                size_t proto_end = base_url.find("://");
                if (proto_end != std::string::npos) {
                    size_t host_end = base_url.find('/', proto_end + 3);
                    href = base_url.substr(0, host_end != std::string::npos ? host_end : base_url.length()) + href;
                }
            }
        }

        if (href.find("http") != 0) {
            si = m.suffix().first;
            continue;
        }

        if (seen.count(href)) {
            si = m.suffix().first;
            continue;
        }
        seen.insert(href);

        // Determine intent
        std::string intent;
        std::string haystack = (text + " " + href).substr(0, 120);
        for (const auto& [re, intent_str] : intents) {
            if (std::regex_search(haystack, re)) {
                intent = intent_str;
                break;
            }
        }

        // Determine external
        bool external = false;
        try {
            size_t proto_end = base_url.find("://");
            size_t host_start = proto_end != std::string::npos ? proto_end + 3 : 0;
            size_t host_end = base_url.find('/', host_start);
            std::string base_host = base_url.substr(host_start, host_end - host_start);

            size_t h_proto_end = href.find("://");
            size_t h_host_start = h_proto_end != std::string::npos ? h_proto_end + 3 : 0;
            size_t h_host_end = href.find('/', h_host_start);
            std::string link_host = href.substr(h_host_start, h_host_end - h_host_start);

            external = (base_host != link_host);
        } catch (...) {}

        links.push_back({href, text, intent, external});
        if (links.size() >= 200) break;

        si = m.suffix().first;
    }

    return links;
}

// ── Markdown heading extraction ──────────────────────────────────────

static std::vector<Heading> headings_from_markdown(const std::string& markdown) {
    std::vector<Heading> headings;
    static std::regex h_re(R"(^(#{1,6})\s+(.*)$)");
    std::smatch m;
    auto si = markdown.cbegin();

    while (std::regex_search(si, markdown.cend(), m, h_re)) {
        int level = static_cast<int>(m[1].str().length());
        std::string text = strip_tags(m[2].str());
        if (text.empty()) { si = m.suffix().first; continue; }

        std::string id;
        for (char c : text) {
            c = std::tolower(static_cast<unsigned char>(c));
            if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '-') {
                id += c;
            }
        }
        // Trim and replace spaces with hyphens
        auto start = id.find_first_not_of(' ');
        auto end = id.find_last_not_of(' ');
        if (start != std::string::npos) {
            id = id.substr(start, end - start + 1);
            std::replace(id.begin(), id.end(), ' ', '-');
        }

        headings.push_back({level, text, id});
        if (headings.size() >= 80) break;
        si = m.suffix().first;
    }
    return headings;
}

// ── Full HTML to Markdown ────────────────────────────────────────────

ParseResult html_to_markdown(const std::string& html, const std::string& url, int max_chars) {
    ParseResult result;
    result.meta = extract_meta(html);
    result.links = extract_links(html, url);

    std::string region = extract_main_region(html);

    // Convert to rough markdown
    static std::regex element_re(
        R"(<(h[1-6]|p|li|pre|blockquote|td|th|tr|br|hr|strong|em|b|i|a|code)[^>]*>([\s\S]*?)</\1>)",
        std::regex::icase
    );

    std::string markdown;
    std::smatch m;
    auto si = region.cbegin();

    while (std::regex_search(si, region.cend(), m, element_re)) {
        std::string tag = m[1].str();
        std::string content = strip_tags(m[2].str());
        if (content.empty()) { si = m.suffix().first; continue; }

        std::string lower_tag = tag;
        std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), ::tolower);

        if (lower_tag[0] == 'h' && lower_tag.length() == 2 && std::isdigit(lower_tag[1])) {
            int level = lower_tag[1] - '0';
            markdown += std::string(level, '#') + " " + content + "\n\n";
        } else if (lower_tag == "li") {
            markdown += "- " + content + "\n";
        } else if (lower_tag == "pre") {
            markdown += "```\n" + content + "\n```\n\n";
        } else if (lower_tag == "blockquote") {
            std::istringstream ss(content);
            std::string line;
            while (std::getline(ss, line)) {
                markdown += "> " + line + "\n";
            }
            markdown += "\n";
        } else if (lower_tag == "br") {
            markdown += "\n";
        } else if (lower_tag == "hr") {
            markdown += "\n---\n\n";
        } else {
            markdown += content + "\n\n";
        }

        si = m.suffix().first;
    }

    // Truncate
    std::string text = html_to_text(markdown);
    result.text = text;
    result.markdown = markdown;

    if ((int)markdown.length() > max_chars) {
        size_t cut = markdown.rfind('.', max_chars);
        if (cut < (size_t)(max_chars / 2)) cut = max_chars;
        markdown = markdown.substr(0, cut + 1);
        result.markdown = markdown;
        result.truncated = true;
    }
    if ((int)text.length() > max_chars) {
        result.text = text.substr(0, max_chars);
        result.truncated = true;
    }

    result.word_count = 0;
    std::istringstream ws(result.text);
    std::string word;
    while (ws >> word) result.word_count++;

    result.headings = headings_from_markdown(result.markdown);

    return result;
}

// ── JSON to Markdown ─────────────────────────────────────────────────

static std::string format_json_value(const std::string& json, int depth = 0) {
    if (depth > 5) return json.substr(0, 200);
    // Simple recursive formatter
    std::string result;
    std::string trimmed = trim_copy(json);
    if (trimmed.empty()) return "";

    if (trimmed[0] == '{' || trimmed[0] == '[') {
        // Object or array — simplified formatting
        return trimmed.substr(0, std::min((size_t)500, trimmed.length()));
    }
    return trimmed;
}

ParseResult json_to_markdown(const std::string& json_str, const std::string& url, int max_chars) {
    ParseResult result;
    result.meta.title = "JSON Response";

    std::string text = format_json_value(json_str);
    result.text = text.substr(0, max_chars);
    result.markdown = text.substr(0, max_chars);
    result.truncated = (int)json_str.length() > max_chars;

    return result;
}

}  // namespace sweep
