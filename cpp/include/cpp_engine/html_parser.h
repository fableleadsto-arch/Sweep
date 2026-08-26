#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace sweep {

struct Link {
    std::string url;
    std::string text;
    std::string intent;
    bool external = false;
};

struct Heading {
    int level;
    std::string text;
    std::string id;
};

struct PageMeta {
    std::string title;
    std::string description;
    std::string site_name;
    std::string author;
    std::string lang;
    std::string canonical;
};

struct ParseResult {
    std::string text;
    std::string markdown;
    std::vector<Link> links;
    std::vector<Heading> headings;
    PageMeta meta;
    int word_count = 0;
    bool truncated = false;
};

// Strip HTML tags and collapse whitespace to plain text.
std::string html_to_text(const std::string& html);

// Pick the main content region (article, main, or body).
std::string extract_main_region(const std::string& html);

// Extract page metadata from meta tags, OpenGraph, JSON-LD.
PageMeta extract_meta(const std::string& html);

// Extract all links with resolved URLs.
std::vector<Link> extract_links(const std::string& html, const std::string& base_url);

// Full HTML to clean Markdown + text + metadata.
ParseResult html_to_markdown(const std::string& html, const std::string& url, int max_chars = 12000);

// JSON to readable Markdown.
ParseResult json_to_markdown(const std::string& json_str, const std::string& url, int max_chars = 12000);

}  // namespace sweep
