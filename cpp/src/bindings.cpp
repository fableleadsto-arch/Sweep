#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp_engine/html_parser.h"
#include "cpp_engine/text_extractor.h"
#include "cpp_engine/search_ranker.h"
#include "cpp_engine/regex_engine.h"
#include "cpp_engine/ml_engine.h"
#include "cpp_engine/data_engine.h"
#include "cpp_engine/nlp_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(sweep_engine, m) {
    m.doc() = "Sweep C++ performance engine — ML, data, NLP, search, extraction";

    // ── HTML Parser ──────────────────────────────────────────────

    py::class_<sweep::Link>(m, "Link")
        .def_readwrite("url", &sweep::Link::url)
        .def_readwrite("text", &sweep::Link::text)
        .def_readwrite("intent", &sweep::Link::intent)
        .def_readwrite("external", &sweep::Link::external);

    py::class_<sweep::Heading>(m, "Heading")
        .def_readwrite("level", &sweep::Heading::level)
        .def_readwrite("text", &sweep::Heading::text)
        .def_readwrite("id", &sweep::Heading::id);

    py::class_<sweep::PageMeta>(m, "PageMeta")
        .def_readwrite("title", &sweep::PageMeta::title)
        .def_readwrite("description", &sweep::PageMeta::description)
        .def_readwrite("site_name", &sweep::PageMeta::site_name)
        .def_readwrite("author", &sweep::PageMeta::author)
        .def_readwrite("lang", &sweep::PageMeta::lang)
        .def_readwrite("canonical", &sweep::PageMeta::canonical);

    py::class_<sweep::ParseResult>(m, "ParseResult")
        .def_readwrite("text", &sweep::ParseResult::text)
        .def_readwrite("markdown", &sweep::ParseResult::markdown)
        .def_readwrite("links", &sweep::ParseResult::links)
        .def_readwrite("headings", &sweep::ParseResult::headings)
        .def_readwrite("meta", &sweep::ParseResult::meta)
        .def_readwrite("word_count", &sweep::ParseResult::word_count)
        .def_readwrite("truncated", &sweep::ParseResult::truncated);

    m.def("html_to_text", &sweep::html_to_text, "Strip HTML tags to plain text");
    m.def("html_to_markdown", &sweep::html_to_markdown,
          py::arg("html"), py::arg("url"), py::arg("max_chars") = 12000,
          "Convert HTML to Markdown with metadata");
    m.def("json_to_markdown", &sweep::json_to_markdown,
          py::arg("json_str"), py::arg("url"), py::arg("max_chars") = 12000,
          "Convert JSON to Markdown");

    // ── Text Extractor ──────────────────────────────────────────

    py::class_<sweep::ExtractedFields>(m, "ExtractedFields")
        .def_readwrite("emails", &sweep::ExtractedFields::emails)
        .def_readwrite("phones", &sweep::ExtractedFields::phones)
        .def_readwrite("social_urls", &sweep::ExtractedFields::social_urls)
        .def_readwrite("fields_found", &sweep::ExtractedFields::fields_found);

    m.def("extract_heuristic", &sweep::extract_heuristic, "Extract emails/phones/URLs");
    m.def("score_relevance", &sweep::score_relevance);
    m.def("extract_top_sentences", &sweep::extract_top_sentences);
    m.def("detect_injection", &sweep::detect_injection);

    // ── Search Ranker ───────────────────────────────────────────

    py::class_<sweep::RankedHit>(m, "RankedHit")
        .def_readwrite("url", &sweep::RankedHit::url)
        .def_readwrite("title", &sweep::RankedHit::title)
        .def_readwrite("snippet", &sweep::RankedHit::snippet)
        .def_readwrite("engine", &sweep::RankedHit::engine)
        .def_readwrite("score", &sweep::RankedHit::score);

    m.def("tokenize", &sweep::tokenize);
    m.def("tfidf_score", &sweep::tfidf_score);
    m.def("rank_hits", &sweep::rank_hits);
    m.def("dedup_hits", &sweep::dedup_hits);

    // ── Regex Engine ────────────────────────────────────────────

    py::class_<sweep::PatternSet>(m, "PatternSet")
        .def(py::init<>())
        .def("find_emails", &sweep::PatternSet::find_emails)
        .def("find_phones", &sweep::PatternSet::find_phones)
        .def("find_urls", &sweep::PatternSet::find_urls)
        .def("find_social_urls", &sweep::PatternSet::find_social_urls)
        .def("has_injection_signals", &sweep::PatternSet::has_injection_signals)
        .def("looks_blocked", &sweep::PatternSet::looks_blocked);

    // ── ML Engine ───────────────────────────────────────────────

    py::class_<sweep::StatsResult>(m, "StatsResult")
        .def_readwrite("mean", &sweep::StatsResult::mean)
        .def_readwrite("median", &sweep::StatsResult::median)
        .def_readwrite("std_dev", &sweep::StatsResult::std_dev)
        .def_readwrite("variance", &sweep::StatsResult::variance)
        .def_readwrite("min_val", &sweep::StatsResult::min_val)
        .def_readwrite("max_val", &sweep::StatsResult::max_val)
        .def_readwrite("count", &sweep::StatsResult::count);

    m.def("compute_stats", &sweep::compute_stats, "Compute statistics for a vector");
    m.def("vector_add", &sweep::vector_add);
    m.def("vector_subtract", &sweep::vector_subtract);
    m.def("vector_multiply", &sweep::vector_multiply);
    m.def("scalar_multiply", &sweep::scalar_multiply);
    m.def("dot_product", &sweep::dot_product);
    m.def("vector_norm", &sweep::vector_norm);
    m.def("normalize", &sweep::normalize);
    m.def("matrix_multiply", &sweep::matrix_multiply);
    m.def("matrix_transpose", &sweep::matrix_transpose);
    m.def("matrix_add", &sweep::matrix_add);
    m.def("matrix_determinant", &sweep::matrix_determinant);
    m.def("matrix_inverse", &sweep::matrix_inverse);
    m.def("matrix_vector_multiply", &sweep::matrix_vector_multiply);
    m.def("correlation", &sweep::correlation);
    m.def("covariance", &sweep::covariance);
    m.def("fft_magnitude", &sweep::fft_magnitude);
    m.def("convolution", &sweep::convolution);
    m.def("random_normal", &sweep::random_normal);
    m.def("random_uniform", &sweep::random_uniform);
    m.def("random_matrix", &sweep::random_matrix);
    m.def("vector_to_json", &sweep::vector_to_json);
    m.def("matrix_to_json", &sweep::matrix_to_json);
    m.def("json_to_vector", &sweep::json_to_vector);
    m.def("json_to_matrix", &sweep::json_to_matrix);

    // ── Data Engine ─────────────────────────────────────────────

    py::class_<sweep::DataFrame>(m, "DataFrame")
        .def_readwrite("columns", &sweep::DataFrame::columns)
        .def_readwrite("rows", &sweep::DataFrame::rows)
        .def("num_rows", &sweep::DataFrame::num_rows)
        .def("num_cols", &sweep::DataFrame::num_cols)
        .def("get_numeric", &sweep::DataFrame::get_numeric)
        .def("get_strings", &sweep::DataFrame::get_strings);

    py::class_<sweep::ColumnStats>(m, "ColumnStats")
        .def_readwrite("name", &sweep::ColumnStats::name)
        .def_readwrite("count", &sweep::ColumnStats::count)
        .def_readwrite("null_count", &sweep::ColumnStats::null_count)
        .def_readwrite("mean", &sweep::ColumnStats::mean)
        .def_readwrite("std_dev", &sweep::ColumnStats::std_dev)
        .def_readwrite("min_val", &sweep::ColumnStats::min_val)
        .def_readwrite("max_val", &sweep::ColumnStats::max_val)
        .def_readwrite("median", &sweep::ColumnStats::median);

    m.def("parse_csv", &sweep::parse_csv);
    m.def("dataframe_to_csv", &sweep::dataframe_to_csv);
    m.def("df_head", &sweep::df_head);
    m.def("df_tail", &sweep::df_tail);
    m.def("df_select", &sweep::df_select);
    m.def("df_filter", &sweep::df_filter);
    m.def("df_sort", &sweep::df_sort);
    m.def("df_describe", &sweep::df_describe);
    m.def("df_correlation", &sweep::df_correlation);
    m.def("df_fill_na", &sweep::df_fill_na);
    m.def("df_drop_na", &sweep::df_drop_na);
    m.def("df_drop_duplicates", &sweep::df_drop_duplicates);
    m.def("df_rename", &sweep::df_rename);
    m.def("column_sum", &sweep::column_sum);
    m.def("column_mean", &sweep::column_mean);
    m.def("column_min", &sweep::column_min);
    m.def("column_max", &sweep::column_max);
    m.def("column_std", &sweep::column_std);

    // ── NLP Engine ──────────────────────────────────────────────

    py::class_<sweep::Token>(m, "Token")
        .def_readwrite("text", &sweep::Token::text)
        .def_readwrite("lemma", &sweep::Token::lemma)
        .def_readwrite("pos", &sweep::Token::pos)
        .def_readwrite("ner", &sweep::Token::ner)
        .def_readwrite("start", &sweep::Token::start)
        .def_readwrite("end", &sweep::Token::end);

    py::class_<sweep::Entity>(m, "Entity")
        .def_readwrite("text", &sweep::Entity::text)
        .def_readwrite("label", &sweep::Entity::label)
        .def_readwrite("start", &sweep::Entity::start)
        .def_readwrite("end", &sweep::Entity::end)
        .def_readwrite("confidence", &sweep::Entity::confidence);

    py::class_<sweep::NLPResult>(m, "NLPResult")
        .def_readwrite("tokens", &sweep::NLPResult::tokens)
        .def_readwrite("entities", &sweep::NLPResult::entities)
        .def_readwrite("sentences", &sweep::NLPResult::sentences)
        .def_readwrite("keywords", &sweep::NLPResult::keywords)
        .def_readwrite("language", &sweep::NLPResult::language);

    py::class_<sweep::TFIDFResult>(m, "TFIDFResult")
        .def_readwrite("vocabulary", &sweep::TFIDFResult::vocabulary)
        .def_readwrite("matrix", &sweep::TFIDFResult::matrix);

    m.def("tokenize_words", &sweep::tokenize_words);
    m.def("tokenize_sentences", &sweep::tokenize_sentences);
    m.def("tokenize_full", &sweep::tokenize_full);
    m.def("cpp_extract_entities", &sweep::extract_entities);
    m.def("cpp_extract_emails", &sweep::extract_emails);
    m.def("cpp_extract_phones", &sweep::extract_phones);
    m.def("cpp_extract_urls", &sweep::extract_urls);
    m.def("extract_keywords", &sweep::extract_keywords);
    m.def("text_similarity", &sweep::text_similarity);
    m.def("summarize", &sweep::summarize);
    m.def("sentiment", &sweep::sentiment);
    m.def("stem", &sweep::stem);
    m.def("lemmatize", &sweep::lemmatize);
    m.def("to_lower", &sweep::to_lower);
    m.def("remove_stopwords", &sweep::remove_stopwords);
    m.def("normalize_text", &sweep::normalize_text);
    m.def("compute_tfidf", &sweep::compute_tfidf);
    m.def("vectorize_text", &sweep::vectorize_text);
}
