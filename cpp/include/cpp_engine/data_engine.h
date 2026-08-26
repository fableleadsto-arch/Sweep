#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

namespace sweep {

using Value = std::variant<double, std::string, bool, std::monostate>;
using Row = std::vector<Value>;
using Column = std::vector<Value>;

struct DataFrame {
    std::vector<std::string> columns;
    std::vector<Row> rows;
    std::unordered_map<std::string, Column> column_data;

    int num_rows() const { return rows.size(); }
    int num_cols() const { return columns.size(); }

    // Get column as numeric values (skipping non-numeric)
    std::vector<double> get_numeric(const std::string& col) const;

    // Get column as strings
    std::vector<std::string> get_strings(const std::string& col) const;
};

// ── CSV Parsing ──────────────────────────────────────────────────────

DataFrame parse_csv(const std::string& csv_text, bool has_header = true);
std::string dataframe_to_csv(const DataFrame& df);

// ── DataFrame Operations ─────────────────────────────────────────────

DataFrame df_head(const DataFrame& df, int n = 5);
DataFrame df_tail(const DataFrame& df, int n = 5);
DataFrame df_select(const DataFrame& df, const std::vector<std::string>& columns);
DataFrame df_filter(const DataFrame& df, const std::string& column, const std::string& op, double value);
DataFrame df_sort(const DataFrame& df, const std::string& column, bool ascending = true);
DataFrame df_group_by(const DataFrame& df, const std::string& column, const std::string& agg_func);

// ── Statistics ───────────────────────────────────────────────────────

struct ColumnStats {
    std::string name;
    int count = 0;
    int null_count = 0;
    double mean = 0;
    double std_dev = 0;
    double min_val = 0;
    double max_val = 0;
    double median = 0;
};

std::vector<ColumnStats> df_describe(const DataFrame& df);
std::vector<std::vector<double>> df_correlation(const DataFrame& df);

// ── Aggregation ──────────────────────────────────────────────────────

double column_sum(const Column& col);
double column_mean(const Column& col);
double column_min(const Column& col);
double column_max(const Column& col);
double column_std(const Column& col);

// ── Cleaning ─────────────────────────────────────────────────────────

DataFrame df_fill_na(const DataFrame& df, const std::string& column, const Value& fill_value);
DataFrame df_drop_na(const DataFrame& df);
DataFrame df_drop_duplicates(const DataFrame& df);
DataFrame df_rename(const DataFrame& df, const std::string& old_name, const std::string& new_name);

}  // namespace sweep
