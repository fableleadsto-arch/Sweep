#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <any>

namespace sweep {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// ── Numeric Computing ────────────────────────────────────────────────

struct StatsResult {
    double mean = 0;
    double median = 0;
    double std_dev = 0;
    double variance = 0;
    double min_val = 0;
    double max_val = 0;
    int count = 0;
};

StatsResult compute_stats(const Vector& data);

Vector vector_add(const Vector& a, const Vector& b);
Vector vector_subtract(const Vector& a, const Vector& b);
Vector vector_multiply(const Vector& a, const Vector& b);
Vector scalar_multiply(const Vector& a, double scalar);
double dot_product(const Vector& a, const Vector& b);
double vector_norm(const Vector& a);
Vector normalize(const Vector& a);

// ── Matrix Operations ────────────────────────────────────────────────

Matrix matrix_create(int rows, int cols, double fill = 0.0);
Matrix matrix_multiply(const Matrix& a, const Matrix& b);
Matrix matrix_transpose(const Matrix& m);
Matrix matrix_add(const Matrix& a, const Matrix& b);
Matrix matrix_scalar_multiply(const Matrix& m, double scalar);
double matrix_determinant(const Matrix& m);
Matrix matrix_inverse(const Matrix& m);
Vector matrix_vector_multiply(const Matrix& m, const Vector& v);

// ── Linear Algebra ───────────────────────────────────────────────────

struct EigenResult {
    Vector eigenvalues;
    std::vector<Vector> eigenvectors;
};

EigenResult eigen_decomposition(const Matrix& m);
std::vector<double> solve_linear_system(const Matrix& A, const Vector& b);

// ── Signal Processing ────────────────────────────────────────────────

Vector fft_magnitude(const Vector& signal);
Vector convolution(const Vector& signal, const Vector& kernel);

// ── Statistics ───────────────────────────────────────────────────────

double correlation(const Vector& x, const Vector& y);
double covariance(const Vector& x, const Vector& y);
Vector percentile(const Matrix& data, double p);

// ── Random Number Generation ─────────────────────────────────────────

Vector random_normal(int n, double mean = 0.0, double std = 1.0);
Vector random_uniform(int n, double low = 0.0, double high = 1.0);
Matrix random_matrix(int rows, int cols, double low = 0.0, double high = 1.0);

// ── Serialization helpers ────────────────────────────────────────────

std::string vector_to_json(const Vector& v);
std::string matrix_to_json(const Matrix& m);
Vector json_to_vector(const std::string& json);
Matrix json_to_matrix(const std::string& json);

}  // namespace sweep
