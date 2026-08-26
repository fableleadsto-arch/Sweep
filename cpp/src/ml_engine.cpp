#include "cpp_engine/ml_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sweep {

static std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// ── Stats ────────────────────────────────────────────────────────────

StatsResult compute_stats(const Vector& data) {
    StatsResult r;
    if (data.empty()) return r;
    r.count = data.size();
    r.min_val = *std::min_element(data.begin(), data.end());
    r.max_val = *std::max_element(data.begin(), data.end());
    r.mean = std::accumulate(data.begin(), data.end(), 0.0) / r.count;

    double sq_sum = 0;
    for (double x : data) sq_sum += (x - r.mean) * (x - r.mean);
    r.variance = sq_sum / r.count;
    r.std_dev = std::sqrt(r.variance);

    Vector sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t mid = sorted.size() / 2;
    r.median = sorted.size() % 2 == 0
        ? (sorted[mid - 1] + sorted[mid]) / 2.0
        : sorted[mid];

    return r;
}

// ── Vector ops ───────────────────────────────────────────────────────

Vector vector_add(const Vector& a, const Vector& b) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++) r[i] = a[i] + b[i];
    return r;
}

Vector vector_subtract(const Vector& a, const Vector& b) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++) r[i] = a[i] - b[i];
    return r;
}

Vector vector_multiply(const Vector& a, const Vector& b) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size() && i < b.size(); i++) r[i] = a[i] * b[i];
    return r;
}

Vector scalar_multiply(const Vector& a, double scalar) {
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] * scalar;
    return r;
}

double dot_product(const Vector& a, const Vector& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) sum += a[i] * b[i];
    return sum;
}

double vector_norm(const Vector& a) {
    return std::sqrt(dot_product(a, a));
}

Vector normalize(const Vector& a) {
    double n = vector_norm(a);
    if (n < 1e-10) return Vector(a.size(), 0.0);
    return scalar_multiply(a, 1.0 / n);
}

// ── Matrix ops ───────────────────────────────────────────────────────

Matrix matrix_create(int rows, int cols, double fill) {
    return Matrix(rows, Vector(cols, fill));
}

Matrix matrix_multiply(const Matrix& a, const Matrix& b) {
    int rows = a.size(), cols = b[0].size(), inner = b.size();
    Matrix result = matrix_create(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < inner; k++)
                result[i][j] += a[i][k] * b[k][j];
    return result;
}

Matrix matrix_transpose(const Matrix& m) {
    if (m.empty()) return {};
    Matrix result = matrix_create(m[0].size(), m.size());
    for (size_t i = 0; i < m.size(); i++)
        for (size_t j = 0; j < m[i].size(); j++)
            result[j][i] = m[i][j];
    return result;
}

Matrix matrix_add(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    for (size_t i = 0; i < a.size(); i++)
        for (size_t j = 0; j < a[i].size(); j++)
            result[i][j] += b[i][j];
    return result;
}

Matrix matrix_scalar_multiply(const Matrix& m, double scalar) {
    Matrix result = m;
    for (auto& row : result)
        for (auto& val : row)
            val *= scalar;
    return result;
}

double matrix_determinant(const Matrix& m) {
    int n = m.size();
    if (n == 1) return m[0][0];
    if (n == 2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];

    double det = 0;
    for (int col = 0; col < n; col++) {
        Matrix minor_matrix;
        for (int i = 1; i < n; i++) {
            Vector row;
            for (int j = 0; j < n; j++) {
                if (j != col) row.push_back(m[i][j]);
            }
            minor_matrix.push_back(row);
        }
        det += (col % 2 == 0 ? 1 : -1) * m[0][col] * matrix_determinant(minor_matrix);
    }
    return det;
}

Matrix matrix_inverse(const Matrix& m) {
    int n = m.size();
    Matrix augmented = m;
    Matrix identity = matrix_create(n, n);
    for (int i = 0; i < n; i++) identity[i][i] = 1.0;

    for (int col = 0; col < n; col++) {
        // Find pivot
        int max_row = col;
        for (int row = col + 1; row < n; row++) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[max_row][col]))
                max_row = row;
        }
        std::swap(augmented[col], augmented[max_row]);
        std::swap(identity[col], identity[max_row]);

        double pivot = augmented[col][col];
        if (std::abs(pivot) < 1e-10) throw std::runtime_error("Matrix is singular");

        for (int j = 0; j < n; j++) {
            augmented[col][j] /= pivot;
            identity[col][j] /= pivot;
        }

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = augmented[row][col];
            for (int j = 0; j < n; j++) {
                augmented[row][j] -= factor * augmented[col][j];
                identity[row][j] -= factor * identity[col][j];
            }
        }
    }
    return identity;
}

Vector matrix_vector_multiply(const Matrix& m, const Vector& v) {
    Vector result(m.size());
    for (size_t i = 0; i < m.size(); i++)
        result[i] = dot_product(m[i], v);
    return result;
}

// ── Linear Algebra ───────────────────────────────────────────────────

EigenResult eigen_decomposition(const Matrix& m) {
    // Simplified: for 2x2 matrices only (full QR needed for larger)
    EigenResult r;
    int n = m.size();
    if (n == 2) {
        double a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
        double trace = a + d;
        double det = a * d - b * c;
        double disc = trace * trace - 4 * det;
        if (disc >= 0) {
            r.eigenvalues = {(trace + std::sqrt(disc)) / 2, (trace - std::sqrt(disc)) / 2};
        } else {
            r.eigenvalues = {trace / 2, trace / 2};
        }
    }
    return r;
}

std::vector<double> solve_linear_system(const Matrix& A, const Vector& b) {
    Matrix inv = matrix_inverse(A);
    return matrix_vector_multiply(inv, b);
}

// ── Signal Processing ────────────────────────────────────────────────

Vector fft_magnitude(const Vector& signal) {
    // Simplified DFT (not true FFT)
    int n = signal.size();
    Vector magnitudes(n / 2 + 1);
    for (int k = 0; k <= n / 2; k++) {
        double real = 0, imag = 0;
        for (int t = 0; t < n; t++) {
            double angle = 2.0 * M_PI * k * t / n;
            real += signal[t] * std::cos(angle);
            imag -= signal[t] * std::sin(angle);
        }
        magnitudes[k] = std::sqrt(real * real + imag * imag) / n;
    }
    return magnitudes;
}

Vector convolution(const Vector& signal, const Vector& kernel) {
    int n = signal.size();
    int k = kernel.size();
    Vector result(n + k - 1, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            result[i + j] += signal[i] * kernel[j];
    return result;
}

// ── Statistics ───────────────────────────────────────────────────────

double correlation(const Vector& x, const Vector& y) {
    if (x.size() != y.size() || x.empty()) return 0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double my = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    double num = 0, dx = 0, dy = 0;
    for (size_t i = 0; i < x.size(); i++) {
        double a = x[i] - mx, b = y[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    double denom = std::sqrt(dx * dy);
    return denom < 1e-10 ? 0 : num / denom;
}

double covariance(const Vector& x, const Vector& y) {
    if (x.size() != y.size() || x.empty()) return 0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double my = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double sum = 0;
    for (size_t i = 0; i < x.size(); i++)
        sum += (x[i] - mx) * (y[i] - my);
    return sum / x.size();
}

Vector percentile(const Matrix& data, double p) {
    Vector result;
    if (data.empty()) return result;
    for (size_t col = 0; col < data[0].size(); col++) {
        Vector column;
        for (const auto& row : data) {
            if (col < row.size()) column.push_back(row[col]);
        }
        std::sort(column.begin(), column.end());
        size_t idx = static_cast<size_t>(p / 100.0 * (column.size() - 1));
        result.push_back(column[idx]);
    }
    return result;
}

// ── Random ───────────────────────────────────────────────────────────

Vector random_normal(int n, double mean, double std) {
    std::normal_distribution<double> dist(mean, std);
    Vector r(n);
    for (auto& x : r) x = dist(rng());
    return r;
}

Vector random_uniform(int n, double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    Vector r(n);
    for (auto& x : r) x = dist(rng());
    return r;
}

Matrix random_matrix(int rows, int cols, double low, double high) {
    Matrix m(rows);
    for (auto& row : m) row = random_uniform(cols, low, high);
    return m;
}

// ── JSON serialization ───────────────────────────────────────────────

std::string vector_to_json(const Vector& v) {
    std::string r = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) r += ",";
        r += std::to_string(v[i]);
    }
    return r + "]";
}

std::string matrix_to_json(const Matrix& m) {
    std::string r = "[";
    for (size_t i = 0; i < m.size(); i++) {
        if (i > 0) r += ",";
        r += vector_to_json(m[i]);
    }
    return r + "]";
}

Vector json_to_vector(const std::string& json) {
    Vector v;
    std::string num;
    for (char c : json) {
        if (c == '[' || c == ']') continue;
        if (c == ',' || c == ' ') {
            if (!num.empty()) {
                v.push_back(std::stod(num));
                num.clear();
            }
        } else {
            num += c;
        }
    }
    if (!num.empty()) v.push_back(std::stod(num));
    return v;
}

Matrix json_to_matrix(const std::string& json) {
    Matrix m;
    // Simplified: expects [[1,2],[3,4]] format
    std::string inner;
    bool in_array = false;
    for (char c : json) {
        if (c == '[') {
            in_array = true;
            inner.clear();
        } else if (c == ']' && in_array) {
            if (!inner.empty()) {
                m.push_back(json_to_vector("[" + inner + "]"));
            }
            in_array = false;
        } else if (in_array) {
            inner += c;
        }
    }
    return m;
}

}  // namespace sweep
