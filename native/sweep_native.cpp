#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

double rms_energy(const std::vector<double>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    long double acc = 0.0L;
    for (double s : samples) {
        acc += static_cast<long double>(s) * s;
    }
    return std::sqrt(static_cast<double>(acc / static_cast<long double>(samples.size())));
}

std::vector<double> moving_average(const std::vector<double>& values, std::size_t window) {
    if (window == 0) {
        throw std::invalid_argument("window must be >= 1");
    }
    const std::size_t n = values.size();
    std::vector<double> out(n, 0.0);
    if (n == 0) {
        return out;
    }
    double rolling = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        rolling += values[i];
        if (i >= window) {
            rolling -= values[i - window];
        }
        const std::size_t count = (i < window - 1) ? i + 1 : window;
        out[i] = rolling / static_cast<double>(count);
    }
    return out;
}

double frame_diff_score(const std::string& prev, const std::string& curr) {
    const std::size_t max_len = std::max(prev.size(), curr.size());
    if (max_len == 0) {
        return 0.0;
    }
    const std::size_t min_len = std::min(prev.size(), curr.size());
    std::size_t diff = max_len - min_len;
    for (std::size_t i = 0; i < min_len; ++i) {
        if (prev[i] != curr[i]) {
            ++diff;
        }
    }
    return static_cast<double>(diff) / static_cast<double>(max_len);
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vectors must have equal length");
    }
    long double acc = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<long double>(a[i]) * b[i];
    }
    return static_cast<double>(acc);
}

double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || a.size() != b.size()) {
        return 0.0;
    }
    long double dot = 0.0L;
    long double norm_a = 0.0L;
    long double norm_b = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<long double>(a[i]) * b[i];
        norm_a += static_cast<long double>(a[i]) * a[i];
        norm_b += static_cast<long double>(b[i]) * b[i];
    }
    if (norm_a == 0.0L || norm_b == 0.0L) {
        return 0.0;
    }
    return static_cast<double>(dot / std::sqrt(norm_a * norm_b));
}

PYBIND11_MODULE(sweep_native, m) {
    m.doc() = "Sweep native fast-path primitives for visual and audio processing";
    m.def("rms_energy", &rms_energy, py::arg("samples"),
          "Root-mean-square energy of an audio sample buffer.");
    m.def("moving_average", &moving_average, py::arg("values"), py::arg("window"),
          "Trailing-window moving average with partial-window warmup.");
    m.def("frame_diff_score", &frame_diff_score, py::arg("prev"), py::arg("curr"),
          "Fraction of differing bytes between two raw frames [0.0..1.0].");
    m.def("dot_product", &dot_product, py::arg("a"), py::arg("b"),
          "Euclidean dot product of two equal-length vectors.");
    m.def("cosine_similarity", &cosine_similarity, py::arg("a"), py::arg("b"),
          "Cosine similarity of two vectors; 0.0 for empty, mismatched or zero inputs.");
}
