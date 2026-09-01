cmake_minimum_required(VERSION 3.20)
project(sweep_engine VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find pybind11 ────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Source files ──────────────────────────────────────────────────────
set(SOURCES
    cpp/src/html_parser.cpp
    cpp/src/text_extractor.cpp
    cpp/src/search_ranker.cpp
    cpp/src/regex_engine.cpp
    cpp/src/ml_engine.cpp
    cpp/src/data_engine.cpp
    cpp/src/nlp_engine.cpp
)

# ── Static library ───────────────────────────────────────────────────
add_library(sweep_core STATIC ${SOURCES})
target_include_directories(sweep_core PUBLIC cpp/include)
target_compile_features(sweep_core PUBLIC cxx_std_20)

# ── pybind11 module ──────────────────────────────────────────────────
pybind11_add_module(sweep_engine cpp/src/bindings.cpp ${SOURCES})
target_include_directories(sweep_engine PRIVATE cpp/include)
target_compile_features(sweep_engine PRIVATE cxx_std_20)

# ── Terminal UI ───────────────────────────────────────────────────────
add_executable(sweep_ui cpp/ui/main.cpp)
target_link_libraries(sweep_ui PRIVATE sweep_core)
target_compile_features(sweep_ui PRIVATE cxx_std_20)

# ── Install ──────────────────────────────────────────────────────────
install(TARGETS sweep_engine DESTINATION .)
