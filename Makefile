# Sweep — Makefile for C++ components
#
# Targets:
#   make engine      — build C++ shared library (lib sweep_core)
#   make python      — build pybind11 Python module (sweep_engine.so)
#   make ui          — build C++ terminal UI (sweep_ui)
#   make all         — build everything
#   make clean       — remove build artifacts
#   make install     — pip install -e . (builds Python module)

CXX      := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Icpp/include
LDFLAGS  :=

# Source files
ENGINE_SRCS := cpp/src/html_parser.cpp cpp/src/text_extractor.cpp cpp/src/search_ranker.cpp cpp/src/regex_engine.cpp
UI_SRCS     := cpp/ui/main.cpp

# ── Engine (static library) ──────────────────────────────────────────

build/sweep_core.a: $(ENGINE_SRCS) | build
	$(CXX) $(CXXFLAGS) -c $(ENGINE_SRCS)
	ar rcs $@ *.o
	rm -f *.o

# ── Terminal UI ───────────────────────────────────────────────────────

build/sweep_ui: $(UI_SRCS) build/sweep_core.a
	$(CXX) $(CXXFLAGS) -o $@ $(UI_SRCS) build/sweep_core.a $(LDFLAGS)

# ── Python module (via pip) ──────────────────────────────────────────

python:
	pip install -e .

# ── Convenience targets ──────────────────────────────────────────────

all: build/sweep_core.a build/sweep_ui python

ui: build/sweep_ui

engine: build/sweep_core.a

build:
	mkdir -p build

clean:
	rm -rf build/ *.o *.so sweep_engine*.pyd
	rm -rf *.egg-info dist/

# ── Run targets ──────────────────────────────────────────────────────

run-ui: build/sweep_ui
	./build/sweep_ui

run-api:
	uvicorn app.main:app --reload --port 8787

.PHONY: all engine python ui clean run-ui run-api
