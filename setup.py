"""Setup.py — build C++ engine + pybind11 bindings.

Usage:
    pip install -e .
    python setup.py build_ext --inplace
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sweep_engine",
        sources=[
            "cpp/src/bindings.cpp",
            "cpp/src/html_parser.cpp",
            "cpp/src/text_extractor.cpp",
            "cpp/src/search_ranker.cpp",
            "cpp/src/regex_engine.cpp",
            "cpp/src/ml_engine.cpp",
            "cpp/src/data_engine.cpp",
            "cpp/src/nlp_engine.cpp",
        ],
        include_dirs=["cpp/include"],
        language="c++",
        cxx_std=20,
    ),
]

setup(
    name="sweep",
    version="2.0.0",
    description="Sweep — Python/C++ web intelligence platform with ML",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "sweep-benchmark=benchmarks.cli:main",
        ],
    },
    install_requires=[
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.29,<1.0",
        "pydantic>=2.8,<3.0",
        "pydantic-settings>=2.4,<3.0",
        "httpx>=0.27,<1.0",
        "beautifulsoup4>=4.12,<5.0",
        "trafilatura>=1.12,<2.0",
        "lxml>=5.0,<6.0",
        "torch>=2.0",
        "tensorflow>=2.15",
        "transformers>=4.40,<5.0",
        "accelerate>=0.30,<2.0",
        "diffusers>=0.28,<1.0",
        "sentence-transformers>=3.0,<4.0",
        "numpy>=1.26,<3.0",
        "pandas>=2.2,<3.0",
        "scipy>=1.13,<2.0",
        "scikit-learn>=1.5,<2.0",
        "networkx>=3.3,<4.0",
        "sympy>=1.13,<2.0",
        "matplotlib>=3.9,<4.0",
        "pillow>=10.4,<12.0",
        "xgboost>=2.0,<3.0",
        "lightgbm>=4.3,<5.0",
        "python-dotenv>=1.0,<2.0",
        "pybind11>=2.12,<3.0",
    ],
)
