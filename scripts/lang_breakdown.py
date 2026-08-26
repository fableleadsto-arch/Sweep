#!/usr/bin/env python3
"""Language breakdown — Python/C++ architecture."""

from pathlib import Path
from collections import defaultdict

EXTENSIONS = {
    ".py": "Python",
    ".cpp": "C++",
    ".h": "C++ Headers",
    ".css": "CSS",
    ".js": "JavaScript",
    ".html": "HTML",
    ".md": "Markdown",
    ".json": "JSON",
}

SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build", ".cache", "__pycache__", "desktop"}


def count_lines(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def main():
    root = Path(".")
    counts = defaultdict(int)
    files = defaultdict(int)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        ext = path.suffix.lower()
        if ext not in EXTENSIONS:
            continue
        lines = count_lines(path)
        lang = EXTENSIONS[ext]
        counts[lang] += lines
        files[lang] += 1

    total = sum(counts.values())
    if total == 0:
        print("No source files found.")
        return

    print("=" * 55)
    print("  SWEEP — Language Breakdown (Python/C++ Architecture)")
    print("=" * 55)
    print()
    print(f"  {'Language':<20} {'Lines':>8} {'Files':>6} {'Percent':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8}")

    for lang, lines in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (lines / total) * 100
        print(f"  {lang:<20} {lines:>8,} {files[lang]:>6} {pct:>7.1f}%")

    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8}")
    print(f"  {'TOTAL':<20} {total:>8,} {sum(files.values()):>6} {'100.0%':>8}")
    print()

    py = counts.get("Python", 0)
    cpp = counts.get("C++", 0) + counts.get("C++ Headers", 0)
    native = py + cpp
    native_pct = (native / total) * 100 if total else 0

    print(f"  Python + C++:     {native_pct:.1f}%  ({native:,} lines)")
    print(f"  Python:           {(py/total*100) if total else 0:.1f}%  ({py:,} lines)")
    print(f"  C++ (engine+UI):  {(cpp/total*100) if total else 0:.1f}%  ({cpp:,} lines)")

    web = counts.get("JavaScript", 0) + counts.get("CSS", 0) + counts.get("HTML", 0)
    if web > 0:
        print(f"  Web (JS/CSS/HTML): {(web/total*100):.1f}%  ({web:,} lines)")
    print()


if __name__ == "__main__":
    main()
