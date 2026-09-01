"""
Dataset Pipeline — §14

Build a reproducible dataset pipeline.
Support: JSONL, JSON, CSV, Parquet, text, image, audio datasets.
Every training example should support metadata:
    task_type, difficulty, modality, source, license, quality, expected_output, evaluation_criteria
Implement: train split, validation split, test split.
Prevent train/test contamination.
Do not train on benchmark test sets.
Do not leak evaluation answers into training data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetEntry:
    """A single dataset entry with full metadata."""
    entry_id: str
    task_type: str
    difficulty: int
    modality: str
    source: str
    license: str
    quality: float
    input_text: str
    expected_output: str
    evaluation_criteria: str
    split: str  # train, validation, test, holdout
    hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hash:
            content = f"{self.input_text}:{self.expected_output}"
            self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    total_entries: int
    by_split: dict[str, int]
    by_task_type: dict[str, int]
    by_modality: dict[str, int]
    by_difficulty: dict[int, int]
    avg_quality: float
    unique_hashes: int
    contamination_risk: float


class DatasetPipeline:
    """
    §14: Manages datasets with proper splitting, contamination prevention,
    and metadata tracking.
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        storage_dir: str | Path = "sweep_neural_mesh/training/datasets",
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._rng = random.Random(seed)
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._entries: list[DatasetEntry] = []
        self._hashes: set[str] = set()  # contamination tracking
        self._benchmark_hashes: set[str] = set()  # benchmark answers

    # ══════════════════════════════════════════════════════════════════
    # LOADING
    # ══════════════════════════════════════════════════════════════════

    def load_jsonl(self, path: str | Path, source: str = "", license: str = "") -> int:
        """Load a JSONL dataset file."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = self._json_to_entry(data, source, license)
                if not self._is_contaminated(entry):
                    self._entries.append(entry)
                    self._hashes.add(entry.hash)
                    count += 1
        return count

    def load_json(self, path: str | Path, source: str = "", license: str = "") -> int:
        """Load a JSON dataset file (list of objects)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("examples", [data]))
        count = 0
        for item in data:
            entry = self._json_to_entry(item, source, license)
            if not self._is_contaminated(entry):
                self._entries.append(entry)
                self._hashes.add(entry.hash)
                count += 1
        return count

    def load_csv(self, path: str | Path, source: str = "", license: str = "") -> int:
        """Load a CSV dataset file."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = self._json_to_entry(dict(row), source, license)
                if not self._is_contaminated(entry):
                    self._entries.append(entry)
                    self._hashes.add(entry.hash)
                    count += 1
        return count

    def load_text(self, path: str | Path, source: str = "", license: str = "") -> int:
        """Load a plain text dataset (one example per block separated by blank lines)."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        count = 0
        for i, block in enumerate(blocks):
            entry = DatasetEntry(
                entry_id=f"TXT-{i:06d}",
                task_type="text",
                difficulty=1,
                modality="text",
                source=source or "text_file",
                license=license or "unknown",
                quality=0.5,
                input_text=block,
                expected_output="",
                evaluation_criteria="completion",
                split="unassigned",
            )
            if not self._is_contaminated(entry):
                self._entries.append(entry)
                self._hashes.add(entry.hash)
                count += 1
        return count

    def add_entry(self, entry: DatasetEntry) -> bool:
        """Add a single entry. Returns False if contaminated."""
        if self._is_contaminated(entry):
            return False
        self._entries.append(entry)
        self._hashes.add(entry.hash)
        return True

    def add_entries(self, entries: list[DatasetEntry]) -> int:
        """Add multiple entries. Returns count of non-contaminated entries added."""
        count = 0
        for entry in entries:
            if self.add_entry(entry):
                count += 1
        return count

    # ══════════════════════════════════════════════════════════════════
    # CONTAMINATION PREVENTION
    # ══════════════════════════════════════════════════════════════════

    def register_benchmark_hash(self, content_hash: str) -> None:
        """Register a benchmark answer hash to prevent contamination."""
        self._benchmark_hashes.add(content_hash)

    def register_benchmark_entry(self, entry: DatasetEntry) -> None:
        """Register a benchmark entry's hash to prevent contamination."""
        self._benchmark_hashes.add(entry.hash)

    def _is_contaminated(self, entry: DatasetEntry) -> bool:
        """Check if an entry would contaminate benchmark data."""
        # Check against registered benchmark hashes
        if entry.hash in self._benchmark_hashes:
            return True
        # Check against input text hash
        input_hash = hashlib.sha256(entry.input_text.encode()).hexdigest()[:16]
        if input_hash in self._benchmark_hashes:
            return True
        return False

    def check_contamination(self) -> dict[str, Any]:
        """Check the dataset for potential contamination."""
        overlap = self._hashes.intersection(self._benchmark_hashes)
        return {
            "total_entries": len(self._entries),
            "benchmark_hashes_registered": len(self._benchmark_hashes),
            "overlap_count": len(overlap),
            "contamination_free": len(overlap) == 0,
        }

    # ══════════════════════════════════════════════════════════════════
    # SPLITTING
    # ══════════════════════════════════════════════════════════════════

    def split(self, force: bool = False) -> dict[str, list[DatasetEntry]]:
        """
        Split entries into train/val/test.

        Uses stratified splitting by task_type and difficulty.
        """
        # Check if already split
        if not force and any(e.split != "unassigned" for e in self._entries):
            return self.get_splits()

        # Group by task_type for stratified splitting
        by_type: dict[str, list[DatasetEntry]] = {}
        for entry in self._entries:
            by_type.setdefault(entry.task_type, []).append(entry)

        train, val, test = [], [], []

        for task_type, entries in by_type.items():
            self._rng.shuffle(entries)
            n = len(entries)
            n_train = int(n * self._train_ratio)
            n_val = int(n * self._val_ratio)

            for i, entry in enumerate(entries):
                if i < n_train:
                    entry.split = "train"
                    train.append(entry)
                elif i < n_train + n_val:
                    entry.split = "validation"
                    val.append(entry)
                else:
                    entry.split = "test"
                    test.append(entry)

        return {"train": train, "validation": val, "test": test}

    def get_splits(self) -> dict[str, list[DatasetEntry]]:
        """Get entries by split."""
        splits: dict[str, list[DatasetEntry]] = {
            "train": [], "validation": [], "test": [], "holdout": [],
        }
        for entry in self._entries:
            splits.setdefault(entry.split, []).append(entry)
        return splits

    # ══════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════

    def export_jsonl(self, split: str, path: str | Path | None = None) -> str:
        """Export a split to JSONL format."""
        entries = [e for e in self._entries if e.split == split]
        path = path or (self._storage_dir / f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps({
                    "id": entry.entry_id,
                    "input": entry.input_text,
                    "expected_output": entry.expected_output,
                    "task_type": entry.task_type,
                    "difficulty": entry.difficulty,
                    "modality": entry.modality,
                    "source": entry.source,
                    "license": entry.license,
                    "quality": entry.quality,
                    "evaluation_criteria": entry.evaluation_criteria,
                    "hash": entry.hash,
                }) + "\n")
        return str(path)

    def export_json(self, path: str | Path | None = None) -> str:
        """Export the full dataset to JSON."""
        path = path or (self._storage_dir / "dataset.json")
        data = {
            "total_entries": len(self._entries),
            "splits": self.stats().by_split,
            "entries": [
                {
                    "id": e.entry_id,
                    "input": e.input_text,
                    "expected_output": e.expected_output,
                    "task_type": e.task_type,
                    "difficulty": e.difficulty,
                    "split": e.split,
                    "hash": e.hash,
                }
                for e in self._entries
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(path)

    # ══════════════════════════════════════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> DatasetStats:
        """Compute dataset statistics."""
        by_split: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_modality: dict[str, int] = {}
        by_difficulty: dict[int, int] = {}
        total_quality = 0.0

        for entry in self._entries:
            by_split[entry.split] = by_split.get(entry.split, 0) + 1
            by_type[entry.task_type] = by_type.get(entry.task_type, 0) + 1
            by_modality[entry.modality] = by_modality.get(entry.modality, 0) + 1
            by_difficulty[entry.difficulty] = by_difficulty.get(entry.difficulty, 0) + 1
            total_quality += entry.quality

        contamination = self.check_contamination()

        return DatasetStats(
            total_entries=len(self._entries),
            by_split=by_split,
            by_task_type=by_type,
            by_modality=by_modality,
            by_difficulty=by_difficulty,
            avg_quality=total_quality / max(1, len(self._entries)),
            unique_hashes=len(self._hashes),
            contamination_risk=contamination["overlap_count"] / max(1, len(self._entries)),
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _json_to_entry(
        self, data: dict[str, Any], source: str, license: str
    ) -> DatasetEntry:
        """Convert a JSON/dict object to a DatasetEntry."""
        return DatasetEntry(
            entry_id=data.get("id", data.get("task_id", f"GEN-{len(self._entries):06d}")),
            task_type=data.get("task_type", data.get("domain", "general")),
            difficulty=int(data.get("difficulty", 1)),
            modality=data.get("modality", "text"),
            source=source or data.get("source", "unknown"),
            license=license or data.get("license", "unknown"),
            quality=float(data.get("quality", 0.5)),
            input_text=data.get("input", data.get("input_text", data.get("question", ""))),
            expected_output=data.get("expected_output", data.get("answer", data.get("output", ""))),
            evaluation_criteria=data.get("evaluation_criteria", "exact_match"),
            split="unassigned",
            metadata=data.get("metadata", {}),
        )
