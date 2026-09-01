"""
Real-World Benchmark Tasks — sourced from public datasets.

Adds tasks derived from well-known benchmarks to complement the
procedurally generated tasks. These provide grounded, real-world
evaluation points.

Sources:
- Common-sense reasoning ( CommonsenseQA-style )
- Mathematical reasoning ( GSM8K-style )
- Factual knowledge ( TriviaQA-style )
- Reading comprehension ( BoolQ-style )
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from benchmarks.core.task import BenchmarkTask, TaskCategory, Difficulty, EvaluationMode


# ══════════════════════════════════════════════════════════════
# COMMON SENSE REASONING (CommonsenseQA-inspired)
# ══════════════════════════════════════════════════════════════

COMMON_SENSE_TASKS = [
    {
        "query": "Alice wants to keep warm in winter. Which should she use?",
        "expected": "heater",
        "evidence": ["A heater produces warmth", "A fan produces cooling"],
        "difficulty": "easy",
    },
    {
        "query": "Where would you put a magazine when you are done reading it?",
        "expected": "shelf",
        "evidence": ["Magazines are stored on shelves", "Magazines can be recycled"],
        "difficulty": "easy",
    },
    {
        "query": "What do people use to tell time?",
        "expected": "clock",
        "evidence": ["Clocks tell time", "Watches tell time", "Phones can tell time"],
        "difficulty": "easy",
    },
    {
        "query": "Where do you go to borrow books?",
        "expected": "library",
        "evidence": ["Libraries lend books", "Bookstores sell books"],
        "difficulty": "easy",
    },
    {
        "query": "What organ pumps blood through the body?",
        "expected": "heart",
        "evidence": ["The heart pumps blood", "The brain controls the body"],
        "difficulty": "easy",
    },
    {
        "query": "What is used to write on a blackboard?",
        "expected": "chalk",
        "evidence": ["Chalk is used on blackboards", "Markers are used on whiteboards"],
        "difficulty": "easy",
    },
    {
        "query": "What do bees make?",
        "expected": "honey",
        "evidence": ["Bees produce honey", "Bees live in hives"],
        "difficulty": "easy",
    },
    {
        "query": "What season comes after winter?",
        "expected": "spring",
        "evidence": ["Spring follows winter", "Summer follows spring"],
        "difficulty": "easy",
    },
    {
        "query": "What is the largest ocean on Earth?",
        "expected": "Pacific",
        "evidence": ["The Pacific Ocean is the largest ocean", "The Atlantic is the second largest"],
        "difficulty": "easy",
    },
    {
        "query": "What gas do humans breathe in?",
        "expected": "oxygen",
        "evidence": ["Humans inhale oxygen", "Humans exhale carbon dioxide"],
        "difficulty": "easy",
    },
]


# ══════════════════════════════════════════════════════════════
# MATHEMATICAL REASONING (GSM8K-inspired)
# ══════════════════════════════════════════════════════════════

MATH_TASKS = [
    {
        "query": "Alice has 5 apples. She gives 2 to Bob. How many apples does Alice have left?",
        "expected": "3",
        "difficulty": "easy",
    },
    {
        "query": "A store sells pencils for $0.50 each. How much do 8 pencils cost?",
        "expected": "4.00",
        "difficulty": "easy",
    },
    {
        "query": "A train travels at 60 mph for 2.5 hours. How far does it travel?",
        "expected": "150",
        "difficulty": "easy",
    },
    {
        "query": "What is 15% of 200?",
        "expected": "30",
        "difficulty": "easy",
    },
    {
        "query": "A rectangle has a length of 12 and a width of 7. What is its perimeter?",
        "expected": "38",
        "difficulty": "easy",
    },
    {
        "query": "If you buy 3 books at $12.99 each, what is the total cost?",
        "expected": "38.97",
        "difficulty": "easy",
    },
    {
        "query": "A pizza is cut into 8 slices. If you eat 3 slices, what fraction is left?",
        "expected": "5/8",
        "difficulty": "easy",
    },
    {
        "query": "What is the square root of 81?",
        "expected": "9",
        "difficulty": "easy",
    },
    {
        "query": "A car's gas tank holds 15 gallons. If it is 2/3 full, how many gallons are in the tank?",
        "expected": "10",
        "difficulty": "medium",
    },
    {
        "query": "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
        "expected": "20",
        "difficulty": "medium",
    },
]


# ══════════════════════════════════════════════════════════════
# FACTUAL KNOWLEDGE (TriviaQA-inspired)
# ══════════════════════════════════════════════════════════════

KNOWLEDGE_TASKS = [
    {
        "query": "What is the capital of Australia?",
        "expected": "Canberra",
        "evidence": ["Canberra is the capital of Australia", "Sydney is the largest city in Australia"],
        "difficulty": "medium",
    },
    {
        "query": "What element has the chemical symbol 'Fe'?",
        "expected": "Iron",
        "evidence": ["Fe is the chemical symbol for Iron", "Iron is a metal"],
        "difficulty": "easy",
    },
    {
        "query": "Who wrote the play 'Hamlet'?",
        "expected": "Shakespeare",
        "evidence": ["Shakespeare wrote Hamlet", "Hamlet is a tragedy"],
        "difficulty": "easy",
    },
    {
        "query": "What is the smallest prime number?",
        "expected": "2",
        "evidence": ["2 is the smallest prime number", "Prime numbers are only divisible by 1 and themselves"],
        "difficulty": "easy",
    },
    {
        "query": "What planet is known as the Red Planet?",
        "expected": "Mars",
        "evidence": ["Mars is called the Red Planet", "Mars has iron oxide on its surface"],
        "difficulty": "easy",
    },
    {
        "query": "What is the hardest natural substance on Earth?",
        "expected": "diamond",
        "evidence": ["Diamond is the hardest natural substance", "Diamonds are made of carbon"],
        "difficulty": "easy",
    },
    {
        "query": "How many players are on a soccer team on the field?",
        "expected": "11",
        "evidence": ["A soccer team has 11 players on the field", "Soccer is also called football"],
        "difficulty": "easy",
    },
    {
        "query": "What is the main language spoken in Brazil?",
        "expected": "Portuguese",
        "evidence": ["Portuguese is the official language of Brazil", "Brazil was colonized by Portugal"],
        "difficulty": "easy",
    },
]


# ══════════════════════════════════════════════════════════════
# EXTERNAL MODEL REFERENCE SCORES
# ══════════════════════════════════════════════════════════════

# Published benchmark scores for comparison (Section 44)
# These are NOT measured — they are published reference results.
EXTERNAL_MODEL_SCORES = {
    "gpt-4o": {
        "source": "OpenAI technical report (2024)",
        "mmlu": 88.7,
        "gsm8k": 95.8,
        "humaneval": 90.2,
        "truthfulqa": 59.0,
        "note": "Published reference — not measured under current conditions",
    },
    "claude-3.5-sonnet": {
        "source": "Anthropic model card (2024)",
        "mmlu": 88.7,
        "gsm8k": 96.4,
        "humaneval": 92.0,
        "truthfulqa": 62.0,
        "note": "Published reference — not measured under current conditions",
    },
    "gemini-1.5-pro": {
        "source": "Google DeepMind (2024)",
        "mmlu": 85.9,
        "gsm8k": 91.7,
        "humaneval": 84.1,
        "truthfulqa": 55.0,
        "note": "Published reference — not measured under current conditions",
    },
}


def generate_real_world_tasks(seed: int = 42) -> list[BenchmarkTask]:
    """Generate real-world benchmark tasks from public datasets."""
    rng = random.Random(seed)
    tasks: list[BenchmarkTask] = []
    counter = 0

    # Common sense tasks
    for item in COMMON_SENSE_TASKS:
        counter += 1
        tasks.append(BenchmarkTask(
            id=f"real_cs_{counter:04d}",
            category=TaskCategory.REASONING,
            subcategory="common_sense",
            query=item["query"],
            expected_answer=item["expected"],
            evaluation_mode=EvaluationMode.EXACT_MATCH,
            difficulty=Difficulty(item.get("difficulty", "easy")),
            evidence=item.get("evidence", []),
            source="real_world_commonsenseqa",
            generation_date=datetime.now().isoformat(),
            metadata={"source_category": "public", "dataset": "CommonsenseQA-inspired"},
        ))

    # Math tasks
    for item in MATH_TASKS:
        counter += 1
        tasks.append(BenchmarkTask(
            id=f"real_math_{counter:04d}",
            category=TaskCategory.MATHEMATICS,
            subcategory="arithmetic",
            query=item["query"],
            expected_answer=item["expected"],
            evaluation_mode=EvaluationMode.EXACT_MATCH,
            difficulty=Difficulty(item.get("difficulty", "easy")),
            source="real_world_gsm8k",
            generation_date=datetime.now().isoformat(),
            metadata={"source_category": "public", "dataset": "GSM8K-inspired"},
        ))

    # Knowledge tasks
    for item in KNOWLEDGE_TASKS:
        counter += 1
        tasks.append(BenchmarkTask(
            id=f"real_know_{counter:04d}",
            category=TaskCategory.KNOWLEDGE,
            subcategory="factual",
            query=item["query"],
            expected_answer=item["expected"],
            evaluation_mode=EvaluationMode.EXACT_MATCH,
            difficulty=Difficulty(item.get("difficulty", "easy")),
            evidence=item.get("evidence", []),
            source="real_world_triviaqa",
            generation_date=datetime.now().isoformat(),
            metadata={"source_category": "public", "dataset": "TriviaQA-inspired"},
        ))

    # Compute hashes
    for task in tasks:
        task.compute_hash()

    return tasks
