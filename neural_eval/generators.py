"""
Task Generators — Produce test tasks for all benchmark domains.

Each generator creates tasks with known ground truth but requires
actual reasoning to solve (no deterministic shortcuts in the task itself).
"""
from __future__ import annotations

import random
import math
from typing import Any

from .core import Task


NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
    "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi",
    "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi",
    "Chi", "Psi", "Omega", "Phoenix", "Nova", "Stella", "Cosmos",
]

EVENTS = [
    "The meeting", "The presentation", "The report", "The decision",
    "The analysis", "The review", "The approval", "The launch",
    "The experiment", "The test", "The evaluation", "The audit",
]


def _names(rng: random.Random, n: int) -> list[str]:
    pool = list(NAMES)
    rng.shuffle(pool)
    return pool[:n]


def _events(rng: random.Random, n: int) -> list[str]:
    pool = list(EVENTS)
    rng.shuffle(pool)
    return pool[:n]


# ═══════════════════════════════════════════════════════════════════
#  1. LOGICAL DEDUCTION
# ═══════════════════════════════════════════════════════════════════

def gen_logical_deduction(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        entities = _names(rng, 4)
        A, B, C, D = entities

        if difficulty <= 2:
            premises = [f"{A} implies {B}.", f"{B} implies {C}."]
            answer = "YES"
            if rng.random() < 0.3:
                premises[1] = f"{C} implies {B}."
                answer = "NO"
            conclusion = f"Does {A} imply {C}?"
        elif difficulty <= 4:
            premises = [f"{A} implies {B}.", f"{B} implies {C}.", f"{C} implies {D}."]
            answer = "YES"
            if rng.random() < 0.3:
                premises[1] = f"{C} implies {B}."
                answer = "NO"
            conclusion = f"Does {A} imply {D}?"
        else:
            E = _names(rng, 1)[0]
            premises = [
                f"{A} implies {B}.", f"{B} implies {C}.",
                f"{C} implies {D}.", f"{D} implies {E}.",
            ]
            answer = "YES"
            if rng.random() < 0.3:
                premises[2] = f"{D} implies {C}."
                answer = "NO"
            conclusion = f"Does {A} imply {E}?"

        text = " ".join(premises) + f"\n\n{conclusion}\nAnswer YES or NO."
        tasks.append(Task(
            task_id=f"LD-{seed}-{i}",
            domain="logical_deduction",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
            metadata={"premises": premises},
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  2. MULTI-STEP REASONING
# ═══════════════════════════════════════════════════════════════════

def gen_multistep_reasoning(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        n_steps = 3 + difficulty
        entities = _names(rng, n_steps + 1)
        relations = []
        for j in range(n_steps):
            relations.append(f"{entities[j]} is older than {entities[j+1]}")

        query_i = rng.randint(0, n_steps - 1)
        query_j = rng.randint(query_i + 1, n_steps)
        answer = "YES"

        if difficulty >= 4 and rng.random() < 0.25:
            k = rng.randint(1, n_steps - 1)
            relations[k] = f"{entities[k+1]} is older than {entities[k]}"
            answer = "YES" if query_j > query_i and all(
                "older than" in r for r in relations
            ) else "NO"

        text = ". ".join(relations) + f".\n\nIs {entities[query_i]} older than {entities[query_j]}?\nAnswer YES or NO."
        tasks.append(Task(
            task_id=f"MSR-{seed}-{i}",
            domain="multistep_reasoning",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  3. CONTRADICTION DETECTION
# ═══════════════════════════════════════════════════════════════════

def gen_contradiction(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        entities = _names(rng, 3)
        A, B, C = entities

        if difficulty <= 3:
            statements = [
                f"{A} is faster than {B}",
                f"{B} is faster than {C}",
            ]
            if rng.random() < 0.4:
                statements.append(f"{C} is faster than {A}")
                answer = "CONTRADICTION"
            else:
                answer = "CONSISTENT"
        else:
            statements = [
                f"{A} is faster than {B}",
                f"{B} is faster than {C}",
            ]
            if rng.random() < 0.3:
                statements.append(f"{C} is faster than {A}")
                answer = "CONTRADICTION"
            elif rng.random() < 0.5:
                statements.append(f"{C} is slower than {A}")
                answer = "CONSISTENT"
            else:
                statements.append(f"{A} is faster than {C}")
                answer = "CONSISTENT"

        text = "Statements:\n" + "\n".join(f"- {s}" for s in statements)
        text += "\n\nAre these statements consistent or contradictory?\nAnswer CONSISTENT or CONTRADICTION."
        tasks.append(Task(
            task_id=f"CD-{seed}-{i}",
            domain="contradiction_detection",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  4. EVIDENCE AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def gen_evidence(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    claims = [
        "exercise improves memory", "sleep deprivation affects cognition",
        "reading improves vocabulary", "practice improves performance",
        "stress affects productivity", "music enhances concentration",
    ]
    for i in range(20):
        claim = rng.choice(claims)
        evidence = []

        if difficulty >= 1:
            evidence.append("Study 1 supports the claim that " + claim)
        if difficulty >= 2:
            evidence.append("Research shows mixed results on whether " + claim)
        if difficulty >= 3:
            evidence.append("A contradicting study found that " + claim + " is false")
        if difficulty >= 4:
            evidence.append("An irrelevant study about weather patterns was published")
        if difficulty >= 5:
            evidence.append("A strong meta-analysis confirms " + claim)
            evidence.append("But a recent study with larger sample size refutes it")

        supports = sum(1 for e in evidence if "supports" in e or "confirms" in e)
        refutes = sum(1 for e in evidence if "refutes" in e or "contradicting" in e)

        if supports > refutes:
            answer = "SUPPORTED"
        elif refutes > supports:
            answer = "REFUTED"
        else:
            answer = "AMBIGUOUS"

        text = f"Claim: {claim}\n\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence)
        text += "\n\nIs the claim SUPPORTED, REFUTED, AMBIGUOUS, or INSUFFICIENT?"
        tasks.append(Task(
            task_id=f"EV-{seed}-{i}",
            domain="evidence_aggregation",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  5. AMBIGUITY RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def gen_ambiguity(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        people = _names(rng, 3)
        P1, P2, P3 = people

        if difficulty <= 2:
            text = f"{P1} told {P2} that they had completed the task.\n\nWho does 'they' refer to?\nAnswer with all possible referents, comma-separated."
            answer = f"{P1}, {P2}"
        elif difficulty <= 4:
            text = f"{P1} told {P2} that they had completed the task.\n\nIs this sentence ambiguous?\nAnswer YES or NO."
            answer = "YES"
        else:
            text = f"{P1} told {P2} that they had completed the task.\n\nHow many possible interpretations exist for 'they'?\nAnswer with a number."
            answer = "2"

        tasks.append(Task(
            task_id=f"AM-{seed}-{i}",
            domain="ambiguity_resolution",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  6. NOVEL STRUCTURES
# ═══════════════════════════════════════════════════════════════════

def gen_novel_structures(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        stype = rng.choice(["tree", "grid", "cycle", "star"])
        if stype == "tree":
            entities = _names(rng, 4 + difficulty)
            root = entities[0]
            children = entities[1:3]
            grandchildren = entities[3:]
            stmts = [f"{root} is parent of {c}" for c in children]
            for j, gc in enumerate(grandchildren):
                stmts.append(f"{children[j % len(children)]} is parent of {gc}")
            text = "\n".join(stmts) + f"\n\nHow many descendants does {root} have?\nAnswer with a number."
            answer = str(len(entities) - 1)
        elif stype == "grid":
            size = 2 + difficulty
            text = f"A {size}x{size} grid. From any cell move right or down. Paths from (1,1) to ({size},{size})?\nAnswer with a number."
            answer = str(math.comb(2 * (size - 1), size - 1))
        elif stype == "cycle":
            entities = _names(rng, 3 + difficulty)
            edges = [(entities[j], entities[(j + 1) % len(entities)]) for j in range(len(entities))]
            text = "Edges:\n" + "\n".join(f"  {a} -> {b}" for a, b in edges)
            text += f"\n\nStarting from {entities[0]}, can you return to {entities[0]}?\nAnswer YES or NO."
            answer = "YES"
        else:
            entities = _names(rng, 3 + difficulty)
            center = entities[0]
            spokes = entities[1:]
            text = f"{center} is connected to: {', '.join(spokes)}\n\nHow many connections does {center} have?\nAnswer with a number."
            answer = str(len(spokes))

        tasks.append(Task(
            task_id=f"NS-{seed}-{i}",
            domain="novel_structures",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  7. CAUSAL REASONING
# ═══════════════════════════════════════════════════════════════════

def gen_causal(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        events = _events(rng, 2 + min(difficulty, 3))
        chain = [f"{events[j]} caused {events[j+1]}" for j in range(len(events) - 1)]

        if difficulty >= 4 and rng.random() < 0.3:
            unrelated = _events(rng, 1)[0]
            chain.append(f"{unrelated} caused {events[0]}")
            text = ". ".join(chain) + f".\n\nDid {events[0]} cause {events[-1]}?\nAnswer YES or NO."
            answer = "NO"
        else:
            text = ". ".join(chain) + f".\n\nDid {events[0]} indirectly cause {events[-1]}?\nAnswer YES or NO."
            answer = "YES"

        tasks.append(Task(
            task_id=f"CR-{seed}-{i}",
            domain="causal_reasoning",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  8. TEMPORAL REASONING
# ═══════════════════════════════════════════════════════════════════

def gen_temporal(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        events = _events(rng, 3 + min(difficulty, 3))
        stmts = []
        for j in range(len(events) - 1):
            if rng.random() < 0.6:
                stmts.append(f"{events[j]} occurred before {events[j+1]}")
            else:
                stmts.append(f"{events[j+1]} occurred after {events[j]}")

        if difficulty >= 5 and rng.random() < 0.4:
            stmts.append(f"{events[-1]} occurred before {events[0]}")
            text = ". ".join(stmts) + ".\n\nIs this timeline possible?\nAnswer POSSIBLE or IMPOSSIBLE."
            answer = "IMPOSSIBLE"
        else:
            qi = rng.randint(0, len(events) - 2)
            text = ". ".join(stmts) + f".\n\nDid {events[qi]} occur before {events[qi+1]}?\nAnswer YES or NO."
            answer = "YES"

        tasks.append(Task(
            task_id=f"TR-{seed}-{i}",
            domain="temporal_reasoning",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  9. RELATIONAL REASONING
# ═══════════════════════════════════════════════════════════════════

def gen_relational(seed: int, difficulty: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        entities = _names(rng, 3 + min(difficulty, 3))
        relations = []
        for j in range(len(entities) - 1):
            if rng.random() < 0.5:
                relations.append(f"{entities[j]} is connected to {entities[j+1]}")
            else:
                relations.append(f"{entities[j]} is parent of {entities[j+1]}")

        text = ". ".join(relations) + f".\n\nIs there a path from {entities[0]} to {entities[-1]}?\nAnswer YES or NO."
        answer = "YES"

        tasks.append(Task(
            task_id=f"RR-{seed}-{i}",
            domain="relational_reasoning",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  10. PARALLEL BRANCH INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def gen_parallel_branches(seed: int, difficulty: int, num_branches: int = 4) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(20):
        branches = []
        required_facts = {}
        for b in range(num_branches):
            a, c = _names(rng, 2)
            fact_type = rng.choice(["age", "speed", "height"])
            if fact_type == "age":
                stmt = f"{a} is older than {c}"
                branches.append(stmt)
                required_facts[a] = "older"
            elif fact_type == "speed":
                stmt = f"{a} is faster than {c}"
                branches.append(stmt)
                required_facts[a] = "faster"
            else:
                stmt = f"{a} is taller than {c}"
                branches.append(stmt)
                required_facts[a] = "taller"

        branch_names = [f"Branch {b+1}" for b in range(num_branches)]
        text = "Reasoning branches:\n"
        for b_name, b_stmt in zip(branch_names, branches):
            text += f"  {b_name}: {b_stmt}\n"

        first_entity = list(required_facts.keys())[0]
        relation = required_facts[first_entity]
        text += f"\nConclusion: {first_entity} is {relation} than others in their branches.\nIs this conclusion supported?\nAnswer YES or NO."
        answer = "YES"

        tasks.append(Task(
            task_id=f"PB-{seed}-{i}",
            domain="parallel_branches",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
            metadata={"num_branches": num_branches},
        ))
    return tasks


# ═══════════════════════════════════════════════════════════════════
#  MASTER GENERATOR
# ═══════════════════════════════════════════════════════════════════

ALL_GENERATORS = {
    "logical_deduction": gen_logical_deduction,
    "multistep_reasoning": gen_multistep_reasoning,
    "contradiction_detection": gen_contradiction,
    "evidence_aggregation": gen_evidence,
    "ambiguity_resolution": gen_ambiguity,
    "novel_structures": gen_novel_structures,
    "causal_reasoning": gen_causal,
    "temporal_reasoning": gen_temporal,
    "relational_reasoning": gen_relational,
    "parallel_branches": gen_parallel_branches,
}


def generate_all_tasks(seed: int, difficulty: int, tasks_per_domain: int = 20) -> list[Task]:
    """Generate tasks for all domains."""
    all_tasks = []
    for domain, gen_fn in ALL_GENERATORS.items():
        tasks = gen_fn(seed, difficulty)
        all_tasks.extend(tasks[:tasks_per_domain])
    return all_tasks


def generate_branching_tasks(seed: int, branch_counts: list[int]) -> dict[int, list[Task]]:
    """Generate parallel branch tasks with varying branch counts."""
    result = {}
    for n in branch_counts:
        result[n] = gen_parallel_branches(seed, difficulty=3, num_branches=n)
    return result
