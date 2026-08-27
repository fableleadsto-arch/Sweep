"""
Extended Test Generators — Distractors, Conflicts, Novel Structures, Ablation.
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

IRRELEVANT_FACTS = [
    "The sky is blue on clear days.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits the Sun in approximately 365 days.",
    "Diamonds are the hardest known natural material.",
    "Sound travels faster in water than in air.",
    "The human body contains about 206 bones.",
    "Light travels at approximately 299,792 km/s in vacuum.",
    "Gold is a chemical element with symbol Au.",
    "The Amazon is the largest river by discharge volume.",
    "Mount Everest is the tallest mountain above sea level.",
    "Octopuses have three hearts.",
    "Bananas are technically berries.",
    "A group of crows is called a murder.",
    "The Great Wall of China is visible from low Earth orbit.",
    "Honey never spoils.",
    "Venus rotates backwards compared to other planets.",
    "The shortest war in history lasted 38 minutes.",
    "A jiffy is an actual unit of time: 1/100th of a second.",
    "The Eiffel Tower can grow 6 inches in summer.",
    "Neutron stars can spin at up to 600 rotations per second.",
]


def _names(rng: random.Random, n: int) -> list[str]:
    pool = list(NAMES)
    rng.shuffle(pool)
    return pool[:n]


# ═══════════════════════════════════════════════════════════════════
#  DISTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════

def gen_distractor_tasks(
    seed: int, difficulty: int,
    relevance_pct: float = 0.1,
    count: int = 20,
) -> list[Task]:
    """
    Generate tasks with varying signal-to-noise ratio.

    relevance_pct: fraction of content that is actually relevant.
    """
    rng = random.Random(seed)
    tasks = []

    for i in range(count):
        entities = _names(rng, 4)
        A, B, C, D = entities

        core_facts = [
            f"{A} implies {B}.",
            f"{B} implies {C}.",
            f"{C} implies {D}.",
        ]
        conclusion = f"Does {A} imply {D}?"

        num_distractors = int(len(core_facts) * (1.0 / max(relevance_pct, 0.01) - 1))
        distractors = [rng.choice(IRRELEVANT_FACTS) for _ in range(num_distractors)]

        all_facts = core_facts + distractors
        rng.shuffle(all_facts)

        text = " ".join(all_facts) + f"\n\n{conclusion}\nAnswer YES or NO."

        tasks.append(Task(
            task_id=f"DIST-{seed}-{i}",
            domain="distractor_resistance",
            difficulty=difficulty,
            input_text=text,
            expected_output="YES",
            metadata={"relevance_pct": relevance_pct, "num_distractors": num_distractors},
        ))

    return tasks


# ═══════════════════════════════════════════════════════════════════
#  CONFLICT TESTS
# ═══════════════════════════════════════════════════════════════════

def gen_conflict_tasks(seed: int, difficulty: int, count: int = 20) -> list[Task]:
    """Generate tasks with conflicting information across branches."""
    rng = random.Random(seed)
    tasks = []

    for i in range(count):
        entities = _names(rng, 4)
        A, B, C, D = entities

        supporting = [
            f"Source 1: {A} is faster than {B}",
            f"Source 2: {B} is faster than {C}",
            f"Source 3: {A} is faster than {C}",
        ]

        if difficulty <= 3:
            text = "Statements:\n" + "\n".join(f"- {s}" for s in supporting)
            text += f"\n\nBased on these statements, is {A} faster than {C}?\nAnswer SUPPORTED, REFUTED, AMBIGUOUS, or INSUFFICIENT."
            answer = "SUPPORTED"
        elif difficulty <= 5:
            contradicting = [
                f"Source 4: {C} is faster than {A}",
            ]
            all_stmts = supporting + contradicting
            rng.shuffle(all_stmts)
            text = "Statements:\n" + "\n".join(f"- {s}" for s in all_stmts)
            text += f"\n\nBased on these statements, is {A} faster than {C}?\nAnswer SUPPORTED, REFUTED, AMBIGUOUS, or INSUFFICIENT."
            answer = "AMBIGUOUS"
        else:
            irrelevant = [
                f"Source 4: The weather is nice today",
                f"Source 5: {D} likes coffee",
            ]
            all_stmts = supporting + irrelevant
            rng.shuffle(all_stmts)
            text = "Statements:\n" + "\n".join(f"- {s}" for s in all_stmts)
            text += f"\n\nBased on these statements, is {A} faster than {C}?\nAnswer SUPPORTED, REFUTED, AMBIGUOUS, or INSUFFICIENT."
            answer = "SUPPORTED"

        tasks.append(Task(
            task_id=f"CON-{seed}-{i}",
            domain="conflict_resolution",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
        ))

    return tasks


# ═══════════════════════════════════════════════════════════════════
#  NOVEL STRUCTURE TESTS (unseen topologies)
# ═══════════════════════════════════════════════════════════════════

def gen_novel_topology_tasks(seed: int, difficulty: int, count: int = 20) -> list[Task]:
    """
    Generate tasks with topologies different from training:
    DAGs, multi-root trees, bidirectional edges, weighted graphs.
    """
    rng = random.Random(seed)
    tasks = []

    for i in range(count):
        topo = rng.choice(["dag", "multi_root", "bidirectional", "weighted", "layered"])

        if topo == "dag":
            entities = _names(rng, 6)
            edges = []
            for j in range(len(entities) - 1):
                edges.append(f"{entities[j]} leads to {entities[j+1]}")
            if len(entities) > 3:
                edges.append(f"{entities[0]} leads to {entities[2]}")
            text = "Edges:\n" + "\n".join(f"  {e}" for e in edges)
            text += f"\n\nCan {entities[0]} reach {entities[-1]}?\nAnswer YES or NO."
            answer = "YES"
        elif topo == "multi_root":
            roots = _names(rng, 3)
            children = _names(rng, 3)
            stmts = []
            for r, c in zip(roots, children):
                stmts.append(f"{r} is parent of {c}")
            text = "\n".join(stmts)
            target = children[0]
            text += f"\n\nDoes {roots[0]} have a child named {target}?\nAnswer YES or NO."
            answer = "YES"
        elif topo == "bidirectional":
            entities = _names(rng, 4)
            edges = []
            for j in range(len(entities) - 1):
                edges.append(f"{entities[j]} is linked to {entities[j+1]}")
            text = "Bidirectional links:\n" + "\n".join(f"  {e}" for e in edges)
            text += f"\n\nIs there a path from {entities[0]} to {entities[-1]}?\nAnswer YES or NO."
            answer = "YES"
        elif topo == "weighted":
            entities = _names(rng, 4)
            edges = []
            for j in range(len(entities) - 1):
                w = rng.randint(1, 10)
                edges.append(f"{entities[j]} --{w}--> {entities[j+1]}")
            text = "Weighted edges:\n" + "\n".join(f"  {e}" for e in edges)
            text += f"\n\nWhat is the total weight from {entities[0]} to {entities[-1]}?\nAnswer with a number."
            weights = [int(e.split("--")[1].split("--")[0]) for e in edges]
            answer = str(sum(weights))
        else:
            layers = []
            for layer_idx in range(3):
                layer_entities = _names(rng, 2 + layer_idx)
                layers.append(layer_entities)
            stmts = []
            for l in range(len(layers) - 1):
                for a in layers[l]:
                    for b in layers[l + 1]:
                        if rng.random() < 0.5:
                            stmts.append(f"{a} is connected to {b}")
            text = "Layered connections:\n" + "\n".join(f"  {s}" for s in stmts)
            text += f"\n\nHow many layers are there?\nAnswer with a number."
            answer = str(len(layers))

        tasks.append(Task(
            task_id=f"NOV-{seed}-{i}",
            domain="novel_topology",
            difficulty=difficulty,
            input_text=text,
            expected_output=answer,
            metadata={"topology": topo},
        ))

    return tasks


# ═══════════════════════════════════════════════════════════════════
#  ABLATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "full_mesh": {
        "description": "Full Sweep neural mesh with all components",
        "num_candidates": 3,
        "use_cortex": True,
    },
    "reduced_mesh_75": {
        "description": "75% of mesh cores active",
        "num_candidates": 2,
        "use_cortex": True,
    },
    "reduced_mesh_50": {
        "description": "50% of mesh cores active",
        "num_candidates": 2,
        "use_cortex": True,
    },
    "reduced_mesh_25": {
        "description": "25% of mesh cores active",
        "num_candidates": 1,
        "use_cortex": True,
    },
    "reduced_mesh_10": {
        "description": "10% of mesh cores active",
        "num_candidates": 1,
        "use_cortex": True,
    },
    "single_path": {
        "description": "Single processing path, no parallelism",
        "num_candidates": 1,
        "use_cortex": True,
    },
}
