"""
Task Generator — Produces unlimited synthetic tasks across all domains and difficulty levels.

§4: Task metadata structure.
§5-§11: Domain-specific task types.
§6 levels of difficulty.
"""
from __future__ import annotations

import random
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single training/evaluation task."""
    task_id: str
    domain: str
    difficulty: int
    input: str
    expected_output: str
    reasoning_type: str
    generation_seed: int
    verification_method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "input": self.input,
            "expected_output": self.expected_output,
            "reasoning_type": self.reasoning_type,
            "generation_seed": self.generation_seed,
            "verification_method": self.verification_method,
            "metadata": self.metadata,
        }


class TaskGenerator:
    """
    Generates unlimited synthetic tasks across all domains.

    §4: Every task has task_id, domain, difficulty, input, expected_output,
        reasoning_type, generation_seed, verification_method.
    §11: Novel structures via randomized generation.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._base_seed = seed

    def _next_id(self, domain: str) -> str:
        self._counter += 1
        return f"TASK-{domain[:3].upper()}-{self._counter:06d}"

    def _seed(self) -> int:
        return self._base_seed + self._counter

    def generate(
        self,
        domain: str,
        difficulty: int = 1,
        count: int = 1,
    ) -> list[Task]:
        """Generate tasks for a specific domain and difficulty."""
        gen = getattr(self, f"_gen_{domain}", self._gen_generic)
        tasks = []
        for _ in range(count):
            seed = self._seed()
            task = gen(domain, difficulty, seed)
            tasks.append(task)
        return tasks

    def generate_batch(
        self,
        domains: list[str] | None = None,
        difficulties: list[int] | None = None,
        per_domain_per_level: int = 5,
    ) -> list[Task]:
        """Generate a balanced batch across domains and difficulties."""
        from sweep_neural_mesh.training.domains import DEFAULT_DOMAINS
        domains = domains or DEFAULT_DOMAINS
        difficulties = difficulties or [1, 2, 3, 4, 5, 6]
        tasks = []
        for d in domains:
            for lvl in difficulties:
                tasks.extend(self.generate(d, lvl, per_domain_per_level))
        self._rng.shuffle(tasks)
        return tasks

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: logic (§5)
    # ══════════════════════════════════════════════════════════════════

    def _gen_logic(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        if difficulty <= 2:
            return self._gen_syllogism(domain, difficulty, rng, seed)
        elif difficulty <= 4:
            return self._gen_conditional(domain, difficulty, rng, seed)
        else:
            return self._gen_nested_logic(domain, difficulty, rng, seed)

    def _gen_syllogism(self, domain: str, difficulty: int, rng: random.Random, seed: int = 0) -> Task:
        entities = self._random_entities(rng, 4)
        A, B, C, D = entities[:4]

        premises = [f"{A} implies {B}.", f"{B} implies {C}."]
        chain = [A, B, C]
        if difficulty >= 2:
            premises.append(f"{C} implies {D}.")
            chain.append(D)

        conclusion = f"Does {chain[0]} imply {chain[-1]}?"
        answer = "YES"

        if rng.random() < 0.3:
            premises[1] = f"{C} implies {B}."
            answer = "NO"

        text = " ".join(premises) + f"\n\n{conclusion}\nAnswer YES or NO."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="syllogism",
            generation_seed=seed,
            verification_method="symbolic_chain",
            metadata={"premises": premises, "chain": chain},
        )

    def _gen_conditional(self, domain: str, difficulty: int, rng: random.Random, seed: int = 0) -> Task:
        entities = self._random_entities(rng, 5)
        A, B, C, D, E = entities[:5]

        if difficulty == 3:
            text = (
                f"If {A} then {B}.\nIf {B} then {C}.\n"
                f"{A} is true.\n\nWhat can be concluded about {C}?\n"
                f"Answer TRUE or FALSE."
            )
            answer = "TRUE"
        elif difficulty == 4:
            text = (
                f"If {A} then {B}.\nIf {B} then {C}.\n"
                f"If {C} then {D}.\n"
                f"Not {D}.\n\nWhat can be concluded about {A}?\n"
                f"Answer TRUE or FALSE."
            )
            answer = "FALSE"
        else:
            text = (
                f"If {A} then {B}.\nIf {B} then {C}.\nIf {C} then {D}.\nIf {D} then {E}.\n"
                f"Not {E}.\n\nWhat can be concluded about {A}?\n"
                f"Answer TRUE or FALSE."
            )
            answer = "FALSE"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="conditional",
            generation_seed=seed,
            verification_method="symbolic_chain",
        )

    def _gen_nested_logic(self, domain: str, difficulty: int, rng: random.Random, seed: int = 0) -> Task:
        entities = self._random_entities(rng, 6)
        A, B, C, D, E, F = entities[:6]

        chain = f"If {A} then {B}. If {B} then {C}. If {C} then {D}. If {D} then {E}. If {E} then {F}."
        not_f = f"Not {F}."

        text = f"{chain}\n{not_f}\n\nWhat can be concluded about {A}?\nAnswer TRUE or FALSE."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output="FALSE",
            reasoning_type="nested_conditional",
            generation_seed=seed,
            verification_method="symbolic_chain",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: reasoning (§5 general)
    # ══════════════════════════════════════════════════════════════════

    def _gen_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        return self._gen_syllogism(domain, difficulty, rng)

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: deduction (§5)
    # ══════════════════════════════════════════════════════════════════

    def _gen_deduction(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        return self._gen_conditional(domain, difficulty, rng)

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: induction (§5)
    # ══════════════════════════════════════════════════════════════════

    def _gen_induction(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        length = 3 + difficulty
        pattern_type = rng.choice(["arithmetic", "geometric", "fibonacci_like"])

        if pattern_type == "arithmetic":
            start = rng.randint(1, 10)
            step = rng.randint(1, 5)
            seq = [start + step * i for i in range(length + 1)]
            answer = str(seq[-1])
            seq_display = seq[:-1]
        elif pattern_type == "geometric":
            start = rng.randint(1, 5)
            mult = rng.randint(2, 3)
            seq = [start * (mult ** i) for i in range(length + 1)]
            answer = str(seq[-1])
            seq_display = seq[:-1]
        else:
            a, b = rng.randint(1, 5), rng.randint(1, 5)
            seq = [a, b]
            for _ in range(length - 1):
                seq.append(seq[-1] + seq[-2])
            answer = str(seq[-1])
            seq_display = seq[:-1]

        text = f"Sequence: {', '.join(str(x) for x in seq_display)}, ?\nWhat is the next number?"
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="pattern_induction",
            generation_seed=seed,
            verification_method="exact_value",
            metadata={"pattern_type": pattern_type, "full_sequence": seq},
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: transitivity (§5)
    # ══════════════════════════════════════════════════════════════════

    def _gen_transitivity(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        chain_len = 2 + difficulty
        entities = self._random_entities(rng, chain_len)

        relations = []
        for i in range(chain_len - 1):
            relations.append(f"{entities[i]} is greater than {entities[i+1]}")

        if difficulty >= 4 and rng.random() < 0.4:
            query_entities = [entities[-1], entities[0]]
            answer = "NO"
            reason = "reverse"
        else:
            query_entities = [entities[0], entities[-1]]
            answer = "YES"
            reason = "forward"

        text = ". ".join(relations) + f".\n\nIs {query_entities[0]} greater than {query_entities[1]}?\nAnswer YES or NO."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="transitivity",
            generation_seed=seed,
            verification_method="chain_verification",
            metadata={"chain": entities, "reason": reason},
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: relational_reasoning (§6)
    # ══════════════════════════════════════════════════════════════════

    def _gen_relational_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        n = 3 + difficulty
        entities = self._random_entities(rng, n)
        relations = []
        for i in range(n - 1):
            if rng.random() < 0.5:
                relations.append(f"{entities[i]} is connected to {entities[i+1]}")
            else:
                relations.append(f"{entities[i]} is parent of {entities[i+1]}")

        if rng.random() < 0.5 and difficulty >= 3:
            a, b = entities[0], entities[-1]
            answer = "YES" if self._connected_forward(entities, rng) else "NO"
            text = ". ".join(relations) + f".\n\nIs there a path from {a} to {b}?\nAnswer YES or NO."
        else:
            common = entities[rng.randint(0, n-1)]
            answer = common
            text = ". ".join(relations) + f".\n\nWho is directly connected to both {entities[0]} and {entities[1]}?\nGive one name."

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="relational",
            generation_seed=seed,
            verification_method="graph_traversal",
        )

    def _connected_forward(self, entities: list[str], rng: random.Random) -> bool:
        return True

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: temporal_reasoning (§8)
    # ══════════════════════════════════════════════════════════════════

    def _gen_temporal_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        events = self._random_events(rng, 3 + min(difficulty, 3))

        statements = []
        for i in range(len(events) - 1):
            if rng.random() < 0.6:
                statements.append(f"{events[i]} occurred before {events[i+1]}")
            else:
                statements.append(f"{events[i+1]} occurred after {events[i]}")

        if difficulty >= 4:
            mid = rng.randint(1, len(events) - 2)
            statements.append(f"{events[mid]} occurred between {events[mid-1]} and {events[mid+1]}")

        if difficulty >= 5 and rng.random() < 0.4:
            statements.append(f"{events[-1]} occurred before {events[0]}")
            answer = "IMPOSSIBLE"
            text = ". ".join(statements) + ".\n\nIs this timeline possible?\nAnswer POSSIBLE or IMPOSSIBLE."
        else:
            query_idx = rng.randint(0, len(events) - 2)
            text = ". ".join(statements) + f".\n\nDid {events[query_idx]} occur before {events[query_idx+1]}?\nAnswer YES or NO."
            answer = "YES"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="temporal",
            generation_seed=seed,
            verification_method="temporal_chain",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: spatial_reasoning
    # ══════════════════════════════════════════════════════════════════

    def _gen_spatial_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        objects = self._random_entities(rng, 3 + min(difficulty, 3))

        statements = []
        for i in range(len(objects) - 1):
            if rng.random() < 0.5:
                statements.append(f"{objects[i]} is to the left of {objects[i+1]}")
            else:
                statements.append(f"{objects[i]} is above {objects[i+1]}")

        text = ". ".join(statements) + f".\n\n"
        if rng.random() < 0.5:
            text += f"Is {objects[0]} to the left of {objects[-1]}?\nAnswer YES or NO."
            answer = "YES"
        else:
            text += f"Is {objects[-1]} to the left of {objects[0]}?\nAnswer YES or NO."
            answer = "NO"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="spatial",
            generation_seed=seed,
            verification_method="spatial_chain",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: evidence_evaluation (§10)
    # ══════════════════════════════════════════════════════════════════

    def _gen_evidence_evaluation(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        claim = self._random_claim(rng)

        evidence = []
        if difficulty >= 1:
            evidence.append(f"Study 1 supports the claim that {claim}")
        if difficulty >= 2:
            evidence.append(f"Research shows mixed results on whether {claim}")
        if difficulty >= 3:
            evidence.append(f"A contradicting study found that {claim} is false")
        if difficulty >= 4:
            evidence.append(f"An irrelevant study about weather patterns was published")
        if difficulty >= 5:
            evidence.append(f"A strong meta-analysis confirms {claim}")
            evidence.append(f"But a recent study with larger sample size refutes it")

        supports = sum(1 for e in evidence if "supports" in e or "confirms" in e)
        refutes = sum(1 for e in evidence if "refutes" in e or "false" in e or "contradicting" in e)

        if supports > refutes:
            answer = "SUPPORTED"
        elif refutes > supports:
            answer = "REFUTED"
        else:
            answer = "AMBIGUOUS"

        if difficulty >= 5 and supports == 0 and refutes == 0:
            answer = "INSUFFICIENT"

        text = f"Claim: {claim}\n\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence)
        text += f"\n\nBased on the evidence, is the claim SUPPORTED, REFUTED, AMBIGUOUS, or INSUFFICIENT?"
        text += f"\nAnswer:"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="evidence",
            generation_seed=seed,
            verification_method="evidence_count",
            metadata={"evidence": evidence, "supports": supports, "refutes": refutes},
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: contradiction_detection
    # ══════════════════════════════════════════════════════════════════

    def _gen_contradiction_detection(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        entities = self._random_entities(rng, 3)

        if difficulty <= 3:
            statements = [
                f"{entities[0]} is faster than {entities[1]}",
                f"{entities[1]} is faster than {entities[2]}",
            ]
            if difficulty >= 2:
                statements.append(f"{entities[2]} is faster than {entities[0]}")
                answer = "CONTRADICTION"
            else:
                answer = "CONSISTENT"
        else:
            statements = [
                f"{entities[0]} is faster than {entities[1]}",
                f"{entities[1]} is faster than {entities[2]}",
                f"{entities[2]} is slower than {entities[0]}",
            ]
            if difficulty >= 5 and rng.random() < 0.3:
                statements.append(f"{entities[0]} is equal in speed to {entities[1]}")
                answer = "CONTRADICTION"
            else:
                answer = "CONSISTENT"

        text = "Statements:\n" + "\n".join(f"- {s}" for s in statements)
        text += "\n\nAre these statements consistent or contradictory?\nAnswer CONSISTENT or CONTRADICTION."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="contradiction",
            generation_seed=seed,
            verification_method="transitivity_check",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: ambiguity_resolution (§9)
    # ══════════════════════════════════════════════════════════════════

    def _gen_ambiguity_resolution(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        people = self._random_entities(rng, 3)
        P1, P2, P3 = people[:3]

        templates = [
            f"{P1} told {P2} that they had completed the task.",
            f"When {P1} asked {P2} about {P3}, they said it was ready.",
            f"{P1} told {P2} that {P3} would help them with the project.",
        ]

        if difficulty <= 2:
            text = rng.choice(templates) + f"\n\nWho does 'they' refer to in this sentence?\nAnswer with all possible referents, comma-separated."
            answer = f"{P1}, {P2}"
        elif difficulty <= 4:
            text = templates[0] + f"\n\nIs this sentence ambiguous?\nAnswer YES or NO."
            answer = "YES"
        else:
            text = templates[0] + f"\n\nAnalyze: How many possible interpretations exist for 'they'?\nAnswer with a number."
            answer = "2"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="ambiguity",
            generation_seed=seed,
            verification_method="ambiguity_check",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: uncertainty
    # ══════════════════════════════════════════════════════════════════

    def _gen_uncertainty(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        entities = self._random_entities(rng, 3)

        statements = [
            f"{entities[0]} might be taller than {entities[1]}",
            f"{entities[1]} is definitely taller than {entities[2]}",
        ]
        if difficulty >= 3:
            statements.append(f"It is uncertain whether {entities[0]} is taller than {entities[2]}")

        text = "Given:\n" + "\n".join(f"- {s}" for s in statements)
        text += f"\n\nCan we definitively determine if {entities[0]} is taller than {entities[2]}?\nAnswer YES, NO, or UNCERTAIN."

        if difficulty <= 2:
            answer = "UNCERTAIN"
        else:
            answer = "UNCERTAIN"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="uncertainty",
            generation_seed=seed,
            verification_method="knowledge_completeness",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: pattern_recognition
    # ══════════════════════════════════════════════════════════════════

    def _gen_pattern_recognition(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        if difficulty <= 2:
            return self._gen_induction(domain, difficulty, seed)
        else:
            length = 5 + difficulty
            letters = [chr(ord('A') + rng.randint(0, 5)) for _ in range(length)]
            answer = letters[-1]
            text = f"Pattern: {' '.join(letters[:-1])} ?\nWhat comes next?"
            return Task(
                task_id=self._next_id(domain),
                domain=domain,
                difficulty=difficulty,
                input=text,
                expected_output=answer,
                reasoning_type="pattern",
                generation_seed=seed,
                verification_method="pattern_match",
            )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: multi_step_planning
    # ══════════════════════════════════════════════════════════════════

    def _gen_multi_step_planning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        steps = 2 + difficulty
        actions = [f"Step {i+1}: {self._random_action(rng)}" for i in range(steps)]

        text = "To complete the task, you must:\n" + "\n".join(actions)
        text += f"\n\nHow many steps are required?\nAnswer with a number."
        answer = str(steps)

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="planning",
            generation_seed=seed,
            verification_method="count_steps",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: graph_reasoning (§7)
    # ══════════════════════════════════════════════════════════════════

    def _gen_graph_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        from graph_benchmark.generator.graph_generator import GraphGenerator
        from graph_benchmark.generator.task_generator import TaskGenerator as GraphTaskGen

        rng = random.Random(seed)
        n = min(10 + difficulty * 5, 50)
        g_gen = GraphGenerator(seed=seed)
        graph = g_gen.generate(num_nodes=n, difficulty="medium")

        t_gen = GraphTaskGen(seed=seed + 1)
        tasks = t_gen.generate_all(graph, tasks_per_type=1)

        if tasks:
            gt = tasks[0].ground_truth
            if isinstance(gt, list):
                gt_str = ", ".join(gt) if gt else "NONE"
            else:
                gt_str = str(gt)
            return Task(
                task_id=self._next_id(domain),
                domain=domain,
                difficulty=difficulty,
                input=tasks[0].prompt,
                expected_output=gt_str,
                reasoning_type=tasks[0].task_type,
                generation_seed=seed,
                verification_method="graph_algorithm",
                metadata={"graph_id": graph.id, "task_type": tasks[0].task_type},
            )

        return self._gen_generic(domain, difficulty, seed)

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: causal_reasoning
    # ══════════════════════════════════════════════════════════════════

    def _gen_causal_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        causes = self._random_events(rng, 2 + min(difficulty, 3))

        chain = []
        for i in range(len(causes) - 1):
            chain.append(f"{causes[i]} caused {causes[i+1]}")

        text = ". ".join(chain) + ".\n\n"
        text += f"Did {causes[0]} indirectly cause {causes[-1]}?\nAnswer YES or NO."
        answer = "YES"

        if difficulty >= 4 and rng.random() < 0.3:
            unrelated = self._random_events(rng, 1)[0]
            chain.append(f"{unrelated} caused {causes[0]}")
            text = ". ".join(chain) + ".\n\n"
            text += f"Did {causes[0]} cause {causes[-1]}?\nAnswer YES or NO."
            answer = "NO"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="causal",
            generation_seed=seed,
            verification_method="causal_chain",
        )

    # ══════════════════════════════════════════════════════════════════
    # DOMAIN: novel_structure_reasoning (§11)
    # ══════════════════════════════════════════════════════════════════

    def _gen_novel_structure_reasoning(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        structure_type = rng.choice(["tree", "grid", "cycle", "star", "chain_with_branch"])

        if structure_type == "tree":
            return self._gen_tree_reasoning(domain, difficulty, rng, seed)
        elif structure_type == "grid":
            return self._gen_grid_reasoning(domain, difficulty, rng, seed)
        elif structure_type == "cycle":
            return self._gen_cycle_reasoning(domain, difficulty, rng, seed)
        elif structure_type == "star":
            return self._gen_star_reasoning(domain, difficulty, rng, seed)
        else:
            return self._gen_branch_reasoning(domain, difficulty, rng, seed)

    def _gen_tree_reasoning(self, domain: str, difficulty: int, rng: random.Random, seed: int) -> Task:
        entities = self._random_entities(rng, 4 + difficulty)
        root = entities[0]
        children = entities[1:3]
        grandchildren = entities[3:]

        statements = [f"{root} is parent of {c}" for c in children]
        for i, gc in enumerate(grandchildren):
            statements.append(f"{children[i % len(children)]} is parent of {gc}")

        text = "\n".join(statements)
        text += f"\n\nHow many descendants does {root} have?\nAnswer with a number."
        answer = str(len(entities) - 1)

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="tree_traversal",
            generation_seed=seed,
            verification_method="count_descendants",
        )

    def _gen_grid_reasoning(self, domain: str, difficulty: int, rng: random.Random, seed: int) -> Task:
        size = 2 + difficulty
        text = f"A {size}x{size} grid has cells labeled (row,col) from (1,1) to ({size},{size}).\n"
        text += f"From any cell, you can move right or down.\n"
        text += f"How many paths exist from (1,1) to ({size},{size})?\n"

        import math
        answer = str(math.comb(2 * (size - 1), size - 1))

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="combinatorial",
            generation_seed=seed,
            verification_method="exact_computation",
        )

    def _gen_cycle_reasoning(self, domain: str, difficulty: int, rng: random.Random, seed: int) -> Task:
        n = 3 + difficulty
        entities = self._random_entities(rng, n)
        edges = [(entities[i], entities[(i+1) % n]) for i in range(n)]

        text = "Edges:\n" + "\n".join(f"  {a} -> {b}" for a, b in edges)
        text += f"\n\nStarting from {entities[0]}, can you return to {entities[0]} following directed edges?\nAnswer YES or NO."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output="YES",
            reasoning_type="cycle_detection",
            generation_seed=seed,
            verification_method="cycle_check",
        )

    def _gen_star_reasoning(self, domain: str, difficulty: int, rng: random.Random, seed: int) -> Task:
        entities = self._random_entities(rng, 3 + difficulty)
        center = entities[0]
        spokes = entities[1:]

        text = f"{center} is connected to: {', '.join(spokes)}\n"
        text += f"\nHow many direct connections does {center} have?\nAnswer with a number."
        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=str(len(spokes)),
            reasoning_type="star_counting",
            generation_seed=seed,
            verification_method="count_connections",
        )

    def _gen_branch_reasoning(self, domain: str, difficulty: int, rng: random.Random, seed: int) -> Task:
        entities = self._random_entities(rng, 5 + difficulty)
        edges = [(entities[0], entities[1]), (entities[0], entities[2])]
        for i in range(3, len(entities)):
            edges.append((entities[i-2], entities[i]))

        text = "Edges:\n" + "\n".join(f"  {a} -> {b}" for a, b in edges)
        text += f"\n\nHow many paths of length 2 exist starting from {entities[0]}?\nAnswer with a number."
        answer = "2" if len(entities) >= 5 else "1"

        return Task(
            task_id=self._next_id(domain),
            domain=domain,
            difficulty=difficulty,
            input=text,
            expected_output=answer,
            reasoning_type="branch_counting",
            generation_seed=seed,
            verification_method="path_counting",
        )

    # ══════════════════════════════════════════════════════════════════
    # NEW CAPABILITY DOMAINS (Items 7, 13, 21, 22, 24, 26, 27)
    # ══════════════════════════════════════════════════════════════════

    # ── #7 Recursive Investigation ──
    def _gen_recursive_investigation(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        people = ["John Smith", "Alice Chen", "Bob Wilson", "Carol Davis", "Eve Johnson"]
        orgs = ["TechCorp", "DataInc", "ResearchLab", "MediaGroup", "FinServ"]
        cities = ["Delhi", "London", "Tokyo", "New York", "Berlin"]
        events = ["conference", "summit", "workshop", "meeting", "launch"]

        person = rng.choice(people)
        org = rng.choice(orgs)
        city = rng.choice(cities)
        event = rng.choice(events)

        if difficulty <= 2:
            text = f"{person} works at {org} in {city}." \
                   f"\n\nWhat entities are connected to {person}?\nList all connected entities."
            answer = f"{org}, {city}"
        elif difficulty <= 4:
            text = f"{person} works at {org} in {city}. " \
                   f"{org} sponsored a {event}. " \
                   f"The {event} was held in {rng.choice(cities)}." \
                   f"\n\nTraverse the investigation graph starting from {person}. " \
                   f"How many entities can be reached within 2 hops?"
            answer = "3"
        else:
            text = f"{person} works at {org} in {city}. " \
                   f"{org} sponsored a {event}. " \
                   f"The {event} featured {rng.choice(people)}. " \
                   f"{rng.choice(people)} is from {rng.choice(cities)}." \
                   f"\n\nBuild the investigation graph and identify all leaf nodes (entities with no outgoing connections)."
            answer = rng.choice(cities)

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="graph_traversal",
            generation_seed=seed, verification_method="graph_traversal",
        )

    # ── #13 Evidence Graph ──
    def _gen_evidence_graph(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        claims = [
            ("Person was at Location A", "Person was at Location B", "CONTRADICTS"),
            ("Drug is effective", "Drug shows results", "CORROBORATES"),
            ("Revenue increased 15%", "Revenue decreased 15%", "CONTRADICTS"),
            ("Study confirms X", "Research supports X", "CORROBORATES"),
            ("Person works at Company A", "Person employed by Company A", "CORROBORATES"),
        ]

        claim_pair = rng.choice(claims)
        ev1, ev2, expected_rel = claim_pair

        if difficulty <= 2:
            text = f"Evidence A: {ev1}\nEvidence B: {ev2}\n\n" \
                   f"What is the relationship between these two pieces of evidence?\n" \
                   f"Answer CORROBORATES, CONTRADICTS, or UNRELATED."
            answer = expected_rel
        elif difficulty <= 4:
            text = f"Evidence A: {ev1}\nEvidence B: {ev2}\n" \
                   f"Evidence C: {rng.choice([c[0] for c in claims])}\n\n" \
                   f"Build an evidence graph. Which evidence nodes form a cluster?\n" \
                   f"List the cluster members."
            answer = f"A, B"
        else:
            text = f"Evidence A: {ev1}\nEvidence B: {ev2}\n" \
                   f"Evidence C: {rng.choice([c[0] for c in claims])}\n" \
                   f"Evidence D: {rng.choice([c[1] for c in claims])}\n\n" \
                   f"Compute evidence importance (PageRank). Which node has highest importance?\n" \
                   f"Answer with the node label."
            answer = "A"

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="evidence_graph",
            generation_seed=seed, verification_method="graph_traversal",
        )

    # ── #21 Location Intelligence ──
    def _gen_location_intelligence(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        locations = [
            ("Delhi", 28.7, 77.1, "India"),
            ("London", 51.5, -0.1, "UK"),
            ("Tokyo", 35.7, 139.7, "Japan"),
            ("New York", 40.7, -74.0, "USA"),
            ("Berlin", 52.5, 13.4, "Germany"),
        ]

        loc1, loc2 = rng.sample(locations, 2)

        if difficulty <= 2:
            text = f"The event took place in {loc1[0]}, {loc1[3]}.\n" \
                   f"\nWhat country is this location in?\nAnswer with the country name."
            answer = loc1[3]
        elif difficulty <= 4:
            text = f"Person was seen in {loc1[0]}, {loc1[3]} on Monday.\n" \
                   f"Person was seen in {loc2[0]}, {loc2[3]} on Tuesday.\n" \
                   f"\nAre these locations in the same region?\nAnswer YES or NO."
            answer = "NO" if loc1[3] != loc2[3] else "YES"
        else:
            text = f"Person was seen in {loc1[0]}, {loc1[3]} on Monday.\n" \
                   f"Person was seen in {loc2[0]}, {loc2[3]} on Tuesday.\n" \
                   f"\nCompute the approximate distance between these locations.\n" \
                   f"Is it feasible to travel between them in 24 hours?\nAnswer YES or NO."
            answer = "NO"  # Different continents

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="geographic",
            generation_seed=seed, verification_method="location_check",
        )

    # ── #22 Search Strategy ──
    def _gen_search_strategy(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        targets = ["John Smith", "TechCorp Inc", "Project Alpha"]
        target = rng.choice(targets)

        known_aspects = rng.sample(["identity", "location", "affiliation", "timeline"], k=min(difficulty, 3))
        unknown_aspects = ["activities", "associates", "online_presence"][:max(1, 5-difficulty)]

        text = f"Investigation target: {target}\n\n"
        text += "Known aspects:\n"
        for a in known_aspects:
            text += f"  - {a}: CONFIRMED\n"
        text += "\nUnknown aspects:\n"
        for a in unknown_aspects:
            text += f"  - {a}: NO DATA\n"
        text += f"\n\nWhat should be the next search priority?\n"
        text += f"Answer with the aspect name that needs the most attention."
        answer = unknown_aspects[0]

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="search_planning",
            generation_seed=seed, verification_method="aspect_match",
        )

    # ── #24 Evidence Reports ──
    def _gen_evidence_reporting(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        n_supporting = rng.randint(1, 5)
        n_contradicting = rng.randint(0, 2) if difficulty >= 3 else 0

        text = f"Investigation: Is X effective?\n\n"
        text += f"Supporting evidence: {n_supporting} sources\n"
        text += f"Contradicting evidence: {n_contradicting} sources\n"
        text += f"Independent sources: {n_supporting + n_contradicting}\n\n"
        text += f"\nWhat is the overall confidence level?\n"
        text += f"Answer: CONFIRMED, LIKELY, POSSIBLE, UNCERTAIN, or CONTRADICTED."

        if n_contradicting > n_supporting:
            answer = "CONTRADICTED"
        elif n_supporting >= 4 and n_contradicting == 0:
            answer = "CONFIRMED"
        elif n_supporting >= 2:
            answer = "LIKELY"
        elif n_supporting >= 1:
            answer = "POSSIBLE"
        else:
            answer = "UNCERTAIN"

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="report_assessment",
            generation_seed=seed, verification_method="confidence_check",
        )

    # ── #26 Deduplication ──
    def _gen_deduplication(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        sources = [
            ("BBC News", "Company X reported record profits today.", "major_news"),
            ("Reuters", "Company X announced record-breaking profits.", "wire_service"),
            ("Blog", "Wow! Company X just made record profits!", "blog"),
            ("Twitter", "Company X profits are through the roof!", "social_media"),
            ("CNN", "Company X achieves historic profit milestone.", "major_news"),
        ]

        selected = rng.sample(sources, min(difficulty + 2, len(sources)))

        text = "Sources:\n"
        for i, (name, content, stype) in enumerate(selected, 1):
            text += f"  {i}. [{name}] ({stype}): {content}\n"
        text += f"\n\nHow many truly independent sources are there?\n"
        text += f"(Consider that news agencies may share the same underlying source)\n"
        text += f"Answer with a number."

        # Count unique source types
        unique_types = len({s[2] for s in selected})
        answer = str(unique_types)

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="deduplication",
            generation_seed=seed, verification_method="count_unique",
        )

    # ── #27 Source Independence ──
    def _gen_source_independence(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        origins = ["Press Release X", "Official Statement Y"]
        origin = rng.choice(origins)

        n_derived = rng.randint(2, 4)
        derived = [f"Article {chr(65+i)}" for i in range(n_derived)]

        text = f"Origin source: {origin}\n\n"
        text += f"Derived articles: {', '.join(derived)}\n"
        text += f"All derived articles contain similar language and facts.\n\n"
        text += f"How many independent confirmations does this represent?\n"
        text += f"(Articles from the same origin count as one)\n"
        text += f"Answer with a number."
        answer = "1"

        return Task(
            task_id=self._next_id(domain), domain=domain, difficulty=difficulty,
            input=text, expected_output=answer, reasoning_type="source_independence",
            generation_seed=seed, verification_method="independence_count",
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _gen_generic(self, domain: str, difficulty: int, seed: int) -> Task:
        rng = random.Random(seed)
        return self._gen_syllogism(domain, difficulty, rng)

    def _random_entities(self, rng: random.Random, n: int) -> list[str]:
        names = [
            "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
            "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi",
            "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi",
            "Chi", "Psi", "Omega", "Phoenix", "Nova", "Stella", "Cosmos",
        ]
        rng.shuffle(names)
        return names[:n]

    def _random_events(self, rng: random.Random, n: int) -> list[str]:
        events = [
            "The meeting", "The presentation", "The report", "The decision",
            "The analysis", "The review", "The approval", "The launch",
            "The experiment", "The test", "The evaluation", "The audit",
        ]
        rng.shuffle(events)
        return events[:n]

    def _random_claim(self, rng: random.Random) -> str:
        claims = [
            "exercise improves memory",
            "sleep deprivation affects cognition",
            "reading improves vocabulary",
            "practice improves performance",
            "stress affects productivity",
            "music enhances concentration",
            "exercise reduces anxiety",
            "meditation improves focus",
        ]
        return rng.choice(claims)

    def _random_action(self, rng: random.Random) -> str:
        actions = [
            "Gather requirements",
            "Analyze data",
            "Design solution",
            "Implement changes",
            "Test thoroughly",
            "Review results",
            "Deploy update",
            "Document findings",
            "Verify correctness",
            "Communicate status",
        ]
        return rng.choice(actions)
