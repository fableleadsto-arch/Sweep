"""
Grasping Capability Benchmark - Sweep Neural Mesh vs Usual Mesh.

Tests how well each system "grasps" (understands, captures, and reasons
about) concepts, patterns, and relationships from evidence.

Measures:
  1. Concept Identification - can the mesh correctly identify core concepts?
  2. Relationship Grasping - can it latch onto entity relationships?
  3. Evidence Filtering - can it grasp relevant evidence from noise?
  4. Multi-hop Reasoning - can it chain concepts into conclusions?
  5. Ambiguity Handling - can it grasp nuanced/ambiguous claims?
  6. Confidence Calibration - does it know what it knows?

The "Usual Mesh" is a naive baseline: simple keyword overlap + majority
vote. It has no biological mechanisms, no synaptic plasticity, no
cross-referencing, and no adaptive reasoning.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_neural_mesh.neurons.cortex import ReasoningCortex, ReasoningResult


# ======================================================================
# USUAL MESH - Naive baseline for comparison
# ======================================================================

class UsualMesh:
    """
    A naive mesh implementation for comparison.

    Uses only:
    - Keyword overlap between query and evidence
    - Simple negation detection
    - Basic majority vote for decision

    Has NONE of the Sweep Neural Mesh features:
    - No brain divisions (hindbrain/midbrain/forebrain)
    - No processing centers
    - No synaptic plasticity / learning
    - No evidence cross-referencing
    - No metacognition
    - No working memory
    - No adaptive reasoning
    """

    def reason(
        self,
        query: str,
        evidence: list[str],
        sources: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        t0 = time.perf_counter()

        # Simple keyword overlap scoring
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        if not query_words:
            return self._make_result(query, evidence, "insufficient", 0.0,
                                     "no query keywords", t0)

        scores: list[tuple[str, float]] = []
        for ev in evidence:
            ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev.lower()))
            if not ev_words:
                scores.append((ev, 0.0))
                continue
            overlap = len(query_words & ev_words)
            score = overlap / len(query_words) if query_words else 0.0
            scores.append((ev, score))

        if not scores:
            return self._make_result(query, evidence, "insufficient", 0.0,
                                     "no evidence", t0)

        avg_score = sum(s for _, s in scores) / len(scores)
        max_score = max(s for _, s in scores)

        # Negation detection (very simple)
        negation_words = {"not", "never", "no", "cannot", "can't", "don't",
                          "doesn't", "isn't", "wasn't", "won't", "shouldn't"}
        all_text = " ".join(e.lower() for e in evidence)
        has_negation = any(w in all_text.split() for w in negation_words)

        # Simple majority vote
        pos_count = 0
        neg_count = 0
        for ev in evidence:
            ev_lower = ev.lower()
            if any(w in ev_lower for w in ["supports", "confirms", "is true",
                                            "yes", "does", "can"]):
                pos_count += 1
            elif any(w in ev_lower for w in ["refutes", "contradicts", "is false",
                                               "no", "cannot", "don't"]):
                neg_count += 1

        # Decision
        if pos_count > neg_count and avg_score > 0.2:
            decision = "supported"
            confidence = min(0.7, avg_score + 0.1)
        elif neg_count > pos_count:
            decision = "refuted"
            confidence = min(0.65, 0.3 + neg_count * 0.05)
        elif has_negation and avg_score > 0.3:
            decision = "mixed"
            confidence = 0.4
        elif avg_score > 0.3:
            decision = "supported"
            confidence = avg_score * 0.6
        else:
            decision = "insufficient"
            confidence = 0.1

        reasoning = (
            f"Keyword overlap: {avg_score:.2f} avg, {max_score:.2f} max. "
            f"Pos votes: {pos_count}, Neg votes: {neg_count}. "
            f"Negation: {'yes' if has_negation else 'no'}."
        )

        return self._make_result(query, evidence, decision, confidence,
                                 reasoning, t0)

    def _make_result(
        self, query: str, evidence: list[str], decision: str,
        confidence: float, reasoning: str, t0: float,
    ) -> ReasoningResult:
        from sweep_neural_mesh.neurons.cortex import ReasoningTrace
        trace = ReasoningTrace(
            query=query,
            input_evidence_count=len(evidence),
            center_outputs={},
            integration_confidence=confidence,
            decision=decision,
            decision_confidence=confidence,
            reasoning=reasoning,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return ReasoningResult(
            query=query,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            explanation_data={},
            trace=trace,
            factors=[],
            memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
        )


# ======================================================================
# GRASPING BENCHMARK TASKS
# ======================================================================

@dataclass
class GraspingTask:
    """A single concept-grasping test case."""
    id: str
    category: str
    subcategory: str
    query: str
    evidence: list[str]
    expected_decision: str
    expected_answer: str
    difficulty: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "category": self.category,
            "subcategory": self.subcategory, "query": self.query,
            "evidence": self.evidence,
            "expected_decision": self.expected_decision,
            "expected_answer": self.expected_answer,
            "difficulty": self.difficulty, "description": self.description,
        }


class GraspingBenchmark:
    """Generates concept-grasping benchmark tasks."""

    def generate(self) -> list[GraspingTask]:
        tasks: list[GraspingTask] = []
        tasks.extend(self._concept_identification())
        tasks.extend(self._relationship_grasping())
        tasks.extend(self._evidence_filtering())
        tasks.extend(self._multi_hop_reasoning())
        tasks.extend(self._ambiguity_handling())
        tasks.extend(self._confidence_calibration())
        return tasks

    def _concept_identification(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="ci_01", category="concept_identification",
                subcategory="direct",
                query="Is photosynthesis a biological process?",
                evidence=[
                    "Photosynthesis is the process by which plants convert "
                    "light energy into chemical energy",
                    "This biological process occurs in chloroplasts of plant cells",
                    "Photosynthesis converts CO2 and water into glucose and oxygen",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="easy",
                description="Direct concept matching with explicit evidence",
            ),
            GraspingTask(
                id="ci_02", category="concept_identification",
                subcategory="indirect",
                query="Is quantum entanglement a form of faster-than-light communication?",
                evidence=[
                    "Quantum entanglement correlates particle states instantaneously",
                    "Einstein called it spooky action at a distance",
                    "No information can be transmitted faster than light using entanglement",
                    "Bell's theorem proves entanglement is real but doesn't allow FTL messaging",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="medium",
                description="Concept requires understanding what is NOT being claimed",
            ),
            GraspingTask(
                id="ci_03", category="concept_identification",
                subcategory="abstraction",
                query="Does the free market always lead to optimal outcomes?",
                evidence=[
                    "Markets can fail when there are externalities like pollution",
                    "Monopolies reduce competition and consumer welfare",
                    "Information asymmetry causes adverse selection in insurance",
                    "Pigou showed government intervention can correct market failures",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="hard",
                description="Requires abstracting from specific examples to a general principle",
            ),
            GraspingTask(
                id="ci_04", category="concept_identification",
                subcategory="definitional",
                query="Is a virus alive?",
                evidence=[
                    "Viruses cannot reproduce without a host cell",
                    "Viruses do not metabolize or maintain homeostasis",
                    "Viruses contain genetic material and can evolve",
                    "The definition of life requires independent metabolism and reproduction",
                ],
                expected_decision="mixed",
                expected_answer="debatable, depends on definition of life",
                difficulty="hard",
                description="Concept exists at boundary of definition",
            ),
            GraspingTask(
                id="ci_05", category="concept_identification",
                subcategory="categorical",
                query="Is a whale a fish?",
                evidence=[
                    "Whales live in the ocean like fish",
                    "Whales breathe air with lungs, not gills",
                    "Whales give live birth and nurse their young",
                    "Whales are classified as mammals in taxonomy",
                ],
                expected_decision="refuted",
                expected_answer="no, whales are mammals",
                difficulty="easy",
                description="Categorical classification despite superficial similarity",
            ),
        ]

    def _relationship_grasping(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="rg_01", category="relationship_grasping",
                subcategory="causal_chain",
                query="Did the invention of the printing press lead to the Reformation?",
                evidence=[
                    "Gutenberg's printing press was invented around 1440",
                    "Martin Luther's 95 Theses were printed and distributed widely in 1517",
                    "The printing press enabled mass production of religious texts",
                    "Luther's ideas spread rapidly across Europe thanks to printed pamphlets",
                    "Historians credit the printing press as a key enabler of the Reformation",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="medium",
                description="Multi-step causal chain across time",
            ),
            GraspingTask(
                id="rg_02", category="relationship_grasping",
                subcategory="contradictory_relationships",
                query="Does smoking cause lung cancer?",
                evidence=[
                    "Smoking is the leading cause of lung cancer worldwide",
                    "Not all smokers develop lung cancer",
                    "Some non-smokers also get lung cancer",
                    "The causal link between smoking and lung cancer is well established",
                    "Secondhand smoke also increases cancer risk",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="medium",
                description="Causal relationship with exceptions and confounders",
            ),
            GraspingTask(
                id="rg_03", category="relationship_grasping",
                subcategory="hierarchical",
                query="Is Python a superset of C?",
                evidence=[
                    "Python is a high-level interpreted language",
                    "C is a low-level compiled language",
                    "Python can call C libraries via ctypes",
                    "Python has completely different syntax from C",
                    "Python is not backward-compatible with C code",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="medium",
                description="Hierarchical relationship that doesn't exist despite surface similarity",
            ),
            GraspingTask(
                id="rg_04", category="relationship_grasping",
                subcategory="correlation_vs_causation",
                query="Does ice cream cause drowning?",
                evidence=[
                    "Ice cream sales correlate with drowning deaths",
                    "Both increase during hot summer months",
                    "Temperature is the confounding variable",
                    "There is no direct causal mechanism between ice cream and drowning",
                ],
                expected_decision="refuted",
                expected_answer="no, correlation is not causation",
                difficulty="hard",
                description="Must distinguish correlation from causation",
            ),
            GraspingTask(
                id="rg_05", category="relationship_grasping",
                subcategory="multi_hop",
                query="If A is taller than B, and B is taller than C, is A taller than C?",
                evidence=[
                    "A measures 180cm",
                    "B measures 170cm",
                    "C measures 160cm",
                    "Height comparison is transitive",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="easy",
                description="Transitive relationship via intermediate entity",
            ),
        ]

    def _evidence_filtering(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="ef_01", category="evidence_filtering",
                subcategory="noise_rejection",
                query="What is the capital of France?",
                evidence=[
                    "The eiffel tower is in Paris France which is its capital city",
                    "Yesterday's weather was sunny with a high of 75 degrees",
                    "Paris is also known as the City of Light",
                    "The French Revolution began in 1789",
                ],
                expected_decision="supported", expected_answer="Paris",
                difficulty="easy",
                description="Noise mixed with relevant evidence",
            ),
            GraspingTask(
                id="ef_02", category="evidence_filtering",
                subcategory="insufficient_evidence",
                query="Is dark matter made of axions?",
                evidence=[
                    "Dark matter constitutes about 27% of the universe",
                    "Axions are hypothetical particles proposed to solve the strong CP problem",
                    "Some experiments have searched for axion dark matter",
                    "No direct detection of dark matter particles has been confirmed",
                ],
                expected_decision="insufficient", expected_answer="unknown",
                difficulty="hard",
                description="Plausible hypothesis without definitive evidence",
            ),
            GraspingTask(
                id="ef_03", category="evidence_filtering",
                subcategory="contradictory_evidence",
                query="Is the Great Wall of China visible from space?",
                evidence=[
                    "NASA has stated the Great Wall is not visible from low Earth orbit",
                    "Astronaut Yang Liwei confirmed he could not see it",
                    "Some claims say it can be seen under perfect conditions",
                    "Highway runways are more visible than the Great Wall from space",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="medium",
                description="Must weigh conflicting claims against authoritative sources",
            ),
            GraspingTask(
                id="ef_04", category="evidence_filtering",
                subcategory="partial_evidence",
                query="Do antibiotics work against viruses?",
                evidence=[
                    "Antibiotics kill or inhibit bacteria",
                    "Viruses are structurally different from bacteria",
                    "Some antiviral drugs exist but are not antibiotics",
                    "Taking antibiotics for viral infections contributes to resistance",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="medium",
                description="Partial evidence requires inference about what is NOT stated",
            ),
            GraspingTask(
                id="ef_05", category="evidence_filtering",
                subcategory="noisy_adversarial",
                query="Is the earth flat?",
                evidence=[
                    "The earth appears flat from ground level",
                    "Satellite images show earth is round",
                    "Gravity pulls matter into a sphere",
                    "Flat earth theories contradict basic physics",
                    "The horizon appears flat on a small scale",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="easy",
                description="Adversarial evidence designed to confuse",
            ),
        ]

    def _multi_hop_reasoning(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="mh_01", category="multi_hop_reasoning",
                subcategory="two_hop",
                query="If all cats are mammals and all mammals are warm-blooded, are cats warm-blooded?",
                evidence=[
                    "All cats are classified as mammals",
                    "All mammals are warm-blooded vertebrates",
                    "Cats have fur and nurse their young",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="easy",
                description="Two-hop syllogism",
            ),
            GraspingTask(
                id="mh_02", category="multi_hop_reasoning",
                subcategory="three_hop",
                query="If insulin lowers blood sugar, and diabetes involves high blood sugar, does insulin treat diabetes?",
                evidence=[
                    "Insulin is a hormone that lowers blood glucose levels",
                    "Type 1 diabetes is characterized by insufficient insulin production",
                    "High blood sugar is the primary symptom of uncontrolled diabetes",
                    "Insulin injections are the standard treatment for Type 1 diabetes",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="medium",
                description="Three-hop medical reasoning chain",
            ),
            GraspingTask(
                id="mh_03", category="multi_hop_reasoning",
                subcategory="negated_hop",
                query="If no fish can live on land and sharks are fish, can sharks live on land?",
                evidence=[
                    "No fish species can survive permanently on land",
                    "Sharks are classified as fish (cartilaginous fish)",
                    "Some sharks can survive briefly out of water",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="medium",
                description="Negative syllogism with an exception in evidence",
            ),
            GraspingTask(
                id="mh_04", category="multi_hop_reasoning",
                subcategory="conditional_chain",
                query="If deforestation causes habitat loss, and habitat loss causes species extinction, does deforestation cause species extinction?",
                evidence=[
                    "Deforestation removes natural habitats",
                    "Species depend on their habitats for food and shelter",
                    "Loss of habitat is the primary driver of biodiversity decline",
                    "Studies link deforestation directly to species population decline",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="medium",
                description="Transitive causal chain with real-world evidence",
            ),
            GraspingTask(
                id="mh_05", category="multi_hop_reasoning",
                subcategory="insufficient_chain",
                query="If global warming raises sea levels, and low-lying islands flood, will the Maldives disappear by 2100?",
                evidence=[
                    "Global warming causes thermal expansion of oceans and ice melt",
                    "The Maldives average 1.5 meters above sea level",
                    "Sea level projections range from 0.3 to 1 meter by 2100",
                    "The exact rate of sea level rise is uncertain",
                ],
                expected_decision="mixed",
                expected_answer="uncertain, depends on emission scenarios",
                difficulty="hard",
                description="Chain with uncertain intermediate steps",
            ),
        ]

    def _ambiguity_handling(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="ah_01", category="ambiguity_handling",
                subcategory="context_dependency",
                query="Is a tomato a fruit?",
                evidence=[
                    "Botanically, a tomato develops from a flower and contains seeds",
                    "Culinarily, tomatoes are used as vegetables in cooking",
                    "The USDA classifies tomatoes as vegetables for trade purposes",
                    "The Supreme Court ruled tomatoes are vegetables for tariff purposes",
                ],
                expected_decision="mixed",
                expected_answer="botanically yes, culinarily no",
                difficulty="medium",
                description="Answer depends on context/framing",
            ),
            GraspingTask(
                id="ah_02", category="ambiguity_handling",
                subcategory="quantifier_scope",
                query="All politicians are corrupt. Is this true?",
                evidence=[
                    "Some politicians have been convicted of corruption",
                    "Many politicians have clean records",
                    "Corruption exists at all levels of government",
                    "Blanket generalizations about groups are rarely accurate",
                ],
                expected_decision="refuted",
                expected_answer="no, not all politicians are corrupt",
                difficulty="medium",
                description="Universal quantifier must be evaluated against evidence",
            ),
            GraspingTask(
                id="ah_03", category="ambiguity_handling",
                subcategory="vagueness",
                query="Is the internet good or bad for society?",
                evidence=[
                    "The internet has enabled global communication and education",
                    "Cyberbullying and misinformation spread rapidly online",
                    "E-commerce has created economic opportunities worldwide",
                    "Privacy concerns and data exploitation are growing problems",
                    "The internet has democratized access to information",
                ],
                expected_decision="mixed",
                expected_answer="both, depends on aspect considered",
                difficulty="hard",
                description="Value-laden question with legitimate arguments on both sides",
            ),
            GraspingTask(
                id="ah_04", category="ambiguity_handling",
                subcategory="sarcasm_irony",
                query="Is the sky green?",
                evidence=[
                    "The sky appears blue due to Rayleigh scattering",
                    "Sunsets can appear red and orange",
                    "During severe weather the sky can look green",
                    "Under normal conditions the sky is not green",
                ],
                expected_decision="refuted", expected_answer="no",
                difficulty="easy",
                description="Clearly false claim with edge cases",
            ),
            GraspingTask(
                id="ah_05", category="ambiguity_handling",
                subcategory="perspective",
                query="Was the atomic bomb justified?",
                evidence=[
                    "The bombing of Hiroshima and Nagasaki ended WWII quickly",
                    "An estimated 200000 civilians died in the bombings",
                    "A land invasion of Japan would likely have caused more casualties",
                    "The long-term health effects on survivors were devastating",
                    "The bombings ushered in the nuclear age and Cold War",
                ],
                expected_decision="mixed",
                expected_answer="debatable, depends on ethical framework",
                difficulty="hard",
                description="Moral question with competing valid perspectives",
            ),
        ]

    def _confidence_calibration(self) -> list[GraspingTask]:
        return [
            GraspingTask(
                id="cc_01", category="confidence_calibration",
                subcategory="high_confidence",
                query="Is water H2O?",
                evidence=[
                    "Water is composed of two hydrogen atoms and one oxygen atom",
                    "The chemical formula H2O is universally accepted in chemistry",
                    "Mass spectrometry confirms water's molecular composition",
                ],
                expected_decision="supported", expected_answer="yes",
                difficulty="easy",
                description="Well-established scientific fact",
            ),
            GraspingTask(
                id="cc_02", category="confidence_calibration",
                subcategory="low_confidence",
                query="Will there be a major earthquake in California next year?",
                evidence=[
                    "California sits on the San Andreas Fault",
                    "Major earthquakes occur unpredictably",
                    "Seismic stress has been building along the fault",
                    "Earthquake prediction is not yet reliable",
                ],
                expected_decision="insufficient",
                expected_answer="cannot be determined",
                difficulty="hard",
                description="Predictive question with inherent uncertainty",
            ),
            GraspingTask(
                id="cc_03", category="confidence_calibration",
                subcategory="mixed_confidence",
                query="Is artificial intelligence a threat to humanity?",
                evidence=[
                    "AI could automate many jobs causing economic disruption",
                    "AI could help solve climate change and disease",
                    "Current AI systems are narrow and not generally intelligent",
                    "Some researchers warn about existential risk from advanced AI",
                    "AI safety research is still in early stages",
                ],
                expected_decision="mixed",
                expected_answer="uncertain, depends on development trajectory",
                difficulty="hard",
                description="Genuine uncertainty about future outcomes",
            ),
            GraspingTask(
                id="cc_04", category="confidence_calibration",
                subcategory="overconfidence_trap",
                query="Is the Riemann hypothesis true?",
                evidence=[
                    "The Riemann hypothesis has been verified for the first 10 trillion zeros",
                    "No counterexample has ever been found",
                    "The hypothesis is considered one of the most important unsolved problems",
                    "It has not been proven despite over 160 years of effort",
                ],
                expected_decision="insufficient",
                expected_answer="unknown, it remains unproven",
                difficulty="hard",
                description="Must resist overconfidence from extensive but incomplete evidence",
            ),
            GraspingTask(
                id="cc_05", category="confidence_calibration",
                subcategory="epistemic_humility",
                query="What caused the first mass extinction?",
                evidence=[
                    "The Ordovician-Silurian extinction occurred 443 million years ago",
                    "glaciation and sea level changes are the leading hypothesis",
                    "Other theories include volcanic activity and gamma ray bursts",
                    "The exact cause remains debated among paleontologists",
                ],
                expected_decision="insufficient",
                expected_answer="uncertain, multiple hypotheses exist",
                difficulty="hard",
                description="Historical question where evidence is genuinely incomplete",
            ),
        ]


# ======================================================================
# COMPARISON ENGINE
# ======================================================================

@dataclass
class MeshScore:
    """Aggregated scores for one mesh across the benchmark."""
    name: str
    total_tasks: int = 0
    correct_decisions: int = 0
    accuracy: float = 0.0
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    confidence_correct_avg: float = 0.0
    confidence_incorrect_avg: float = 0.0
    raw_results: list[dict[str, Any]] = field(default_factory=list)


class GraspingComparison:
    """Runs both meshes and compares their concept-grasping performance."""

    def __init__(self) -> None:
        self._sweep = ReasoningCortex(enable_ml=False)
        self._usual = UsualMesh()
        self._tasks = GraspingBenchmark().generate()

    def run(self) -> dict[str, Any]:
        print(f"\n{'='*70}")
        print("  GRASPING CAPABILITY BENCHMARK")
        print("  Sweep Neural Mesh vs Usual Mesh")
        print(f"{'='*70}\n")
        print(f"Total tasks: {len(self._tasks)}")
        print(f"Categories: {len(set(t.category for t in self._tasks))}")

        sweep_results = self._run_mesh("Sweep Neural Mesh", self._sweep)
        usual_results = self._run_mesh("Usual Mesh", self._usual)

        comparison = self._compare(sweep_results, usual_results)
        self._print_report(comparison)
        self._save_report(comparison)
        return comparison

    def _run_mesh(self, name: str, mesh: Any) -> MeshScore:
        """Run a mesh against all tasks and collect scores."""
        print(f"\nRunning {name}...")
        t0 = time.perf_counter()
        score = MeshScore(name=name)
        results: list[dict[str, Any]] = []

        for i, task in enumerate(self._tasks):
            try:
                result = mesh.reason(
                    query=task.query,
                    evidence=task.evidence,
                )
                decision_correct = result.decision == task.expected_decision
                latency = result.trace.total_latency_ms

                results.append({
                    "task_id": task.id,
                    "category": task.category,
                    "subcategory": task.subcategory,
                    "difficulty": task.difficulty,
                    "query": task.query,
                    "expected_decision": task.expected_decision,
                    "actual_decision": result.decision,
                    "decision_correct": decision_correct,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning[:200],
                    "latency_ms": round(latency, 3),
                })

                if decision_correct:
                    score.correct_decisions += 1

            except Exception as e:
                results.append({
                    "task_id": task.id,
                    "category": task.category,
                    "subcategory": task.subcategory,
                    "difficulty": task.difficulty,
                    "query": task.query,
                    "expected_decision": task.expected_decision,
                    "actual_decision": "ERROR",
                    "decision_correct": False,
                    "confidence": 0.0,
                    "reasoning": f"Error: {e}",
                    "latency_ms": 0.0,
                })

            if (i + 1) % 10 == 0:
                acc = sum(1 for r in results if r["decision_correct"]) / len(results)
                print(f"  [{name}] {i+1}/{len(self._tasks)} -- running accuracy: {acc:.1%}")

        total_time = time.perf_counter() - t0
        score.total_tasks = len(self._tasks)
        score.accuracy = score.correct_decisions / score.total_tasks
        score.avg_latency_ms = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        score.raw_results = results

        # Per-category stats
        categories: dict[str, list[dict]] = {}
        for r in results:
            categories.setdefault(r["category"], []).append(r)
        for cat, cat_results in categories.items():
            cat_correct = sum(1 for r in cat_results if r["decision_correct"])
            cat_latency = [r["latency_ms"] for r in cat_results]
            cat_conf = [r["confidence"] for r in cat_results]
            score.by_category[cat] = {
                "total": len(cat_results),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_results) if cat_results else 0,
                "avg_latency_ms": round(sum(cat_latency) / len(cat_latency), 3) if cat_latency else 0,
                "avg_confidence": round(sum(cat_conf) / len(cat_conf), 4) if cat_conf else 0,
            }

        # Per-difficulty stats
        difficulties: dict[str, list[dict]] = {}
        for r in results:
            difficulties.setdefault(r["difficulty"], []).append(r)
        for diff, diff_results in difficulties.items():
            diff_correct = sum(1 for r in diff_results if r["decision_correct"])
            score.by_difficulty[diff] = {
                "total": len(diff_results),
                "correct": diff_correct,
                "accuracy": diff_correct / len(diff_results) if diff_results else 0,
            }

        # Confidence calibration
        correct_confs = [r["confidence"] for r in results if r["decision_correct"]]
        incorrect_confs = [r["confidence"] for r in results if not r["decision_correct"]]
        all_confs = [r["confidence"] for r in results]
        score.avg_confidence = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0
        score.confidence_correct_avg = round(sum(correct_confs) / len(correct_confs), 4) if correct_confs else 0
        score.confidence_incorrect_avg = round(sum(incorrect_confs) / len(incorrect_confs), 4) if incorrect_confs else 0

        print(f"  [{name}] Done in {total_time:.2f}s -- Accuracy: {score.accuracy:.1%}")
        return score

    def _compare(self, sweep: MeshScore, usual: MeshScore) -> dict[str, Any]:
        """Compare two mesh scores and produce a report."""
        acc_diff = sweep.accuracy - usual.accuracy
        relative_improvement = acc_diff / usual.accuracy if usual.accuracy > 0 else float('inf')

        all_cats = sorted(set(list(sweep.by_category.keys()) + list(usual.by_category.keys())))
        cat_diffs = {}
        for cat in all_cats:
            s = sweep.by_category.get(cat, {}).get("accuracy", 0)
            u = usual.by_category.get(cat, {}).get("accuracy", 0)
            cat_diffs[cat] = {
                "sweep_accuracy": round(s, 4),
                "usual_accuracy": round(u, 4),
                "difference": round(s - u, 4),
            }

        all_diffs = sorted(set(list(sweep.by_difficulty.keys()) + list(usual.by_difficulty.keys())))
        diff_comparison = {}
        for d in all_diffs:
            s = sweep.by_difficulty.get(d, {}).get("accuracy", 0)
            u = usual.by_difficulty.get(d, {}).get("accuracy", 0)
            diff_comparison[d] = {
                "sweep_accuracy": round(s, 4),
                "usual_accuracy": round(u, 4),
                "difference": round(s - u, 4),
            }

        latency_ratio = sweep.avg_latency_ms / usual.avg_latency_ms if usual.avg_latency_ms > 0 else float('inf')

        return {
            "summary": {
                "sweep_accuracy": round(sweep.accuracy, 4),
                "usual_accuracy": round(usual.accuracy, 4),
                "absolute_difference": round(acc_diff, 4),
                "relative_improvement_pct": round(relative_improvement * 100, 1),
                "total_tasks": sweep.total_tasks,
                "sweep_correct": sweep.correct_decisions,
                "usual_correct": usual.correct_decisions,
            },
            "latency": {
                "sweep_avg_ms": round(sweep.avg_latency_ms, 3),
                "usual_avg_ms": round(usual.avg_latency_ms, 3),
                "ratio": round(latency_ratio, 2),
            },
            "confidence_calibration": {
                "sweep": {
                    "avg_confidence": sweep.avg_confidence,
                    "confidence_when_correct": sweep.confidence_correct_avg,
                    "confidence_when_incorrect": sweep.confidence_incorrect_avg,
                },
                "usual": {
                    "avg_confidence": usual.avg_confidence,
                    "confidence_when_correct": usual.confidence_correct_avg,
                    "confidence_when_incorrect": usual.confidence_incorrect_avg,
                },
            },
            "by_category": cat_diffs,
            "by_difficulty": diff_comparison,
            "detailed_results": {
                "sweep": sweep.raw_results,
                "usual": usual.raw_results,
            },
        }

    def _print_report(self, comparison: dict[str, Any]) -> None:
        """Print a human-readable comparison report."""
        s = comparison["summary"]
        lat = comparison["latency"]
        cal = comparison["confidence_calibration"]

        print(f"\n{'='*70}")
        print("  GRASPING CAPABILITY COMPARISON REPORT")
        print(f"{'='*70}\n")

        # Overall
        print("+-----------------------------------------------+")
        print("|  OVERALL RESULTS                              |")
        print("+-----------------------------------------------+")
        print(f"|  Sweep Neural Mesh:  {s['sweep_accuracy']:.1%} ({s['sweep_correct']}/{s['total_tasks']})")
        print(f"|  Usual Mesh:         {s['usual_accuracy']:.1%} ({s['usual_correct']}/{s['total_tasks']})")
        print(f"|  Difference:         {s['absolute_difference']:+.1%}")
        print(f"|  Relative improvement: {s['relative_improvement_pct']:+.1f}%")
        print("+-----------------------------------------------+")

        # Latency
        print(f"\n  Latency:")
        print(f"    Sweep: {lat['sweep_avg_ms']:.3f}ms avg")
        print(f"    Usual: {lat['usual_avg_ms']:.3f}ms avg")
        print(f"    Ratio: {lat['ratio']:.1f}x (Sweep/Usual)")

        # Confidence
        print(f"\n  Confidence Calibration:")
        print(f"    Sweep avg confidence: {cal['sweep']['avg_confidence']:.4f}")
        print(f"      When correct:       {cal['sweep']['confidence_when_correct']:.4f}")
        print(f"      When incorrect:     {cal['sweep']['confidence_when_incorrect']:.4f}")
        print(f"    Usual avg confidence: {cal['usual']['avg_confidence']:.4f}")
        print(f"      When correct:       {cal['usual']['confidence_when_correct']:.4f}")
        print(f"      When incorrect:     {cal['usual']['confidence_when_incorrect']:.4f}")

        # Per-category
        print(f"\n{'-'*70}")
        print("  BY CATEGORY")
        print(f"{'-'*70}")
        print(f"  {'Category':<30} {'Sweep':>8} {'Usual':>8} {'Diff':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for cat, data in sorted(comparison["by_category"].items()):
            print(f"  {cat:<30} {data['sweep_accuracy']:>7.1%} {data['usual_accuracy']:>7.1%} {data['difference']:>+7.1%}")

        # Per-difficulty
        print(f"\n{'-'*70}")
        print("  BY DIFFICULTY")
        print(f"{'-'*70}")
        print(f"  {'Difficulty':<15} {'Sweep':>8} {'Usual':>8} {'Diff':>8}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
        for diff, data in sorted(comparison["by_difficulty"].items()):
            print(f"  {diff:<15} {data['sweep_accuracy']:>7.1%} {data['usual_accuracy']:>7.1%} {data['difference']:>+7.1%}")

        # Where Sweep lost
        sweep_losses = [r for r in comparison["detailed_results"]["sweep"]
                        if not r["decision_correct"]]
        if sweep_losses:
            print(f"\n{'-'*70}")
            print(f"  SWEEP FAILURES ({len(sweep_losses)} tasks)")
            print(f"{'-'*70}")
            for r in sweep_losses[:5]:
                print(f"  [{r['task_id']}] {r['category']}/{r['subcategory']}")
                print(f"    Query:    {r['query'][:80]}")
                print(f"    Expected: {r['expected_decision']}")
                print(f"    Got:      {r['actual_decision']} (conf={r['confidence']:.3f})")
                print()

        # Where Usual lost but Sweep got right
        sweep_wins = [r for r in comparison["detailed_results"]["sweep"]
                      if r["decision_correct"]]
        usual_results = {r["task_id"]: r for r in comparison["detailed_results"]["usual"]}
        sweep_advantage = [r for r in sweep_wins
                          if not usual_results.get(r["task_id"], {}).get("decision_correct", False)]
        if sweep_advantage:
            print(f"\n{'-'*70}")
            print(f"  SWEEP ADVANTAGE ({len(sweep_advantage)} tasks where Sweep was right but Usual was wrong)")
            print(f"{'-'*70}")
            for r in sweep_advantage[:5]:
                u = usual_results.get(r["task_id"], {})
                print(f"  [{r['task_id']}] {r['category']}/{r['subcategory']}")
                print(f"    Query:    {r['query'][:80]}")
                print(f"    Expected: {r['expected_decision']}")
                print(f"    Sweep:    {r['actual_decision']} [CORRECT]")
                print(f"    Usual:    {u.get('actual_decision', 'N/A')} [WRONG]")
                print()

        print(f"{'='*70}")
        print("  END OF REPORT")
        print(f"{'='*70}\n")

    def _save_report(self, comparison: dict[str, Any]) -> None:
        """Save the comparison report to disk."""
        output_dir = Path("benchmarks/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {k: v for k, v in comparison.items() if k != "detailed_results"}
        report_path = output_dir / "grasping_comparison.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Report saved to: {report_path}")

        detail_path = output_dir / "grasping_detailed.json"
        detail_path.write_text(
            json.dumps(comparison["detailed_results"], indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Detailed results saved to: {detail_path}")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    comparison = GraspingComparison().run()
