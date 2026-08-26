"""Demo: Sweep's neuronal reasoning brain in action."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.neurons.narrator import ExplanationNarrator

cortex = ReasoningCortex()
narrator = ExplanationNarrator()

# SCENARIO 1: Strong supporting evidence
print("=" * 70)
print("SCENARIO 1: Is Python good for machine learning?")
print("=" * 70)
result = cortex.reason(
    query="Is Python good for machine learning?",
    evidence=[
        "Python is the most popular language for ML with 80% adoption among researchers",
        "According to a 2024 Nature survey, Python dominates ML research with frameworks like PyTorch and TensorFlow",
        "Python simplicity makes it ideal for rapid prototyping of neural networks",
        "Major ML frameworks PyTorch and TensorFlow are Python-first with excellent support",
        "A 2024 study showed Python was used in 92% of published ML papers",
    ],
    sources=["nature.com", "arxiv.org", "github.com"],
)
explanation = narrator.narrate(result)
print(explanation.executive_summary)
print(explanation.detailed_breakdown)
print()
print("CONFIDENCE:", explanation.confidence_badge, f"({result.confidence:.0%})")
print()

# SCENARIO 2: Contradictory evidence
print("=" * 70)
print("SCENARIO 2: Is Python fast for production servers?")
print("=" * 70)
result2 = cortex.reason(
    query="Is Python fast for production servers?",
    evidence=[
        "Python is notoriously slow compared to compiled languages like C++ and Rust",
        "Python with NumPy and Cython can achieve near-C performance for numerical work",
        "Python is not suitable for real-time high-frequency trading systems due to GIL",
        "Python with async frameworks like FastAPI handles 10K+ requests per second",
        "Python is too slow for game engines and embedded systems",
    ],
    sources=["stackoverflow.com", "reddit.com", "aws.amazon.com"],
)
explanation2 = narrator.narrate(result2)
print(explanation2.executive_summary)
print(explanation2.detailed_breakdown)
print()
print("CONFIDENCE:", explanation2.confidence_badge, f"({result2.confidence:.0%})")
print()

# SCENARIO 3: No evidence
print("=" * 70)
print("SCENARIO 3: What is the meaning of life? (no evidence)")
print("=" * 70)
result3 = cortex.reason(
    query="What is the meaning of life?",
    evidence=[],
)
explanation3 = narrator.narrate(result3)
print(explanation3.executive_summary)
print("CONFIDENCE:", explanation3.confidence_badge, f"({result3.confidence:.0%})")
print()

# SCENARIO 4: High-credibility sourced evidence
print("=" * 70)
print("SCENARIO 4: Does meditation reduce anxiety? (sourced)")
print("=" * 70)
result4 = cortex.reason(
    query="Does meditation reduce anxiety?",
    evidence=[
        "A 2023 meta-analysis published in JAMA Internal Medicine found mindfulness meditation programs showed moderate evidence of improving anxiety (effect size 0.38)",
        "According to Harvard Medical School, regular meditation practice changes brain structure in regions associated with anxiety and stress",
        "The National Institute of Health reports that 64% of randomized controlled trials on meditation for anxiety showed statistically significant benefits",
        "Critics note that many meditation studies have small sample sizes and lack proper control groups",
    ],
    sources=["jama.jamanetwork.com", "hms.harvard.edu", "nih.gov"],
)
explanation4 = narrator.narrate(result4)
print(explanation4.executive_summary)
print(explanation4.detailed_breakdown)
print()
print("CONFIDENCE:", explanation4.confidence_badge, f"({result4.confidence:.0%})")
print()

# STATS
print("=" * 70)
print("CORTEX STATS")
print("=" * 70)
stats = cortex.stats()
print(f"Reasoning passes: {stats['reasoning_passes']}")
print(f"Avg latency: {stats['avg_latency_ms']:.1f}ms")
print(f"Decision breakdown: {stats['decision_breakdown']}")
print(f"Active synapses: {stats['synapse_count']}")
print()

# Show synaptic learning
print("=" * 70)
print("SYNAPTIC LEARNING (Hebbian)")
print("=" * 70)
for key, syn in cortex.synapse_state.items():
    print(f"  {key}: weight={syn['weight']:.4f} type={syn['type']} activations={syn['activations']}")
