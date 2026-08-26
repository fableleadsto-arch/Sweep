"""
Synaptic Plasticity — the mechanism of learning and mastery.

Implements the biological process of how the brain transforms
from novice (effortful, slow, error-prone) to expert (automatic, fast, "flow").

Based on neuroscience:

    1. LTP/LTD: Long-Term Potentiation/Depression
       - Use strengthens synapses, disuse weakens them
       - Timing-dependent: pre before post = LTP, post before pre = LTD

    2. Myelination: Speed up frequently used pathways
       - Frequently used axons get wrapped in myelin (insulation)
       - Increases signal speed from 1m/s to 100m/s
       - Turns slow conscious thought into rapid automatic action

    3. Circuit Reorganization:
       - Novice: Heavy prefrontal cortex (all centers active, high effort)
       - Practice: Activity shifts to specialized centers (striatum)
       - Mastery: Basal ganglia & cerebellum (automated shortcuts, "flow")

    4. Dendritic Growth: New connections form
       - Frequently co-activated centers grow new synapses
       - Creates specialized pathways that bypass general processing

Phase Transition:

    Novice Phase:
        - High activation across all centers
        - Slow processing, high error rate
        - All signals pass through all centers
        - Prefrontal cortex (explanation_builder) heavily involved

    Practice Phase:
        - Some centers become dominant
        - Processing speeds up
        - Error rate decreases
        - Some shortcuts emerge

    Mastery Phase:
        - Automated shortcuts handle routine queries
        - Only novel/complex queries need full processing
        - Fast, low-effort, "flow state"
        - Basal ganglia handles most decisions automatically
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("sweep.neurons.plasticity")


class MasteryPhase(Enum):
    """The three phases of learning mastery."""
    NOVICE = "novice"           # learning, high effort, slow
    PRACTICE = "practice"       # improving, moderate effort
    MASTERY = "mastery"         # automatic, fast, "flow"


@dataclass
class SynapticState:
    """State of a single synapse including plasticity information."""
    from_center: str
    to_center: str
    weight: float = 1.0
    base_weight: float = 1.0     # original weight before learning
    myelination: float = 0.0     # 0.0–1.0: speed boost from myelination
    activation_count: int = 0
    last_activation: float = 0.0
    ltp_strength: float = 0.0    # accumulated LTP
    ltd_strength: float = 0.0    # accumulated LTD
    dendritic_growth: float = 0.0  # new connection growth
    # ── STDP fields ──
    pre_times: list[float] = field(default_factory=list)  # recent pre-synaptic spike times
    post_times: list[float] = field(default_factory=list)  # recent post-synaptic spike times
    stdp_delta: float = 0.0      # accumulated STDP weight change
    # ── Homeostatic fields ──
    homeostatic_offset: float = 0.0  # homeostatic scaling offset
    target_activity: float = 0.5     # target activity level
    metaplasticity: float = 0.0      # how plasticity itself changes


@dataclass
class CenterState:
    """State of a processing center including phase information."""
    center_name: str
    activation_count: int = 0
    total_processing_time_ms: float = 0.0
    error_count: int = 0         # times this center produced low-quality output
    success_count: int = 0       # times this center produced high-quality output
    current_phase: MasteryPhase = MasteryPhase.NOVICE
    specialization_score: float = 0.0  # how specialized this center has become


@dataclass
class LearningMetrics:
    """Metrics about the learning state of the entire system."""
    overall_phase: MasteryPhase
    phase_scores: dict[str, float]  # center_name → 0.0–1.0 (0=novice, 1=mastery)
    total_activations: int
    total_ltp_events: int
    total_ltd_events: int
    avg_myelination: float
    shortcut_count: int           # automated shortcuts that bypass full processing
    efficiency_gain: float        # how much faster we are vs novice baseline
    # ── STDP metrics ──
    total_stdp_events: int = 0
    avg_stdp_delta: float = 0.0
    avg_metaplasticity: float = 0.0
    # ── Homeostatic metrics ──
    homeostatic_scaled: int = 0
    avg_homeostatic_offset: float = 0.0


class SynapticPlasticity:
    """
    Implements biological learning mechanisms.

    This is the engine that transforms Sweep from a novice (slow, effortful)
    to an expert (fast, automatic, "flow state") through:

    1. LTP/LTD: Synapses strengthen with use, weaken with disuse
    2. Myelination: Frequently used pathways speed up
    3. Circuit reorganization: Processing shifts from general to specialized
    4. Dendritic growth: New connections form between co-activated centers

    The system tracks which centers are used for which types of queries
    and gradually builds specialized pathways that handle routine queries
    automatically, freeing up compute for novel/complex queries.
    """

    def __init__(self) -> None:
        self._synapses: dict[str, SynapticState] = {}
        self._centers: dict[str, CenterState] = {}
        self._learning_rate = 0.1
        self._myelination_rate = 0.05
        self._ltd_decay_rate = 0.02
        self._shortcut_threshold = 10  # activations before shortcut forms
        self._phase_thresholds = {
            MasteryPhase.NOVICE: 0.0,
            MasteryPhase.PRACTICE: 0.3,
            MasteryPhase.MASTERY: 0.7,
        }
        self._total_activations = 0
        self._total_ltp = 0
        self._total_ltd = 0
        self._shortcuts: dict[str, dict[str, Any]] = {}
        # ── STDP parameters ──
        self._stdp_tau_plus = 20.0    # LTP time constant (ms)
        self._stdp_tau_minus = 20.0   # LTD time constant (ms)
        self._stdp_a_plus = 0.01      # LTP learning rate
        self._stdp_a_minus = 0.012    # LTD learning rate (slightly stronger for stability)
        self._stdp_window = 100.0     # max timing window (ms)
        # ── Homeostatic parameters ──
        self._homeostatic_rate = 0.005  # scaling rate
        self._homeostatic_target = 0.5  # target activity level
        self._homeostatic_threshold = 0.3  # min activity before scaling kicks in
        # ── Metaplasticity ──
        self._metaplasticity_rate = 0.01

    def record_activation(
        self,
        from_center: str,
        to_center: str,
        output_quality: float,
        processing_time_ms: float,
    ) -> None:
        """
        Record that a synapse was activated with given quality.

        This drives LTP/LTD, myelination, and phase transitions.
        """
        key = f"{from_center}->{to_center}"
        now = time.time()

        # Get or create synapse state
        if key not in self._synapses:
            self._synapses[key] = SynapticState(
                from_center=from_center,
                to_center=to_center,
            )
        syn = self._synapses[key]
        syn.activation_count += 1
        syn.last_activation = now
        self._total_activations += 1

        # Get or create center state
        for center_name in [from_center, to_center]:
            if center_name not in self._centers:
                self._centers[center_name] = CenterState(center_name=center_name)
        from_state = self._centers[from_center]
        to_state = self._centers[to_center]
        from_state.activation_count += 1
        to_state.activation_count += 1
        to_state.total_processing_time_ms += processing_time_ms

        # ── LTP/LTD: Strengthen or weaken based on quality ──
        if output_quality > 0.5:
            # LTP: Good output → strengthen this pathway
            ltp_delta = self._learning_rate * output_quality
            syn.weight = min(2.0, syn.weight + ltp_delta)
            syn.ltp_strength += ltp_delta
            self._total_ltp += 1
            to_state.success_count += 1
        elif output_quality < 0.3:
            # LTD: Poor output → weaken this pathway
            ltd_delta = self._learning_rate * (1.0 - output_quality)
            syn.weight = max(0.1, syn.weight - ltd_delta)
            syn.ltd_strength += ltd_delta
            self._total_ltd += 1
            to_state.error_count += 1

        # ── Myelination: Speed up frequently used pathways ──
        if syn.activation_count > self._shortcut_threshold:
            # Myelination increases with use, caps at 1.0
            myelination_delta = self._myelination_rate * (
                1.0 - syn.myelination
            )
            syn.myelination = min(1.0, syn.myelination + myelination_delta)

        # ── Phase transition: Update center's mastery phase ──
        self._update_phase(from_state)
        self._update_phase(to_state)

        # ── Dendritic growth: Create new connections for co-activated centers ──
        self._dendritic_growth(from_center, to_center, output_quality)
        logger.debug(
            f"Plasticity record: {from_center}→{to_center} quality={output_quality:.3f} "
            f"(total_act={self._total_activations}, LTP={self._total_ltp}, LTD={self._total_ltd})"
        )

    def _update_phase(self, center: CenterState) -> None:
        """Update a center's mastery phase based on its performance."""
        total = center.success_count + center.error_count
        if total < 5:
            return  # not enough data

        success_rate = center.success_count / total
        avg_time = (
            center.total_processing_time_ms / center.activation_count
            if center.activation_count > 0 else 100.0
        )

        # Compute specialization score (0.0 = novice, 1.0 = mastery)
        # Factors: success rate, speed, activation count
        speed_factor = max(0.0, 1.0 - avg_time / 100.0)  # faster = more mastery
        experience_factor = min(1.0, center.activation_count / 100.0)
        score = success_rate * 0.5 + speed_factor * 0.3 + experience_factor * 0.2
        center.specialization_score = score

        # Determine phase
        if score >= self._phase_thresholds[MasteryPhase.MASTERY]:
            center.current_phase = MasteryPhase.MASTERY
        elif score >= self._phase_thresholds[MasteryPhase.PRACTICE]:
            center.current_phase = MasteryPhase.PRACTICE
        else:
            center.current_phase = MasteryPhase.NOVICE

    def _dendritic_growth(
        self,
        from_center: str,
        to_center: str,
        quality: float,
    ) -> None:
        """
        Grow new connections between co-activated centers.

        Like biological dendritic growth, frequently co-activated
        centers develop direct connections that bypass intermediate processing.
        """
        if quality < 0.6:
            return  # only grow from good activations

        key = f"{from_center}->{to_center}"
        if key in self._synapses:
            self._synapses[key].dendritic_growth = min(
                1.0,
                self._synapses[key].dendritic_growth + 0.05,
            )

    def apply_myelination_speedup(
        self,
        center_name: str,
        base_latency_ms: float,
    ) -> float:
        """
        Apply myelination speedup to a center's processing.

        Returns the adjusted latency after myelination effects.
        """
        if center_name not in self._centers:
            return base_latency_ms

        center = self._centers[center_name]
        # Find all synapses feeding into this center
        incoming_myelination = 0.0
        count = 0
        for key, syn in self._synapses.items():
            if syn.to_center == center_name:
                incoming_myelination += syn.myelination
                count += 1

        if count == 0:
            return base_latency_ms

        avg_myelination = incoming_myelination / count
        # Myelination reduces latency: 0.0 = no speedup, 1.0 = 10x faster
        speedup = 1.0 + avg_myelination * 9.0  # 1x to 10x
        return base_latency_ms / speedup

    def get_shortcut_path(
        self,
        query_type: str,
    ) -> list[str] | None:
        """
        Check if an automated shortcut exists for this query type.

        In the mastery phase, the basal ganglia handles routine queries
        automatically via shortcuts, bypassing the full processing pipeline.
        """
        return self._shortcuts.get(query_type, {}).get("path")

    def register_shortcut(
        self,
        query_type: str,
        path: list[str],
        confidence: float,
    ) -> None:
        """Register a new automated shortcut."""
        self._shortcuts[query_type] = {
            "path": path,
            "confidence": confidence,
            "created_at": time.time(),
            "use_count": 0,
        }

    def decay_unused_synapses(self, decay_rate: float = 0.01) -> None:
        """
        Decay synapses that haven't been used recently.

        Like biological synaptic decay, unused connections weaken over time.
        """
        now = time.time()
        for key, syn in self._synapses.items():
            hours_since = (now - syn.last_activation) / 3600 if syn.last_activation > 0 else 100
            if hours_since > 24:  # unused for 24+ hours
                decay = decay_rate * min(1.0, hours_since / 168)  # caps at 1 week
                syn.weight = max(0.1, syn.weight - decay)

    # ════════════════════════════════════════════════════════════════
    # STDP: Spike-Timing-Dependent Plasticity
    # ════════════════════════════════════════════════════════════════

    def record_stdp_event(
        self,
        from_center: str,
        to_center: str,
        pre_time: float,
        post_time: float,
    ) -> float:
        """
        Record a spike-timing event for STDP learning.

        STDP rule:
        - If pre fires BEFORE post (cause→effect): LTP (strengthen)
        - If post fires BEFORE pre (effect→cause): LTD (weaken)
        - The magnitude decays exponentially with the time difference

        Returns the weight change applied.
        """
        key = f"{from_center}->{to_center}"
        if key not in self._synapses:
            self._synapses[key] = SynapticState(
                from_center=from_center, to_center=to_center,
            )
        syn = self._synapses[key]

        # Record spike times
        syn.pre_times.append(pre_time)
        syn.post_times.append(post_time)
        # Keep only recent spikes (last 20)
        syn.pre_times = syn.pre_times[-20:]
        syn.post_times = syn.post_times[-20:]

        # Compute timing difference
        delta_t = post_time - pre_time  # positive if pre before post

        # Apply STDP rule with exponential decay
        if abs(delta_t) > self._stdp_window / 1000.0:
            return 0.0  # outside timing window

        if delta_t > 0:
            # Pre before post → LTP (causal, strengthen)
            weight_change = self._stdp_a_plus * math.exp(-delta_t / (self._stdp_tau_plus / 1000.0))
            # Metaplasticity: reduce plasticity if synapse is already strong
            plasticity_factor = 1.0 / (1.0 + syn.metaplasticity)
            weight_change *= plasticity_factor
            syn.weight = min(2.0, syn.weight + weight_change)
        else:
            # Post before pre → LTD (anti-causal, weaken)
            weight_change = -self._stdp_a_minus * math.exp(delta_t / (self._stdp_tau_minus / 1000.0))
            plasticity_factor = 1.0 / (1.0 + syn.metaplasticity)
            weight_change *= plasticity_factor
            syn.weight = max(0.1, syn.weight + weight_change)

        syn.stdp_delta += weight_change
        syn.metaplasticity += self._metaplasticity_rate * abs(weight_change)
        logger.debug(f"STDP: {from_center}→{to_center} Δt={delta_t:.4f}s, Δw={weight_change:.6f}")
        return weight_change

    def get_stdp_stats(self) -> dict[str, dict[str, Any]]:
        """Get STDP statistics for all synapses."""
        return {
            key: {
                "stdp_delta": round(syn.stdp_delta, 6),
                "metaplasticity": round(syn.metaplasticity, 4),
                "pre_spikes": len(syn.pre_times),
                "post_spikes": len(syn.post_times),
            }
            for key, syn in self._synapses.items()
        }

    # ════════════════════════════════════════════════════════════════
    # HOMEOSTATIC PLASTICITY: Maintain stability
    # ════════════════════════════════════════════════════════════════

    def homeostatic_scaling(self) -> int:
        """
        Apply homeostatic scaling to all synapses.

        Like biological neurons maintaining stable activity:
        - If a center is overactive → scale down all incoming weights
        - If a center is underactive → scale up all incoming weights
        - This prevents runaway excitation or silencing

        Returns the number of synapses scaled.
        """
        scaled = 0
        for center_name, center in self._centers.items():
            if center.activation_count < 5:
                continue  # not enough data

            # Compute center's current activity level
            success_rate = center.success_count / max(1, center.success_count + center.error_count)
            activity = success_rate

            # Compute scaling factor
            if activity > self._homeostatic_target + self._homeostatic_threshold:
                # Overactive → scale down
                scaling = 1.0 - self._homeostatic_rate * (activity - self._homeostatic_target)
            elif activity < self._homeostatic_target - self._homeostatic_threshold:
                # Underactive → scale up
                scaling = 1.0 + self._homeostatic_rate * (self._homeostatic_target - activity)
            else:
                continue  # within acceptable range

            # Apply scaling to all synapses feeding into this center
            for key, syn in self._synapses.items():
                if syn.to_center == center_name:
                    old_weight = syn.weight
                    syn.weight = max(0.1, min(2.0, syn.weight * scaling))
                    syn.homeostatic_offset += syn.weight - old_weight
                    scaled += 1

        logger.debug(f"Homeostatic scaling: {scaled} synapses adjusted")
        return scaled

    def get_homeostatic_stats(self) -> dict[str, Any]:
        """Get homeostatic scaling statistics."""
        offsets = [syn.homeostatic_offset for syn in self._synapses.values()]
        return {
            "total_synapses_scaled": sum(1 for o in offsets if abs(o) > 0.001),
            "avg_homeostatic_offset": round(
                sum(offsets) / max(1, len(offsets)), 6
            ),
            "max_homeostatic_offset": round(max(offsets) if offsets else 0.0, 6),
        }

    def get_metrics(self) -> LearningMetrics:
        """Get current learning metrics."""
        if not self._centers:
            return LearningMetrics(
                overall_phase=MasteryPhase.NOVICE,
                phase_scores={},
                total_activations=0,
                total_ltp_events=0,
                total_ltd_events=0,
                avg_myelination=0.0,
                shortcut_count=0,
                efficiency_gain=0.0,
            )

        # Determine overall phase
        phase_scores = {
            name: center.specialization_score
            for name, center in self._centers.items()
        }
        avg_score = sum(phase_scores.values()) / len(phase_scores)

        if avg_score >= self._phase_thresholds[MasteryPhase.MASTERY]:
            overall = MasteryPhase.MASTERY
        elif avg_score >= self._phase_thresholds[MasteryPhase.PRACTICE]:
            overall = MasteryPhase.PRACTICE
        else:
            overall = MasteryPhase.NOVICE

        # Average myelination
        myelinations = [syn.myelination for syn in self._synapses.values()]
        avg_myel = sum(myelinations) / len(myelinations) if myelinations else 0.0

        # Efficiency gain: how much faster than novice baseline
        efficiency = avg_myel * 0.5 + avg_score * 0.5  # combined metric

        return LearningMetrics(
            overall_phase=overall,
            phase_scores=phase_scores,
            total_activations=self._total_activations,
            total_ltp_events=self._total_ltp,
            total_ltd_events=self._total_ltd,
            avg_myelination=avg_myel,
            shortcut_count=len(self._shortcuts),
            efficiency_gain=efficiency,
            # STDP metrics
            total_stdp_events=sum(
                len(syn.pre_times) for syn in self._synapses.values()
            ),
            avg_stdp_delta=round(
                sum(syn.stdp_delta for syn in self._synapses.values())
                / max(1, len(self._synapses)), 6
            ),
            avg_metaplasticity=round(
                sum(syn.metaplasticity for syn in self._synapses.values())
                / max(1, len(self._synapses)), 4
            ),
            # Homeostatic metrics
            homeostatic_scaled=sum(
                1 for syn in self._synapses.values()
                if abs(syn.homeostatic_offset) > 0.001
            ),
            avg_homeostatic_offset=round(
                sum(syn.homeostatic_offset for syn in self._synapses.values())
                / max(1, len(self._synapses)), 6
            ),
        )

    @property
    def synapse_states(self) -> dict[str, dict[str, Any]]:
        """Get all synapse states."""
        return {
            key: {
                "weight": round(syn.weight, 4),
                "base_weight": syn.base_weight,
                "myelination": round(syn.myelination, 4),
                "activations": syn.activation_count,
                "ltp": round(syn.ltp_strength, 4),
                "ltd": round(syn.ltd_strength, 4),
                "dendritic_growth": round(syn.dendritic_growth, 4),
            }
            for key, syn in self._synapses.items()
        }

    @property
    def center_states(self) -> dict[str, dict[str, Any]]:
        """Get all center states."""
        return {
            name: {
                "phase": center.current_phase.value,
                "specialization": round(center.specialization_score, 4),
                "activations": center.activation_count,
                "success_rate": round(
                    center.success_count / max(1, center.success_count + center.error_count),
                    4,
                ),
                "avg_latency_ms": round(
                    center.total_processing_time_ms / max(1, center.activation_count),
                    2,
                ),
            }
            for name, center in self._centers.items()
        }
