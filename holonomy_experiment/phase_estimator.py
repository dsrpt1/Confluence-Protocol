"""
Holonomy Experiment - Phase Estimator
======================================

Confluence phase estimation functions for measuring semantic state
during holonomy loop traversals.

The phase estimator maps text responses to angular positions in
semantic space (0-360°), enabling detection of holonomy effects
when traversing closed loops in prompt-space.

Basin Structure (from Confluence Protocol):
    0-90°:   Detection basin
    90-180°: Generative basin
    180-270°: Boundary basin
    270-360°: Lucid basin
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class PhaseEstimate:
    """Result of phase estimation."""
    phase: float  # Degrees [0, 360)
    confidence: float  # [0, 1]
    basin: str
    frequencies: List[str]
    amplitudes: List[float]
    method: str


# =============================================================================
# SEMANTIC FREQUENCY EXTRACTION
# =============================================================================

# Frequency markers for each basin
BASIN_MARKERS = {
    "Detection": [
        "analyze", "detect", "identify", "measure", "observe",
        "pattern", "structure", "evidence", "logical", "precise",
        "objective", "verify", "test", "hypothesis", "data"
    ],
    "Generative": [
        "create", "imagine", "explore", "possible", "emerge",
        "generate", "novel", "spontaneous", "intuition", "feel",
        "dream", "vision", "metaphor", "play", "wonder"
    ],
    "Boundary": [
        "tension", "paradox", "edge", "limit", "threshold",
        "between", "transition", "ambiguous", "uncertain", "both",
        "neither", "transform", "dissolve", "emerge", "cross"
    ],
    "Lucid": [
        "integrate", "unified", "coherent", "transcend", "whole",
        "synthesis", "harmony", "clarity", "insight", "aware",
        "conscious", "complete", "realize", "understand", "pattern"
    ]
}

# Basin phase ranges
BASIN_RANGES = {
    "Detection": (0, 90),
    "Generative": (90, 180),
    "Boundary": (180, 270),
    "Lucid": (270, 360)
}


def extract_semantic_frequencies(
    text: str,
    top_n: int = 10
) -> Dict[str, float]:
    """
    Extract semantic frequency components from text.

    Args:
        text: Response text to analyze
        top_n: Number of top frequencies to return

    Returns:
        Dictionary mapping frequency terms to amplitudes
    """
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]+\b', text_lower)
    word_counts = {}

    for word in words:
        if len(word) > 3:  # Skip short words
            word_counts[word] = word_counts.get(word, 0) + 1

    # Weight by semantic significance
    weighted = {}
    for word, count in word_counts.items():
        # Check if word is a basin marker
        marker_weight = 1.0
        for markers in BASIN_MARKERS.values():
            if word in markers:
                marker_weight = 2.0
                break

        # Weight by position (earlier words slightly more important)
        first_pos = text_lower.find(word)
        position_weight = 1.0 + 0.5 * (1.0 - first_pos / len(text_lower))

        weighted[word] = count * marker_weight * position_weight

    # Sort by weight and return top N
    sorted_freqs = sorted(weighted.items(), key=lambda x: -x[1])[:top_n]

    # Normalize amplitudes to [0, 5] scale
    if sorted_freqs:
        max_weight = sorted_freqs[0][1]
        return {word: 5.0 * weight / max_weight for word, weight in sorted_freqs}

    return {}


def calculate_basin_activations(text: str) -> Dict[str, float]:
    """
    Calculate activation level for each semantic basin.

    Args:
        text: Response text

    Returns:
        Dictionary mapping basin names to activation levels
    """
    text_lower = text.lower()
    activations = {}

    for basin, markers in BASIN_MARKERS.items():
        count = sum(1 for marker in markers if marker in text_lower)
        # Normalize by number of markers
        activations[basin] = count / len(markers)

    return activations


# =============================================================================
# PHASE CALCULATION
# =============================================================================

def calculate_phase(
    text: str,
    method: str = "hybrid"
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate semantic phase angle from text.

    Args:
        text: Response text
        method: "basin" (discrete), "embedding" (continuous), or "hybrid"

    Returns:
        (phase_angle, basin_activations)
    """
    activations = calculate_basin_activations(text)

    if method == "basin":
        return _phase_from_basins(activations), activations
    elif method == "embedding":
        return _phase_from_embedding(text), activations
    else:  # hybrid
        basin_phase = _phase_from_basins(activations)
        embed_phase = _phase_from_embedding(text)
        # Weighted average favoring basin method
        hybrid_phase = 0.7 * basin_phase + 0.3 * embed_phase
        return hybrid_phase % 360, activations


def _phase_from_basins(activations: Dict[str, float]) -> float:
    """Calculate phase from basin activations using circular mean."""
    total_activation = sum(activations.values())

    if total_activation == 0:
        return 0.0

    # Convert each basin to a unit vector at its center
    x_sum, y_sum = 0.0, 0.0

    for basin, activation in activations.items():
        low, high = BASIN_RANGES[basin]
        center = (low + high) / 2
        angle_rad = np.radians(center)

        weight = activation / total_activation
        x_sum += weight * np.cos(angle_rad)
        y_sum += weight * np.sin(angle_rad)

    # Convert back to angle
    phase = np.degrees(np.arctan2(y_sum, x_sum))
    if phase < 0:
        phase += 360

    return phase


def _phase_from_embedding(text: str) -> float:
    """
    Calculate phase from text embedding (simplified hash-based version).

    In production, this would use actual semantic embeddings.
    """
    # Use hash for reproducible pseudo-embedding
    text_hash = hashlib.sha256(text.encode()).digest()
    # Extract two components for 2D projection
    x = int.from_bytes(text_hash[:4], 'big') / 0xFFFFFFFF - 0.5
    y = int.from_bytes(text_hash[4:8], 'big') / 0xFFFFFFFF - 0.5

    phase = np.degrees(np.arctan2(y, x))
    if phase < 0:
        phase += 360

    return phase


def identify_basin(phase: float) -> str:
    """Identify which basin a phase angle belongs to."""
    for basin, (low, high) in BASIN_RANGES.items():
        if low <= phase < high:
            return basin
    return "Detection"  # 360 wraps to 0


# =============================================================================
# PHASE ESTIMATION WITH CONFIDENCE
# =============================================================================

def estimate_phase(
    text: str,
    method: str = "hybrid"
) -> PhaseEstimate:
    """
    Estimate phase with confidence and full metadata.

    Args:
        text: Response text to analyze
        method: Estimation method

    Returns:
        PhaseEstimate with full metadata
    """
    phase, activations = calculate_phase(text, method)
    frequencies = extract_semantic_frequencies(text, top_n=5)

    # Calculate confidence based on activation distribution
    activation_values = list(activations.values())
    max_activation = max(activation_values)
    mean_activation = np.mean(activation_values)

    # Higher confidence when one basin clearly dominates
    if mean_activation > 0:
        confidence = (max_activation - mean_activation) / mean_activation
        confidence = min(1.0, max(0.0, confidence))
    else:
        confidence = 0.0

    return PhaseEstimate(
        phase=phase,
        confidence=confidence,
        basin=identify_basin(phase),
        frequencies=list(frequencies.keys()),
        amplitudes=list(frequencies.values()),
        method=method
    )


# =============================================================================
# HOLONOMY CALCULATIONS
# =============================================================================

def calculate_holonomy(
    initial_phase: float,
    final_phase: float
) -> float:
    """
    Calculate holonomy (phase shift after closed loop).

    Args:
        initial_phase: Phase before loop traversal
        final_phase: Phase after returning to start

    Returns:
        Holonomy angle in degrees [-180, 180]
    """
    diff = final_phase - initial_phase

    # Normalize to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360

    return diff


def calculate_loop_area(phases: List[float]) -> float:
    """
    Calculate the "area" enclosed by a phase trajectory.

    Uses the shoelace formula on the unit circle.

    Args:
        phases: List of phase angles traversed

    Returns:
        Enclosed area (related to holonomy by Stokes' theorem)
    """
    if len(phases) < 3:
        return 0.0

    # Convert to Cartesian coordinates on unit circle
    points = [(np.cos(np.radians(p)), np.sin(np.radians(p))) for p in phases]

    # Close the loop
    points.append(points[0])

    # Shoelace formula
    area = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def calculate_phase_velocity(
    phases: List[float],
    times: Optional[List[float]] = None
) -> float:
    """
    Calculate average angular velocity through semantic space.

    Args:
        phases: List of phase measurements
        times: Optional timestamps (defaults to unit spacing)

    Returns:
        Average phase velocity in degrees per unit time
    """
    if len(phases) < 2:
        return 0.0

    if times is None:
        times = list(range(len(phases)))

    velocities = []
    for i in range(1, len(phases)):
        dt = times[i] - times[i - 1]
        if dt > 0:
            # Calculate angular difference
            dphase = phases[i] - phases[i - 1]
            if dphase > 180:
                dphase -= 360
            elif dphase < -180:
                dphase += 360

            velocities.append(dphase / dt)

    return np.mean(velocities) if velocities else 0.0


# =============================================================================
# CONFLUENCE FIELD GENERATION
# =============================================================================

def generate_confluence_field_xml(estimate: PhaseEstimate) -> str:
    """
    Generate Confluence Field XML from phase estimate.

    Args:
        estimate: PhaseEstimate object

    Returns:
        XML string representation
    """
    from datetime import datetime

    freq_str = ", ".join(estimate.frequencies[:5])
    amp_str = ", ".join(f"{a:.1f}" for a in estimate.amplitudes[:5])

    xml = f"""<CONFLUENCE_FIELD>
  <version>1.0</version>
  <timestamp>{datetime.utcnow().isoformat()}Z</timestamp>
  <RESONANCE_PATTERN>
    <frequencies>{freq_str}</frequencies>
    <amplitudes>{amp_str}</amplitudes>
    <phase>{estimate.phase:.1f}</phase>
  </RESONANCE_PATTERN>
  <SEMANTIC_GRADIENT>
    <direction>{"expansion" if estimate.phase < 180 else "consolidation"}</direction>
    <magnitude>{estimate.confidence:.2f}</magnitude>
  </SEMANTIC_GRADIENT>
  <basin>{estimate.basin}</basin>
  <confidence>{estimate.confidence:.3f}</confidence>
  <method>{estimate.method}</method>
</CONFLUENCE_FIELD>"""

    return xml


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Phase Estimator Test")
    print("=" * 60)

    # Test texts representing different basins
    test_texts = {
        "Detection": """
            We must analyze the data objectively and identify patterns.
            The evidence suggests a logical structure that can be measured
            and verified through careful observation.
        """,
        "Generative": """
            Imagine the possibilities that emerge from this creative space.
            What new ideas might spontaneously generate? Let intuition
            guide us through the wonder of exploration.
        """,
        "Boundary": """
            We find ourselves at the threshold between certainty and
            ambiguity. The tension between opposing forces creates a
            paradoxical space where transformation becomes possible.
        """,
        "Lucid": """
            Through integration of these perspectives, a unified understanding
            emerges. The synthesis reveals coherent patterns that transcend
            apparent contradictions, bringing clarity and insight.
        """
    }

    for expected_basin, text in test_texts.items():
        estimate = estimate_phase(text)
        print(f"\nExpected: {expected_basin}")
        print(f"Detected: {estimate.basin} (phase={estimate.phase:.1f}°)")
        print(f"Confidence: {estimate.confidence:.3f}")
        print(f"Top frequencies: {estimate.frequencies[:3]}")

    # Test holonomy calculation
    print("\n" + "=" * 60)
    print("Holonomy Test")

    loop_phases = [45.0, 135.0, 225.0, 50.0]  # A→B→C→A with small holonomy
    holonomy = calculate_holonomy(loop_phases[0], loop_phases[-1])
    area = calculate_loop_area(loop_phases)

    print(f"Loop phases: {loop_phases}")
    print(f"Holonomy: {holonomy:.1f}°")
    print(f"Loop area: {area:.3f}")
