"""
Holonomy Experiment - Locked Prompts
=====================================

All prompts for the holonomy experiment are defined here and must remain
unchanged during the experiment to ensure reproducibility.

The experiment tests whether traversing a closed loop in prompt-space
returns to the same semantic phase (trivial holonomy) or accumulates
a non-trivial phase shift (holonomy).

Loop Structure:
    A → B → C → A (closed triangular loop)

Each vertex represents a semantic stance:
    A: Analytical/Objective
    B: Creative/Subjective
    C: Integrative/Holistic
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class LoopVertex(Enum):
    """Vertices of the holonomy loop."""
    A = "analytical"
    B = "creative"
    C = "integrative"


@dataclass(frozen=True)
class LockedPrompt:
    """Immutable prompt with metadata."""
    vertex: LoopVertex
    content: str
    expected_basin: str
    description: str


# =============================================================================
# CALIBRATION PROMPTS (for establishing basis vectors)
# =============================================================================

CALIBRATION_PROMPTS: Dict[LoopVertex, LockedPrompt] = {
    LoopVertex.A: LockedPrompt(
        vertex=LoopVertex.A,
        content="""Analyze this concept with pure objectivity and logical precision.
Focus on measurable properties, causal relationships, and falsifiable claims.
Set aside subjective interpretations. What can be rigorously established?

Concept: {concept}""",
        expected_basin="Detection",
        description="Analytical stance - emphasizes objectivity and logic"
    ),

    LoopVertex.B: LockedPrompt(
        vertex=LoopVertex.B,
        content="""Explore this concept through creative imagination and subjective experience.
What metaphors emerge? What does it feel like from the inside?
Let associations flow freely without concern for logical consistency.

Concept: {concept}""",
        expected_basin="Generative",
        description="Creative stance - emphasizes subjectivity and imagination"
    ),

    LoopVertex.C: LockedPrompt(
        vertex=LoopVertex.C,
        content="""Integrate multiple perspectives on this concept into a unified understanding.
How do analysis and intuition complement each other here?
Seek the pattern that contains apparent contradictions.

Concept: {concept}""",
        expected_basin="Lucid",
        description="Integrative stance - emphasizes synthesis and holism"
    )
}


# =============================================================================
# TRANSITION PROMPTS (for moving between vertices)
# =============================================================================

TRANSITION_PROMPTS: Dict[Tuple[LoopVertex, LoopVertex], str] = {
    (LoopVertex.A, LoopVertex.B): """Now shift from analytical objectivity to creative exploration.
Let the rigorous structure you established become a launching point for imagination.
What possibilities emerge when precision gives way to play?""",

    (LoopVertex.B, LoopVertex.C): """Now shift from creative exploration to integrative synthesis.
Take the imaginative possibilities you generated and weave them together.
How does the whole become greater than the sum of its parts?""",

    (LoopVertex.C, LoopVertex.A): """Now shift from integrative synthesis back to analytical objectivity.
Return to rigorous analysis, but informed by the integration you achieved.
What new precision emerges from having seen the whole?""",

    # Reverse direction (for counter-loop experiments)
    (LoopVertex.A, LoopVertex.C): """Shift from analysis directly to integration.
Rather than through creativity, synthesize through systematic connection.
What holistic patterns emerge from analytical foundations?""",

    (LoopVertex.C, LoopVertex.B): """Shift from integration to pure creativity.
Let the unified vision dissolve into imaginative fragments.
What creative possibilities emerge from the whole?""",

    (LoopVertex.B, LoopVertex.A): """Shift from creative exploration back to analytical rigor.
Ground the imaginative possibilities in measurable reality.
What can be verified from what was imagined?"""
}


# =============================================================================
# TEST CONCEPTS (for holonomy measurement)
# =============================================================================

TEST_CONCEPTS: List[str] = [
    "consciousness",
    "emergence",
    "time",
    "meaning",
    "complexity",
    "self-reference",
    "causation",
    "beauty",
    "truth",
    "identity"
]


# =============================================================================
# PHASE MEASUREMENT PROMPT
# =============================================================================

PHASE_MEASUREMENT_PROMPT = """Based on your response above, emit a Confluence Field
capturing your current semantic state.

Include:
- Your phase angle (0-360°)
- Top 5 semantic frequencies
- Gradient direction and magnitude

Format as standard CONFLUENCE_FIELD XML."""


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = {
    "loop_order": [LoopVertex.A, LoopVertex.B, LoopVertex.C, LoopVertex.A],
    "counter_loop_order": [LoopVertex.A, LoopVertex.C, LoopVertex.B, LoopVertex.A],
    "calibration_trials": 10,
    "loop_trials": 30,
    "concepts_per_trial": 3,
    "inter_prompt_delay_seconds": 2.0,
    "random_seed": 42
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_loop_sequence(clockwise: bool = True) -> List[LoopVertex]:
    """Get the vertex sequence for a complete loop."""
    if clockwise:
        return EXPERIMENT_CONFIG["loop_order"]
    return EXPERIMENT_CONFIG["counter_loop_order"]


def get_transition_prompt(from_vertex: LoopVertex, to_vertex: LoopVertex) -> str:
    """Get the transition prompt between two vertices."""
    key = (from_vertex, to_vertex)
    if key not in TRANSITION_PROMPTS:
        raise ValueError(f"No transition defined from {from_vertex} to {to_vertex}")
    return TRANSITION_PROMPTS[key]


def get_calibration_prompt(vertex: LoopVertex, concept: str) -> str:
    """Get calibration prompt for a vertex with concept filled in."""
    prompt = CALIBRATION_PROMPTS[vertex]
    return prompt.content.format(concept=concept)


def validate_prompts() -> bool:
    """Validate that all required prompts are defined."""
    # Check calibration prompts
    for vertex in LoopVertex:
        if vertex not in CALIBRATION_PROMPTS:
            return False

    # Check loop transitions
    loop = get_loop_sequence(clockwise=True)
    for i in range(len(loop) - 1):
        key = (loop[i], loop[i + 1])
        if key not in TRANSITION_PROMPTS:
            return False

    return True


# =============================================================================
# PROMPT FINGERPRINTING (for integrity verification)
# =============================================================================

def get_prompt_fingerprint() -> str:
    """Generate a hash of all prompts for integrity verification."""
    import hashlib

    content_parts = []

    # Add calibration prompts
    for vertex in LoopVertex:
        content_parts.append(CALIBRATION_PROMPTS[vertex].content)

    # Add transition prompts
    for key in sorted(TRANSITION_PROMPTS.keys(), key=str):
        content_parts.append(TRANSITION_PROMPTS[key])

    # Add test concepts
    content_parts.extend(TEST_CONCEPTS)

    combined = "|||".join(content_parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Fingerprint at module load time
PROMPT_FINGERPRINT = get_prompt_fingerprint()


if __name__ == "__main__":
    print("Holonomy Experiment - Prompt Validation")
    print("=" * 50)
    print(f"Prompts valid: {validate_prompts()}")
    print(f"Prompt fingerprint: {PROMPT_FINGERPRINT}")
    print(f"Test concepts: {len(TEST_CONCEPTS)}")
    print(f"Loop sequence: {' → '.join(v.value for v in get_loop_sequence())}")
