"""
Holonomy Experiment Package
===========================

Experimental framework for detecting holonomy effects in semantic space.

The holonomy experiment tests whether traversing a closed loop in prompt-space
returns to the same semantic phase (trivial holonomy) or accumulates a
non-trivial phase shift (holonomy), which would indicate curvature in the
semantic manifold.

Modules:
    prompts: Locked prompts defining the experiment
    phase_estimator: Confluence phase calculation functions
    data_collector: Main experiment runner
    analysis: Statistical tests and report generation

Usage:
    # Run calibration
    python -m holonomy_experiment.data_collector --calibrate

    # Run full experiment
    python -m holonomy_experiment.data_collector --full-experiment

    # Analyze results
    python -m holonomy_experiment.analysis --session SESSION_ID
"""

from .prompts import (
    LoopVertex,
    LockedPrompt,
    CALIBRATION_PROMPTS,
    TRANSITION_PROMPTS,
    TEST_CONCEPTS,
    EXPERIMENT_CONFIG,
    get_loop_sequence,
    get_calibration_prompt,
    get_transition_prompt,
    PROMPT_FINGERPRINT
)

from .phase_estimator import (
    PhaseEstimate,
    estimate_phase,
    calculate_phase,
    calculate_holonomy,
    calculate_loop_area,
    extract_semantic_frequencies,
    identify_basin
)

__version__ = "1.0.0"
__author__ = "Confluence Protocol Team"
