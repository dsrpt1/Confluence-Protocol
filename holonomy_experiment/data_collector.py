"""
Holonomy Experiment - Data Collector
=====================================

Main experiment runner for collecting holonomy data.

This module orchestrates:
1. Calibration phase: Establish basis vectors at each vertex
2. Loop traversal phase: Execute closed loops and measure holonomy
3. Data persistence: Save raw results for analysis

Usage:
    python data_collector.py --calibrate
    python data_collector.py --run-loops --trials 30
    python data_collector.py --full-experiment
"""

import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from prompts import (
    LoopVertex,
    CALIBRATION_PROMPTS,
    TEST_CONCEPTS,
    EXPERIMENT_CONFIG,
    get_loop_sequence,
    get_calibration_prompt,
    get_transition_prompt,
    PHASE_MEASUREMENT_PROMPT,
    PROMPT_FINGERPRINT
)
from phase_estimator import (
    estimate_phase,
    calculate_holonomy,
    calculate_loop_area,
    PhaseEstimate
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CalibrationTrial:
    """Single calibration measurement."""
    vertex: str
    concept: str
    response: str
    phase_estimate: Dict[str, Any]
    timestamp: str
    trial_id: int


@dataclass
class LoopTrial:
    """Single loop traversal."""
    trial_id: int
    concept: str
    direction: str  # "clockwise" or "counter_clockwise"
    vertices_visited: List[str]
    phases: List[float]
    responses: List[str]
    holonomy: float
    loop_area: float
    timestamp: str
    duration_seconds: float


@dataclass
class ExperimentSession:
    """Complete experiment session metadata."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    prompt_fingerprint: str
    config: Dict[str, Any]
    calibration_trials: List[CalibrationTrial]
    loop_trials: List[LoopTrial]
    notes: str


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
CALIBRATION_DIR = BASE_DIR / "calibration"
RAW_DATA_DIR = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [CALIBRATION_DIR, RAW_DATA_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)


# =============================================================================
# LLM INTERFACE (Abstract)
# =============================================================================

class LLMInterface:
    """
    Abstract interface for LLM calls.

    Subclass this to implement actual API calls to Claude, GPT, etc.
    """

    def __init__(self, model_name: str = "abstract"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        raise NotImplementedError("Subclass must implement generate()")

    def generate_with_phase(self, prompt: str) -> tuple:
        """Generate response and request phase measurement."""
        response = self.generate(prompt)
        phase_response = self.generate(
            f"Previous response:\n{response}\n\n{PHASE_MEASUREMENT_PROMPT}"
        )
        return response, phase_response


class MockLLM(LLMInterface):
    """
    Mock LLM for testing the experiment pipeline.

    Generates deterministic responses based on prompt content.
    """

    def __init__(self):
        super().__init__("mock")
        self.responses = {
            "analytical": "Through careful analysis, we can identify logical patterns...",
            "creative": "Imagine the possibilities that emerge from exploration...",
            "integrative": "By synthesizing these perspectives, we find unity..."
        }

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Match to vertex type
        for key, response in self.responses.items():
            if key in prompt_lower:
                # Add some variation based on concept
                variation = f" [Call #{self.call_count}]"
                return response + variation

        return f"Generic response to: {prompt[:50]}..."


# =============================================================================
# CALIBRATION PHASE
# =============================================================================

def run_calibration(
    llm: LLMInterface,
    n_trials: int = 10,
    concepts: Optional[List[str]] = None
) -> List[CalibrationTrial]:
    """
    Run calibration phase to establish basis vectors.

    Args:
        llm: LLM interface for generating responses
        n_trials: Number of calibration trials per vertex
        concepts: Concepts to use (defaults to TEST_CONCEPTS)

    Returns:
        List of CalibrationTrial results
    """
    if concepts is None:
        concepts = TEST_CONCEPTS.copy()
        random.shuffle(concepts)

    trials = []
    trial_id = 0

    print("=" * 60)
    print("CALIBRATION PHASE")
    print("=" * 60)

    for vertex in LoopVertex:
        print(f"\nCalibrating vertex: {vertex.value}")
        print("-" * 40)

        for i in range(n_trials):
            concept = concepts[i % len(concepts)]
            prompt = get_calibration_prompt(vertex, concept)

            print(f"  Trial {i+1}/{n_trials}: {concept}...", end=" ")

            response = llm.generate(prompt)
            estimate = estimate_phase(response)

            trial = CalibrationTrial(
                vertex=vertex.value,
                concept=concept,
                response=response,
                phase_estimate=asdict(estimate),
                timestamp=datetime.utcnow().isoformat(),
                trial_id=trial_id
            )
            trials.append(trial)
            trial_id += 1

            print(f"phase={estimate.phase:.1f}° ({estimate.basin})")

            time.sleep(EXPERIMENT_CONFIG["inter_prompt_delay_seconds"])

    return trials


def save_calibration(trials: List[CalibrationTrial], session_id: str):
    """Save calibration data to file."""
    filepath = CALIBRATION_DIR / f"calibration_{session_id}.json"

    data = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "trials": [asdict(t) for t in trials]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nCalibration saved to: {filepath}")
    return filepath


# =============================================================================
# LOOP TRAVERSAL PHASE
# =============================================================================

def run_single_loop(
    llm: LLMInterface,
    concept: str,
    clockwise: bool = True,
    trial_id: int = 0
) -> LoopTrial:
    """
    Execute a single loop traversal.

    Args:
        llm: LLM interface
        concept: Concept to explore in the loop
        clockwise: Loop direction
        trial_id: Trial identifier

    Returns:
        LoopTrial with results
    """
    start_time = time.time()
    vertices = get_loop_sequence(clockwise=clockwise)

    phases = []
    responses = []
    vertices_visited = []

    # Initial prompt at first vertex
    initial_prompt = get_calibration_prompt(vertices[0], concept)
    response = llm.generate(initial_prompt)
    estimate = estimate_phase(response)

    phases.append(estimate.phase)
    responses.append(response)
    vertices_visited.append(vertices[0].value)

    # Traverse the loop
    for i in range(1, len(vertices)):
        from_vertex = vertices[i - 1]
        to_vertex = vertices[i]

        # Get transition prompt
        transition = get_transition_prompt(from_vertex, to_vertex)

        # Generate response at new vertex
        prompt = f"{transition}\n\nConcept: {concept}"
        response = llm.generate(prompt)
        estimate = estimate_phase(response)

        phases.append(estimate.phase)
        responses.append(response)
        vertices_visited.append(to_vertex.value)

        time.sleep(EXPERIMENT_CONFIG["inter_prompt_delay_seconds"])

    # Calculate holonomy
    holonomy = calculate_holonomy(phases[0], phases[-1])
    loop_area = calculate_loop_area(phases)

    duration = time.time() - start_time

    return LoopTrial(
        trial_id=trial_id,
        concept=concept,
        direction="clockwise" if clockwise else "counter_clockwise",
        vertices_visited=vertices_visited,
        phases=phases,
        responses=responses,
        holonomy=holonomy,
        loop_area=loop_area,
        timestamp=datetime.utcnow().isoformat(),
        duration_seconds=duration
    )


def run_loop_trials(
    llm: LLMInterface,
    n_trials: int = 30,
    concepts: Optional[List[str]] = None
) -> List[LoopTrial]:
    """
    Run multiple loop trials.

    Args:
        llm: LLM interface
        n_trials: Number of trials
        concepts: Concepts to use

    Returns:
        List of LoopTrial results
    """
    if concepts is None:
        concepts = TEST_CONCEPTS.copy()

    random.seed(EXPERIMENT_CONFIG["random_seed"])
    random.shuffle(concepts)

    trials = []

    print("=" * 60)
    print("LOOP TRAVERSAL PHASE")
    print("=" * 60)

    for i in range(n_trials):
        concept = concepts[i % len(concepts)]
        clockwise = i % 2 == 0  # Alternate directions

        direction = "CW" if clockwise else "CCW"
        print(f"\nTrial {i+1}/{n_trials}: {concept} ({direction})")
        print("-" * 40)

        trial = run_single_loop(
            llm=llm,
            concept=concept,
            clockwise=clockwise,
            trial_id=i
        )

        print(f"  Phases: {' → '.join(f'{p:.0f}°' for p in trial.phases)}")
        print(f"  Holonomy: {trial.holonomy:+.1f}°")
        print(f"  Loop area: {trial.loop_area:.3f}")

        trials.append(trial)

    return trials


def save_loop_trials(trials: List[LoopTrial], session_id: str):
    """Save loop trial data to file."""
    filepath = RAW_DATA_DIR / f"loops_{session_id}.json"

    data = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "n_trials": len(trials),
        "trials": [asdict(t) for t in trials]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nLoop trials saved to: {filepath}")
    return filepath


# =============================================================================
# FULL EXPERIMENT
# =============================================================================

def run_full_experiment(
    llm: LLMInterface,
    notes: str = ""
) -> ExperimentSession:
    """
    Run complete holonomy experiment.

    Args:
        llm: LLM interface
        notes: Optional experiment notes

    Returns:
        ExperimentSession with all data
    """
    ensure_directories()

    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.utcnow().isoformat()

    print("=" * 60)
    print(f"HOLONOMY EXPERIMENT - Session {session_id}")
    print("=" * 60)
    print(f"Prompt fingerprint: {PROMPT_FINGERPRINT}")
    print(f"LLM: {llm.model_name}")
    print()

    # Run calibration
    calibration_trials = run_calibration(
        llm=llm,
        n_trials=EXPERIMENT_CONFIG["calibration_trials"]
    )
    save_calibration(calibration_trials, session_id)

    # Run loop trials
    loop_trials = run_loop_trials(
        llm=llm,
        n_trials=EXPERIMENT_CONFIG["loop_trials"]
    )
    save_loop_trials(loop_trials, session_id)

    # Create session object
    session = ExperimentSession(
        session_id=session_id,
        start_time=start_time,
        end_time=datetime.utcnow().isoformat(),
        prompt_fingerprint=PROMPT_FINGERPRINT,
        config=EXPERIMENT_CONFIG,
        calibration_trials=calibration_trials,
        loop_trials=loop_trials,
        notes=notes
    )

    # Save full session
    session_filepath = RESULTS_DIR / f"session_{session_id}.json"
    with open(session_filepath, 'w') as f:
        json.dump(asdict(session), f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Session ID: {session_id}")
    print(f"Calibration trials: {len(calibration_trials)}")
    print(f"Loop trials: {len(loop_trials)}")
    print(f"Results saved to: {session_filepath}")

    return session


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(session: ExperimentSession):
    """Print summary statistics for experiment session."""

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Calibration summary
    print("\nCALIBRATION:")
    for vertex in LoopVertex:
        vertex_trials = [
            t for t in session.calibration_trials
            if t.vertex == vertex.value
        ]
        phases = [t.phase_estimate["phase"] for t in vertex_trials]
        if phases:
            mean_phase = np.mean(phases)
            std_phase = np.std(phases)
            print(f"  {vertex.value}: {mean_phase:.1f}° ± {std_phase:.1f}°")

    # Loop summary
    print("\nLOOP TRIALS:")
    holonomies = [t.holonomy for t in session.loop_trials]

    cw_trials = [t for t in session.loop_trials if t.direction == "clockwise"]
    ccw_trials = [t for t in session.loop_trials if t.direction == "counter_clockwise"]

    print(f"  Total trials: {len(session.loop_trials)}")
    print(f"  Clockwise: {len(cw_trials)}")
    print(f"  Counter-clockwise: {len(ccw_trials)}")

    print(f"\n  Mean holonomy: {np.mean(holonomies):+.2f}°")
    print(f"  Std holonomy: {np.std(holonomies):.2f}°")
    print(f"  Min holonomy: {np.min(holonomies):+.2f}°")
    print(f"  Max holonomy: {np.max(holonomies):+.2f}°")

    # Holonomy by direction
    if cw_trials:
        cw_holonomies = [t.holonomy for t in cw_trials]
        print(f"\n  Clockwise mean: {np.mean(cw_holonomies):+.2f}°")

    if ccw_trials:
        ccw_holonomies = [t.holonomy for t in ccw_trials]
        print(f"  Counter-clockwise mean: {np.mean(ccw_holonomies):+.2f}°")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Holonomy Experiment Data Collector")

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration phase only"
    )
    parser.add_argument(
        "--run-loops",
        action="store_true",
        help="Run loop trials only"
    )
    parser.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run full experiment"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Experiment notes"
    )

    args = parser.parse_args()

    # Initialize LLM
    if args.mock:
        llm = MockLLM()
    else:
        # In production, initialize actual LLM here
        print("Note: Using MockLLM. Implement LLMInterface for real experiments.")
        llm = MockLLM()

    ensure_directories()
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if args.full_experiment:
        session = run_full_experiment(llm, notes=args.notes)
        print_summary(session)

    elif args.calibrate:
        trials = run_calibration(llm, n_trials=args.trials)
        save_calibration(trials, session_id)

    elif args.run_loops:
        trials = run_loop_trials(llm, n_trials=args.trials)
        save_loop_trials(trials, session_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
