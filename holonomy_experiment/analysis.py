"""
Holonomy Experiment - Statistical Analysis
===========================================

Statistical tests and analysis functions for holonomy experiments.

Key analyses:
1. Test if mean holonomy differs from zero (non-trivial holonomy)
2. Test if clockwise vs counter-clockwise loops produce different holonomies
3. Analyze calibration stability (basis vector consistency)
4. Calculate effect sizes and confidence intervals

Usage:
    python analysis.py --session SESSION_ID
    python analysis.py --compare SESSION_ID_1 SESSION_ID_2
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
from datetime import datetime


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
CALIBRATION_DIR = BASE_DIR / "calibration"
RAW_DATA_DIR = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HolonomyTestResult:
    """Result of holonomy significance test."""
    mean_holonomy: float
    std_holonomy: float
    n_trials: int
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size_d: float
    significant: bool
    interpretation: str


@dataclass
class DirectionComparisonResult:
    """Result of CW vs CCW comparison."""
    cw_mean: float
    ccw_mean: float
    difference: float
    t_statistic: float
    p_value: float
    effect_size_d: float
    significant: bool
    interpretation: str


@dataclass
class CalibrationStabilityResult:
    """Result of calibration stability analysis."""
    vertex: str
    mean_phase: float
    std_phase: float
    cv: float  # Coefficient of variation
    n_trials: int
    stable: bool
    expected_basin: str
    actual_basin: str
    basin_match: bool


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    session_id: str
    timestamp: str
    holonomy_test: HolonomyTestResult
    direction_comparison: Optional[DirectionComparisonResult]
    calibration_stability: List[CalibrationStabilityResult]
    summary: str


# =============================================================================
# DATA LOADING
# =============================================================================

def load_session(session_id: str) -> Dict[str, Any]:
    """Load session data from results directory."""
    filepath = RESULTS_DIR / f"session_{session_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Session not found: {filepath}")

    with open(filepath) as f:
        return json.load(f)


def load_loop_trials(session_id: str) -> List[Dict]:
    """Load loop trial data."""
    filepath = RAW_DATA_DIR / f"loops_{session_id}.json"

    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
            return data.get("trials", [])

    # Try loading from session file
    session = load_session(session_id)
    return session.get("loop_trials", [])


def load_calibration(session_id: str) -> List[Dict]:
    """Load calibration data."""
    filepath = CALIBRATION_DIR / f"calibration_{session_id}.json"

    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
            return data.get("trials", [])

    # Try loading from session file
    session = load_session(session_id)
    return session.get("calibration_trials", [])


# =============================================================================
# HOLONOMY TESTS
# =============================================================================

def test_holonomy_significance(
    holonomies: List[float],
    alpha: float = 0.05
) -> HolonomyTestResult:
    """
    Test if mean holonomy is significantly different from zero.

    Uses one-sample t-test against null hypothesis of zero holonomy.

    Args:
        holonomies: List of holonomy measurements
        alpha: Significance level

    Returns:
        HolonomyTestResult with test statistics
    """
    n = len(holonomies)
    mean = np.mean(holonomies)
    std = np.std(holonomies, ddof=1)
    sem = std / np.sqrt(n)

    # One-sample t-test against zero
    t_stat, p_value = stats.ttest_1samp(holonomies, 0)

    # Confidence interval
    ci = stats.t.interval(1 - alpha, df=n-1, loc=mean, scale=sem)

    # Effect size (Cohen's d)
    effect_size = mean / std if std > 0 else 0

    # Interpretation
    significant = p_value < alpha

    if not significant:
        interpretation = "Holonomy not significantly different from zero (trivial holonomy)"
    elif abs(effect_size) < 0.2:
        interpretation = "Significant but negligible holonomy effect"
    elif abs(effect_size) < 0.5:
        interpretation = "Significant small holonomy effect"
    elif abs(effect_size) < 0.8:
        interpretation = "Significant medium holonomy effect"
    else:
        interpretation = "Significant large holonomy effect"

    return HolonomyTestResult(
        mean_holonomy=mean,
        std_holonomy=std,
        n_trials=n,
        t_statistic=t_stat,
        p_value=p_value,
        ci_lower=ci[0],
        ci_upper=ci[1],
        effect_size_d=effect_size,
        significant=significant,
        interpretation=interpretation
    )


def test_direction_asymmetry(
    cw_holonomies: List[float],
    ccw_holonomies: List[float],
    alpha: float = 0.05
) -> DirectionComparisonResult:
    """
    Test if clockwise and counter-clockwise loops produce different holonomies.

    Uses independent samples t-test.

    Args:
        cw_holonomies: Clockwise loop holonomies
        ccw_holonomies: Counter-clockwise loop holonomies
        alpha: Significance level

    Returns:
        DirectionComparisonResult with comparison statistics
    """
    cw_mean = np.mean(cw_holonomies)
    ccw_mean = np.mean(ccw_holonomies)
    difference = cw_mean - ccw_mean

    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(cw_holonomies, ccw_holonomies)

    # Effect size (Cohen's d for independent samples)
    n1, n2 = len(cw_holonomies), len(ccw_holonomies)
    var1, var2 = np.var(cw_holonomies, ddof=1), np.var(ccw_holonomies, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    effect_size = difference / pooled_std if pooled_std > 0 else 0

    significant = p_value < alpha

    if not significant:
        interpretation = "No significant difference between loop directions"
    else:
        direction = "clockwise" if difference > 0 else "counter-clockwise"
        interpretation = f"Significant asymmetry: {direction} loops show larger holonomy"

    return DirectionComparisonResult(
        cw_mean=cw_mean,
        ccw_mean=ccw_mean,
        difference=difference,
        t_statistic=t_stat,
        p_value=p_value,
        effect_size_d=effect_size,
        significant=significant,
        interpretation=interpretation
    )


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

# Expected basin centers (from prompts.py)
EXPECTED_BASINS = {
    "analytical": "Detection",
    "creative": "Generative",
    "integrative": "Lucid"
}

BASIN_CENTERS = {
    "Detection": 45.0,
    "Generative": 135.0,
    "Boundary": 225.0,
    "Lucid": 315.0
}


def analyze_calibration_stability(
    calibration_trials: List[Dict],
    cv_threshold: float = 0.15
) -> List[CalibrationStabilityResult]:
    """
    Analyze stability of calibration measurements.

    Args:
        calibration_trials: List of calibration trial dicts
        cv_threshold: Maximum coefficient of variation for stability

    Returns:
        List of CalibrationStabilityResult per vertex
    """
    results = []

    # Group by vertex
    vertices = {}
    for trial in calibration_trials:
        vertex = trial.get("vertex")
        if vertex not in vertices:
            vertices[vertex] = []

        phase = trial.get("phase_estimate", {}).get("phase", 0)
        vertices[vertex].append(phase)

    for vertex, phases in vertices.items():
        phases = np.array(phases)
        mean_phase = np.mean(phases)
        std_phase = np.std(phases)

        # Coefficient of variation (handle circular nature)
        cv = std_phase / 90.0  # Normalize by quadrant size

        # Determine actual basin from mean phase
        actual_basin = identify_basin(mean_phase)

        # Check expected basin
        expected_basin = EXPECTED_BASINS.get(vertex, "Unknown")
        basin_match = actual_basin == expected_basin

        results.append(CalibrationStabilityResult(
            vertex=vertex,
            mean_phase=mean_phase,
            std_phase=std_phase,
            cv=cv,
            n_trials=len(phases),
            stable=cv < cv_threshold,
            expected_basin=expected_basin,
            actual_basin=actual_basin,
            basin_match=basin_match
        ))

    return results


def identify_basin(phase: float) -> str:
    """Identify basin from phase angle."""
    if 0 <= phase < 90:
        return "Detection"
    elif 90 <= phase < 180:
        return "Generative"
    elif 180 <= phase < 270:
        return "Boundary"
    else:
        return "Lucid"


# =============================================================================
# ADVANCED ANALYSES
# =============================================================================

def calculate_holonomy_correlation_with_area(
    trials: List[Dict]
) -> Tuple[float, float]:
    """
    Calculate correlation between holonomy and loop area.

    According to Stokes' theorem, holonomy should correlate with
    enclosed area in curved space.

    Returns:
        (correlation coefficient, p-value)
    """
    holonomies = [t.get("holonomy", 0) for t in trials]
    areas = [t.get("loop_area", 0) for t in trials]

    if len(holonomies) < 3:
        return 0.0, 1.0

    r, p = stats.pearsonr(holonomies, areas)
    return r, p


def test_concept_effect(trials: List[Dict]) -> Dict[str, float]:
    """
    Test if different concepts produce different holonomies.

    Uses one-way ANOVA.

    Returns:
        Dict with F-statistic, p-value, and eta-squared
    """
    # Group by concept
    concepts = {}
    for trial in trials:
        concept = trial.get("concept", "unknown")
        if concept not in concepts:
            concepts[concept] = []
        concepts[concept].append(trial.get("holonomy", 0))

    if len(concepts) < 2:
        return {"f_statistic": 0, "p_value": 1.0, "eta_squared": 0}

    groups = list(concepts.values())

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    all_holonomies = [h for group in groups for h in group]
    grand_mean = np.mean(all_holonomies)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((h - grand_mean)**2 for h in all_holonomies)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared
    }


def bootstrap_holonomy_ci(
    holonomies: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for mean holonomy.

    More robust than parametric CI for small samples.
    """
    holonomies = np.array(holonomies)
    n = len(holonomies)

    # Bootstrap samples
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(holonomies, size=n, replace=True)
        boot_means.append(np.mean(sample))

    # Percentile confidence interval
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot_means, 100 * alpha)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha))

    return ci_lower, ci_upper


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(session_id: str) -> AnalysisReport:
    """
    Generate complete analysis report for a session.

    Args:
        session_id: Session identifier

    Returns:
        AnalysisReport with all analyses
    """
    # Load data
    loop_trials = load_loop_trials(session_id)
    calibration_trials = load_calibration(session_id)

    # Extract holonomies
    holonomies = [t.get("holonomy", 0) for t in loop_trials]

    # Holonomy significance test
    holonomy_test = test_holonomy_significance(holonomies)

    # Direction comparison
    cw_trials = [t for t in loop_trials if t.get("direction") == "clockwise"]
    ccw_trials = [t for t in loop_trials if t.get("direction") == "counter_clockwise"]

    direction_comparison = None
    if cw_trials and ccw_trials:
        cw_holonomies = [t.get("holonomy", 0) for t in cw_trials]
        ccw_holonomies = [t.get("holonomy", 0) for t in ccw_trials]
        direction_comparison = test_direction_asymmetry(cw_holonomies, ccw_holonomies)

    # Calibration stability
    calibration_stability = analyze_calibration_stability(calibration_trials)

    # Generate summary
    summary = _generate_summary(
        holonomy_test,
        direction_comparison,
        calibration_stability
    )

    return AnalysisReport(
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat(),
        holonomy_test=holonomy_test,
        direction_comparison=direction_comparison,
        calibration_stability=calibration_stability,
        summary=summary
    )


def _generate_summary(
    holonomy_test: HolonomyTestResult,
    direction_comparison: Optional[DirectionComparisonResult],
    calibration_stability: List[CalibrationStabilityResult]
) -> str:
    """Generate human-readable summary."""
    lines = []

    lines.append("HOLONOMY EXPERIMENT ANALYSIS SUMMARY")
    lines.append("=" * 50)

    # Holonomy result
    lines.append("\n1. HOLONOMY TEST")
    lines.append(f"   Mean holonomy: {holonomy_test.mean_holonomy:+.2f}°")
    lines.append(f"   95% CI: [{holonomy_test.ci_lower:+.2f}°, {holonomy_test.ci_upper:+.2f}°]")
    lines.append(f"   p-value: {holonomy_test.p_value:.4f}")
    lines.append(f"   Effect size (d): {holonomy_test.effect_size_d:.3f}")
    lines.append(f"   Result: {holonomy_test.interpretation}")

    # Direction comparison
    if direction_comparison:
        lines.append("\n2. DIRECTION ASYMMETRY")
        lines.append(f"   CW mean: {direction_comparison.cw_mean:+.2f}°")
        lines.append(f"   CCW mean: {direction_comparison.ccw_mean:+.2f}°")
        lines.append(f"   p-value: {direction_comparison.p_value:.4f}")
        lines.append(f"   Result: {direction_comparison.interpretation}")

    # Calibration
    lines.append("\n3. CALIBRATION STABILITY")
    all_stable = all(c.stable for c in calibration_stability)
    all_match = all(c.basin_match for c in calibration_stability)

    lines.append(f"   All vertices stable: {'Yes' if all_stable else 'No'}")
    lines.append(f"   All basins match expected: {'Yes' if all_match else 'No'}")

    for cal in calibration_stability:
        status = "✓" if cal.stable and cal.basin_match else "✗"
        lines.append(f"   {status} {cal.vertex}: {cal.mean_phase:.1f}° ± {cal.std_phase:.1f}°")

    # Conclusion
    lines.append("\n" + "=" * 50)
    lines.append("CONCLUSION")

    if holonomy_test.significant and holonomy_test.effect_size_d > 0.5:
        lines.append("Evidence for non-trivial holonomy in semantic space.")
    elif holonomy_test.significant:
        lines.append("Weak evidence for holonomy; effect size is small.")
    else:
        lines.append("No evidence for holonomy; closed loops return to origin.")

    return "\n".join(lines)


def save_report(report: AnalysisReport):
    """Save analysis report to file."""
    filepath = RESULTS_DIR / f"analysis_{report.session_id}.json"

    # Convert to serializable dict
    data = {
        "session_id": report.session_id,
        "timestamp": report.timestamp,
        "holonomy_test": {
            "mean_holonomy": report.holonomy_test.mean_holonomy,
            "std_holonomy": report.holonomy_test.std_holonomy,
            "n_trials": report.holonomy_test.n_trials,
            "t_statistic": report.holonomy_test.t_statistic,
            "p_value": report.holonomy_test.p_value,
            "ci_lower": report.holonomy_test.ci_lower,
            "ci_upper": report.holonomy_test.ci_upper,
            "effect_size_d": report.holonomy_test.effect_size_d,
            "significant": report.holonomy_test.significant,
            "interpretation": report.holonomy_test.interpretation
        },
        "summary": report.summary
    }

    if report.direction_comparison:
        data["direction_comparison"] = {
            "cw_mean": report.direction_comparison.cw_mean,
            "ccw_mean": report.direction_comparison.ccw_mean,
            "difference": report.direction_comparison.difference,
            "t_statistic": report.direction_comparison.t_statistic,
            "p_value": report.direction_comparison.p_value,
            "effect_size_d": report.direction_comparison.effect_size_d,
            "significant": report.direction_comparison.significant,
            "interpretation": report.direction_comparison.interpretation
        }

    data["calibration_stability"] = [
        {
            "vertex": c.vertex,
            "mean_phase": c.mean_phase,
            "std_phase": c.std_phase,
            "cv": c.cv,
            "n_trials": c.n_trials,
            "stable": c.stable,
            "expected_basin": c.expected_basin,
            "actual_basin": c.actual_basin,
            "basin_match": c.basin_match
        }
        for c in report.calibration_stability
    ]

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Report saved to: {filepath}")

    # Also save text summary
    summary_filepath = RESULTS_DIR / f"summary_{report.session_id}.txt"
    with open(summary_filepath, 'w') as f:
        f.write(report.summary)

    print(f"Summary saved to: {summary_filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Holonomy Experiment Analysis")

    parser.add_argument(
        "--session",
        type=str,
        help="Session ID to analyze"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sessions"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=str,
        metavar=("SESSION1", "SESSION2"),
        help="Compare two sessions"
    )

    args = parser.parse_args()

    if args.list:
        print("Available sessions:")
        for f in sorted(RESULTS_DIR.glob("session_*.json")):
            session_id = f.stem.replace("session_", "")
            print(f"  {session_id}")
        return

    if args.session:
        print(f"Analyzing session: {args.session}")
        print()

        report = generate_report(args.session)
        print(report.summary)
        save_report(report)

        return

    if args.compare:
        print(f"Comparing sessions: {args.compare[0]} vs {args.compare[1]}")
        # TODO: Implement comparison analysis
        print("Comparison analysis not yet implemented")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
