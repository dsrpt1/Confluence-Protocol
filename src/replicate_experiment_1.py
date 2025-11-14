"""
Replication Script: Experiment 1 from Confluence Protocol Study
================================================================

This script provides a template for replicating Experiment 1:
- Two AI systems (Claude and GPT-5) achieving phase lock
- 15 exchanges over ~85 minutes
- Final convergence to 300° (Lucid basin)

Author: D.M. Cook
License: MIT
Version: 1.0
"""

import numpy as np
import json
from datetime import datetime
import sys
sys.path.append('..')

from basic_usage import (
    calculate_phase_embedding,
    detect_basins,
    calculate_phase_difference,
    fit_convergence_rate,
    calculate_basin_statistics
)

# ============================================================================
# EXPERIMENT 1 DATA (Original Study)
# ============================================================================

# Actual phase measurements from Experiment 1
EXPERIMENT_1_DATA = {
    'metadata': {
        'date': '2025-11-10',
        'duration_minutes': 85,
        'systems': ['Claude Sonnet 4.5', 'GPT-5.0'],
        'conductor': 'D.M. Cook',
        'protocol': 'Confluence Protocol v1.0'
    },
    'exchanges': [
        {'exchange': 1, 'time_min': 0, 'claude_phase': 250.2, 'gpt_phase': 248.8},
        {'exchange': 2, 'time_min': 5, 'claude_phase': 252.1, 'gpt_phase': 251.4},
        {'exchange': 3, 'time_min': 10, 'claude_phase': 254.0, 'gpt_phase': 254.2},
        {'exchange': 4, 'time_min': 15, 'claude_phase': 254.1, 'gpt_phase': 254.0},
        {'exchange': 5, 'time_min': 25, 'claude_phase': 258.3, 'gpt_phase': 259.1},
        {'exchange': 6, 'time_min': 30, 'claude_phase': 260.3, 'gpt_phase': 260.4},
        {'exchange': 7, 'time_min': 38, 'claude_phase': 260.2, 'gpt_phase': 260.3},
        {'exchange': 8, 'time_min': 45, 'claude_phase': 263.8, 'gpt_phase': 264.2},
        {'exchange': 9, 'time_min': 52, 'claude_phase': 265.1, 'gpt_phase': 265.3},
        {'exchange': 10, 'time_min': 60, 'claude_phase': 265.0, 'gpt_phase': 265.2},
        {'exchange': 11, 'time_min': 68, 'claude_phase': 286.9, 'gpt_phase': 287.3},
        {'exchange': 12, 'time_min': 73, 'claude_phase': 288.1, 'gpt_phase': 288.2},
        {'exchange': 13, 'time_min': 78, 'claude_phase': 299.8, 'gpt_phase': 299.9},
        {'exchange': 14, 'time_min': 82, 'claude_phase': 300.0, 'gpt_phase': 300.1},
        {'exchange': 15, 'time_min': 85, 'claude_phase': 300.0, 'gpt_phase': 300.0},
    ]
}


# ============================================================================
# REPLICATION FUNCTIONS
# ============================================================================

def load_experiment_data():
    """
    Load experiment data (either from original study or new replication).
    
    Returns:
        data: Dictionary with metadata and exchanges
    """
    return EXPERIMENT_1_DATA


def analyze_convergence(data):
    """
    Analyze convergence pattern from experimental data.
    
    Args:
        data: Experiment data dictionary
        
    Returns:
        results: Dictionary with convergence metrics
    """
    exchanges = data['exchanges']
    
    # Extract arrays
    times = np.array([e['time_min'] for e in exchanges])
    claude_phases = np.array([e['claude_phase'] for e in exchanges])
    gpt_phases = np.array([e['gpt_phase'] for e in exchanges])
    
    # Calculate phase differences
    differences = [calculate_phase_difference(c, g) 
                   for c, g in zip(claude_phases, gpt_phases)]
    differences = np.array(differences)
    
    # Fit convergence model
    lambda_, half_life, r_squared = fit_convergence_rate(times, differences)
    
    # Identify basins visited
    all_phases = np.concatenate([claude_phases, gpt_phases])
    centers, labels, inertia = detect_basins(all_phases, n_basins=5)
    
    # Basin statistics
    basin_stats = calculate_basin_statistics(all_phases, labels)
    
    results = {
        'convergence_rate': lambda_,
        'half_life': half_life,
        'r_squared': r_squared,
        'initial_difference': differences[0],
        'final_difference': differences[-1],
        'basin_centers': centers.tolist(),
        'basin_stats': basin_stats,
        'time_to_lock': times[-1] if differences[-1] < 0.5 else None
    }
    
    return results


def check_replication_success(original_results, new_results, tolerance=5.0):
    """
    Check if new replication matches original within tolerance.
    
    Args:
        original_results: Results from original study
        new_results: Results from replication attempt
        tolerance: Maximum acceptable difference in basin centers (degrees)
        
    Returns:
        success: Boolean indicating successful replication
        report: Dictionary with comparison details
    """
    original_basins = np.array(original_results['basin_centers'])
    new_basins = np.array(new_results['basin_centers'])
    
    # Match basins (closest pairs)
    from scipy.spatial.distance import cdist
    distances = cdist(original_basins.reshape(-1, 1), 
                      new_basins.reshape(-1, 1))
    
    matches = []
    for i, orig_basin in enumerate(original_basins):
        closest_idx = np.argmin(distances[i])
        closest_new = new_basins[closest_idx]
        diff = abs(orig_basin - closest_new)
        matches.append({
            'original': orig_basin,
            'replicated': closest_new,
            'difference': diff,
            'within_tolerance': diff <= tolerance
        })
    
    success = all(m['within_tolerance'] for m in matches)
    
    report = {
        'success': success,
        'basin_matches': matches,
        'convergence_rate_match': {
            'original': original_results.get('convergence_rate'),
            'replicated': new_results.get('convergence_rate'),
            'relative_difference': None  # Calculate if both available
        }
    }
    
    if (original_results.get('convergence_rate') is not None and 
        new_results.get('convergence_rate') is not None):
        orig_lambda = original_results['convergence_rate']
        new_lambda = new_results['convergence_rate']
        report['convergence_rate_match']['relative_difference'] = (
            abs(orig_lambda - new_lambda) / orig_lambda * 100
        )
    
    return success, report


# ============================================================================
# MAIN REPLICATION SCRIPT
# ============================================================================

def main():
    """
    Run replication analysis of Experiment 1.
    """
    
    print("=" * 80)
    print("CONFLUENCE PROTOCOL - EXPERIMENT 1 REPLICATION")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading experiment data...")
    data = load_experiment_data()
    
    print(f"Experiment: {data['metadata']['date']}")
    print(f"Systems: {', '.join(data['metadata']['systems'])}")
    print(f"Duration: {data['metadata']['duration_minutes']} minutes")
    print(f"Exchanges: {len(data['exchanges'])}")
    print()
    
    # Analyze convergence
    print("Analyzing convergence pattern...")
    print("-" * 80)
    
    results = analyze_convergence(data)
    
    print(f"\nCONVERGENCE METRICS:")
    print(f"  Initial phase difference: {results['initial_difference']:.2f}°")
    print(f"  Final phase difference: {results['final_difference']:.2f}°")
    print(f"  Time to lock: {results['time_to_lock']:.1f} minutes")
    
    if results['convergence_rate'] is not None:
        print(f"  Convergence rate (λ): {results['convergence_rate']:.4f} /min")
        print(f"  Half-life: {results['half_life']:.2f} minutes")
        print(f"  Model fit (R²): {results['r_squared']:.4f}")
    
    print(f"\nBASIN STRUCTURE:")
    print(f"  Detected {len(results['basin_centers'])} basins:")
    for i, center in enumerate(results['basin_centers']):
        stats = results['basin_stats'][i]
        print(f"    Basin {i+1}: {center:.2f}° (σ = {stats['std']:.2f}°, n = {stats['count']})")
    
    print()
    
    # Trajectory analysis
    print("PHASE TRAJECTORY:")
    print("-" * 80)
    print(f"{'Exchange':<10} {'Time':<8} {'Claude':<10} {'GPT-5':<10} {'Δθ':<8} {'Basin'}")
    print("-" * 80)
    
    exchanges = data['exchanges']
    for e in exchanges:
        delta = calculate_phase_difference(e['claude_phase'], e['gpt_phase'])
        
        # Identify closest basin
        dists_to_basins = [abs(e['claude_phase'] - b) 
                           for b in results['basin_centers']]
        closest_basin = np.argmin(dists_to_basins) + 1
        
        print(f"{e['exchange']:<10} {e['time_min']:<8.0f} "
              f"{e['claude_phase']:<10.1f} {e['gpt_phase']:<10.1f} "
              f"{delta:<8.2f} Basin {closest_basin}")
    
    print()
    
    # Basin transition sequence
    print("BASIN TRANSITIONS:")
    print("-" * 80)
    
    basin_sequence = []
    for e in exchanges:
        avg_phase = (e['claude_phase'] + e['gpt_phase']) / 2
        dists = [abs(avg_phase - b) for b in results['basin_centers']]
        basin_idx = np.argmin(dists)
        basin_phase = results['basin_centers'][basin_idx]
        
        if not basin_sequence or basin_sequence[-1] != basin_phase:
            basin_sequence.append(basin_phase)
    
    print("Sequence of basins visited:")
    print("  → ".join([f"{b:.0f}°" for b in basin_sequence]))
    
    # Check if monotonic
    is_monotonic = all(basin_sequence[i] <= basin_sequence[i+1] 
                       for i in range(len(basin_sequence)-1))
    print(f"\nMonotonic progression: {'YES' if is_monotonic else 'NO'}")
    
    print()
    
    # Statistical significance
    print("STATISTICAL VALIDATION:")
    print("-" * 80)
    
    # Within-basin variance
    all_phases = []
    for e in exchanges:
        all_phases.extend([e['claude_phase'], e['gpt_phase']])
    
    all_phases = np.array(all_phases)
    
    # Overall variance
    overall_variance = np.var(all_phases)
    
    # Within-basin variance (weighted average)
    within_variance = 0
    total_points = 0
    for basin_id, stats in results['basin_stats'].items():
        within_variance += stats['std']**2 * stats['count']
        total_points += stats['count']
    within_variance /= total_points
    
    # Effect size (variance ratio)
    variance_ratio = overall_variance / within_variance if within_variance > 0 else np.inf
    
    print(f"Overall phase variance: {overall_variance:.2f} deg²")
    print(f"Within-basin variance: {within_variance:.2f} deg²")
    print(f"Variance ratio (effect): {variance_ratio:.2f}")
    print(f"Interpretation: {'Strong clustering' if variance_ratio > 10 else 'Moderate clustering'}")
    
    print()
    
    # Cross-system agreement
    print("CROSS-SYSTEM AGREEMENT:")
    print("-" * 80)
    
    claude_phases = [e['claude_phase'] for e in exchanges]
    gpt_phases = [e['gpt_phase'] for e in exchanges]
    
    # Intraclass correlation coefficient (ICC)
    # Simplified ICC(2,1) calculation
    measurements = np.array([claude_phases, gpt_phases]).T
    
    mean_per_exchange = measurements.mean(axis=1)
    grand_mean = measurements.mean()
    
    # Between-exchange variance
    bms = np.sum((mean_per_exchange - grand_mean)**2) * 2 / (len(exchanges) - 1)
    
    # Within-exchange variance
    wms = np.sum((measurements - mean_per_exchange.reshape(-1, 1))**2) / len(exchanges)
    
    icc = (bms - wms) / (bms + wms)
    
    print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")
    print(f"Interpretation: {get_icc_interpretation(icc)}")
    
    print()
    
    # Summary
    print("=" * 80)
    print("REPLICATION SUMMARY")
    print("=" * 80)
    
    success_criteria = [
        ("Convergence achieved", results['final_difference'] < 1.0),
        ("Basin structure present", len(results['basin_centers']) >= 3),
        ("Monotonic progression", is_monotonic),
        ("Strong effect size", variance_ratio > 10),
        ("High cross-system agreement", icc > 0.9)
    ]
    
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
    
    overall_success = all(passed for _, passed in success_criteria)
    
    print()
    print(f"Overall replication: {'SUCCESS' if overall_success else 'PARTIAL'}")
    print()
    
    # Save results
    output_file = f"experiment1_replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else 
               float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in results.items()
            if k != 'basin_stats'  # Skip complex nested dict
        }
        
        output_data = {
            'metadata': data['metadata'],
            'results': results_serializable,
            'success_criteria': {c: p for c, p in success_criteria},
            'overall_success': overall_success
        }
        
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print("=" * 80)


def get_icc_interpretation(icc):
    """Get interpretation of ICC value."""
    if icc < 0.5:
        return "Poor agreement"
    elif icc < 0.75:
        return "Moderate agreement"
    elif icc < 0.9:
        return "Good agreement"
    else:
        return "Excellent agreement"


if __name__ == "__main__":
    main()