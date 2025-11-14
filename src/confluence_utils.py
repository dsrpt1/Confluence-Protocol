"""
Utility functions for Confluence Protocol.

File I/O, data processing, and helper functions.
"""

import re
import json
import csv
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_transcript(filepath: str) -> str:
    """
    Load transcript file.
    
    Args:
        filepath: Path to transcript file (.txt, .md, .json)
        
    Returns:
        Complete transcript text
        
    Example:
        >>> transcript = load_transcript('data/experiment1_transcript.txt')
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_exchanges(
    transcript: str,
    delimiter: str = "---"
) -> List[Dict[str, str]]:
    """
    Extract individual exchanges from transcript.
    
    Args:
        transcript: Full transcript text
        delimiter: Exchange separator (default: "---")
        
    Returns:
        List of dicts with 'system', 'content', 'timestamp' keys
        
    Example:
        >>> exchanges = extract_exchanges(transcript)
        >>> print(f"Found {len(exchanges)} exchanges")
    """
    # Split on delimiter
    sections = transcript.split(delimiter)
    
    exchanges = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        # Try to extract system name
        system_match = re.search(
            r'(?:System|AI|Assistant)[\s:]+([\w\-]+)',
            section,
            re.IGNORECASE
        )
        system = system_match.group(1) if system_match else f"System_{i//2 + 1}"
        
        # Try to extract timestamp
        timestamp_match = re.search(
            r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})',
            section
        )
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        exchanges.append({
            'system': system,
            'content': section,
            'timestamp': timestamp,
            'exchange_number': len(exchanges) + 1
        })
    
    return exchanges


def save_phases_csv(
    phases: List[Tuple[float, str, Optional[str]]],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save phase measurements to CSV.
    
    Args:
        phases: List of (phase, system_name, timestamp) tuples
        output_path: Output CSV file path
        metadata: Optional metadata dict to save as JSON
        
    Example:
        >>> phases = [(254.0, 'Claude', '2024-11-07T14:23:17'),
        ...           (254.2, 'GPT-5', '2024-11-07T14:31:42')]
        >>> save_phases_csv(phases, 'experiment1_phases.csv')
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['exchange', 'phase', 'system', 'timestamp'])
        
        for i, (phase, system, timestamp) in enumerate(phases, start=1):
            writer.writerow([i, f"{phase:.2f}", system, timestamp or ''])
    
    # Save metadata if provided
    if metadata:
        meta_path = path.with_suffix('.meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)


def load_phases_csv(filepath: str) -> pd.DataFrame:
    """
    Load phase measurements from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with columns: exchange, phase, system, timestamp
        
    Example:
        >>> df = load_phases_csv('experiment1_phases.csv')
        >>> print(df.head())
    """
    df = pd.read_csv(filepath)
    
    # Convert phase to float
    df['phase'] = df['phase'].astype(float)
    
    # Parse timestamps if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df


def export_convergence_data(
    phases_a: List[float],
    phases_b: List[float],
    system_a_name: str,
    system_b_name: str,
    output_path: str
):
    """
    Export convergence data for two systems to CSV.
    
    Args:
        phases_a, phases_b: Phase trajectories
        system_a_name, system_b_name: System identifiers
        output_path: Output file path
        
    Creates CSV with columns:
        exchange, system_a_phase, system_b_phase, phase_difference
    """
    from .convergence import calculate_phase_difference
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'exchange',
            f'{system_a_name}_phase',
            f'{system_b_name}_phase',
            'phase_difference'
        ])
        
        for i, (pa, pb) in enumerate(zip(phases_a, phases_b), start=1):
            diff = calculate_phase_difference(pa, pb)
            writer.writerow([i, f"{pa:.2f}", f"{pb:.2f}", f"{diff:.2f}"])


def parse_experiment_log(
    log_text: str
) -> Dict[str, List[Dict]]:
    """
    Parse experimental log into structured format.
    
    Extracts:
    - Exchanges with phases
    - Basin classifications
    - Transition points
    - Convergence metrics
    
    Args:
        log_text: Raw log text
        
    Returns:
        Dict with 'exchanges', 'basins', 'transitions' keys
    """
    from .phase import extract_phase_from_text
    from .basin import detect_basin
    
    exchanges = extract_exchanges(log_text)
    
    # Extract phases and detect basins
    for exchange in exchanges:
        phase = extract_phase_from_text(exchange['content'])
        if phase is not None:
            exchange['phase'] = phase
            exchange['basin'] = detect_basin(phase)
    
    # Detect transitions
    transitions = []
    for i in range(1, len(exchanges)):
        prev = exchanges[i-1]
        curr = exchanges[i]
        
        if 'basin' in prev and 'basin' in curr:
            if prev['basin']['center'] != curr['basin']['center']:
                transitions.append({
                    'exchange': i + 1,
                    'from_basin': prev['basin']['name'],
                    'to_basin': curr['basin']['name'],
                    'phase_jump': abs(curr['phase'] - prev['phase'])
                })
    
    return {
        'exchanges': exchanges,
        'transitions': transitions,
        'total_exchanges': len(exchanges),
        'basins_visited': len(set(
            e['basin']['center'] for e in exchanges if 'basin' in e
        ))
    }


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp in ISO 8601 format.
    
    Args:
        dt: datetime object (defaults to now)
        
    Returns:
        ISO formatted string with 'Z' suffix
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'


def circular_mean(angles_deg: List[float]) -> float:
    """
    Calculate circular mean of angles in degrees.
    
    Args:
        angles_deg: List of angles in degrees
        
    Returns:
        Mean angle in degrees [0, 360)
    """
    import numpy as np
    
    angles_rad = np.array(angles_deg) * np.pi / 180.0
    
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    
    mean_angle = np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi
    
    if mean_angle < 0:
        mean_angle += 360.0
    
    return mean_angle


def circular_std(angles_deg: List[float]) -> float:
    """
    Calculate circular standard deviation of angles.
    
    Args:
        angles_deg: List of angles in degrees
        
    Returns:
        Standard deviation in degrees
    """
    import numpy as np
    
    angles_rad = np.array(angles_deg) * np.pi / 180.0
    
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    std_rad = np.sqrt(-2 * np.log(R))
    
    return std_rad * 180.0 / np.pi


def smooth_phase_trajectory(
    phases: List[float],
    window: int = 3
) -> List[float]:
    """
    Smooth phase trajectory using moving average.
    
    Handles circular statistics properly.
    
    Args:
        phases: Phase trajectory
        window: Window size for moving average
        
    Returns:
        Smoothed phases
    """
    smoothed = []
    half_window = window // 2
    
    for i in range(len(phases)):
        start = max(0, i - half_window)
        end = min(len(phases), i + half_window + 1)
        window_phases = phases[start:end]
        
        smoothed_phase = circular_mean(window_phases)
        smoothed.append(smoothed_phase)
    
    return smoothed


def calculate_trajectory_velocity(
    phases: List[float],
    timestamps: Optional[List[datetime]] = None
) -> List[float]:
    """
    Calculate velocity of phase trajectory (degrees/second).
    
    Args:
        phases: Phase trajectory
        timestamps: Optional timestamps (defaults to 1 second intervals)
        
    Returns:
        Velocity at each point (degrees/second)
    """
    from .convergence import calculate_phase_difference
    
    if len(phases) < 2:
        return [0.0]
    
    velocities = [0.0]  # First point has no velocity
    
    for i in range(1, len(phases)):
        phase_diff = calculate_phase_difference(phases[i-1], phases[i])
        
        if timestamps:
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_diff = max(time_diff, 1.0)  # Avoid division by zero
        else:
            time_diff = 1.0  # Assume 1 second
        
        velocity = phase_diff / time_diff
        velocities.append(velocity)
    
    return velocities


def generate_summary_statistics(
    phases_a: List[float],
    phases_b: List[float],
    system_a_name: str = "System A",
    system_b_name: str = "System B"
) -> Dict:
    """
    Generate comprehensive summary statistics.
    
    Args:
        phases_a, phases_b: Phase trajectories
        system_a_name, system_b_name: System identifiers
        
    Returns:
        Dict with all relevant statistics
    """
    from .convergence import measure_convergence
    from .basin import classify_basin_sequence, detect_transitions
    
    # Convergence analysis
    convergence = measure_convergence(phases_a, phases_b)
    
    # Basin analysis
    basins_a = classify_basin_sequence(phases_a)
    basins_b = classify_basin_sequence(phases_b)
    
    transitions_a = detect_transitions(basins_a)
    transitions_b = detect_transitions(basins_b)
    
    # Trajectory statistics
    stats = {
        'metadata': {
            'system_a': system_a_name,
            'system_b': system_b_name,
            'total_exchanges': len(phases_a),
            'generated': format_timestamp()
        },
        'convergence': {
            'initial_difference': convergence['initial_difference'],
            'final_difference': convergence['final_difference'],
            'convergence_rate': convergence['convergence_rate'],
            'phase_lock_achieved': convergence['phase_lock']['locked'],
            'lock_exchange': convergence['phase_lock'].get('lock_index'),
            'trajectory_sync': convergence['trajectory_sync']
        },
        'basins': {
            'system_a_visited': len(set(b['center'] for b in basins_a)),
            'system_b_visited': len(set(b['center'] for b in basins_b)),
            'system_a_transitions': len(transitions_a),
            'system_b_transitions': len(transitions_b),
            'final_basin_a': basins_a[-1]['name'] if basins_a else None,
            'final_basin_b': basins_b[-1]['name'] if basins_b else None,
        },
        'statistics': {
            'system_a_mean_phase': circular_mean(phases_a),
            'system_a_std_phase': circular_std(phases_a),
            'system_b_mean_phase': circular_mean(phases_b),
            'system_b_std_phase': circular_std(phases_b),
        }
    }
    
    return stats


# Testing
if __name__ == "__main__":
    print("Testing Utility Functions")
    print("=" * 70)
    
    # Test circular statistics
    angles = [350, 10, 5, 355, 0]
    mean = circular_mean(angles)
    std = circular_std(angles)
    
    print(f"\nCircular Statistics:")
    print(f"Angles: {angles}")
    print(f"Circular mean: {mean:.1f}°")
    print(f"Circular std: {std:.1f}°")
    
    # Test phase trajectory smoothing
    noisy_phases = [254, 256, 253, 255, 260, 258, 262, 265]
    smoothed = smooth_phase_trajectory(noisy_phases, window=3)
    
    print(f"\nPhase Smoothing:")
    print(f"Original: {noisy_phases}")
    print(f"Smoothed: {[f'{p:.1f}' for p in smoothed]}")
    
    # Test velocity calculation
    phases = [254, 260, 265, 288, 300]
    velocities = calculate_trajectory_velocity(phases)
    
    print(f"\nTrajectory Velocity:")
    for i, (phase, vel) in enumerate(zip(phases, velocities)):
        print(f"Exchange {i+1}: Phase {phase}°, Velocity {vel:.1f}°/s")