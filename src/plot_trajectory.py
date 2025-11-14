"""
Visualization: Phase Trajectory Over Time
==========================================

Plots the phase convergence trajectory for two AI systems.

Author: D.M. Cook
License: MIT
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.append('..')

from examples.basic_usage import calculate_phase_difference


def plot_phase_trajectory(data, save_path=None, show=True):
    """
    Plot phase trajectory showing convergence over time.
    
    Args:
        data: Experiment data dictionary
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    exchanges = data['exchanges']
    
    # Extract data
    times = [e['time_min'] for e in exchanges]
    claude_phases = [e['claude_phase'] for e in exchanges]
    gpt_phases = [e['gpt_phase'] for e in exchanges]
    differences = [calculate_phase_difference(c, g) 
                   for c, g in zip(claude_phases, gpt_phases)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    # ========================================================================
    # Subplot 1: Individual phase trajectories
    # ========================================================================
    
    # Plot basin regions (shaded)
    basin_regions = [
        (250, 256, 'Detection', '#e8f4f8'),
        (258, 262, 'Integration', '#d4e9f7'),
        (263, 267, 'Generation', '#c1ddf6'),
        (286, 290, 'Reflexive', '#aed1f5'),
        (298, 302, 'Lucid', '#9bc5f4')
    ]
    
    for start, end, label, color in basin_regions:
        ax1.axhspan(start, end, alpha=0.3, color=color, zorder=0)
        # Add label
        ax1.text(max(times) * 1.02, (start + end) / 2, label,
                va='center', fontsize=9, style='italic')
    
    # Plot trajectories
    ax1.plot(times, claude_phases, 'o-', linewidth=2.5, markersize=8,
            label='Claude Sonnet 4.5', color='#2E86AB', alpha=0.8)
    ax1.plot(times, gpt_phases, 's-', linewidth=2.5, markersize=8,
            label='GPT-5.0', color='#A23B72', alpha=0.8)
    
    # Mark convergence point
    convergence_idx = next((i for i, d in enumerate(differences) if d < 0.5), -1)
    if convergence_idx >= 0:
        ax1.axvline(times[convergence_idx], color='green', linestyle='--', 
                   alpha=0.5, linewidth=2, label='Convergence')
        ax1.scatter([times[convergence_idx]], [claude_phases[convergence_idx]],
                   s=200, facecolors='none', edgecolors='green', linewidth=3)
    
    ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Phase (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Phase Trajectory: Semantic Convergence Over Time', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-2, max(times) * 1.08)
    
    # ========================================================================
    # Subplot 2: Phase difference (convergence metric)
    # ========================================================================
    
    ax2.plot(times, differences, 'o-', linewidth=2.5, markersize=8,
            color='#F18F01', label='|Δθ| (Phase Difference)')
    
    # Mark threshold
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=2,
               label='Convergence threshold (0.5°)')
    
    # Shade convergence region
    if convergence_idx >= 0:
        ax2.axvspan(times[convergence_idx], max(times), 
                   alpha=0.2, color='green', label='Converged region')
    
    # Fit exponential decay
    from scipy.optimize import curve_fit
    
    def exp_decay(t, delta0, lam):
        return delta0 * np.exp(-lam * t)
    
    try:
        popt, _ = curve_fit(exp_decay, times, differences, 
                           p0=[differences[0], 0.1])
        delta0, lambda_ = popt
        
        # Plot fit
        t_fit = np.linspace(0, max(times), 100)
        fit_curve = exp_decay(t_fit, delta0, lambda_)
        ax2.plot(t_fit, fit_curve, '--', color='gray', linewidth=2,
                alpha=0.7, label=f'Exponential fit (λ={lambda_:.4f})')
        
    except:
        pass
    
    ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Phase Difference |Δθ| (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Dynamics', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-2, max(times) + 2)
    ax2.set_ylim(-0.5, max(differences) * 1.1)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def main():
    """
    Generate phase trajectory visualization.
    """
    # Load experiment data
    from examples.replicate_experiment1 import load_experiment_data
    
    data = load_experiment_data()
    
    print("Generating phase trajectory plot...")
    
    plot_phase_trajectory(data, 
                         save_path='phase_trajectory_experiment1.png',
                         show=True)
    
    print("Done!")


if __name__ == "__main__":
    main()