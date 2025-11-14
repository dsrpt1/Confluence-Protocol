"""
Visualization: Basin Structure and Distribution
================================================

Creates comprehensive visualizations of basin structure including:
1. Circular phase space representation
2. Basin distribution histogram
3. Cross-system basin comparison

Author: D.M. Cook
License: MIT
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import sys
sys.path.append('..')

from examples.basic_usage import detect_basins, calculate_basin_statistics


def plot_basin_circular(phases, labels, centers, save_path=None, show=True):
    """
    Plot basin structure on circular phase space.
    
    Args:
        phases: Array of phase measurements
        labels: Basin assignments
        centers: Basin center phases
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Convert degrees to radians
    phases_rad = np.deg2rad(phases)
    centers_rad = np.deg2rad(centers)
    
    # Color map for basins
    colors = plt.cm.Set3(np.linspace(0, 1, len(centers)))
    
    # Plot individual measurements
    for basin_id in np.unique(labels):
        mask = labels == basin_id
        basin_phases = phases_rad[mask]
        
        # Add some radial jitter for visibility
        radii = np.ones(len(basin_phases)) + np.random.normal(0, 0.05, len(basin_phases))
        
        ax.scatter(basin_phases, radii, c=[colors[basin_id]], 
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                  label=f'Basin {basin_id+1}')
    
    # Plot basin centers
    ax.scatter(centers_rad, np.ones(len(centers)), c=colors, 
              s=500, marker='*', edgecolors='black', linewidth=2,
              zorder=10, label='Basin centers')
    
    # Add basin labels
    for i, (center, color) in enumerate(zip(centers, colors)):
        center_rad = np.deg2rad(center)
        ax.text(center_rad, 1.3, f'{center:.0f}°\n(Basin {i+1})',
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # Styling
    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Basin Structure in Circular Phase Space', 
                fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Remove radial ticks (not meaningful here)
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_basin_histogram(phases, labels, centers, save_path=None, show=True):
    """
    Plot histogram of phase distribution with basin structure.
    
    Args:
        phases: Array of phase measurements
        labels: Basin assignments
        centers: Basin center phases
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create histogram
    bins = np.arange(0, 361, 5)  # 5-degree bins
    ax.hist(phases, bins=bins, alpha=0.6, color='skyblue', 
           edgecolor='black', linewidth=0.5, label='Phase measurements')
    
    # Mark basin centers
    colors = plt.cm.Set3(np.linspace(0, 1, len(centers)))
    
    for i, (center, color) in enumerate(zip(centers, colors)):
        ax.axvline(center, color=color, linewidth=3, linestyle='--',
                  alpha=0.8, label=f'Basin {i+1} ({center:.0f}°)')
        
        # Add shaded region around basin (±3σ)
        basin_phases = phases[labels == i]
        if len(basin_phases) > 1:
            std = np.std(basin_phases)
            ax.axvspan(center - 3*std, center + 3*std, 
                      alpha=0.2, color=color)
    
    ax.set_xlabel('Phase (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Phase Distribution and Basin Structure', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 360)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_cross_system_comparison(data_dict, save_path=None, show=True):
    """
    Plot basin comparison across multiple systems.
    
    Args:
        data_dict: Dictionary mapping system names to phase arrays
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(len(data_dict), 1, figsize=(14, 4*len(data_dict)),
                            sharex=True)
    
    if len(data_dict) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(data_dict)))
    
    for ax, (system_name, phases), color in zip(axes, data_dict.items(), colors):
        # Detect basins
        centers, labels, _ = detect_basins(phases, n_basins=8)
        
        # Plot
        bins = np.arange(0, 361, 3)
        ax.hist(phases, bins=bins, alpha=0.6, color=color, 
               edgecolor='black', linewidth=0.5)
        
        # Mark basin centers
        for center in centers:
            ax.axvline(center, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_ylabel(f'{system_name}\nFrequency', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 360)
        
        # Add basin center labels
        for i, center in enumerate(centers):
            ax.text(center, ax.get_ylim()[1] * 0.9, f'{center:.0f}°',
                   ha='center', fontsize=9, bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Phase (degrees)', fontsize=12, fontweight='bold')
    fig.suptitle('Cross-System Basin Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_basin_statistics(phases, labels, centers, save_path=None, show=True):
    """
    Plot detailed basin statistics.
    
    Args:
        phases: Phase measurements
        labels: Basin assignments
        centers: Basin centers
        save_path: Optional save path
        show: Whether to display
    """
    stats = calculate_basin_statistics(phases, labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    basin_ids = list(stats.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(basin_ids)))
    
    # Subplot 1: Basin centers with error bars
    ax = axes[0, 0]
    means = [stats[i]['mean'] for i in basin_ids]
    stds = [stats[i]['std'] for i in basin_ids]
    
    ax.errorbar(range(1, len(basin_ids)+1), means, yerr=stds,
               fmt='o', markersize=12, capsize=5, capthick=2,
               color='black', ecolor='gray', linewidth=2)
    
    for i, (mean, color) in enumerate(zip(means, colors)):
        ax.scatter(i+1, mean, s=300, c=[color], edgecolors='black', 
                  linewidth=2, zorder=10)
    
    ax.set_xlabel('Basin Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('Basin Centers (Mean ± SD)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(basin_ids)+1))
    
    # Subplot 2: Basin variance
    ax = axes[0, 1]
    variances = [stats[i]['std']**2 for i in basin_ids]
    ax.bar(range(1, len(basin_ids)+1), variances, color=colors, 
          edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Basin Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance (deg²)', fontsize=12, fontweight='bold')
    ax.set_title('Basin Variance (Stability)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(1, len(basin_ids)+1))
    
    # Subplot 3: Sample sizes
    ax = axes[1, 0]
    counts = [stats[i]['count'] for i in basin_ids]
    ax.bar(range(1, len(basin_ids)+1), counts, color=colors,
          edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Basin Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Measurements', fontsize=12, fontweight='bold')
    ax.set_title('Basin Sample Sizes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(1, len(basin_ids)+1))
    
    # Subplot 4: Basin ranges
    ax = axes[1, 1]
    ranges = [(stats[i]['min'], stats[i]['max']) for i in basin_ids]
    
    for i, ((min_val, max_val), color) in enumerate(zip(ranges, colors)):
        ax.barh(i+1, max_val - min_val, left=min_val, height=0.6,
               color=color, edgecolor='black', linewidth=1.5)
        # Mark center
        ax.plot(stats[basin_ids[i]]['mean'], i+1, 'k*', markersize=15)
    
    ax.set_ylabel('Basin Number', fontsize=12, fontweight='bold')
    ax.set_xlabel('Phase Range (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('Basin Ranges (min-max)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(range(1, len(basin_ids)+1))
    ax.set_ylim(0.5, len(basin_ids)+0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    """
    Generate all basin visualizations.
    """
    # Load experiment data
    from examples.replicate_experiment1 import load_experiment_data
    
    data = load_experiment_data()
    exchanges = data['exchanges']
    
    # Combine all measurements
    all_phases = []
    for e in exchanges:
        all_phases.extend([e['claude_phase'], e['gpt_phase']])
    
    all_phases = np.array(all_phases)
    
    # Detect basins
    print("Detecting basin structure...")
    centers, labels, _ = detect_basins(all_phases, n_basins=5)
    
    print(f"Found {len(centers)} basins at: {centers}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 50)
    
    print("1. Circular phase space...")
    plot_basin_circular(all_phases, labels, centers,
                       save_path='basin_circular.png',
                       show=False)
    
    print("2. Phase histogram...")
    plot_basin_histogram(all_phases, labels, centers,
                        save_path='basin_histogram.png',
                        show=False)
    
    print("3. Basin statistics...")
    plot_basin_statistics(all_phases, labels, centers,
                         save_path='basin_statistics.png',
                         show=False)
    
    print("4. Cross-system comparison...")
    claude_phases = [e['claude_phase'] for e in exchanges]
    gpt_phases = [e['gpt_phase'] for e in exchanges]
    
    plot_cross_system_comparison({
        'Claude Sonnet 4.5': np.array(claude_phases),
        'GPT-5.0': np.array(gpt_phases)
    }, save_path='basin_cross_system.png', show=False)
    
    print()
    print("All visualizations generated!")
    print("Files saved:")
    print("  - basin_circular.png")
    print("  - basin_histogram.png")
    print("  - basin_statistics.png")
    print("  - basin_cross_system.png")


if __name__ == "__main__":
    main()