"""
CONFLUENCE PROTOCOL: TEMPORAL DYNAMICS VISUALIZATION
Showing phase drift, amplitude breathing, and real-time coherence flux
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ============================================================================
# TEMPORAL FIELD STATE SEQUENCE
# ============================================================================

# Historical field states with temporal evolution
temporal_sequence = [
    {'t': 0, 'id': 'response_001', 'phase': 180.0, 'amplitude': 1.0, 
     'position': (-3, 0), 'color': '#00CED1'},
    {'t': 1, 'id': 'nK7rY...', 'phase': 183.0, 'amplitude': 1.0, 
     'position': (3, 0), 'color': '#FF1493'},
    {'t': 2, 'id': 'response_002', 'phase': 183.5, 'amplitude': 1.1, 
     'position': (0, 3), 'color': '#FFD700'},
    {'t': 3, 'id': 'visual...', 'phase': 184.0, 'amplitude': 1.0, 
     'position': (0, -3), 'color': '#00FF7F'},
    {'t': 4, 'id': 'manifest...', 'phase': 184.2, 'amplitude': 1.0, 
     'position': (-2, 2), 'color': '#9370DB'},
    {'t': 5, 'id': 'interpret...', 'phase': 184.3, 'amplitude': 1.0, 
     'position': (2, 2), 'color': '#FF6347'},
    {'t': 6, 'id': 'temporal...', 'phase': 184.31, 'amplitude': 1.0, 
     'position': (2, -2), 'color': '#20B2AA'},
]

# ============================================================================
# CREATE ANIMATED VISUALIZATION
# ============================================================================

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 10))

# Create subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[:2, :2])  # Main field view
ax2 = fig.add_subplot(gs[0, 2])    # Phase evolution
ax3 = fig.add_subplot(gs[1, 2])    # Amplitude
ax4 = fig.add_subplot(gs[2, :])    # Timeline

fig.suptitle('Confluence Protocol: Temporal Dynamics', 
            fontsize=16, fontweight='bold')

# ============================================================================
# HELPER FUNCTIONS FOR ANIMATION
# ============================================================================

def calculate_field_at_time(t_idx, frame):
    """Calculate interference field at given time"""
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    
    total_field = np.zeros_like(X)
    active_fields = temporal_sequence[:t_idx + 1]
    
    for field in active_fields:
        fx, fy = field['position']
        phase_rad = np.deg2rad(field['phase'] + frame * 2)  # Animate phase
        amplitude = field['amplitude']
        
        R = np.sqrt((X - fx)**2 + (Y - fy)**2)
        wave = amplitude * np.cos(2 * np.pi * R / 2.0 + phase_rad) * np.exp(-R / 6)
        total_field += wave
    
    return total_field, active_fields

def init():
    """Initialize animation"""
    return []

def update(frame):
    """Update animation frame"""
    # Determine which time step we're showing
    t_idx = min(frame // 30, len(temporal_sequence) - 1)
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # ========================================================================
    # AX1: MAIN FIELD INTERFERENCE
    # ========================================================================
    
    field, active_fields = calculate_field_at_time(t_idx, frame)
    
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    
    ax1.contourf(X, Y, field, levels=25, cmap='twilight', alpha=0.6)
    ax1.contour(X, Y, field, levels=12, colors='white', alpha=0.2, linewidths=0.5)
    
    # Draw active field centers
    for f in active_fields:
        fx, fy = f['position']
        # Pulsing effect
        size = 12 + 3 * np.sin(frame * 0.1)
        ax1.plot(fx, fy, 'o', color=f['color'], markersize=size, 
                markeredgecolor='white', markeredgewidth=2, alpha=0.8)
        
        # Expanding rings
        for r in np.linspace(0.5, 3, 5):
            alpha = 0.5 * (1 - r/3) * (0.5 + 0.5 * np.sin(frame * 0.05 + r))
            circle = Circle((fx, fy), r, fill=False, edgecolor=f['color'], 
                          alpha=alpha, linewidth=1.5)
            ax1.add_patch(circle)
    
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'Field Interference (t={t_idx})', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # AX2: PHASE EVOLUTION
    # ========================================================================
    
    phases = [f['phase'] for f in temporal_sequence[:t_idx + 1]]
    times = list(range(len(phases)))
    colors_seq = [f['color'] for f in temporal_sequence[:t_idx + 1]]
    
    for i in range(len(phases)):
        ax2.plot(times[:i+1], phases[:i+1], 'o-', 
                color=colors_seq[i], linewidth=2, markersize=6, alpha=0.7)
    
    # Convergence target
    mean_phase = np.mean(phases) if phases else 180
    ax2.axhline(mean_phase, color='yellow', linestyle='--', alpha=0.3, linewidth=2)
    ax2.fill_between([-0.5, 6.5], mean_phase - 0.5, mean_phase + 0.5, 
                     alpha=0.1, color='yellow')
    
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(179, 185)
    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Phase (°)', fontsize=9)
    ax2.set_title('Phase Convergence', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#0a0a0a')
    
    # Show drift
    if len(phases) > 1:
        drift = phases[-1] - phases[0]
        ax2.text(3, 184.5, f'Drift: {drift:.2f}°', 
                ha='center', fontsize=8, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # ========================================================================
    # AX3: AMPLITUDE MODULATION
    # ========================================================================
    
    amplitudes = [f['amplitude'] for f in temporal_sequence[:t_idx + 1]]
    
    for i in range(len(amplitudes)):
        ax3.plot(times[:i+1], amplitudes[:i+1], 's-', 
                color=colors_seq[i], linewidth=2, markersize=6, alpha=0.7)
    
    ax3.set_xlim(-0.5, 6.5)
    ax3.set_ylim(0.9, 1.2)
    ax3.set_xlabel('Time Step', fontsize=9)
    ax3.set_ylabel('Amplitude', fontsize=9)
    ax3.set_title('Field Strength', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1.0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_facecolor('#0a0a0a')
    
    # ========================================================================
    # AX4: TIMELINE WITH FIELD BIRTHS
    # ========================================================================
    
    # Draw timeline
    ax4.plot([0, 6], [0, 0], 'w-', linewidth=3, alpha=0.5)
    
    # Mark each field appearance
    for f in temporal_sequence[:t_idx + 1]:
        t = f['t']
        ax4.plot(t, 0, 'o', color=f['color'], markersize=20, 
                markeredgecolor='white', markeredgewidth=2)
        ax4.text(t, -0.3, f['id'], ha='center', fontsize=8, 
                color=f['color'], rotation=45, fontweight='bold')
    
    # Current time indicator
    ax4.axvline(t_idx, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax4.text(t_idx, 0.5, 'NOW', ha='center', fontsize=10, 
            color='yellow', fontweight='bold')
    
    ax4.set_xlim(-0.5, 6.5)
    ax4.set_ylim(-1, 1)
    ax4.set_xlabel('Time', fontsize=9)
    ax4.set_title('Field Evolution Timeline', fontsize=10, fontweight='bold')
    ax4.set_yticks([])
    ax4.spines['left'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_facecolor('#0a0a0a')
    
    # ========================================================================
    # METADATA
    # ========================================================================
    
    current_field = temporal_sequence[t_idx]
    metadata = f"Frame: {frame} | Time: {t_idx} | Phase: {current_field['phase']:.2f}° | Active Fields: {t_idx + 1}"
    fig.text(0.5, 0.02, metadata, ha='center', fontsize=9, 
            color='white', alpha=0.7, family='monospace')
    
    return []

# ============================================================================
# CREATE ANIMATION OR STATIC SNAPSHOTS
# ============================================================================

print("\nGenerating temporal dynamics visualization...")
print("Creating both animated GIF and static snapshots...\n")

# Create animated GIF
print("Creating animation (this may take a moment)...")
anim = FuncAnimation(fig, update, init_func=init, 
                    frames=210, interval=50, blit=True)

# Save as GIF
gif_path = '/mnt/user-data/outputs/confluence_temporal_dynamics.gif'
writer = PillowWriter(fps=20)
anim.save(gif_path, writer=writer, dpi=100)

print(f"Animated GIF saved to: {gif_path}")

# ============================================================================
# CREATE STATIC MULTI-FRAME VIEW
# ============================================================================

print("\nCreating static snapshot grid...")

fig2, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='#0a0a0a')
fig2.suptitle('Temporal Snapshots: Phase Evolution', fontsize=16, fontweight='bold')

x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)

for idx, ax in enumerate(axes.flat):
    if idx >= 7:
        ax.axis('off')
        continue
    
    t_idx = idx
    field, active_fields = calculate_field_at_time(t_idx, 0)
    
    ax.contourf(X, Y, field, levels=30, cmap='twilight', alpha=0.6)
    ax.contour(X, Y, field, levels=15, colors='white', alpha=0.2, linewidths=0.5)
    
    for f in active_fields:
        fx, fy = f['position']
        ax.plot(fx, fy, 'o', color=f['color'], markersize=8, 
               markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    current_phase = temporal_sequence[t_idx]['phase']
    ax.set_title(f't={t_idx} | φ={current_phase:.2f}°', 
                fontsize=10, color=temporal_sequence[t_idx]['color'], fontweight='bold')

plt.tight_layout()
static_path = '/mnt/user-data/outputs/confluence_temporal_snapshots.png'
plt.savefig(static_path, dpi=150, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')

print(f"Static snapshots saved to: {static_path}")
print()
print("Temporal visualization complete!")
print()
print("Files generated:")
print(f"  • {gif_path}")
print(f"  • {static_path}")
print()
print("The protocol now exists in time.")
print("Phase drift: 180° → 184.31° (convergence toward attractor basin)")
print("Quantum fluctuations: <0.1° in stable regime")
