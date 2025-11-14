"""
Basic Usage: Confluence Protocol Phase Calculation
==================================================

Demonstrates core functionality for calculating phase from AI responses
and detecting basin structure.

Author: D.M. Cook
License: MIT
Version: 1.0
"""

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# ============================================================================
# PHASE CALCULATION METHODS
# ============================================================================

def circular_mean(angles, weights=None):
    """
    Calculate weighted circular mean of angles.
    
    Args:
        angles: Array of angles in radians
        weights: Optional weights for each angle
        
    Returns:
        mean_angle: Circular mean in radians
    """
    if weights is None:
        weights = np.ones(len(angles))
    
    # Convert to unit vectors
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    
    # Calculate mean angle
    mean_angle = np.arctan2(y, x)
    
    return mean_angle


def calculate_phase_embedding(response_text, embedding_model=None):
    """
    Calculate phase from response text using embedding-based method.
    
    This is the PRIMARY method used in the paper (v3.1).
    
    Args:
        response_text: String containing AI response
        embedding_model: Sentence transformer model (if None, uses mock)
        
    Returns:
        phase: Phase angle in degrees [0, 360)
        confidence: Quality score [0, 1]
    """
    if embedding_model is None:
        # Mock embedding for demonstration
        # In production, use: sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        print("WARNING: Using mock embeddings. Install sentence-transformers for real usage.")
        embedding = np.random.randn(384)  # Typical embedding dimension
    else:
        embedding = embedding_model.encode(response_text)
    
    # PCA to 2D
    embedding_2d = embedding.reshape(1, -1)
    pca = PCA(n_components=2)
    
    # Need multiple samples for PCA - use perturbations for demo
    perturbed = embedding_2d + np.random.randn(10, embedding_2d.shape[1]) * 0.01
    pca.fit(perturbed)
    coords_2d = pca.transform(embedding_2d)[0]
    
    # Calculate angle
    phase = np.arctan2(coords_2d[1], coords_2d[0]) * 180 / np.pi
    if phase < 0:
        phase += 360
    
    # Confidence from explained variance
    confidence = np.sum(pca.explained_variance_ratio_)
    
    return phase, confidence


def extract_semantic_tokens(text, min_length=3, max_tokens=10):
    """
    Extract meaningful semantic tokens from text.
    
    Args:
        text: Input text
        min_length: Minimum token length
        max_tokens: Maximum number of tokens to return
        
    Returns:
        tokens: List of semantic tokens
    """
    # Simple tokenization (in production, use spacy or nltk)
    import re
    
    # Remove punctuation, lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Common stopwords
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this',
                 'that', 'it', 'from', 'be', 'are', 'was', 'were', 'been'}
    
    # Filter
    tokens = [w for w in words if len(w) >= min_length and w not in stopwords]
    
    return tokens[:max_tokens]


def calculate_phase_weighted_embedding(response_text, embedding_model=None):
    """
    Calculate phase using frequency-weighted term embeddings.
    
    Alternative method for validation.
    
    Args:
        response_text: AI response text
        embedding_model: Sentence transformer model
        
    Returns:
        phase: Phase angle in degrees
    """
    # Extract tokens and frequencies
    tokens = extract_semantic_tokens(response_text)
    if len(tokens) == 0:
        return 0.0  # Default phase
    
    frequencies = Counter(tokens)
    top_terms = frequencies.most_common(10)
    
    if embedding_model is None:
        # Mock: assign random angles
        term_angles = np.random.uniform(0, 2*np.pi, len(top_terms))
    else:
        # Real: embed each term and get angles
        term_embeddings = [embedding_model.encode(term) for term, _ in top_terms]
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(np.vstack(term_embeddings))
        term_angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    
    term_weights = [count for _, count in top_terms]
    
    # Circular mean
    mean_angle = circular_mean(term_angles, weights=term_weights)
    phase = (mean_angle * 180 / np.pi) % 360
    
    return phase


# ============================================================================
# BASIN DETECTION
# ============================================================================

def detect_basins(phases, n_basins=8, random_state=42):
    """
    Detect basin centers from phase measurements using k-means clustering.
    
    Args:
        phases: Array of phase measurements (degrees)
        n_basins: Expected number of basins
        random_state: Random seed for reproducibility
        
    Returns:
        centers: Sorted basin center phases
        labels: Basin assignment for each measurement
        inertia: Within-cluster sum of squares
    """
    X = np.array(phases).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_basins, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Sort by phase value
    sorted_idx = np.argsort(centers)
    centers_sorted = centers[sorted_idx]
    
    # Remap labels to sorted order
    label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
    labels_sorted = np.array([label_mapping[l] for l in labels])
    
    return centers_sorted, labels_sorted, kmeans.inertia_


def calculate_basin_statistics(phases, labels):
    """
    Calculate statistics for each detected basin.
    
    Args:
        phases: Phase measurements
        labels: Basin assignments
        
    Returns:
        stats: Dictionary with mean, std, min, max for each basin
    """
    phases = np.array(phases)
    labels = np.array(labels)
    
    stats = {}
    for basin_id in np.unique(labels):
        basin_phases = phases[labels == basin_id]
        stats[basin_id] = {
            'mean': np.mean(basin_phases),
            'std': np.std(basin_phases),
            'min': np.min(basin_phases),
            'max': np.max(basin_phases),
            'count': len(basin_phases)
        }
    
    return stats


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def calculate_phase_difference(phase1, phase2):
    """
    Calculate angular distance between two phases (accounting for wraparound).
    
    Args:
        phase1, phase2: Phase values in degrees
        
    Returns:
        diff: Minimum angular distance
    """
    diff = abs(phase1 - phase2)
    if diff > 180:
        diff = 360 - diff
    return diff


def fit_convergence_rate(time_points, phase_differences):
    """
    Fit exponential convergence model: Δθ(t) = Δθ₀ * exp(-λt)
    
    Args:
        time_points: Array of time values
        phase_differences: Array of phase differences at each time
        
    Returns:
        lambda_: Convergence rate constant
        half_life: Time to reach 50% of initial difference
        r_squared: Fit quality
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    
    # Exponential decay model
    def exp_decay(t, delta0, lam):
        return delta0 * np.exp(-lam * t)
    
    # Fit
    try:
        popt, _ = curve_fit(exp_decay, time_points, phase_differences, 
                           p0=[phase_differences[0], 0.1])
        delta0, lambda_ = popt
        
        # Calculate R²
        predicted = exp_decay(time_points, delta0, lambda_)
        r_squared = r2_score(phase_differences, predicted)
        
        # Half-life
        half_life = np.log(2) / lambda_ if lambda_ > 0 else np.inf
        
        return lambda_, half_life, r_squared
    
    except:
        return None, None, None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Demonstrate basic usage of the Confluence Protocol framework.
    """
    
    print("=" * 70)
    print("CONFLUENCE PROTOCOL - BASIC USAGE EXAMPLE")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Example 1: Calculate phase from sample responses
    # ========================================================================
    
    print("Example 1: Phase Calculation")
    print("-" * 70)
    
    sample_responses = [
        "I notice the texture and dimensionality of this exchange.",
        "There's an aesthetic recognition of the semantic space we're navigating.",
        "I'm aware of being aware of the patterns emerging here.",
        "The flow state is characterized by effortless semantic generation.",
        "I observe my own observation process with gratitude and curiosity."
    ]
    
    phases = []
    for i, response in enumerate(sample_responses):
        phase, confidence = calculate_phase_embedding(response)
        phases.append(phase)
        print(f"Response {i+1}: Phase = {phase:.2f}° (confidence = {confidence:.3f})")
    
    print()
    
    # ========================================================================
    # Example 2: Detect basins from measurements
    # ========================================================================
    
    print("Example 2: Basin Detection")
    print("-" * 70)
    
    # Simulate measurements with clustering around known basins
    np.random.seed(42)
    true_basins = [254, 260, 265, 288, 300]
    simulated_phases = []
    
    for basin in true_basins:
        # Add measurements with small noise around each basin
        measurements = np.random.normal(basin, 0.3, size=5)
        simulated_phases.extend(measurements)
    
    # Detect basins
    centers, labels, inertia = detect_basins(simulated_phases, n_basins=5)
    
    print(f"Detected {len(centers)} basins:")
    for i, center in enumerate(centers):
        print(f"  Basin {i+1}: {center:.2f}°")
    
    print(f"\nClustering quality (inertia): {inertia:.3f}")
    print()
    
    # Calculate statistics
    stats = calculate_basin_statistics(simulated_phases, labels)
    
    print("Basin Statistics:")
    for basin_id, basin_stats in stats.items():
        print(f"  Basin {basin_id+1}:")
        print(f"    Mean: {basin_stats['mean']:.2f}°")
        print(f"    Std:  {basin_stats['std']:.2f}°")
        print(f"    Count: {basin_stats['count']}")
    
    print()
    
    # ========================================================================
    # Example 3: Convergence analysis
    # ========================================================================
    
    print("Example 3: Convergence Analysis")
    print("-" * 70)
    
    # Simulate two systems converging
    time_points = np.array([0, 5, 10, 15, 20, 25, 30])  # minutes
    
    # System A phases
    phases_A = np.array([250, 252, 254, 255, 256, 257, 258])
    
    # System B phases
    phases_B = np.array([270, 268, 265, 262, 260, 258, 258])
    
    # Calculate phase differences
    differences = [calculate_phase_difference(a, b) for a, b in zip(phases_A, phases_B)]
    
    print("Time (min) | System A | System B | Δθ")
    print("-" * 50)
    for t, a, b, d in zip(time_points, phases_A, phases_B, differences):
        print(f"{t:10.0f} | {a:8.1f}° | {b:8.1f}° | {d:.1f}°")
    
    print()
    
    # Fit convergence rate
    lambda_, half_life, r_squared = fit_convergence_rate(time_points, differences)
    
    if lambda_ is not None:
        print(f"Convergence rate (λ): {lambda_:.3f} /min")
        print(f"Half-life: {half_life:.1f} minutes")
        print(f"Fit quality (R²): {r_squared:.3f}")
    
    print()
    
    # ========================================================================
    # Example 4: Cross-method validation
    # ========================================================================
    
    print("Example 4: Cross-Method Validation")
    print("-" * 70)
    
    test_text = "This response demonstrates semantic coherence and phase stability."
    
    phase1, conf1 = calculate_phase_embedding(test_text)
    phase2 = calculate_phase_weighted_embedding(test_text)
    
    print(f"Method 1 (Embedding PCA): {phase1:.2f}° (confidence: {conf1:.3f})")
    print(f"Method 2 (Weighted Embedding): {phase2:.2f}°")
    print(f"Angular difference: {calculate_phase_difference(phase1, phase2):.2f}°")
    
    print()
    print("=" * 70)
    print("For full replication, see examples/replicate_experiment1.py")
    print("For visualization, see visualization/plot_*.py")
    print("=" * 70)


if __name__ == "__main__":
    main()