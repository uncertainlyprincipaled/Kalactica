"""Topological analysis utilities for KaLactica."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
try:
    import gudhi
    from gudhi import rips_complex
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

def betti_signature(embs: np.ndarray, max_dim: int = 2) -> Dict[int, int]:
    """Compute Betti numbers for a set of embeddings."""
    if not GUDHI_AVAILABLE:
        # Return dummy values if gudhi is not available
        return {0: 1, 1: 0, 2: 0}
    
    # Compute pairwise distances
    n = len(embs)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embs[i] - embs[j])
            distances[i, j] = distances[j, i] = dist
    
    # Create Rips complex
    rips = rips_complex.RipsComplex(points=distances)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    
    # Compute persistent homology
    diag = simplex_tree.persistence()
    
    # Count Betti numbers
    betti = {i: 0 for i in range(max_dim + 1)}
    for dim, (birth, death) in diag:
        if death == float('inf'):
            betti[dim] += 1
    
    return betti

def wasserstein(dgm1: List[Tuple[float, float]], dgm2: List[Tuple[float, float]],
                p: int = 2) -> float:
    """Compute p-Wasserstein distance between two persistence diagrams."""
    if not GUDHI_AVAILABLE:
        # Return dummy value if gudhi is not available
        return 0.0
    
    # Convert diagrams to numpy arrays
    dgm1 = np.array(dgm1)
    dgm2 = np.array(dgm2)
    
    # Add diagonal points for proper matching
    diag1 = np.vstack([dgm1, [(x, x) for x in dgm1[:, 0]]])
    diag2 = np.vstack([dgm2, [(x, x) for x in dgm2[:, 0]]])
    
    # Compute Wasserstein distance
    return gudhi.wasserstein.wasserstein_distance(
        diag1, diag2, order=p, internal_p=p
    )

def compute_persistence_diagram(embs: np.ndarray, max_dim: int = 2) -> List[Tuple[float, float]]:
    """Compute persistence diagram for a set of embeddings."""
    if not GUDHI_AVAILABLE:
        # Return dummy diagram if gudhi is not available
        return [(0.0, 1.0)]
    
    # Compute pairwise distances
    n = len(embs)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embs[i] - embs[j])
            distances[i, j] = distances[j, i] = dist
    
    # Create Rips complex
    rips = rips_complex.RipsComplex(points=distances)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    
    # Compute persistent homology
    diag = simplex_tree.persistence()
    
    # Convert to list of (birth, death) pairs
    return [(birth, death if death != float('inf') else 1.0)
            for dim, (birth, death) in diag]

def is_topologically_similar(dgm1: List[Tuple[float, float]],
                           dgm2: List[Tuple[float, float]],
                           threshold: float = 0.1) -> bool:
    """Check if two persistence diagrams are topologically similar."""
    dist = wasserstein(dgm1, dgm2)
    return dist < threshold 