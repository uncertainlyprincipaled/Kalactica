"""Neural collapse metrics for KaLactica."""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def collapse_stats(feats: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute neural collapse statistics for features and labels.
    
    Args:
        feats: Feature matrix of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
    
    Returns:
        Dictionary containing:
        - intra_class_dist: Average intra-class cosine distance
        - inter_class_dist: Average inter-class cosine distance
        - nc_index: Neural collapse index (intra/inter ratio)
    """
    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_classes = len(le.classes_)
    
    # Normalize features
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    
    # Compute class means
    class_means = np.zeros((n_classes, feats.shape[1]))
    for i in range(n_classes):
        class_means[i] = np.mean(feats[labels == i], axis=0)
    class_means = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
    
    # Compute intra-class distances
    intra_dists = []
    for i in range(n_classes):
        class_feats = feats[labels == i]
        if len(class_feats) > 0:
            dists = 1 - cosine_similarity(class_feats, class_means[i:i+1])
            intra_dists.extend(dists.flatten())
    intra_class_dist = np.mean(intra_dists) if intra_dists else 0.0
    
    # Compute inter-class distances
    inter_dists = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            dist = 1 - cosine_similarity(class_means[i:i+1], class_means[j:j+1])
            inter_dists.append(dist[0, 0])
    inter_class_dist = np.mean(inter_dists) if inter_dists else 0.0
    
    # Compute neural collapse index
    nc_index = intra_class_dist / inter_class_dist if inter_class_dist > 0 else float('inf')
    
    return {
        "intra_class_dist": float(intra_class_dist),
        "inter_class_dist": float(inter_class_dist),
        "nc_index": float(nc_index)
    }

def is_collapsed(feats: np.ndarray, labels: np.ndarray,
                threshold: float = 0.05) -> Tuple[bool, Dict[str, float]]:
    """Check if features exhibit neural collapse.
    
    Args:
        feats: Feature matrix of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        threshold: Maximum allowed neural collapse index
    
    Returns:
        Tuple of (is_collapsed, stats)
    """
    stats = collapse_stats(feats, labels)
    return stats["nc_index"] <= threshold, stats

def compute_class_means(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute normalized class means for features.
    
    Args:
        feats: Feature matrix of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
    
    Returns:
        Array of shape (n_classes, n_features) containing normalized class means
    """
    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_classes = len(le.classes_)
    
    # Normalize features
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    
    # Compute class means
    class_means = np.zeros((n_classes, feats.shape[1]))
    for i in range(n_classes):
        class_means[i] = np.mean(feats[labels == i], axis=0)
    
    # Normalize class means
    return class_means / np.linalg.norm(class_means, axis=1, keepdims=True) 