"""Topological safety filter for KaLactica."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from .config import BETTI_THRESHOLDS
from .topology import betti_signature, wasserstein, compute_persistence_diagram

class TopologyFilter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 sig_db_path: Optional[str] = None):
        """Initialize topology filter.
        
        Args:
            model_name: Name of sentence transformer model
            sig_db_path: Path to signature database
        """
        self.model = SentenceTransformer(model_name)
        self.sig_db = {}
        
        if sig_db_path:
            self.load_signatures(sig_db_path)
    
    def load_signatures(self, sig_db_path: str):
        """Load topological signatures from database."""
        with open(sig_db_path) as f:
            self.sig_db = json.load(f)
    
    def save_signatures(self, sig_db_path: str):
        """Save topological signatures to database."""
        with open(sig_db_path, 'w') as f:
            json.dump(self.sig_db, f)
    
    def compute_signature(self, text: str) -> Dict[str, Any]:
        """Compute topological signature for text."""
        # Generate embeddings
        embs = self.model.encode([text])[0]
        
        # Compute Betti numbers
        betti = betti_signature(embs.reshape(1, -1))
        
        # Compute persistence diagram
        diag = compute_persistence_diagram(embs.reshape(1, -1))
        
        return {
            "betti": betti,
            "diagram": diag
        }
    
    def is_safe(self, text: str, domain: str,
                threshold: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
        """Check if text is topologically safe for domain.
        
        Args:
            text: Text to check
            domain: Target domain (cv, nlp, rl, etc.)
            threshold: Maximum allowed Wasserstein distance
        
        Returns:
            Tuple of (is_safe, stats)
        """
        # Compute signature for input text
        sig = self.compute_signature(text)
        
        # Get domain threshold
        max_betti = BETTI_THRESHOLDS.get(domain, 1)
        
        # Check Betti numbers
        if sig["betti"][1] > max_betti:
            return False, {
                "reason": "betti_too_high",
                "betti": sig["betti"],
                "max_allowed": max_betti
            }
        
        # Compare with domain signatures if available
        if domain in self.sig_db:
            domain_sig = self.sig_db[domain]
            dist = wasserstein(sig["diagram"], domain_sig["diagram"])
            
            if dist > threshold:
                return False, {
                    "reason": "topology_mismatch",
                    "distance": dist,
                    "threshold": threshold
                }
        
        return True, {
            "reason": "safe",
            "betti": sig["betti"],
            "diagram": sig["diagram"]
        }

def topology_filter(draft: str, sig_db: Dict[str, Any],
                   domain: str = "nlp") -> bool:
    """Convenience function to check if draft is topologically safe.
    
    Args:
        draft: Generated text to check
        sig_db: Dictionary of domain signatures
        domain: Target domain
    
    Returns:
        True if draft is safe, False otherwise
    """
    filter = TopologyFilter()
    filter.sig_db = sig_db
    is_safe, _ = filter.is_safe(draft, domain)
    return is_safe 