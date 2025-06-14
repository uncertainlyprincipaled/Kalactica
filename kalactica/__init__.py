"""
KaLactica: A Memory- & Topology-Enhanced Successor to Galactica
"""

__version__ = "0.1.0"

from .config import *
from .model import KaLactica, Generator
from .retrieval import build_index, search
from .memory import GraphMemory
from .topology import betti_signature, wasserstein
from .nc_metrics import collapse_stats
from .safety import topology_filter

__all__ = [
    "KaLactica",
    "Generator",
    "build_index",
    "search",
    "GraphMemory",
    "betti_signature",
    "wasserstein",
    "collapse_stats",
    "topology_filter",
] 