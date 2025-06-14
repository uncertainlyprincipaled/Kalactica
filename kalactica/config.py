"""Configuration settings for KaLactica."""

import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("KALACTICA_DATA_DIR", ROOT_DIR / "data")
INDEX_DIR = os.getenv("KALACTICA_INDEX_DIR", ROOT_DIR / "indices")
CKPT_DIR = os.getenv("KALACTICA_CKPT_DIR", ROOT_DIR / "checkpoints")

# Create directories if they don't exist
for dir_path in [DATA_DIR, INDEX_DIR, CKPT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Domain-specific keywords for task classification
DOMAIN_KEYWORDS = {
    "cv": [
        "image", "vision", "cnn", "resnet", "efficientnet",
        "detection", "segmentation", "classification"
    ],
    "nlp": [
        "text", "bert", "transformer", "token", "embedding",
        "sentiment", "classification", "translation"
    ],
    "rl": [
        "reinforcement", "agent", "environment", "reward",
        "policy", "q-learning", "actor-critic"
    ],
    "tabular": [
        "dataframe", "pandas", "feature", "engineering",
        "regression", "classification", "xgboost"
    ]
}

# Betti number thresholds for different domains
BETTI_THRESHOLDS = {
    "cv": 1,    # Simple manifold tasks
    "nlp": 2,   # Text processing tasks
    "rl": 5,    # Complex agent tasks
    "tabular": 0  # Basic tabular tasks
}

# Model configuration
MODEL_CONFIG = {
    "base_model": "meta-llama/Llama-2-13b-hf",
    "max_length": 2048,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "index_type": "faiss",
    "embedding_dim": 768,
    "n_neighbors": 5,
    "search_batch_size": 32
}

# Memory configuration
MEMORY_CONFIG = {
    "max_nodes": 1000,
    "max_edges": 5000,
    "cache_size": 100
} 