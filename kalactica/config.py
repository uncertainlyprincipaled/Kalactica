"""Configuration for KaLactica."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env files for AWS and Kaggle
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env" / "aws" / "credentials", override=True)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env" / "kaggle" / "kaggle.json", override=True)

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
INDEX_DIR = BASE_DIR / "indices"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# AWS Configuration
AWS_CONFIG = {
    "region": os.getenv("AWS_REGION", "us-east-1"),
    "opensearch_host": os.getenv("OPENSEARCH_HOST"),
    "s3_bucket": os.getenv("S3_BUCKET"),
    "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
}

# Kaggle Configuration
KAGGLE_CONFIG = {
    "username": os.getenv("KAGGLE_USERNAME"),
    "key": os.getenv("KAGGLE_KEY"),
}

# Model Configuration
MODEL_CONFIG = {
    "base_model": "facebook/galactica-1.3b",
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "search_batch_size": 32,
    "max_results": 5,
}

# Training Configuration
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
}

# Topology Configuration
TOPOLOGY_CONFIG = {
    "betti_thresholds": {
        "code": 0.8,
        "markdown": 0.6,
        "output": 0.7,
    },
    "max_dimension": 2,
    "persistence_threshold": 0.1,
}

# Notebook Configuration
NOTEBOOK_CONFIG = {
    "max_cells": 20,
    "max_cell_length": 1000,
    "cell_types": ["code", "markdown"],
    "default_imports": [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
    ],
}

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