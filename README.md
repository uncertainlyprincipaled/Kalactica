# KaLactica

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
   - [Local Development](#local-development)
   - [Colab/Kaggle Setup](#colabkaggle-setup)
   - [Lambda Labs Setup](#lambda-labs-setup)
3. [Usage Instructions](#usage-instructions)
   - [Preprocessing](#preprocessing)
   - [Model Training](#model-training)
   - [Retrieval System](#retrieval-system)
   - [Memory System](#memory-system)
4. [Architecture](#architecture)
5. [Features](#features)
6. [Requirements](#requirements)
7. [Security and Best Practices](#security-and-best-practices)
8. [Configuration](#configuration)
9. [License](#license)

## Introduction

KaLactica is a memory- and topology-enhanced successor to Galactica, designed to generate domain-aware scientific prose and code with improved factual grounding and coherence. It combines retrieval-augmented generation, a hierarchical memory system, and topological curriculum learning to produce high-quality, verifiable outputs while maintaining a small computational footprint.

### Key Innovations
- **Memory-Enhanced Generation**: Utilizes a hierarchical memory system to maintain context and improve coherence
- **Topology-Aware Learning**: Employs topological curriculum learning for progressive task complexity
- **Retrieval-Augmented Outputs**: Combines FAISS indexing with dual-crop memory for improved factual grounding
- **Safety-First Design**: Implements multiple layers of safety checks and consistency verification

### Use Cases
- Scientific paper generation and summarization
- Code generation with domain awareness
- Technical documentation creation
- Research assistance and literature review

## Environment Setup

This section provides instructions for setting up KaLactica in different environments. Choose the setup that best matches your use case:
- **Local Development**: For development and testing
- **Colab/Kaggle**: For preprocessing and data preparation
- **Lambda Labs**: For model training and fine-tuning

### Local Development
```bash
# Install in development mode
pip install -e .

# Run the demo
python demo.py
```

### Colab/Kaggle Setup

#### Colab:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install requirements
!pip install -r requirements.txt

# Set up Kaggle credentials
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/Kalactica/.env/kaggle/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download datasets
!kaggle datasets download -d kaggle/meta-kaggle
!kaggle datasets download -d kaggle/meta-kaggle-code
!unzip meta-kaggle.zip -d data/
!unzip meta-kaggle-code.zip -d data/
```

#### Kaggle:
```python
# Clone repository to /kaggle/working
!git clone https://github.com/uncertainlyprincipaled/Kalactica.git /kaggle/working/Kalactica
# Console
%cd /kaggle/working/Kalactica

# Install required dependencies
!pip install opensearch-py
!pip install python-dotenv
!pip install faiss-cpu
!pip install transformers
!pip install peft
!pip install gudhi  # Optional, for topology features

# Install package in development mode
# May need to restart kernel after running
!pip install -e .

# Set up Kaggle credentials (already available in Kaggle environment)
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input/kaggle-api'

# Download datasets (if not already in /kaggle/input)
!kaggle datasets download -d kaggle/meta-kaggle
!kaggle datasets download -d kaggle/meta-kaggle-code
!unzip meta-kaggle.zip -d data/
!unzip meta-kaggle-code.zip -d data/

# Create necessary directories
!mkdir -p data/processed
!mkdir -p .env/kaggle

# Note: Your kaggle.json is automatically available in the Kaggle environment
# No need to manually copy it
```

### Lambda Labs Setup
```bash
# Install Lambda Labs CLI
pip install lambda-cloud

# Configure credentials
lambda configure --api-key your_api_key

# Launch instance
lambda launch --instance-type gpu.1x.a10
```

## Usage Instructions

KaLactica's components can be used in different environments based on their resource requirements. This section provides detailed instructions for each major component.

### Preprocessing
The preprocessing step is designed to run on CPU and can be executed on Colab, Kaggle, or local machines.

#### Colab/Kaggle:
```python
# Run preprocessing
# Note: In Kaggle, we need to set environment variables directly
# as python-dotenv doesn't work well with Kaggle's environment
import os

# Set required environment variables
os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input/kaggle-api'
os.environ['DATA_DIR'] = '/kaggle/working/Kalactica/data'

# Run preprocessing with explicit paths
!python -m kalactica.preprocess \
    --input /kaggle/input/meta-kaggle/KernelVersions.csv \
    --output /kaggle/working/Kalactica/data/processed.jsonl
```

#### Local:
```bash
# Set up environment variables
export DATA_DIR=$(pwd)/data

# Run preprocessing
python -m kalactica.preprocess \
    --input /data/meta-kaggle/KernelVersions.csv \
    --output data/processed.jsonl
```

### Model Training
Training requires GPU resources and should be run on Lambda Labs.

#### Lambda Labs Instance:
```bash
# Clone repository
git clone https://github.com/uncertainlyprincipaled/Kalactica.git
cd Kalactica

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy credentials
mkdir -p .env/lambda
# Copy your credentials.json to .env/lambda/

# Start training
python -m kalactica.train_qlora
```

### Retrieval System
The retrieval system can be run on any environment with sufficient memory.

#### Local/Colab/Kaggle:
```python
from kalactica.retrieval import RetrievalSystem

# Initialize system
retrieval = RetrievalSystem()

# Query the system
results = retrieval.query("your query here")
```

### Memory System
The memory system is designed to work across all environments.

#### Any Environment:
```python
from kalactica.memory import MemorySystem

# Initialize system
memory = MemorySystem()

# Add to memory
memory.add("key", "value")

# Retrieve from memory
value = memory.get("key")
```

## Architecture

KaLactica's architecture is built around four core components that work together to provide enhanced generation capabilities.

```
┌─────────────────────────────────────────────────────────┐
│                      KaLactica Core                     │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  Retrieval  │   Memory    │  Topology   │   Safety     │
│   Layer     │   Layer     │   Layer     │   Filter     │
├─────────────┼─────────────┼─────────────┼──────────────┤
│  FAISS      │  Graph      │  Betti      │  Wasserstein │
│  Index      │  Memory     │  Signature  │  Distance    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                    Base Model (LLaMA)                    │
└─────────────────────────────────────────────────────────┘
```

## Features

KaLactica offers several key features that set it apart from traditional language models:

- **Retrieval-Augmented Generation**: Dual-crop FAISS index for short-term and long-term memory
- **Memory Hierarchy**: Tracks dialogue state, code executions, and consequence notes
- **Topological Curriculum**: Orders tasks by Betti complexity for progressive learning
- **Safety Filter**: Persistent homology checks for generation consistency
- **Parameter-Efficient**: Uses QLoRA and DPO for efficient fine-tuning

## Requirements

To run KaLactica, you'll need the following software and hardware requirements:

- Python 3.8+
- PyTorch 2.0+
- FAISS-CPU
- Transformers
- PEFT
- Gudhi (optional, for topology features)

## Security and Best Practices

KaLactica implements several security measures and best practices to ensure safe and efficient operation:

1. **File Permissions**
   - Set credential files to 600 permissions
   - Never commit credentials to version control

2. **Environment Variables**
   - Use different API keys for different environments
   - Regularly rotate credentials
   - Store sensitive data in environment variables

3. **Cost Optimization**
   - Use CPU runtime for preprocessing
   - Monitor memory and GPU usage
   - Clean up temporary files
   - Use spot instances when possible
   - Shut down instances when not in use

## Configuration

Proper configuration is essential for optimal performance. This section covers the necessary configuration files and settings:

### Environment Files
- `.env/aws/credentials`: AWS credentials for S3 and OpenSearch
- `.env/kaggle/kaggle.json`: Kaggle API credentials
- `.env/lambda/credentials.json`: Lambda Labs configuration
- `.env/colab/config.json`: Colab runtime settings

### .gitignore
```
# Credentials and configs
.env/
*.json
credentials
config.json

# Model files
*.npy
*.pt
*.safetensors

# Data files
*.csv
*.jsonl
*.zip

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
venv/
```

## License

MIT License
