# KaLactica

KaLactica is a memory- and topology-enhanced successor to Galactica, designed to generate domain-aware scientific prose and code with improved factual grounding and coherence. It combines retrieval-augmented generation, a hierarchical memory system, and topological curriculum learning to produce high-quality, verifiable outputs while maintaining a small computational footprint.

## Quickstart

```bash
# Install in development mode
pip install -e .

# Run the demo
python demo.py
```

## Architecture

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

- **Retrieval-Augmented Generation**: Dual-crop FAISS index for short-term and long-term memory
- **Memory Hierarchy**: Tracks dialogue state, code executions, and consequence notes
- **Topological Curriculum**: Orders tasks by Betti complexity for progressive learning
- **Safety Filter**: Persistent homology checks for generation consistency
- **Parameter-Efficient**: Uses QLoRA and DPO for efficient fine-tuning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS-CPU
- Transformers
- PEFT
- Gudhi (optional, for topology features)

## License

MIT License
