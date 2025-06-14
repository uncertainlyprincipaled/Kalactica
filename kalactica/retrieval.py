"""Retrieval system for KaLactica using FAISS."""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import INDEX_DIR, RETRIEVAL_CONFIG

class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embedding_dim = RETRIEVAL_CONFIG["embedding_dim"]
    
    def build_index(self, jsonl_path: str, out_dir: str = None) -> None:
        """Build FAISS index from JSONL file."""
        if out_dir is None:
            out_dir = INDEX_DIR
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process chunks
        chunks = []
        with open(jsonl_path) as f:
            for line in tqdm(f, desc="Loading chunks"):
                chunk = json.loads(line)
                chunks.append(chunk)
        
        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=RETRIEVAL_CONFIG["search_batch_size"],
            show_progress_bar=True
        )
        
        # Build FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype(np.float32))
        
        # Save index and chunks
        faiss.write_index(index, str(out_dir / "index.faiss"))
        with open(out_dir / "chunks.json", "w") as f:
            json.dump(chunks, f)
        
        self.index = index
        self.chunks = chunks
    
    def load_index(self, index_dir: str) -> None:
        """Load existing FAISS index and chunks."""
        index_dir = Path(index_dir)
        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(f"No index found in {index_dir}")
        
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.json") as f:
            self.chunks = json.load(f)
    
    def search(self, text: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar chunks using FAISS."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        
        # Generate query embedding
        query_embedding = self.model.encode([text])[0]
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k
        )
        
        # Return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((float(dist), self.chunks[idx]))
        
        return results

def build_index(jsonl_path: str, out_dir: str = None) -> None:
    """Convenience function to build index."""
    retriever = Retriever()
    retriever.build_index(jsonl_path, out_dir)

def search(text: str, k: int = 5, index_dir: str = None) -> List[Tuple[float, Dict[str, Any]]]:
    """Convenience function to search index."""
    if index_dir is None:
        index_dir = INDEX_DIR
    
    retriever = Retriever()
    try:
        retriever.load_index(index_dir)
    except FileNotFoundError:
        raise RuntimeError(f"No index found in {index_dir}. Please build index first.")
    
    return retriever.search(text, k) 