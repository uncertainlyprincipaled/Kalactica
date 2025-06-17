"""Retrieval module for KaLactica."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .aws import get_vector_db, get_aws_storage

class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 opensearch_host: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 region: str = "us-east-1"):
        """Initialize retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            opensearch_host: OpenSearch endpoint (optional)
            s3_bucket: S3 bucket name (optional)
            region: AWS region
        """
        self.model = SentenceTransformer(model_name)
        self.vector_db = None
        self.storage = None
        
        if opensearch_host:
            self.vector_db = get_vector_db(opensearch_host, region)
        if s3_bucket:
            self.storage = get_aws_storage(s3_bucket, region)
    
    def build_index(self, jsonl_path: str, output_dir: Optional[str] = None):
        """Build vector index from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
            output_dir: Directory to save index (optional)
        """
        documents = []
        embeddings = []
        
        with open(jsonl_path, "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)
                embedding = self.model.encode(doc["content"])
                embeddings.append(embedding)
        
        if self.vector_db:
            self.vector_db.add_documents(documents, embeddings)
        
        if output_dir and not self.vector_db:
            # Fallback to local storage if no vector DB
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / "embeddings.npy", np.array(embeddings))
            with open(output_path / "documents.json", "w") as f:
                json.dump(documents, f)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of similar documents
        """
        query_embedding = self.model.encode(query)
        
        if self.vector_db:
            return self.vector_db.search(query_embedding, k)
        else:
            raise NotImplementedError(
                "Local search not implemented. Please provide OpenSearch host."
            )
    
    def save_to_s3(self, key: str, data: Any):
        """Save data to S3.
        
        Args:
            key: S3 object key
            data: Data to save
        """
        if not self.storage:
            raise ValueError("S3 bucket not configured")
        self.storage.save_file(key, data)
    
    def load_from_s3(self, key: str) -> Any:
        """Load data from S3.
        
        Args:
            key: S3 object key
        
        Returns:
            Loaded data
        """
        if not self.storage:
            raise ValueError("S3 bucket not configured")
        return self.storage.load_file(key)

def build_index(*args, **kwargs):
    """Module-level stub for build_index. Calls Retriever.build_index if possible."""
    retriever = Retriever()
    if args or kwargs:
        return retriever.build_index(*args, **kwargs)
    raise NotImplementedError("build_index requires arguments. See Retriever.build_index for usage.")

def search(*args, **kwargs):
    """Module-level stub for search. Calls Retriever.search if possible."""
    retriever = Retriever()
    if args or kwargs:
        return retriever.search(*args, **kwargs)
    raise NotImplementedError("search requires arguments. See Retriever.search for usage.") 