"""AWS integration for KaLactica.

This module provides helpers for S3 and OpenSearch, including efficient upload/download
and logic to avoid unnecessary S3 requests. Use these helpers to store and retrieve
raw and processed data, minimizing S3 API calls."""

import boto3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np

class AWSStorage:
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        """Initialize AWS storage.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = bucket_name
    
    def save_file(self, key: str, data: Any):
        """Save data to S3. Avoids duplicate uploads by checking if the object exists."""
        if self.exists(key):
            print(f"[S3] {key} already exists, skipping upload.")
            return
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data
        )
        print(f"[S3] Uploaded {key}")
    
    def load_file(self, key: str) -> Any:
        """Load data from S3."""
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=key
        )
        data = response["Body"].read().decode()
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    
    def download_file(self, key: str, dest: str):
        """Download a file from S3 to local path if not already present."""
        dest_path = Path(dest)
        if dest_path.exists():
            print(f"[S3] {dest} already exists locally, skipping download.")
            return
        self.s3.download_file(self.bucket, key, str(dest_path))
        print(f"[S3] Downloaded {key} to {dest}")
    
    def exists(self, key: str) -> bool:
        """Check if a file exists in S3 bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError:
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 bucket."""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )
        return [obj["Key"] for obj in response.get("Contents", [])]

class OpenSearchVectorDB:
    def __init__(self, host: str, region: str = "us-east-1",
                 index_name: str = "kalactica"):
        """Initialize OpenSearch vector database.
        
        Args:
            host: OpenSearch endpoint
            region: AWS region
            index_name: Index name
        """
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",
            session_token=credentials.token
        )
        
        self.client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        self.index_name = index_name
        self._ensure_index()
    
    def _ensure_index(self):
        """Create index if it doesn't exist."""
        if not self.client.indices.exists(self.index_name):
            self.client.indices.create(
                self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 768
                            },
                            "metadata": {"type": "object"}
                        }
                    },
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100
                        }
                    }
                }
            )
    
    def add_documents(self, documents: List[Dict[str, Any]],
                     embeddings: List[np.ndarray]):
        """Add documents with embeddings to index."""
        for doc, emb in zip(documents, embeddings):
            self.client.index(
                index=self.index_name,
                body={
                    "content": doc["content"],
                    "embedding": emb.tolist(),
                    "metadata": doc.get("metadata", {})
                }
            )
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        response = self.client.search(
            index=self.index_name,
            body={
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding.tolist(),
                            "k": k
                        }
                    }
                }
            }
        )
        
        return [hit["_source"] for hit in response["hits"]["hits"]]

def get_aws_storage(bucket_name: str, region: str = "us-east-1") -> AWSStorage:
    """Get AWS storage instance."""
    return AWSStorage(bucket_name, region)

def get_vector_db(host: str, region: str = "us-east-1",
                 index_name: str = "kalactica") -> OpenSearchVectorDB:
    """Get OpenSearch vector database instance."""
    return OpenSearchVectorDB(host, region, index_name) 