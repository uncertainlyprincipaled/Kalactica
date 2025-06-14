"""Graph-based memory system for KaLactica."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .config import MEMORY_CONFIG

class GraphMemory:
    def __init__(self, db_path: str = None):
        """Initialize the graph memory system."""
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id)
                )
            """)
            
            conn.commit()
    
    def store(self, node: Dict[str, Any], typ: str) -> int:
        """Store a new node in the graph."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert node
            cursor.execute("""
                INSERT INTO nodes (type, content, metadata)
                VALUES (?, ?, ?)
            """, (
                typ,
                json.dumps(node.get("content", "")),
                json.dumps(node.get("metadata", {}))
            ))
            
            node_id = cursor.lastrowid
            conn.commit()
            return node_id
    
    def link(self, source_id: int, relation: str, target_id: int,
             metadata: Optional[Dict[str, Any]] = None) -> int:
        """Create an edge between two nodes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert edge
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, relation, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                source_id,
                target_id,
                relation,
                json.dumps(metadata or {})
            ))
            
            edge_id = cursor.lastrowid
            conn.commit()
            return edge_id
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve nodes matching the query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Search nodes by content
            cursor.execute("""
                SELECT id, type, content, metadata
                FROM nodes
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]),
                    "metadata": json.loads(row[3])
                })
            
            return results
    
    def get_neighbors(self, node_id: int, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get neighboring nodes connected by edges."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT n.id, n.type, n.content, n.metadata, e.relation, e.metadata as edge_metadata
                FROM nodes n
                JOIN edges e ON n.id = e.target_id
                WHERE e.source_id = ?
            """
            params = [node_id]
            
            if relation:
                query += " AND e.relation = ?"
                params.append(relation)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]),
                    "metadata": json.loads(row[3]),
                    "relation": row[4],
                    "edge_metadata": json.loads(row[5])
                })
            
            return results
    
    def clear(self):
        """Clear all data from the memory system."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM nodes")
            conn.commit() 