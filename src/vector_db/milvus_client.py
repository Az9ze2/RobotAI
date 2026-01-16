"""
Milvus Vector Database Client
Handles memory storage and retrieval for the robot
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
from loguru import logger


class MilvusClient:
    def __init__(self, host: str = "localhost", port: int = 19530, 
                 collection_name: str = "robot_memory",
                 embedding_model: str = "BAAI/bge-m3"):
        """
        Initialize Milvus client and embedding model
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {embedding_model}, dim={self.embedding_dim}")
        
        # Create collection if not exists
        self._create_collection()
        
    def _create_collection(self):
        """Create Milvus collection for memory storage"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="student_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields=fields, description="Robot memory storage")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "IP",  # Inner Product (for cosine similarity)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection '{self.collection_name}' with index")
        
    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding
    
    def insert_memory(self, text: str, memory_type: str, 
                     student_id: str = "", timestamp: int = None) -> bool:
        """
        Insert a new memory into the vector database
        
        Args:
            text: Memory content (e.g., diary summary)
            memory_type: Type of memory (diary, knowledge, navigation)
            student_id: Student identifier
            timestamp: Unix timestamp
        """
        try:
            import time
            if timestamp is None:
                timestamp = int(time.time())
            
            # Generate embedding
            embedding = self.embed_text(text)
            
            # Prepare data with correct field order
            entities = [
                {"embedding": embedding.tolist(),
                 "text": text,
                 "memory_type": memory_type,
                 "student_id": student_id,
                 "timestamp": timestamp}
            ]
            
            # Insert
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"Inserted memory: type={memory_type}, student={student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert memory: {e}")
            return False
    
    def search_memory(self, query: str, top_k: int = 5, 
                     memory_type: Optional[str] = None,
                     student_id: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant memories
        
        Args:
            query: Search query text
            top_k: Number of results to return
            memory_type: Filter by memory type
            student_id: Filter by student
            
        Returns:
            List of memory dictionaries with text, score, metadata
        """
        try:
            # Load collection
            self.collection.load()
            
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Build filter expression
            expr_parts = []
            if memory_type:
                expr_parts.append(f'memory_type == "{memory_type}"')
            if student_id:
                expr_parts.append(f'student_id == "{student_id}"')
            
            expr = " && ".join(expr_parts) if expr_parts else None
            
            # Search
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "memory_type", "student_id", "timestamp"]
            )
            
            # Format results
            memories = []
            for hits in results:
                for hit in hits:
                    memories.append({
                        "text": hit.entity.get("text"),
                        "score": hit.score,
                        "memory_type": hit.entity.get("memory_type"),
                        "student_id": hit.entity.get("student_id"),
                        "timestamp": hit.entity.get("timestamp")
                    })
            
            logger.info(f"Found {len(memories)} relevant memories for query")
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def close(self):
        """Close connection"""
        connections.disconnect(alias="default")
        logger.info("Disconnected from Milvus")