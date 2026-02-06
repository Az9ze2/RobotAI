"""
Student database management with Milvus vector database.

This module handles student enrollment and identification using Milvus
for efficient similarity search of face embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from loguru import logger
from datetime import datetime


class StudentDatabase:
    """
    Student database using Milvus for face embedding storage and search.
    
    Features:
    - Store multiple embeddings per student for robustness
    - Cosine similarity search
    - Metadata storage (student_id, name, enrollment_date, etc.)
    - CRUD operations for student management
    """
    
    def __init__(
        self,
        collection_name: str = "student_faces",
        similarity_threshold: float = 0.6,
        top_k: int = 1,
        embeddings_per_student: int = 5,
        embedding_dim: int = 512,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        host: str = "localhost",
        port: int = 19530
    ):
        """
        Initialize student database.
        
        Args:
            collection_name: Name of Milvus collection
            similarity_threshold: Minimum similarity for identification
            top_k: Number of top matches to return
            embeddings_per_student: Number of embeddings to store per student
            embedding_dim: Dimension of face embeddings
            index_type: Milvus index type
            metric_type: Distance metric (COSINE, L2, IP)
            host: Milvus host
            port: Milvus port
        """
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.embeddings_per_student = embeddings_per_student
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric_type = metric_type
        
        # Connect to Milvus
        try:
            connections.connect(host=host, port=port)
            logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
        
        # Create or load collection
        self.collection = self._create_or_load_collection()
        
        logger.info(f"StudentDatabase initialized with collection: {collection_name}")
    
    def _create_or_load_collection(self) -> Collection:
        """Create or load Milvus collection."""
        if utility.has_collection(self.collection_name):
            logger.info(f"Loading existing collection: {self.collection_name}")
            collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="student_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="enrollment_date", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding_index", dtype=DataType.INT64),  # Which embedding (0-4)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Student face embeddings for recognition"
            )
            
            collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # Create index
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created index with type: {self.index_type}, metric: {self.metric_type}")
        
        # Load collection
        collection.load()
        return collection
    
    def enroll_student(
        self,
        student_id: str,
        name: str,
        embeddings: List[np.ndarray]
    ) -> bool:
        """
        Enroll a new student with face embeddings.
        
        Args:
            student_id: Unique student identifier
            name: Student name
            embeddings: List of face embeddings (up to embeddings_per_student)
        
        Returns:
            True if successful
        """
        if len(embeddings) == 0:
            logger.error("No embeddings provided for enrollment")
            return False
        
        # Limit to max embeddings per student
        embeddings = embeddings[:self.embeddings_per_student]
        
        # Prepare data
        enrollment_date = datetime.now().isoformat()
        
        data = [
            [student_id] * len(embeddings),  # student_id
            [emb.tolist() for emb in embeddings],  # embedding
            [name] * len(embeddings),  # name
            [enrollment_date] * len(embeddings),  # enrollment_date
            list(range(len(embeddings)))  # embedding_index
        ]
        
        try:
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Enrolled student {student_id} ({name}) with {len(embeddings)} embeddings")
            return True
        
        except Exception as e:
            logger.error(f"Failed to enroll student: {e}")
            return False
    
    def identify_student(self, embedding: np.ndarray) -> Optional[Dict]:
        """
        Identify student from face embedding.
        
        Args:
            embedding: Face embedding to search for
        
        Returns:
            Dictionary with student info and similarity, or None if not found
        """
        try:
            # Search for similar embeddings
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=self.top_k,
                output_fields=["student_id", "name", "enrollment_date"]
            )
            
            # Check if any matches above threshold
            if len(results) > 0 and len(results[0]) > 0:
                top_match = results[0][0]
                
                # Convert distance to similarity based on metric type
                if self.metric_type == "COSINE":
                    similarity = 1.0 - top_match.distance
                elif self.metric_type == "L2":
                    similarity = 1.0 / (1.0 + top_match.distance)
                else:  # IP (Inner Product)
                    similarity = top_match.distance
                
                if similarity >= self.similarity_threshold:
                    return {
                        "student_id": top_match.entity.get("student_id"),
                        "name": top_match.entity.get("name"),
                        "enrollment_date": top_match.entity.get("enrollment_date"),
                        "similarity": float(similarity)
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to identify student: {e}")
            return None
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """
        Get student information by student_id.
        
        Args:
            student_id: Student identifier
        
        Returns:
            Dictionary with student info, or None if not found
        """
        try:
            # Query by student_id
            expr = f'student_id == "{student_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["student_id", "name", "enrollment_date"]
            )
            
            if len(results) > 0:
                return {
                    "student_id": results[0]["student_id"],
                    "name": results[0]["name"],
                    "enrollment_date": results[0]["enrollment_date"]
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get student: {e}")
            return None
    
    def list_students(self) -> List[Dict]:
        """
        List all enrolled students.
        
        Returns:
            List of student dictionaries
        """
        try:
            # Query all unique students
            results = self.collection.query(
                expr="embedding_index == 0",  # Get only first embedding per student
                output_fields=["student_id", "name", "enrollment_date"]
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to list students: {e}")
            return []
    
    def delete_student(self, student_id: str) -> bool:
        """
        Delete a student from the database.
        
        Args:
            student_id: Student identifier
        
        Returns:
            True if successful
        """
        try:
            expr = f'student_id == "{student_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Deleted student: {student_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete student: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        try:
            num_entities = self.collection.num_entities
            num_students = len(self.list_students())
            
            return {
                "total_embeddings": num_entities,
                "total_students": num_students,
                "collection_name": self.collection_name
            }
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"StudentDatabase(collection={self.collection_name}, "
            f"students={stats.get('total_students', 0)}, "
            f"embeddings={stats.get('total_embeddings', 0)})"
        )
