"""
Unit tests for student database with Milvus (using mocks).
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vision.student_db import StudentDatabase


class TestStudentDatabase:
    """Test suite for student database with mocked Milvus."""
    
    @pytest.fixture
    def mock_milvus(self):
        """Mock Milvus connections and collection."""
        with patch('vision.student_db.connections') as mock_conn, \
             patch('vision.student_db.Collection') as mock_coll_class, \
             patch('vision.student_db.FieldSchema') as mock_field, \
             patch('vision.student_db.CollectionSchema') as mock_schema, \
             patch('vision.student_db.DataType') as mock_dtype:
            
            # Mock collection instance
            mock_collection = MagicMock()
            mock_coll_class.return_value = mock_collection
            
            # Mock search results
            mock_collection.search.return_value = [[]]
            mock_collection.query.return_value = []
            
            yield {
                'connections': mock_conn,
                'Collection': mock_coll_class,
                'collection': mock_collection,
                'FieldSchema': mock_field,
                'CollectionSchema': mock_schema,
                'DataType': mock_dtype
            }
    
    @pytest.fixture
    def db_config(self):
        """Database configuration for testing."""
        return {
            "collection_name": "test_student_faces",
            "similarity_threshold": 0.6,
            "top_k": 1,
            "embeddings_per_student": 3,
            "embedding_dim": 512,
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "host": "localhost",
            "port": 19530
        }
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        embeddings = []
        for _ in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings
    
    def test_database_initialization(self, mock_milvus, db_config):
        """Test database initialization with mocked Milvus."""
        db = StudentDatabase(**db_config)
        
        assert db.collection_name == "test_student_faces"
        assert db.similarity_threshold == 0.6
        assert db.embedding_dim == 512
        
        # Verify Milvus connection was attempted
        mock_milvus['connections'].connect.assert_called_once()
    
    def test_enroll_student(self, mock_milvus, db_config, sample_embeddings):
        """Test enrolling a student with mocked Milvus."""
        db = StudentDatabase(**db_config)
        
        # Mock successful insert
        mock_milvus['collection'].insert.return_value = MagicMock()
        
        success = db.enroll_student(
            student_id="TEST001",
            name="Test Student",
            embeddings=sample_embeddings
        )
        
        assert success == True
        mock_milvus['collection'].insert.assert_called_once()
    
    def test_enroll_empty_embeddings(self, mock_milvus, db_config):
        """Test enrolling with empty embeddings."""
        db = StudentDatabase(**db_config)
        
        success = db.enroll_student(
            student_id="TEST002",
            name="Test Student 2",
            embeddings=[]
        )
        
        assert success == False
    
    def test_identify_student(self, mock_milvus, db_config, sample_embeddings):
        """Test identifying a student with mocked results."""
        db = StudentDatabase(**db_config)
        
        # Mock search result with high similarity
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda key: {
            "student_id": "TEST003",
            "name": "Test Student 3",
            "enrollment_date": "2024-01-01"
        }.get(key)
        mock_hit.distance = 0.95  # High similarity
        
        mock_milvus['collection'].search.return_value = [[mock_hit]]
        
        result = db.identify_student(sample_embeddings[0])
        
        assert result is not None
        assert result["student_id"] == "TEST003"
        assert result["name"] == "Test Student 3"
        assert result["similarity"] >= 0.6
    
    def test_identify_unknown(self, mock_milvus, db_config):
        """Test identifying unknown student."""
        db = StudentDatabase(**db_config)
        
        # Mock empty search result
        mock_milvus['collection'].search.return_value = [[]]
        
        random_emb = np.random.randn(512).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb)
        
        result = db.identify_student(random_emb)
        
        assert result is None
    
    def test_get_student(self, mock_milvus, db_config):
        """Test getting student info."""
        db = StudentDatabase(**db_config)
        
        # Mock query result
        mock_milvus['collection'].query.return_value = [{
            "student_id": "TEST004",
            "name": "Test Student 4",
            "enrollment_date": "2024-01-01"
        }]
        
        info = db.get_student("TEST004")
        
        assert info is not None
        assert info["student_id"] == "TEST004"
        assert info["name"] == "Test Student 4"
    
    def test_list_students(self, mock_milvus, db_config):
        """Test listing all students."""
        db = StudentDatabase(**db_config)
        
        # Mock query result with multiple students
        mock_milvus['collection'].query.return_value = [
            {"student_id": "TEST005", "name": "Test Student 5"},
            {"student_id": "TEST006", "name": "Test Student 6"}
        ]
        
        students = db.list_students()
        
        assert len(students) >= 2
        student_ids = [s["student_id"] for s in students]
        assert "TEST005" in student_ids
        assert "TEST006" in student_ids
    
    def test_delete_student(self, mock_milvus, db_config):
        """Test deleting a student."""
        db = StudentDatabase(**db_config)
        
        # Mock successful delete
        mock_milvus['collection'].delete.return_value = MagicMock()
        
        success = db.delete_student("TEST007")
        
        assert success == True
        mock_milvus['collection'].delete.assert_called_once()
    
    def test_get_stats(self, mock_milvus, db_config):
        """Test getting database statistics."""
        db = StudentDatabase(**db_config)
        
        # Mock collection stats
        mock_milvus['collection'].num_entities = 15
        mock_milvus['collection'].query.return_value = [
            {"student_id": "S1"}, {"student_id": "S2"}, {"student_id": "S3"}
        ]
        
        stats = db.get_stats()
        
        assert "total_embeddings" in stats
        assert "total_students" in stats
        assert "collection_name" in stats
        assert stats["collection_name"] == "test_student_faces"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

