"""
Test script for ChromaDB setup and basic functionality.

This test verifies that ChromaDB is properly configured and working.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from ..database.chroma_client import ChromaDBClient
from ..utils.file_utils import validate_file_type, get_file_info


class TestChromaDBSetup:
    """Test ChromaDB configuration and basic operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_client = ChromaDBClient(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_client_initialization(self):
        """Test ChromaDB client initialization."""
        assert self.test_client is not None
        assert self.test_client.collection_name == "test_collection"
        assert self.test_client.persist_directory == Path(self.temp_dir)
    
    def test_health_check(self):
        """Test ChromaDB connection health."""
        health = self.test_client.health_check()
        assert isinstance(health, bool)
    
    def test_collection_creation(self):
        """Test collection creation and retrieval."""
        collection = self.test_client.get_collection()
        assert collection is not None
        assert collection.name == "test_collection"
    
    def test_document_operations(self):
        """Test basic document operations (add, query, update, delete)."""
        # Add test documents
        documents = [
            "This is a test document about data analysis.",
            "Another document discussing machine learning and AI.",
            "A third document about Python programming and databases."
        ]
        
        metadatas = [
            {"source": "test1.txt", "category": "analysis"},
            {"source": "test2.txt", "category": "ml"},
            {"source": "test3.txt", "category": "programming"}
        ]
        
        doc_ids = self.test_client.add_documents(
            documents=documents,
            metadatas=metadatas
        )
        
        assert len(doc_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        
        # Test querying
        results = self.test_client.query_documents(
            query_texts=["data analysis"],
            n_results=2
        )
        
        assert "documents" in results
        assert len(results["documents"][0]) <= 2
        
        # Test semantic search
        search_results = self.test_client.semantic_search(
            query="machine learning",
            n_results=1
        )
        
        assert len(search_results) <= 1
        if search_results:
            assert "document" in search_results[0]
            assert "metadata" in search_results[0]
            assert "similarity" in search_results[0]
        
        # Test update
        self.test_client.update_document(
            document_id=doc_ids[0],
            metadata={"source": "test1_updated.txt", "category": "analysis", "updated": True}
        )
        
        # Test delete
        self.test_client.delete_document(doc_ids[0])
        
        # Verify deletion
        collection_info = self.test_client.get_collection_info()
        assert collection_info["count"] == 2  # Should have 2 documents left
    
    def test_collection_management(self):
        """Test collection management operations."""
        # Test collection info
        info = self.test_client.get_collection_info()
        assert "name" in info
        assert "count" in info
        assert info["name"] == "test_collection"
        
        # Test list collections
        collections = self.test_client.list_collections()
        assert isinstance(collections, list)
        assert "test_collection" in collections
        
        # Test clear collection
        # Add a document first
        self.test_client.add_documents(
            documents=["Test document for clearing"],
            metadatas=[{"source": "test"}]
        )
        
        info = self.test_client.get_collection_info()
        assert info["count"] > 0
        
        # Clear the collection
        self.test_client.clear_collection()
        
        # Verify it's empty
        info = self.test_client.get_collection_info()
        assert info["count"] == 0


class TestFileUtils:
    """Test file utility functions."""
    
    def test_validate_file_type(self):
        """Test file type validation."""
        assert validate_file_type("test.csv") == True
        assert validate_file_type("test.xlsx") == True
        assert validate_file_type("test.xls") == True
        assert validate_file_type("test.txt") == False
        assert validate_file_type("test.pdf") == False
        
        # Test custom allowed types
        assert validate_file_type("test.txt", ["txt", "csv"]) == True
        assert validate_file_type("test.pdf", ["txt", "csv"]) == False
    
    def test_get_file_info_nonexistent(self):
        """Test file info for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_file_info("nonexistent_file.csv")


def test_chromadb_integration():
    """Integration test for ChromaDB setup."""
    # Create a simple test
    with tempfile.TemporaryDirectory() as temp_dir:
        client = ChromaDBClient(
            collection_name="integration_test",
            persist_directory=temp_dir
        )
        
        # Test basic workflow
        docs = ["Sample document for integration test"]
        metadata = [{"test": True}]
        
        doc_ids = client.add_documents(documents=docs, metadatas=metadata)
        assert len(doc_ids) == 1
        
        results = client.semantic_search("sample document")
        assert len(results) == 1
        assert results[0]["metadata"]["test"] == True


def test_sample_data_processing():
    """Test data processing with sample data."""
    # Create sample CSV data
    sample_data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"]
    }
    df = pd.DataFrame(sample_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Test file validation
        assert validate_file_type(temp_file)
        
        # Test file info
        info = get_file_info(temp_file)
        assert info["name"] == Path(temp_file).name
        assert info["extension"] == ".csv"
        assert info["size_bytes"] > 0
        
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    import sys
    
    print("Running ChromaDB setup tests...")
    
    try:
        # Test basic functionality
        test_chromadb_integration()
        print("✅ ChromaDB integration test passed")
        
        # Test file utilities
        test_sample_data_processing()
        print("✅ File utilities test passed")
        
        print("\n🎉 All basic tests passed! ChromaDB setup is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1) 