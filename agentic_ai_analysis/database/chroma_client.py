"""
ChromaDB client implementation for vector database operations.

This module provides a high-level interface to ChromaDB for storing,
retrieving, and managing vector embeddings for semantic search.
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """ChromaDB client for vector database operations."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            host: ChromaDB host (defaults to settings)
            port: ChromaDB port (defaults to settings)
            collection_name: Name of the collection (defaults to settings)
            embedding_model: Embedding model to use (defaults to settings)
            persist_directory: Directory to persist data (optional)
        """
        self.settings = get_settings()
        
        self.host = host or self.settings.chromadb_host
        self.port = port or self.settings.chromadb_port
        self.collection_name = collection_name or self.settings.chromadb_collection_name
        self.embedding_model = embedding_model or self.settings.embedding_model
        
        # Set up persistence directory
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = Path(__file__).parent.parent / "data" / "chroma_db"
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._client = None
        self._collection = None
        self._embedding_function = None
        
        logger.info(f"ChromaDB client initialized for {self.host}:{self.port}")
    
    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                if self.settings.debug:
                    # Use persistent client in development
                    self._client = chromadb.PersistentClient(
                        path=str(self.persist_directory),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    logger.info(f"Connected to persistent ChromaDB at {self.persist_directory}")
                else:
                    # Use HTTP client in production
                    self._client = chromadb.HttpClient(
                        host=self.host,
                        port=self.port,
                        settings=Settings(anonymized_telemetry=False)
                    )
                    logger.info(f"Connected to ChromaDB server at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                raise
        
        return self._client
    
    @property
    def embedding_function(self):
        """Get or create embedding function."""
        if self._embedding_function is None:
            if self.settings.openai_api_key:
                # Use OpenAI embeddings if API key is available
                self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=self.settings.openai_api_key,
                    model_name=self.embedding_model
                )
                logger.info(f"Using OpenAI embedding model: {self.embedding_model}")
            else:
                # Fall back to sentence transformers
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("Using SentenceTransformer embedding model: all-MiniLM-L6-v2")
        
        return self._embedding_function
    
    def get_collection(self, collection_name: Optional[str] = None):
        """Get or create a collection."""
        collection_name = collection_name or self.collection_name
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Agentic AI Data Analysis Collection"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs (auto-generated if None)
            collection_name: Collection to add to
            
        Returns:
            List of document IDs
        """
        collection = self.get_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Ensure metadatas list matches documents length
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        elif len(metadatas) != len(documents):
            metadatas.extend([{"source": "unknown"} for _ in range(len(documents) - len(metadatas))])
        
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection {collection.name}")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def query_documents(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query documents using semantic search.
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return
            where: Metadata filter conditions
            collection_name: Collection to search in
            include: Fields to include in results
            
        Returns:
            Query results dictionary
        """
        collection = self.get_collection(collection_name)
        
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        try:
            # ChromaDB doesn't accept empty where clauses - only include if non-empty
            query_params = {
                "query_texts": query_texts,
                "n_results": n_results,
                "include": include
            }
            
            # Only add where clause if it has content and is not empty
            if where and isinstance(where, dict) and len(where) > 0:
                # Additional validation to ensure proper filter format
                valid_where = {}
                for key, value in where.items():
                    if value is not None and value != "":
                        valid_where[key] = value
                
                if valid_where:  # Only add if we have valid filters
                    query_params["where"] = valid_where
                
            results = collection.query(**query_params)
            # Log actual document count, not result set count
            doc_count = len(results.get('documents', [[]])[0]) if results.get('documents') and results['documents'] else 0
            logger.info(f"Retrieved {doc_count} documents from {len(results.get('documents', []))} query result sets")
            return results
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return formatted results.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Metadata filter conditions
            collection_name: Collection to search in
            
        Returns:
            List of formatted search results
        """
        results = self.query_documents(
            query_texts=[query],
            n_results=n_results,
            where=filters,
            collection_name=collection_name
        )
        

        formatted_results = []
        
        # More robust result processing with better error handling
        if results.get('documents') and len(results['documents']) > 0:
            # Extract documents from first result set
            documents_list = results['documents'][0] if results['documents'] and len(results['documents']) > 0 else []
            
            # Handle case where documents list might be None or empty
            if documents_list:
                # Extract other arrays with proper bounds checking
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') and len(results['metadatas']) > 0 else []
                distances = results.get('distances', [[]])[0] if results.get('distances') and len(results['distances']) > 0 else []
                ids = results.get('ids', [[]])[0] if results.get('ids') and len(results['ids']) > 0 else []
                
                # Ensure all arrays are the same length as documents
                num_docs = len(documents_list)
                metadatas = metadatas[:num_docs] + [{}] * max(0, num_docs - len(metadatas))
                distances = distances[:num_docs] + [0.0] * max(0, num_docs - len(distances))
                ids = ids[:num_docs] + [''] * max(0, num_docs - len(ids))
                
                # Format results
                for i, doc in enumerate(documents_list):
                    if doc:  # Only include non-empty documents
                        distance = distances[i] if i < len(distances) else 0.0
                        # For cosine distance, convert to similarity score properly
                        # ChromaDB cosine distances range from 0 (identical) to 2 (opposite)
                        similarity = max(0.0, (2.0 - distance) / 2.0)  # Normalize to 0-1 range
                        
                        formatted_results.append({
                            'id': ids[i] if i < len(ids) else '',
                            'document': doc,
                            'metadata': metadatas[i] if i < len(metadatas) else {},
                            'distance': distance,
                            'similarity': similarity
                        })
            else:
                logger.warning(f"semantic_search: Empty documents list in results for query: {query[:50]}...")
        else:
            logger.warning(f"semantic_search: No documents found in results for query: {query[:50]}...")
        
        logger.debug(f"semantic_search: Returning {len(formatted_results)} formatted results")
        return formatted_results
    
    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ):
        """
        Update a document in the collection.
        
        Args:
            document_id: ID of the document to update
            document: New document text
            metadata: New metadata
            collection_name: Collection containing the document
        """
        collection = self.get_collection(collection_name)
        
        update_data = {"ids": [document_id]}
        if document is not None:
            update_data["documents"] = [document]
        if metadata is not None:
            update_data["metadatas"] = [metadata]
        
        try:
            collection.update(**update_data)
            logger.info(f"Updated document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            raise
    
    def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ):
        """
        Delete a document from the collection.
        
        Args:
            document_id: ID of the document to delete
            collection_name: Collection containing the document
        """
        collection = self.get_collection(collection_name)
        
        try:
            collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        collection = self.get_collection(collection_name)
        
        try:
            count = collection.count()
            return {
                "name": collection.name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    def delete_collection(self, collection_name: Optional[str] = None):
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    def clear_collection(self, collection_name: Optional[str] = None):
        """
        Clear all documents from a collection.
        
        Args:
            collection_name: Name of the collection to clear
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # Delete and recreate the collection
            self.delete_collection(collection_name)
            self.get_collection(collection_name)
            logger.info(f"Cleared collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if ChromaDB connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


# Global client instance
_chroma_client = None


def get_chroma_client() -> ChromaDBClient:
    """Get global ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = ChromaDBClient()
    return _chroma_client


def init_chroma_client(**kwargs) -> ChromaDBClient:
    """Initialize global ChromaDB client with custom parameters."""
    global _chroma_client
    _chroma_client = ChromaDBClient(**kwargs)
    return _chroma_client 