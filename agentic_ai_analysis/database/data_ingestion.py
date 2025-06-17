"""
Data ingestion module for processing CSV/Excel files and storing them in ChromaDB.

This module handles file processing, data chunking, and vector storage
for semantic search and analysis.
"""

import io
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .chroma_client import ChromaDBClient, get_chroma_client
from config.settings import get_settings
from utils.file_utils import validate_file_type, get_file_info

logger = logging.getLogger(__name__)


class DataIngestionProcessor:
    """Processes and ingests data files into ChromaDB for semantic search."""
    
    def __init__(self, chroma_client: Optional[ChromaDBClient] = None):
        """
        Initialize data ingestion processor.
        
        Args:
            chroma_client: ChromaDB client instance (optional)
        """
        self.settings = get_settings()
        self.chroma_client = chroma_client or get_chroma_client()
        self.chunk_size = self.settings.chunk_size
        
        logger.info("Data ingestion processor initialized")
    
    def process_file(
        self,
        file_path: Union[str, Path, io.BytesIO],
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a file and ingest it into ChromaDB.
        
        Args:
            file_path: Path to the file or BytesIO object
            file_name: Name of the file (required for BytesIO)
            metadata: Additional metadata for the file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Handle different input types
            if isinstance(file_path, io.BytesIO):
                if not file_name:
                    raise ValueError("file_name is required when using BytesIO")
                df = self._read_file_from_bytes(file_path, file_name)
                source_path = file_name
            else:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                df = self._read_file(file_path)
                source_path = str(file_path)
            
            # Validate and prepare the dataframe
            df = self._prepare_dataframe(df)
            
            # Create base metadata
            base_metadata = {
                "source": source_path,
                "file_name": file_name or Path(source_path).name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "file_type": self._get_file_type(source_path),
                "ingestion_timestamp": pd.Timestamp.now().isoformat()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Process and chunk the data
            chunks = self._create_chunks(df, base_metadata)
            
            # Ingest chunks into ChromaDB
            document_ids = self._ingest_chunks(chunks)
            
            result = {
                "success": True,
                "source": source_path,
                "chunks_created": len(chunks),
                "documents_ingested": len(document_ids),
                "document_ids": document_ids,
                "metadata": base_metadata
            }
            
            logger.info(f"Successfully processed file: {source_path}")
            logger.info(f"Created {len(chunks)} chunks, ingested {len(document_ids)} documents")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": str(file_path) if file_path else "unknown"
            }
    
    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read a file into a pandas DataFrame."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                # Try different encodings for CSV files
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Unable to read CSV file with any supported encoding")
                
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _read_file_from_bytes(self, file_bytes: io.BytesIO, file_name: str) -> pd.DataFrame:
        """Read a file from BytesIO object."""
        file_extension = Path(file_name).suffix.lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_bytes)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_bytes)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error reading file from bytes {file_name}: {e}")
            raise
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean the dataframe for processing."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill missing values with empty strings for text processing
        df = df.fillna('')
        
        # Convert all columns to string for consistent processing
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        return df
    
    def _get_file_type(self, file_path: str) -> str:
        """Get the file type from the file path."""
        return Path(file_path).suffix.lower().lstrip('.')
    
    def _create_chunks(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from the dataframe for vector storage.
        
        Args:
            df: The dataframe to chunk
            base_metadata: Base metadata for all chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Strategy 1: Row-based chunking
        chunks.extend(self._create_row_chunks(df, base_metadata))
        
        # Strategy 2: Column-based chunking (for column descriptions)
        chunks.extend(self._create_column_chunks(df, base_metadata))
        
        # Strategy 3: Statistical summary chunks
        chunks.extend(self._create_summary_chunks(df, base_metadata))
        
        return chunks
    
    def _create_row_chunks(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks based on rows of data."""
        chunks = []
        
        for start_idx in range(0, len(df), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Create a text representation of the chunk
            chunk_text = self._dataframe_to_text(chunk_df)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "rows",
                "chunk_start_row": start_idx,
                "chunk_end_row": end_idx - 1,
                "chunk_size": len(chunk_df),
                "chunk_id": str(uuid.uuid4())
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "data": chunk_df.to_dict('records')
            })
        
        return chunks
    
    def _create_column_chunks(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks based on column information."""
        chunks = []
        
        for column in df.columns:
            # Create descriptive text for the column
            unique_values = df[column].unique()[:10]  # Sample of unique values
            
            column_text = f"Column: {column}\n"
            column_text += f"Data type: {df[column].dtype}\n"
            column_text += f"Non-null count: {df[column].count()}\n"
            column_text += f"Unique values: {len(df[column].unique())}\n"
            column_text += f"Sample values: {', '.join(map(str, unique_values))}\n"
            
            # Add basic statistics for numeric columns
            if df[column].dtype in ['int64', 'float64']:
                numeric_series = pd.to_numeric(df[column], errors='coerce')
                column_text += f"Mean: {numeric_series.mean():.2f}\n"
                column_text += f"Median: {numeric_series.median():.2f}\n"
                column_text += f"Min: {numeric_series.min():.2f}\n"
                column_text += f"Max: {numeric_series.max():.2f}\n"
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "column",
                "column_name": column,
                "data_type": str(df[column].dtype),
                "unique_count": len(df[column].unique()),
                "chunk_id": str(uuid.uuid4())
            })
            
            chunks.append({
                "text": column_text,
                "metadata": chunk_metadata,
                "data": {"column_info": {
                    "name": column,
                    "dtype": str(df[column].dtype),
                    "unique_values": unique_values.tolist()
                }}
            })
        
        return chunks
    
    def _create_summary_chunks(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create summary chunks with dataset overview."""
        chunks = []
        
        # Overall dataset summary
        summary_text = f"Dataset Summary:\n"
        summary_text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary_text += f"Columns: {', '.join(df.columns)}\n"
        summary_text += f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n"
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_text += f"\nNumeric columns: {', '.join(numeric_cols)}\n"
            summary_text += "Statistical Summary:\n"
            summary_text += df[numeric_cols].describe().to_string()
        
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_type": "summary",
            "summary_type": "dataset_overview",
            "chunk_id": str(uuid.uuid4())
        })
        
        chunks.append({
            "text": summary_text,
            "metadata": chunk_metadata,
            "data": {"summary_stats": df.describe().to_dict() if len(numeric_cols) > 0 else {}}
        })
        
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert a dataframe chunk to text representation."""
        text_parts = []
        
        # Add header
        text_parts.append("Data Table:")
        text_parts.append("Columns: " + ", ".join(df.columns))
        text_parts.append("")
        
        # Add data rows in a readable format
        for idx, row in df.iterrows():
            row_text = f"Row {idx}:"
            for col, value in row.items():
                if value and str(value).strip():
                    row_text += f" {col}: {value};"
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _ingest_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Ingest chunks into ChromaDB."""
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk["metadata"]["chunk_id"])
        
        # Add documents to ChromaDB
        return self.chroma_client.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_data(
        self,
        query: str,
        n_results: int = 5,
        chunk_types: Optional[List[str]] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search ingested data using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            chunk_types: Filter by chunk types (rows, column, summary)
            source_filter: Filter by source file
            
        Returns:
            List of search results
        """
        filters = {}
        
        if chunk_types:
            filters["chunk_type"] = {"$in": chunk_types}
        
        if source_filter:
            filters["source"] = source_filter
        
        return self.chroma_client.semantic_search(
            query=query,
            n_results=n_results,
            filters=filters if filters else None
        )


def process_uploaded_file(
    file_path: Union[str, Path, io.BytesIO],
    file_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process and ingest an uploaded file.
    
    Args:
        file_path: Path to the file or BytesIO object
        file_name: Name of the file (required for BytesIO)
        metadata: Additional metadata
        
    Returns:
        Processing results dictionary
    """
    processor = DataIngestionProcessor()
    return processor.process_file(file_path, file_name, metadata)
