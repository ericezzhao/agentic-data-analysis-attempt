"""
Enhanced Data Processing Module with Optimized Chunking and Indexing.

This module extends the basic data ingestion with:
- Intelligent chunking strategies based on data characteristics
- Metadata indexing for faster filtering
- Batch processing optimizations
- Content-aware chunk sizing
- Advanced text preprocessing for better embeddings
"""

import io
import uuid
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .data_ingestion import DataIngestionProcessor
from .chroma_client import ChromaDBClient, get_chroma_client
from .query_optimizer import get_query_optimizer, QueryResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Configuration for data chunking strategies."""
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 50
    strategy_type: str = "adaptive"  # "fixed", "adaptive", "content_aware"
    preserve_structure: bool = True


@dataclass
class ProcessingResult:
    """Enhanced processing result with detailed metrics."""
    success: bool
    source: str
    total_chunks: int
    chunk_types: Dict[str, int]
    processing_time: float
    optimization_metrics: Dict[str, Any]
    error_message: Optional[str] = None


class ContentAnalyzer:
    """Analyzes content to determine optimal chunking strategies."""
    
    def __init__(self):
        self.numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        self.date_patterns = [
            re.compile(r'\d{4}-\d{2}-\d{2}'),
            re.compile(r'\d{2}/\d{2}/\d{4}'),
            re.compile(r'\d{2}-\d{2}-\d{4}')
        ]
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataframe characteristics for optimization."""
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "column_types": {},
            "data_density": 0.0,
            "text_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "unique_ratios": {},
            "null_percentages": {}
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            # Calculate null percentage
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            analysis["null_percentages"][col] = null_pct
            
            # Calculate unique ratio
            unique_ratio = len(col_data.unique()) / len(col_data) if len(col_data) > 0 else 0
            analysis["unique_ratios"][col] = unique_ratio
            
            # Classify column type
            col_type = self._classify_column_type(col_data)
            analysis["column_types"][col] = col_type
            
            # Categorize columns
            if col_type == "numeric":
                analysis["numeric_columns"].append(col)
            elif col_type == "date":
                analysis["date_columns"].append(col)
            elif col_type == "text":
                analysis["text_columns"].append(col)
            else:
                analysis["categorical_columns"].append(col)
        
        # Calculate overall data density
        non_null_cells = df.notna().sum().sum()
        total_cells = len(df) * len(df.columns)
        analysis["data_density"] = non_null_cells / total_cells if total_cells > 0 else 0
        
        return analysis
    
    def _classify_column_type(self, series: pd.Series) -> str:
        """Classify the type of a pandas series."""
        if len(series) == 0:
            return "unknown"
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        
        # Convert to string for pattern matching
        str_series = series.astype(str)
        sample_size = min(100, len(str_series))
        sample = str_series.sample(sample_size) if len(str_series) > sample_size else str_series
        
        # Check for dates
        date_matches = sum(1 for val in sample if any(pattern.search(val) for pattern in self.date_patterns))
        if date_matches / len(sample) > 0.7:
            return "date"
        
        # Check for numeric strings
        numeric_matches = sum(1 for val in sample if self.numeric_pattern.match(val.strip()))
        if numeric_matches / len(sample) > 0.8:
            return "numeric"
        
        # Check average text length
        avg_length = sample.str.len().mean()
        unique_ratio = len(series.unique()) / len(series)
        
        if avg_length > 50 or unique_ratio > 0.9:
            return "text"
        else:
            return "categorical"
    
    def recommend_chunking_strategy(self, analysis: Dict[str, Any]) -> ChunkingStrategy:
        """Recommend optimal chunking strategy based on data analysis."""
        row_count = analysis["row_count"]
        memory_mb = analysis["memory_usage_mb"]
        text_cols = len(analysis["text_columns"])
        
        # Base strategy on data size and characteristics
        if memory_mb > 100 or row_count > 10000:
            # Large dataset - use smaller chunks
            return ChunkingStrategy(
                max_chunk_size=500,
                min_chunk_size=100,
                overlap_size=25,
                strategy_type="adaptive",
                preserve_structure=True
            )
        elif text_cols > 2:
            # Text-heavy dataset - preserve text coherence
            return ChunkingStrategy(
                max_chunk_size=200,
                min_chunk_size=50,
                overlap_size=20,
                strategy_type="content_aware",
                preserve_structure=True
            )
        else:
            # Standard dataset
            return ChunkingStrategy(
                max_chunk_size=1000,
                min_chunk_size=200,
                overlap_size=50,
                strategy_type="adaptive",
                preserve_structure=True
            )


class EnhancedTextProcessor:
    """Enhanced text processing for better embeddings."""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def preprocess_text(self, text: str, preserve_structure: bool = True) -> str:
        """Preprocess text for better embedding quality."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Clean up common data artifacts
        text = re.sub(r'\b(nan|null|none|n/a)\b', '[MISSING]', text, flags=re.IGNORECASE)
        
        # Normalize numeric representations
        text = re.sub(r'\b\d+\.\d+\b', '[NUMBER]', text)
        text = re.sub(r'\b\d+\b', '[INTEGER]', text)
        
        return text.strip()
    
    def enhance_chunk_text(self, chunk_data: Dict[str, Any], chunk_type: str) -> str:
        """Enhance chunk text with context and metadata."""
        base_text = chunk_data.get("text", "")
        metadata = chunk_data.get("metadata", {})
        
        # Add context based on chunk type
        if chunk_type == "column":
            column_name = metadata.get("column_name", "unknown")
            data_type = metadata.get("data_type", "unknown")
            prefix = f"Column analysis for '{column_name}' (type: {data_type}): "
            return prefix + base_text
        
        elif chunk_type == "summary":
            source = metadata.get("source", "unknown")
            prefix = f"Dataset summary for '{source}': "
            return prefix + base_text
        
        elif chunk_type == "rows":
            start_row = metadata.get("chunk_start_row", 0)
            end_row = metadata.get("chunk_end_row", 0)
            prefix = f"Data rows {start_row}-{end_row}: "
            return prefix + base_text
        
        return base_text


class EnhancedDataProcessor(DataIngestionProcessor):
    """Enhanced data processor with optimization features."""
    
    def __init__(self, chroma_client: Optional[ChromaDBClient] = None):
        super().__init__(chroma_client)
        self.content_analyzer = ContentAnalyzer()
        self.text_processor = EnhancedTextProcessor()
        self.query_optimizer = get_query_optimizer()
        
        # Batch processing settings
        self.batch_size = 100
        self.enable_optimizations = True
        
        logger.info("Enhanced data processor initialized")
    
    def process_file_optimized(
        self,
        file_path: Union[str, Path, io.BytesIO],
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None
    ) -> ProcessingResult:
        """
        Process file with enhanced optimizations.
        
        Args:
            file_path: Path to file or BytesIO object
            file_name: Name of the file
            metadata: Additional metadata
            chunking_strategy: Custom chunking strategy
            
        Returns:
            Enhanced processing result
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Read and analyze the data
            if isinstance(file_path, io.BytesIO):
                if not file_name:
                    raise ValueError("file_name is required when using BytesIO")
                df = self._read_file_from_bytes(file_path, file_name)
                source_path = file_name
            else:
                file_path = Path(file_path)
                df = self._read_file(file_path)
                source_path = str(file_path)
            
            # Prepare dataframe
            df = self._prepare_dataframe(df)
            
            # Analyze content characteristics
            content_analysis = self.content_analyzer.analyze_dataframe(df)
            
            # Determine optimal chunking strategy
            if chunking_strategy is None:
                chunking_strategy = self.content_analyzer.recommend_chunking_strategy(content_analysis)
            
            # Create base metadata with analysis results
            base_metadata = {
                "source": source_path,
                "file_name": file_name or Path(source_path).name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": ", ".join(df.columns),
                "file_type": self._get_file_type(source_path),
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "content_analysis": content_analysis,
                "chunking_strategy": {
                    "max_chunk_size": chunking_strategy.max_chunk_size,
                    "strategy_type": chunking_strategy.strategy_type
                }
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Create optimized chunks
            chunks = self._create_optimized_chunks(df, base_metadata, chunking_strategy)
            
            # Process chunks in batches for better performance
            document_ids = self._batch_ingest_chunks(chunks)
            
            # Calculate processing metrics
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Count chunk types
            chunk_type_counts = {}
            for chunk in chunks:
                chunk_type = chunk["metadata"].get("chunk_type", "unknown")
                chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            result = ProcessingResult(
                success=True,
                source=source_path,
                total_chunks=len(chunks),
                chunk_types=chunk_type_counts,
                processing_time=processing_time,
                optimization_metrics={
                    "content_analysis": content_analysis,
                    "chunking_strategy": chunking_strategy.__dict__,
                    "documents_ingested": len(document_ids),
                    "avg_chunk_size": np.mean([len(c["text"]) for c in chunks]) if chunks else 0,
                    "embedding_efficiency": len(document_ids) / len(chunks) if chunks else 0
                }
            )
            
            logger.info(f"Enhanced processing completed: {source_path}")
            logger.info(f"Created {len(chunks)} chunks in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"Enhanced processing failed for {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                source=str(file_path) if file_path else "unknown",
                total_chunks=0,
                chunk_types={},
                processing_time=processing_time,
                optimization_metrics={},
                error_message=str(e)
            )
    
    def _create_optimized_chunks(
        self,
        df: pd.DataFrame,
        base_metadata: Dict[str, Any],
        strategy: ChunkingStrategy
    ) -> List[Dict[str, Any]]:
        """Create optimized chunks based on strategy and content analysis."""
        chunks = []
        
        # Create different types of chunks based on strategy
        if strategy.strategy_type == "adaptive":
            chunks.extend(self._create_adaptive_row_chunks(df, base_metadata, strategy))
        else:
            chunks.extend(self._create_row_chunks(df, base_metadata))
        
        # Always create column and summary chunks
        chunks.extend(self._create_enhanced_column_chunks(df, base_metadata))
        chunks.extend(self._create_enhanced_summary_chunks(df, base_metadata))
        
        # Enhance text content for all chunks
        for chunk in chunks:
            chunk_type = chunk["metadata"].get("chunk_type", "unknown")
            enhanced_text = self.text_processor.enhance_chunk_text(chunk, chunk_type)
            chunk["text"] = self.text_processor.preprocess_text(
                enhanced_text, 
                preserve_structure=strategy.preserve_structure
            )
            
            # Flatten metadata for ChromaDB compatibility
            chunk["metadata"] = self._flatten_metadata_for_chromadb(chunk["metadata"])
        
        return chunks
    
    def _flatten_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten metadata to ensure ChromaDB compatibility.
        ChromaDB only accepts str, int, float, bool values in metadata.
        """
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flattened[key] = value
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    flat_key = f"{key}_{nested_key}"
                    if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                        flattened[flat_key] = nested_value
                    elif isinstance(nested_value, list):
                        # Convert lists to comma-separated strings
                        flattened[flat_key] = ", ".join(str(item) for item in nested_value) if nested_value else ""
                    else:
                        flattened[flat_key] = str(nested_value)
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                flattened[key] = ", ".join(str(item) for item in value) if value else ""
            else:
                # Convert other types to strings
                flattened[key] = str(value)
        
        return flattened
    
    def _create_adaptive_row_chunks(
        self,
        df: pd.DataFrame,
        base_metadata: Dict[str, Any],
        strategy: ChunkingStrategy
    ) -> List[Dict[str, Any]]:
        """Create adaptive row chunks based on content density."""
        chunks = []
        content_analysis = base_metadata.get("content_analysis", {})
        
        # Adjust chunk size based on data density
        data_density = content_analysis.get("data_density", 0.5)
        avg_text_length = self._calculate_avg_text_length(df)
        
        # Dynamic chunk sizing
        if data_density > 0.8 and avg_text_length > 100:
            # Dense, text-heavy data - smaller chunks
            chunk_size = max(strategy.min_chunk_size, strategy.max_chunk_size // 4)
        elif data_density < 0.3:
            # Sparse data - larger chunks to maintain context
            chunk_size = strategy.max_chunk_size
        else:
            # Standard chunking
            chunk_size = strategy.max_chunk_size // 2
        
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Enhanced text representation
            chunk_text = self._enhanced_dataframe_to_text(chunk_df, content_analysis)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "rows",
                "chunk_start_row": start_idx,
                "chunk_end_row": end_idx - 1,
                "chunk_size": len(chunk_df),
                "adaptive_chunk_size": chunk_size,
                "chunk_id": str(uuid.uuid4())
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "data": chunk_df.to_dict('records')
            })
        
        return chunks
    
    def _create_enhanced_column_chunks(
        self,
        df: pd.DataFrame,
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create enhanced column chunks with better analysis."""
        chunks = []
        content_analysis = base_metadata.get("content_analysis", {})
        
        for column in df.columns:
            # Enhanced column analysis
            col_analysis = self._analyze_column_deeply(df[column], content_analysis)
            
            # Create comprehensive column text
            column_text = self._create_enhanced_column_text(column, col_analysis)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_type": "column",
                "column_name": column,
                "data_type": col_analysis["type"],
                "unique_count": col_analysis["unique_count"],
                "null_percentage": col_analysis["null_percentage"],
                "column_analysis": col_analysis,
                "chunk_id": str(uuid.uuid4())
            })
            
            chunks.append({
                "text": column_text,
                "metadata": chunk_metadata,
                "data": {"column_analysis": col_analysis}
            })
        
        return chunks
    
    def _create_enhanced_summary_chunks(
        self,
        df: pd.DataFrame,
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create enhanced summary chunks with comprehensive analysis."""
        chunks = []
        content_analysis = base_metadata.get("content_analysis", {})
        
        # Overall dataset summary
        summary_text = self._create_comprehensive_summary(df, content_analysis)
        
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_type": "summary",
            "summary_type": "comprehensive_overview",
            "chunk_id": str(uuid.uuid4())
        })
        
        chunks.append({
            "text": summary_text,
            "metadata": chunk_metadata,
            "data": {"summary_analysis": content_analysis}
        })
        
        return chunks
    
    def _calculate_avg_text_length(self, df: pd.DataFrame) -> float:
        """Calculate average text length across all cells."""
        text_lengths = []
        for col in df.columns:
            lengths = df[col].astype(str).str.len()
            text_lengths.extend(lengths.tolist())
        return np.mean(text_lengths) if text_lengths else 0
    
    def _enhanced_dataframe_to_text(self, df: pd.DataFrame, content_analysis: Dict[str, Any]) -> str:
        """Create enhanced text representation of dataframe chunk."""
        text_parts = []
        
        # Add context header
        text_parts.append(f"Data segment with {len(df)} rows and {len(df.columns)} columns")
        text_parts.append(f"Columns: {', '.join(df.columns)}")
        text_parts.append("")
        
        # Add sample rows with intelligent selection
        sample_size = min(5, len(df))
        if len(df) > sample_size:
            # Select diverse samples
            sample_indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sample_df = df.iloc[sample_indices]
        else:
            sample_df = df
        
        for idx, (orig_idx, row) in enumerate(sample_df.iterrows()):
            row_text = f"Record {orig_idx}:"
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    row_text += f" {col}={value};"
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def _analyze_column_deeply(self, series: pd.Series, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep analysis of a column."""
        analysis = {
            "type": content_analysis.get("column_types", {}).get(series.name, "unknown"),
            "unique_count": len(series.unique()),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "sample_values": series.dropna().unique()[:10].tolist()
        }
        
        # Add type-specific analysis
        if analysis["type"] == "numeric":
            numeric_series = pd.to_numeric(series, errors='coerce')
            analysis.update({
                "mean": float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else 0.0,
                "median": float(numeric_series.median()) if not pd.isna(numeric_series.median()) else 0.0,
                "std": float(numeric_series.std()) if not pd.isna(numeric_series.std()) else 0.0,
                "min": float(numeric_series.min()) if not pd.isna(numeric_series.min()) else 0.0,
                "max": float(numeric_series.max()) if not pd.isna(numeric_series.max()) else 0.0
            })
        elif analysis["type"] == "text":
            text_series = series.astype(str)
            analysis.update({
                "avg_length": float(text_series.str.len().mean()),
                "max_length": int(text_series.str.len().max()),
                "common_words": self._extract_common_words(text_series)
            })
        
        return analysis
    
    def _create_enhanced_column_text(self, column: str, analysis: Dict[str, Any]) -> str:
        """Create enhanced text description for a column."""
        text_parts = [f"Column '{column}' analysis:"]
        text_parts.append(f"Type: {analysis['type']}")
        text_parts.append(f"Unique values: {analysis['unique_count']}")
        text_parts.append(f"Missing data: {analysis['null_percentage']:.1f}%")
        
        if analysis["type"] == "numeric" and "mean" in analysis:
            text_parts.append(f"Statistics: mean={analysis['mean']:.2f}, median={analysis['median']:.2f}")
            text_parts.append(f"Range: {analysis['min']:.2f} to {analysis['max']:.2f}")
        elif analysis["type"] == "text" and "avg_length" in analysis:
            text_parts.append(f"Text length: average={analysis['avg_length']:.1f}, max={analysis['max_length']}")
        
        text_parts.append(f"Sample values: {', '.join(map(str, analysis['sample_values']))}")
        
        return "\n".join(text_parts)
    
    def _create_comprehensive_summary(self, df: pd.DataFrame, content_analysis: Dict[str, Any]) -> str:
        """Create comprehensive dataset summary."""
        summary_parts = ["Comprehensive Dataset Analysis:"]
        summary_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        summary_parts.append(f"Memory usage: {content_analysis.get('memory_usage_mb', 0):.2f} MB")
        summary_parts.append(f"Data density: {content_analysis.get('data_density', 0):.1%}")
        
        # Column type distribution
        col_types = content_analysis.get("column_types", {})
        type_counts = {}
        for col_type in col_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        summary_parts.append("Column types: " + ", ".join([f"{k}: {v}" for k, v in type_counts.items()]))
        
        # Data quality summary
        null_percentages = content_analysis.get("null_percentages", {})
        avg_null_pct = np.mean(list(null_percentages.values())) if null_percentages else 0
        summary_parts.append(f"Average missing data: {avg_null_pct:.1f}%")
        
        return "\n".join(summary_parts)
    
    def _extract_common_words(self, text_series: pd.Series, top_n: int = 5) -> List[str]:
        """Extract common words from text series."""
        all_text = " ".join(text_series.astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_counts = {}
        for word in words:
            if len(word) > 2 and word not in self.text_processor.stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        return sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:top_n]
    
    def _batch_ingest_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Ingest chunks in batches for better performance."""
        document_ids = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            documents = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            
            try:
                batch_ids = self.chroma_client.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                document_ids.extend(batch_ids)
                logger.debug(f"Processed batch {i//self.batch_size + 1}: {len(batch)} chunks")
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                # Continue with next batch
        
        return document_ids
    
    def optimized_search(
        self,
        query: str,
        n_results: int = 5,
        chunk_types: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
        use_optimizer: bool = True
    ) -> List[QueryResult]:
        """
        Perform optimized search using the query optimizer.
        
        Args:
            query: Search query
            n_results: Number of results
            chunk_types: Filter by chunk types
            source_filter: Filter by source file
            use_optimizer: Whether to use the advanced optimizer
            
        Returns:
            List of optimized query results
        """
        filters = {}
        
        if chunk_types:
            filters["chunk_type"] = {"$in": chunk_types}
        
        if source_filter:
            filters["source"] = source_filter
        
        if use_optimizer:
            return self.query_optimizer.optimize_search(
                query=query,
                n_results=n_results,
                filters=filters
            )
        else:
            # Fallback to standard search
            results = self.chroma_client.semantic_search(
                query=query,
                n_results=n_results,
                filters=filters if filters else None
            )
            
            return [
                QueryResult(
                    id=r['id'],
                    document=r['document'],
                    metadata=r['metadata'],
                    semantic_score=r['similarity'],
                    keyword_score=0.0,
                    combined_score=r['similarity'],
                    rank=i + 1
                )
                for i, r in enumerate(results)
            ]


def get_enhanced_processor() -> EnhancedDataProcessor:
    """Get enhanced data processor instance."""
    return EnhancedDataProcessor() 