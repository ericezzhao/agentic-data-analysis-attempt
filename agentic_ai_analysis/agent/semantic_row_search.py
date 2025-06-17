"""Semantic Row Search Engine for Data Analysis Agent.

Enables finding actual data rows based on semantic similarity to natural language queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from database.chroma_client import ChromaDBClient

logger = logging.getLogger(__name__)


class SemanticRowSearchEngine:
    """Search for actual data rows using semantic similarity."""
    
    def __init__(self, dataset_cache: Dict[str, pd.DataFrame], chroma_client: ChromaDBClient):
        """Initialize with reference to agent's dataset cache and ChromaDB client."""
        self._cache = dataset_cache
        self._chroma_client = chroma_client
        self._row_collections = {}  # Track which datasets have row-level embeddings
        
        # Keywords that indicate row-level search intent
        self._row_search_keywords = {
            "customers", "records", "rows", "entries", "people", "individuals",
            "find", "show me", "list", "who", "which", "similar to"
        }
        
        # Semantic patterns for common search intents
        self._search_patterns = {
            "wealthy": ["high income", "rich", "affluent", "premium", "expensive"],
            "young": ["age", "youth", "millennial", "gen z"],
            "tech": ["technology", "IT", "software", "computer", "digital"],
            "industry": ["sector", "business", "company", "field", "domain"]
        }
    
    def detect_row_search_query(self, query: str) -> bool:
        """Detect if a query is asking for specific data rows."""
        query_lower = query.lower().strip()
        
        # Enhanced keywords that indicate row search
        row_keywords = [
            "customers", "people", "employees", "records", "individuals", "persons", "staff", "workers",
            "find", "show", "list", "get", "who", "which", "search", "locate", "identify",
            "engineers", "sales", "marketing", "finance", "hr", "managers", "developers",
            "wealthy", "high", "top", "best", "experienced", "senior", "junior", "new",
            "in", "from", "at", "living", "located", "based", "working"
        ]
        
        # Location and descriptive terms that indicate row searches
        location_keywords = [
            "san francisco", "new york", "california", "texas", "chicago", "boston", 
            "seattle", "austin", "denver", "miami", "portland", "los angeles"
        ]
        
        # Patterns that suggest row searches
        search_patterns = [
            "in ", "from ", "at ", "with ", "who ", "which ", "that ",
            "engineers in", "people in", "employees in", "customers in",
            "staff in", "workers in", "sales in", "marketing in"
        ]
        
        # Check for row keywords
        has_row_keyword = any(keyword in query_lower for keyword in row_keywords)
        
        # Check for location keywords  
        has_location = any(location in query_lower for location in location_keywords)
        
        # Check for search patterns
        has_search_pattern = any(pattern in query_lower for pattern in search_patterns)
        
        # Avoid statistical and visualization queries
        stat_keywords = ["average", "mean", "sum", "count", "total", "max", "min", "median", "std"]
        viz_keywords = ["show histogram", "plot", "chart", "graph", "visualize", "bar chart", "scatter"]
        
        is_stat_query = any(stat in query_lower for stat in stat_keywords)
        is_viz_query = any(viz in query_lower for viz in viz_keywords)
        
        # Enhanced detection: row search if has keywords/patterns/locations but not stat/viz
        is_row_search = (has_row_keyword or has_location or has_search_pattern) and not (is_stat_query or is_viz_query)
        
        # Special case: if query starts with "show" but not visualization, it might be row search
        if query_lower.startswith("show ") and not is_viz_query:
            # "show employees" vs "show histogram" 
            show_terms = ["employees", "people", "customers", "staff", "records"]
            if any(term in query_lower for term in show_terms):
                is_row_search = True
        
        return is_row_search
    
    async def ensure_row_embeddings(self, dataset_name: str) -> bool:
        """Ensure row-level embeddings exist for the dataset."""
        if dataset_name not in self._cache:
            logger.error(f"Dataset '{dataset_name}' not found in cache")
            return False
        
        collection_name = f"rows_{dataset_name.replace('.', '_').replace('/', '_')}"
        
        if collection_name in self._row_collections:
            logger.info(f"Row embeddings already exist for {dataset_name}")
            return True
        
        try:
            df = self._cache[dataset_name]
            logger.info(f"Creating row embeddings for {dataset_name} with {len(df)} rows")
            
            # Create textual representation of each row
            row_texts = []
            row_metadata = []
            
            for idx, row in df.iterrows():
                # Convert row to descriptive text
                row_text = self._row_to_text(row, df.columns)
                row_texts.append(row_text)
                
                # Store metadata for retrieval
                metadata = {
                    "dataset": dataset_name,
                    "row_index": int(idx),
                    "source": f"row_{idx}"
                }
                
                # Add key column values to metadata for filtering
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        metadata[f"col_{col}"] = str(val)
                
                row_metadata.append(metadata)
            
            # Store in ChromaDB with special collection for rows
            document_ids = self._chroma_client.add_documents(
                collection_name=collection_name,
                documents=row_texts,
                metadatas=row_metadata,
                ids=[f"{dataset_name}_row_{i}" for i in range(len(row_texts))]
            )
            
            if document_ids and len(document_ids) == len(row_texts):
                self._row_collections[collection_name] = dataset_name
                logger.info(f"Successfully created {len(row_texts)} row embeddings for {dataset_name}")
                return True
            else:
                logger.error(f"Failed to create row embeddings for {dataset_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating row embeddings: {e}")
            return False
    
    def _row_to_text(self, row: pd.Series, columns: List[str]) -> str:
        """Convert a pandas row to descriptive text for embedding."""
        text_parts = []
        
        for col in columns:
            value = row[col]
            if pd.notna(value):
                # Create natural language description
                if isinstance(value, (int, float)):
                    text_parts.append(f"{col} is {value}")
                else:
                    text_parts.append(f"{col} is {value}")
        
        return ". ".join(text_parts)
    
    async def search_rows(self, dataset_name: str, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search for data rows based on semantic similarity."""
        try:
            # Ensure row embeddings exist
            if not await self.ensure_row_embeddings(dataset_name):
                return {"success": False, "error": "Failed to create row embeddings"}
            
            collection_name = f"rows_{dataset_name.replace('.', '_').replace('/', '_')}"
            
            # Perform semantic search on row embeddings
            results = self._chroma_client.semantic_search(
                query=query,
                n_results=max_results,
                collection_name=collection_name,
                filters={"dataset": dataset_name}
            )
            
            if not results:
                return {
                    "success": True,
                    "rows": [],
                    "message": f"No rows found matching query: '{query}'"
                }
            
            # Retrieve actual data rows
            df = self._cache[dataset_name]
            found_rows = []
            
            for result in results:
                try:
                    row_index = result["metadata"]["row_index"]
                    similarity = result["similarity"]
                    
                    # Get the actual row data
                    row_data = df.iloc[row_index].to_dict()
                    
                    found_rows.append({
                        "row_index": row_index,
                        "similarity": similarity,
                        "data": row_data,
                        "text_representation": result["document"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
            
            return {
                "success": True,
                "query": query,
                "dataset": dataset_name,
                "rows": found_rows,
                "total_found": len(found_rows)
            }
            
        except Exception as e:
            logger.error(f"Error in semantic row search: {e}")
            return {"success": False, "error": f"Search failed: {str(e)}"}
    
    def format_row_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format row search results for agent response."""
        if not search_results.get("success", False):
            return search_results
        
        rows = search_results.get("rows", [])
        if not rows:
            return {
                "success": True,
                "results_count": 0,
                "insights": [f"No rows found matching the query"],
                "results": []
            }
        
        # Format for agent response
        formatted_results = []
        insights = []
        
        for row in rows:
            similarity = row["similarity"]
            data = row["data"] 
            
            # Create summary of row
            key_fields = list(data.keys())[:4]  # Show first 4 fields
            summary = ", ".join([f"{k}: {data[k]}" for k in key_fields])
            
            formatted_results.append({
                "content": summary,
                "relevance": similarity,
                "source": search_results["dataset"],
                "type": "data_row",
                "full_data": data,
                "row_index": row["row_index"]
            })
        
        # Generate insights
        avg_similarity = sum(r["similarity"] for r in rows) / len(rows)
        insights.append(f"Found {len(rows)} matching data rows")
        
        if avg_similarity > 0.7:
            insights.append("High similarity scores - strong semantic matches")
        elif avg_similarity < 0.4:
            insights.append("Lower similarity scores - may need to refine query")
        
        # Analyze patterns in results
        if len(rows) > 1:
            insights.append(f"Results show similarity range: {min(r['similarity'] for r in rows):.3f} to {max(r['similarity'] for r in rows):.3f}")
        
        return {
            "success": True,
            "query": search_results["query"],
            "results_count": len(rows),
            "insights": insights,
            "results": formatted_results
        } 