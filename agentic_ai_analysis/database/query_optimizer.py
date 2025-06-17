"""
Query Optimizer for Enhanced ChromaDB Semantic Search.

This module provides advanced query optimization techniques including:
- Query expansion and synonyms
- Hybrid search combining semantic and keyword matching
- Result re-ranking and relevance scoring
- Query caching for improved performance
- Adaptive search strategies based on query patterns
"""

import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

import numpy as np

from .chroma_client import ChromaDBClient, get_chroma_client
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Enhanced query result with additional scoring and context."""
    id: str
    document: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int
    relevance_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchContext:
    """Context information for search optimization."""
    query: str
    expanded_query: str
    query_type: str  # "data_analysis", "column_search", "statistical", "general"
    search_filters: Dict[str, Any]
    search_strategy: str


class QueryCache:
    """Simple in-memory cache for query results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[List[QueryResult], float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, query: str, filters: Dict[str, Any], n_results: int) -> str:
        """Generate cache key from query parameters."""
        key_data = f"{query}:{str(sorted(filters.items()))}:{n_results}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, filters: Dict[str, Any], n_results: int) -> Optional[List[QueryResult]]:
        """Get cached results if available and not expired."""
        key = self._generate_key(query, filters, n_results)
        current_time = time.time()
        
        if key in self.cache:
            results, timestamp = self.cache[key]
            if current_time - timestamp < self.ttl_seconds:
                self.access_times[key] = current_time
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return results
            else:
                # Remove expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def put(self, query: str, filters: Dict[str, Any], n_results: int, results: List[QueryResult]):
        """Cache query results."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used entries
            lru_keys = sorted(self.access_times.items(), key=lambda x: x[1])[:len(self.cache) // 4]
            for key, _ in lru_keys:
                if key in self.cache:
                    del self.cache[key]
                del self.access_times[key]
        
        key = self._generate_key(query, filters, n_results)
        current_time = time.time()
        self.cache[key] = (results, current_time)
        self.access_times[key] = current_time
        logger.debug(f"Cached results for query: {query[:50]}...")


class QueryExpander:
    """Expands queries with synonyms and related terms for better search coverage."""
    
    def __init__(self):
        # Data analysis synonyms and related terms
        self.synonyms = {
            "analyze": ["examine", "study", "investigate", "review"],
            "statistics": ["stats", "statistical", "metrics", "measures"],
            "data": ["dataset", "information", "records", "table"],
            "column": ["field", "attribute", "variable", "feature"],
            "row": ["record", "entry", "observation", "instance"],
            "correlation": ["relationship", "association", "connection"],
            "distribution": ["spread", "pattern", "frequency"],
            "average": ["mean", "typical", "standard"],
            "total": ["sum", "aggregate", "combined"],
            "trend": ["pattern", "direction", "movement"],
            "outlier": ["anomaly", "exception", "unusual"],
            "missing": ["null", "empty", "blank"],
            "unique": ["distinct", "different", "separate"]
        }
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            expanded_terms.append(word)
            # Add synonyms
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word][:2])  # Limit to 2 synonyms
        
        return " ".join(expanded_terms)


class QueryTypeClassifier:
    """Classifies queries to apply appropriate search strategies."""
    
    def __init__(self):
        self.patterns = {
            "data_analysis": [
                r"analyze|analysis|examine|study",
                r"statistics|statistical|stats|distribution",
                r"correlation|relationship|trend|pattern"
            ],
            "column_search": [
                r"column|field|attribute|variable",
                r"what.*columns?|list.*columns?|show.*columns?",
                r"data types?|dtype|schema"
            ],
            "statistical": [
                r"mean|average|median|mode|std|variance",
                r"min|max|count|sum|total",
                r"quartile|percentile|outlier"
            ],
            "filter_search": [
                r"where|filter|select|find.*with",
                r"greater than|less than|equal|contains",
                r"between|range|above|below"
            ]
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type based on patterns."""
        query_lower = query.lower()
        
        scores = {}
        for query_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Return the type with highest score, default to "general"
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general"


class HybridSearchEngine:
    """Combines semantic search with keyword-based search for better results."""
    
    def __init__(self, chroma_client: Optional[ChromaDBClient] = None):
        self.chroma_client = chroma_client or get_chroma_client()
        
    def _calculate_keyword_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calculate simple keyword-based similarity scores."""
        query_words = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            # Simple Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            score = intersection / union if union > 0 else 0.0
            scores.append(score)
        
        return scores
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[QueryResult]:
        """Perform hybrid search combining semantic and keyword matching."""
        # Get semantic search results
        semantic_results = self.chroma_client.semantic_search(
            query=query,
            n_results=min(n_results * 2, 50),  # Get more for better re-ranking
            filters=filters
        )
        
        if not semantic_results:
            return []
        
        # Extract documents for keyword scoring
        documents = [result['document'] for result in semantic_results]
        
        # Calculate keyword scores
        keyword_scores = self._calculate_keyword_scores(query, documents)
        
        # Combine scores and create enhanced results
        enhanced_results = []
        for i, result in enumerate(semantic_results):
            semantic_score = result['similarity']
            keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0.0
            
            # Combined score with weights
            combined_score = (semantic_weight * semantic_score + 
                            keyword_weight * keyword_score)
            
            enhanced_result = QueryResult(
                id=result['id'],
                document=result['document'],
                metadata=result['metadata'],
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
                rank=i + 1,
                relevance_factors={
                    'semantic_weight': semantic_weight,
                    'keyword_weight': keyword_weight,
                    'doc_length': len(result['document']),
                    'chunk_type': result['metadata'].get('chunk_type', 'unknown')
                }
            )
            enhanced_results.append(enhanced_result)
        
        # Re-rank by combined score
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Update ranks after re-ranking
        for i, result in enumerate(enhanced_results[:n_results]):
            result.rank = i + 1
        
        return enhanced_results[:n_results]


class AdvancedQueryOptimizer:
    """Main query optimizer that orchestrates all optimization techniques."""
    
    def __init__(self, chroma_client: Optional[ChromaDBClient] = None):
        self.settings = get_settings()
        self.chroma_client = chroma_client or get_chroma_client()
        
        # Initialize components
        self.cache = QueryCache(max_size=1000, ttl_seconds=3600)
        self.query_expander = QueryExpander()
        self.query_classifier = QueryTypeClassifier()
        self.hybrid_engine = HybridSearchEngine(self.chroma_client)
        
        # Performance tracking
        self.query_stats = defaultdict(int)
        self.response_times = []
        
        logger.info("Advanced Query Optimizer initialized")
    
    def optimize_search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        enable_expansion: bool = True,
        enable_hybrid: bool = True,
        enable_cache: bool = True
    ) -> List[QueryResult]:
        """
        Perform optimized search with all enhancement techniques.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Metadata filters
            enable_expansion: Whether to use query expansion
            enable_hybrid: Whether to use hybrid search
            enable_cache: Whether to use caching
            
        Returns:
            List of enhanced query results
        """
        start_time = time.time()
        filters = filters or {}
        
        # Check cache first
        if enable_cache:
            cached_results = self.cache.get(query, filters, n_results)
            if cached_results:
                self.query_stats['cache_hits'] += 1
                return cached_results[:n_results]
        
        # Classify query type
        query_type = self.query_classifier.classify_query(query)
        
        # Expand query if enabled
        search_query = query
        if enable_expansion:
            search_query = self.query_expander.expand_query(query)
            logger.debug(f"Expanded query: '{query}' -> '{search_query}'")
        
        # Create search context
        context = SearchContext(
            query=query,
            expanded_query=search_query,
            query_type=query_type,
            search_filters=filters,
            search_strategy="hybrid" if enable_hybrid else "semantic"
        )
        
        # Apply query-type specific optimizations
        optimized_filters = self._optimize_filters(filters, context)
        
        # Perform search
        if enable_hybrid:
            results = self.hybrid_engine.hybrid_search(
                query=search_query,
                n_results=n_results,
                filters=optimized_filters,
                semantic_weight=self._get_semantic_weight(query_type),
                keyword_weight=self._get_keyword_weight(query_type)
            )
        else:
            # Fallback to standard semantic search
            semantic_results = self.chroma_client.semantic_search(
                query=search_query,
                n_results=n_results,
                filters=optimized_filters
            )
            results = [
                QueryResult(
                    id=r['id'],
                    document=r['document'],
                    metadata=r['metadata'],
                    semantic_score=r['similarity'],
                    keyword_score=0.0,
                    combined_score=r['similarity'],
                    rank=i + 1
                )
                for i, r in enumerate(semantic_results)
            ]
        
        # Apply post-processing optimizations
        results = self._post_process_results(results, context)
        
        # Cache results
        if enable_cache and results:
            self.cache.put(query, filters, n_results, results)
        
        # Update performance stats
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        self.query_stats['total_queries'] += 1
        
        return results[:n_results]
    
    def _optimize_filters(self, filters: Dict[str, Any], context: SearchContext) -> Optional[Dict[str, Any]]:
        """Optimize filters based on query context."""
        optimized = filters.copy() if filters else {}
        
        # Add chunk type preferences based on query type
        # Note: ChromaDB uses specific filter syntax, not MongoDB-style queries
        if context.query_type == "column_search":
            if "chunk_type" not in optimized:
                optimized["chunk_type"] = "column"  # Prefer column chunks for column searches
        elif context.query_type == "statistical":
            if "chunk_type" not in optimized:
                optimized["chunk_type"] = "summary"  # Prefer summary chunks for statistical queries
        elif context.query_type == "data_analysis":
            # For data analysis, we'll search without chunk type restriction to get broader results
            pass
        elif context.query_type == "filter_search":
            # For filter searches, avoid adding additional constraints
            pass
        
        # Return None if empty to avoid ChromaDB validation issues
        return optimized if optimized else None
    
    def _get_semantic_weight(self, query_type: str) -> float:
        """Get semantic weight based on query type."""
        weights = {
            "data_analysis": 0.8,    # Prefer semantic understanding
            "column_search": 0.6,    # Balance semantic and keyword
            "statistical": 0.7,      # Slightly prefer semantic
            "filter_search": 0.5,    # Equal balance
            "general": 0.7          # Default semantic preference
        }
        return weights.get(query_type, 0.7)
    
    def _get_keyword_weight(self, query_type: str) -> float:
        """Get keyword weight based on query type."""
        return 1.0 - self._get_semantic_weight(query_type)
    
    def _post_process_results(self, results: List[QueryResult], context: SearchContext) -> List[QueryResult]:
        """Apply post-processing optimizations to results."""
        if not results:
            return results
        
        # Boost results based on relevance factors
        for result in results:
            boost_factor = 1.0
            
            # Boost based on chunk type relevance
            chunk_type = result.metadata.get('chunk_type', '')
            if context.query_type == "column_search" and chunk_type == "column":
                boost_factor *= 1.2
            elif context.query_type == "statistical" and chunk_type == "summary":
                boost_factor *= 1.2
            elif context.query_type == "data_analysis" and chunk_type in ["summary", "column"]:
                boost_factor *= 1.1
            
            # Apply boost
            result.combined_score *= boost_factor
            result.relevance_factors['post_process_boost'] = boost_factor
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.response_times:
            return {"message": "No queries processed yet"}
        
        cache_hit_rate = (self.query_stats['cache_hits'] / 
                         max(self.query_stats['total_queries'], 1))
        
        return {
            "total_queries": self.query_stats['total_queries'],
            "avg_response_time": np.mean(self.response_times),
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache.cache)
        }


# Global optimizer instance
_query_optimizer = None


def get_query_optimizer() -> AdvancedQueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = AdvancedQueryOptimizer()
    return _query_optimizer 