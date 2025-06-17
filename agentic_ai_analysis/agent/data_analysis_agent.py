"""
Data Analysis Agent using Agno Framework

This module implements an intelligent data analysis agent that can:
- Reason about data analysis tasks
- Maintain conversation memory
- Collaborate with users through natural language
- Execute complex analysis workflows
- Generate insights and visualizations

The agent integrates with:
- MCP Server for tool access
- ChromaDB for semantic search and memory
- Enhanced query optimization
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

# Agno framework imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.pandas import PandasTools

# Custom tools and integrations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_client import get_chroma_client
from database.enhanced_data_processor import EnhancedDataProcessor
from database.query_optimizer import AdvancedQueryOptimizer, SearchContext
from config.settings import get_settings
from agent.statistics_engine import StatisticsEngine
from agent.visualization_engine import VisualizationEngine
from agent.semantic_row_search import SemanticRowSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Context for data analysis operations."""
    dataset_name: str
    query: str
    analysis_type: str
    user_intent: str
    conversation_id: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DataAnalysisMemory:
    """Custom memory storage for data analysis conversations."""
    
    def __init__(self, chroma_client=None):
        self.chroma_client = chroma_client or get_chroma_client()
        self.collection_name = "agent_memory"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the memory collection exists."""
        try:
            # Simply get the collection - it will be created if it doesn't exist
            self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Memory collection '{self.collection_name}' ensured")
        except Exception as e:
            logger.error(f"Error ensuring memory collection: {e}")
    
    async def read(self, session_id: str) -> List[Dict[str, Any]]:
        """Read conversation history for a session."""
        try:
            results = self.chroma_client.semantic_search(
                query=f"session {session_id}",
                n_results=50,  # Get more history
                filters={"session_id": session_id}
            )
            
            # Sort by timestamp
            memories = []
            for result in results:
                # Parse analysis_context from JSON string back to dict
                analysis_context_str = result["metadata"].get("analysis_context")
                analysis_context = None
                if analysis_context_str:
                    try:
                        analysis_context = json.loads(analysis_context_str)
                    except (json.JSONDecodeError, TypeError):
                        analysis_context = None
                
                memory = {
                    "role": result["metadata"].get("role", "user"),
                    "message": result["document"],
                    "timestamp": result["metadata"].get("timestamp"),
                    "analysis_context": analysis_context
                }
                memories.append(memory)
            
            # Sort by timestamp (handle None values)
            memories.sort(key=lambda x: x.get("timestamp") or "")
            return memories
            
        except Exception as e:
            logger.error(f"Error reading memories: {e}")
            return []
    
    async def write(self, session_id: str, data: Dict[str, Any]) -> None:
        """Write conversation memory."""
        try:
            document = data.get("content", data.get("message", ""))
            
            # Ensure no None values in metadata (ChromaDB requirement)
            analysis_context = data.get("analysis_context", {})
            if analysis_context is None:
                analysis_context = {}
            
            metadata = {
                "session_id": session_id or "unknown",
                "role": data.get("role") or "user",
                "timestamp": data.get("timestamp") or datetime.now().isoformat(),
                "analysis_context": json.dumps(analysis_context)
            }
            
            # Add to ChromaDB
            self.chroma_client.add_documents(
                collection_name=self.collection_name,
                documents=[document],
                metadatas=[metadata],
                ids=[f"{session_id}_{datetime.now().timestamp()}"]
            )
            
        except Exception as e:
            logger.error(f"Error writing memory: {e}")


class DataAnalysisTools:
    """Custom tools for data analysis operations."""
    
    def __init__(self, chroma_client=None, data_processor=None):
        self.chroma_client = chroma_client or get_chroma_client()
        self.data_processor = data_processor or EnhancedDataProcessor()
        self.query_optimizer = AdvancedQueryOptimizer(chroma_client=self.chroma_client)
    
    async def upload_and_analyze_dataset(self, file_path: str, file_name: str, description: str = "") -> Dict[str, Any]:
        """Upload a dataset and perform initial analysis."""
        try:
            # Process the file
            result = self.data_processor.process_file_optimized(
                file_path=file_path,
                file_name=file_name,
                metadata={"description": description}
            )
            
            if result.success:
                # Get initial insights
                insights = await self._generate_initial_insights(file_name, result)
                
                return {
                    "success": True,
                    "message": f"Successfully processed dataset '{file_name}'",
                    "chunks_created": result.total_chunks,
                    "processing_time": result.processing_time,
                    "insights": insights,
                    "optimization_metrics": result.optimization_metrics
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to process dataset: {result.error_message}"
                }
                
        except Exception as e:
            logger.error(f"Error in upload_and_analyze_dataset: {e}")
            return {
                "success": False,
                "message": f"Error processing dataset: {str(e)}"
            }
    
    async def intelligent_query(self, query: str, dataset_name: str = None, max_results: int = 10) -> Dict[str, Any]:
        """Perform intelligent querying with optimization."""
        try:
            # Create search context
            search_context = SearchContext(
                query=query,
                expanded_query="",  # Will be filled by optimizer
                query_type="",     # Will be determined by optimizer
                search_filters={},
                search_strategy=""
            )
            
            # Optimize the search
            results = self.query_optimizer.optimize_search(
                search_context=search_context,
                dataset_filter=dataset_name,
                max_results=max_results
            )
            
            if results:
                # Analyze and rank results
                analyzed_results = await self._analyze_query_results(query, results)
                
                return {
                    "success": True,
                    "query": query,
                    "query_type": search_context.query_type,
                    "results_count": len(results),
                    "results": analyzed_results,
                    "insights": await self._generate_query_insights(query, results)
                }
            else:
                return {
                    "success": False,
                    "message": f"No results found for query: '{query}'"
                }
                
        except Exception as e:
            logger.error(f"Error in intelligent_query: {e}")
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}"
            }
    
    async def perform_statistical_analysis(self, dataset_name: str, columns: List[str] = None, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        try:
            if analysis_types is None:
                analysis_types = ["descriptive", "correlation", "distribution"]
            
            # Search for relevant data
            filters = {"source": dataset_name}
            if columns:
                filters["chunk_type"] = "column"
            
            results = self.chroma_client.semantic_search(
                query=f"statistical analysis {dataset_name} {' '.join(columns or [])}",
                n_results=20,
                filters=filters
            )
            
            if results:
                # Compile statistical insights
                stats_analysis = await self._compile_statistical_analysis(results, analysis_types)
                
                return {
                    "success": True,
                    "dataset": dataset_name,
                    "analysis_types": analysis_types,
                    "columns_analyzed": columns or "all",
                    "statistical_summary": stats_analysis,
                    "recommendations": await self._generate_statistical_recommendations(stats_analysis)
                }
            else:
                return {
                    "success": False,
                    "message": f"No statistical data found for dataset: '{dataset_name}'"
                }
                
        except Exception as e:
            logger.error(f"Error in perform_statistical_analysis: {e}")
            return {
                "success": False,
                "message": f"Error performing statistical analysis: {str(e)}"
            }
    
    async def suggest_visualizations(self, dataset_name: str, analysis_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Suggest appropriate visualizations based on data characteristics."""
        try:
            # Get data characteristics
            results = self.chroma_client.semantic_search(
                query=f"data types columns structure {dataset_name}",
                n_results=15,
                filters={"source": dataset_name, "chunk_type": {"$in": ["column", "summary"]}}
            )
            
            if results:
                viz_suggestions = await self._generate_visualization_suggestions(results, analysis_context)
                
                return {
                    "success": True,
                    "dataset": dataset_name,
                    "visualization_suggestions": viz_suggestions,
                    "reasoning": await self._explain_visualization_reasoning(viz_suggestions)
                }
            else:
                return {
                    "success": False,
                    "message": f"No data structure information found for dataset: '{dataset_name}'"
                }
                
        except Exception as e:
            logger.error(f"Error in suggest_visualizations: {e}")
            return {
                "success": False,
                "message": f"Error suggesting visualizations: {str(e)}"
            }
    
    # Helper methods
    async def _generate_initial_insights(self, file_name: str, result) -> List[str]:
        """Generate initial insights from processed dataset."""
        insights = []
        
        try:
            metrics = result.optimization_metrics
            
            if "content_analysis" in metrics:
                analysis = metrics["content_analysis"]
                insights.append(f"Dataset contains {analysis.get('row_count', 0)} rows and {analysis.get('column_count', 0)} columns")
                
                density = analysis.get('data_density', 0)
                if density > 0.8:
                    insights.append("High data density detected - minimal missing values")
                elif density < 0.5:
                    insights.append("Low data density - significant missing values may require attention")
                
                col_types = analysis.get('column_types', {})
                numeric_cols = len([c for c, t in col_types.items() if t == 'numeric'])
                text_cols = len([c for c, t in col_types.items() if t == 'text'])
                
                if numeric_cols > 0:
                    insights.append(f"Found {numeric_cols} numeric columns suitable for statistical analysis")
                if text_cols > 0:
                    insights.append(f"Found {text_cols} text columns suitable for semantic search")
            
            # Processing efficiency insights
            chunks = result.total_chunks
            time_taken = result.processing_time
            
            if time_taken < 1.0:
                insights.append("Fast processing - dataset is well-structured")
            elif chunks > 50:
                insights.append("Large dataset processed with intelligent chunking")
            
        except Exception as e:
            logger.error(f"Error generating initial insights: {e}")
            insights.append("Dataset processed successfully - ready for analysis")
        
        return insights
    
    async def _analyze_query_results(self, query: str, results) -> List[Dict[str, Any]]:
        """Analyze and structure query results."""
        analyzed = []
        
        for result in results:
            analyzed_result = {
                "content": result.document[:300] + "..." if len(result.document) > 300 else result.document,
                "relevance_score": result.combined_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "source": result.metadata.get("source", "Unknown"),
                "chunk_type": result.metadata.get("chunk_type", "data"),
                "column_name": result.metadata.get("column_name"),
                "data_type": result.metadata.get("data_type")
            }
            analyzed.append(analyzed_result)
        
        return analyzed
    
    async def _generate_query_insights(self, query: str, results) -> List[str]:
        """Generate insights from query results."""
        insights = []
        
        try:
            # Analyze result types
            chunk_types = {}
            sources = set()
            
            for result in results:
                chunk_type = result.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                sources.add(result.metadata.get("source", "Unknown"))
            
            insights.append(f"Found {len(results)} relevant results across {len(sources)} dataset(s)")
            
            if "column" in chunk_types:
                insights.append(f"Results include {chunk_types['column']} column-specific matches")
            if "summary" in chunk_types:
                insights.append(f"Results include {chunk_types['summary']} dataset summaries")
            
            # Score analysis
            avg_score = sum(r.combined_score for r in results) / len(results)
            if avg_score > 0.8:
                insights.append("High relevance scores - results closely match your query")
            elif avg_score < 0.5:
                insights.append("Lower relevance scores - consider refining your query")
            
        except Exception as e:
            logger.error(f"Error generating query insights: {e}")
            insights.append("Query processed successfully")
        
        return insights
    
    async def _compile_statistical_analysis(self, results, analysis_types: List[str]) -> Dict[str, Any]:
        """Compile statistical analysis from search results."""
        analysis = {
            "descriptive": {},
            "correlation": {},
            "distribution": {},
            "outliers": {}
        }
        
        try:
            for result in results:
                chunk_type = result.metadata.get("chunk_type")
                
                if chunk_type == "column":
                    col_name = result.metadata.get("column_name")
                    data_type = result.metadata.get("data_type")
                    
                    if col_name and "descriptive" in analysis_types:
                        # Extract statistical information from the document
                        analysis["descriptive"][col_name] = {
                            "type": data_type,
                            "description": result.document[:200],
                            "unique_count": result.metadata.get("unique_count", 0)
                        }
                
                elif chunk_type == "summary" and "descriptive" in analysis_types:
                    analysis["descriptive"]["dataset_summary"] = result.document
        
        except Exception as e:
            logger.error(f"Error compiling statistical analysis: {e}")
        
        return analysis
    
    async def _generate_statistical_recommendations(self, stats_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistical analysis."""
        recommendations = []
        
        try:
            descriptive = stats_analysis.get("descriptive", {})
            
            # Check for numeric columns
            numeric_cols = [col for col, info in descriptive.items() 
                          if isinstance(info, dict) and info.get("type") == "numeric"]
            
            if len(numeric_cols) >= 2:
                recommendations.append("Consider correlation analysis between numeric columns")
            
            if len(numeric_cols) > 0:
                recommendations.append("Visualize distributions with histograms or box plots")
                recommendations.append("Check for outliers in numeric data")
            
            # Check for categorical columns
            categorical_cols = [col for col, info in descriptive.items() 
                              if isinstance(info, dict) and info.get("type") in ["text", "categorical"]]
            
            if len(categorical_cols) > 0:
                recommendations.append("Analyze categorical distributions with bar charts")
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                recommendations.append("Explore relationships between categorical and numeric variables")
        
        except Exception as e:
            logger.error(f"Error generating statistical recommendations: {e}")
            recommendations.append("Perform exploratory data analysis")
        
        return recommendations
    
    async def _generate_visualization_suggestions(self, results, analysis_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate visualization suggestions based on data characteristics."""
        suggestions = []
        
        try:
            # Analyze column types from results
            column_info = {}
            for result in results:
                if result.metadata.get("chunk_type") == "column":
                    col_name = result.metadata.get("column_name")
                    data_type = result.metadata.get("data_type")
                    if col_name and data_type:
                        column_info[col_name] = data_type
            
            numeric_cols = [col for col, dtype in column_info.items() if dtype == "numeric"]
            categorical_cols = [col for col, dtype in column_info.items() if dtype in ["text", "categorical"]]
            
            # Single variable visualizations
            if numeric_cols:
                suggestions.append({
                    "chart_type": "histogram",
                    "columns": numeric_cols[:3],  # Limit to first 3
                    "description": "Distribution analysis of numeric variables",
                    "purpose": "Understand data distribution and identify patterns"
                })
                
                suggestions.append({
                    "chart_type": "box_plot",
                    "columns": numeric_cols[:3],
                    "description": "Box plots for outlier detection",
                    "purpose": "Identify outliers and understand quartile distributions"
                })
            
            if categorical_cols:
                suggestions.append({
                    "chart_type": "bar_chart",
                    "columns": categorical_cols[:3],
                    "description": "Frequency distribution of categorical variables",
                    "purpose": "Understand category distributions and frequencies"
                })
            
            # Multi-variable visualizations
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "chart_type": "scatter_plot",
                    "columns": numeric_cols[:2],
                    "description": f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}",
                    "purpose": "Explore relationships between numeric variables"
                })
                
                suggestions.append({
                    "chart_type": "correlation_heatmap",
                    "columns": numeric_cols,
                    "description": "Correlation matrix heatmap",
                    "purpose": "Visualize correlations between all numeric variables"
                })
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                suggestions.append({
                    "chart_type": "grouped_bar_chart",
                    "columns": [categorical_cols[0], numeric_cols[0]],
                    "description": f"Grouped analysis of {numeric_cols[0]} by {categorical_cols[0]}",
                    "purpose": "Compare numeric values across categorical groups"
                })
        
        except Exception as e:
            logger.error(f"Error generating visualization suggestions: {e}")
            suggestions.append({
                "chart_type": "basic_exploration",
                "columns": [],
                "description": "Basic data exploration recommended",
                "purpose": "Start with simple data overview"
            })
        
        return suggestions
    
    async def _explain_visualization_reasoning(self, viz_suggestions: List[Dict[str, Any]]) -> List[str]:
        """Explain the reasoning behind visualization suggestions."""
        reasoning = []
        
        try:
            chart_types = [viz["chart_type"] for viz in viz_suggestions]
            
            if "histogram" in chart_types:
                reasoning.append("Histograms help understand the distribution shape and identify skewness")
            
            if "scatter_plot" in chart_types:
                reasoning.append("Scatter plots reveal linear/non-linear relationships between variables")
            
            if "correlation_heatmap" in chart_types:
                reasoning.append("Correlation heatmaps provide a comprehensive view of variable relationships")
            
            if "box_plot" in chart_types:
                reasoning.append("Box plots are excellent for outlier detection and quartile analysis")
            
            if "bar_chart" in chart_types:
                reasoning.append("Bar charts effectively show categorical distributions and frequencies")
            
            reasoning.append("These visualizations follow data analysis best practices for your data types")
        
        except Exception as e:
            logger.error(f"Error explaining visualization reasoning: {e}")
            reasoning.append("Visualizations selected based on data characteristics")
        
        return reasoning


class DataAnalysisAgent(Agent):
    """
    Intelligent Data Analysis Agent using Agno Framework
    
    This agent provides sophisticated data analysis capabilities with:
    - Natural language understanding for data queries
    - Reasoning about data analysis tasks
    - Conversation memory and context retention
    - Integration with MCP server and ChromaDB
    - Collaborative analysis workflows
    """
    
    def __init__(self, session_id: str = None, debug_mode: bool = False):
        """Initialize the Data Analysis Agent."""
        
        # Get settings
        settings = get_settings()
        
        # Setup model
        model = OpenAIChat(
            id="gpt-4",
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize ChromaDB components
        self.chroma_client = get_chroma_client()
        self.data_processor = EnhancedDataProcessor()
        self.query_optimizer = AdvancedQueryOptimizer(chroma_client=self.chroma_client)
        
        # Session management
        init_session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup custom memory
        self.memory_storage = DataAnalysisMemory(chroma_client=self.chroma_client)
        
        # Initialize Agno Agent
        super().__init__(
            model=model,
            instructions=self._get_system_instructions(),
            debug_mode=debug_mode,
            add_history_to_messages=True,
            show_tool_calls=True,
            markdown=True
        )
        
        # Re-assign the session_id AFTER the parent class initializer to ensure it is not overwritten
        self.session_id = init_session_id
        
        # Cache for loaded pandas DataFrames by dataset name
        self._dataset_cache = {}
        from .statistics_engine import StatisticsEngine  # Lazy import to avoid circular
        from .visualization_engine import VisualizationEngine  # Lazy import
        from .semantic_row_search import SemanticRowSearchEngine  # Lazy import
        self._stats_engine = StatisticsEngine(self._dataset_cache)
        self._viz_engine = VisualizationEngine(self._dataset_cache)
        self._row_search_engine = SemanticRowSearchEngine(self._dataset_cache, self.chroma_client)
        self._last_dataset = None  # Keep track of most recently analysed dataset
        
        logger.info(f"DataAnalysisAgent initialized with session_id: {self.session_id}")
    
    def _get_system_instructions(self) -> str:
        """Get system instructions for the agent."""
        return """You are an expert Data Analysis Agent with advanced reasoning capabilities.

Your core competencies include:
1. Data Understanding: Analyze datasets and identify patterns
2. Statistical Analysis: Perform descriptive statistics and correlations  
3. Query Processing: Understand natural language queries about data
4. Visualization Recommendations: Suggest appropriate charts
5. Reasoning & Insights: Provide meaningful insights and explanations

Always provide clear reasoning for your analysis choices and suggest next steps."""
    
    async def analyze_dataset(self, file_path: str, file_name: str, description: str = "") -> Dict[str, Any]:
        """Comprehensive dataset analysis workflow."""
        logger.info(f"Starting analysis of dataset: {file_name}")
        
        try:
            # Process the dataset
            result = self.data_processor.process_file_optimized(
                file_path=file_path,
                file_name=file_name,
                metadata={"description": description}
            )
            
            if result.success:
                # Generate initial insights
                insights = await self._generate_insights(file_name, result)
                
                # Record dataset analysis in memory
                await self.memory_storage.write(self.session_id, {
                    "role": "system",
                    "message": f"Dataset '{file_name}' analyzed successfully",
                    "analysis_context": {
                        "type": "dataset_analysis",
                        "dataset_name": file_name,
                        "chunks_created": result.total_chunks,
                        "processing_time": result.processing_time
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
                # Load dataframe into cache for quick stats access
                try:
                    import pandas as pd  # Local import to avoid module cost when not needed
                    df_cache = pd.read_csv(file_path)
                    self._dataset_cache[file_name] = df_cache
                    self._last_dataset = file_name
                except Exception as df_err:
                    logger.warning(f"Could not cache dataframe for stats: {df_err}")
                
                # Set this as the active dataset
                self._last_dataset = file_name
                
                return {
                    "success": True,
                    "message": f"Successfully processed dataset '{file_name}'",
                    "chunks_created": result.total_chunks,
                    "processing_time": result.processing_time,
                    "insights": insights
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to process dataset: {result.error_message}"
                }
                
        except Exception as e:
            logger.error(f"Error in analyze_dataset: {e}")
            return {
                "success": False,
                "message": f"Error processing dataset: {str(e)}"
            }
    
    async def query_data(self, query: str, dataset_name: str = None) -> Dict[str, Any]:
        """Process natural language queries about data."""
        logger.info(f"Processing query: {query}")
        
        # Check for visualization queries first
        viz_intent = self._viz_engine.detect_visualization_query(query)
        if viz_intent and viz_intent.get("is_visualization"):
            return await self._handle_visualization_query(query, viz_intent, dataset_name)
        
        # Check for semantic row search queries
        if self._row_search_engine.detect_row_search_query(query):
            target_dataset = dataset_name or self._last_dataset
            
            if not target_dataset:
                return {"success": False, "message": "No dataset context available for row search."}
            
            logger.info(f"Detected row search query: '{query}' on dataset '{target_dataset}'")
            
            # Perform semantic row search
            search_results = await self._row_search_engine.search_rows(target_dataset, query, max_results=10)
            
            if search_results.get("success", False):
                formatted_results = self._row_search_engine.format_row_results(search_results)
                return formatted_results
            else:
                return {
                    "success": False,
                    "message": search_results.get("error", "Row search failed")
                }
        
        # Enhanced statistical query detection 
        import re
        query_lower = query.lower().strip()
        
        # Enhanced patterns for statistical queries
        patterns = [
            # Direct pattern: "mean age", "sum income", "average spending", etc.
            r"^(average|avg|mean|median|sum|total|min|max|count|std|variance)\s+(\w+)$",
            # Count pattern: "how many", "count of", "number of"
            r"^(how many|count of|number of)\s+(\w+)",
            # Conditional count: "how many people are older than 30"
            r"^(how many|count)\s+(\w+)\s+(are|is|with|have|having)\s+",
            # Statistical comparison: "what is the average salary"
            r"^what\s+is\s+the\s+(average|mean|sum|total|min|max|count)\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Extract statistical operation and column
                if pattern.startswith(r"^(average|avg"):
                    # Direct stat pattern
                    stat_key = match.group(1)
                    col_name = match.group(2)
                elif pattern.startswith(r"^(how many|count"):
                    # Count pattern - extract the entity being counted
                    stat_key = "count"
                    if len(match.groups()) >= 2:
                        col_name = match.group(2)
                    else:
                        # For complex count queries, default to row count
                        col_name = None  # Will count rows
                elif pattern.startswith(r"^what"):
                    # "what is the average salary" pattern
                    stat_key = match.group(1)
                    col_name = match.group(2)
                else:
                    continue  # Skip unhandled patterns
                
                target_dataset = dataset_name or self._last_dataset
                logger.info(f"Detected statistical query: {stat_key} of {col_name or 'rows'} on dataset {target_dataset}")
                
                if not target_dataset:
                    return {"success": False, "message": "No dataset context available. Please specify dataset."}
                
                # Handle special case for counting with conditions
                if stat_key == "count" and ("older than" in query_lower or "greater than" in query_lower or ">" in query):
                    # This is a conditional count - let's handle it specially
                    return await self._handle_conditional_count(query, target_dataset)
                
                # Handle simple count (count of rows or count of non-null values in column)
                if stat_key == "count" and col_name is None:
                    # Count total rows
                    if target_dataset in self._dataset_cache:
                        df = self._dataset_cache[target_dataset]
                        row_count = len(df)
                        return {
                            "success": True,
                            "query": query,
                            "results_count": 1,
                            "insights": [f"Total number of records in '{target_dataset}' is {row_count}"],
                            "results": [{"content": str(row_count), "relevance": 1.0, "source": target_dataset}]
                        }
                
                # Regular statistical computation
                val, err = self._stats_engine.compute(target_dataset, col_name, stat_key)
                if err:
                    return {"success": False, "message": err}
                
                return {
                    "success": True,
                    "query": query,
                    "results_count": 1,
                    "insights": [f"The {stat_key} of '{col_name}' in dataset '{target_dataset}' is {val}"],
                    "results": [{"content": str(val), "relevance": 1.0, "source": target_dataset}]
                }
        
        try:
            # Perform basic semantic search for now (bypassing optimizer to avoid array length issues)
            filters = {}
            if dataset_name:
                filters["source"] = dataset_name
            
            results = self.chroma_client.semantic_search(
                query=query,
                n_results=10,
                filters=filters if filters else None
            )
            
            # Debug logging to see what's returned
            logger.info(f"Query '{query}' returned {len(results) if results else 0} results")
            if results:
                logger.info(f"First result: {results[0].get('document', '')[:100]}...")

            if results and len(results) > 0:
                insights = await self._analyze_query_results(query, results)
                
                return {
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "insights": insights,
                    "results": [
                        {
                            "content": r["document"][:200] + "...",
                            "relevance": r["similarity"],
                            "source": r["metadata"].get("source", "Unknown")
                        }
                        for r in results
                    ]
                }
            else:
                return {
                    "success": False,
                    "message": f"No results found for query: '{query}'"
                }
                
        except Exception as e:
            logger.error(f"Error in query_data: {e}")
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}"
            }
    
    async def _handle_visualization_query(self, query: str, viz_intent: Dict[str, Any], dataset_name: str = None) -> Dict[str, Any]:
        """Handle visualization-specific queries."""
        target_dataset = dataset_name or self._last_dataset
        
        if not target_dataset:
            return {"success": False, "message": "No dataset context available for visualization."}
        
        if target_dataset not in self._dataset_cache:
            return {"success": False, "message": f"Dataset '{target_dataset}' not loaded for visualization."}
        
        chart_type = viz_intent.get("chart_type")  # Don't default yet, let inference handle it
        potential_columns = viz_intent.get("potential_columns", [])
        
        # Enhanced chart type inference if not detected directly
        if chart_type is None:
            query_lower = query.lower()
            if "vs" in query_lower or "versus" in query_lower:
                chart_type = "scatter"
            elif "distribution" in query_lower:
                chart_type = "histogram"
            elif "bar" in query_lower or "count" in query_lower:
                chart_type = "bar"
            else:
                chart_type = "histogram"  # Safe default
        
        # Try to map potential column names to actual dataset columns
        df = self._dataset_cache[target_dataset]
        actual_columns = []
        
        for potential_col in potential_columns:
            # Look for exact matches or partial matches
            exact_matches = [col for col in df.columns if col.lower() == potential_col.lower()]
            if exact_matches:
                actual_columns.extend(exact_matches)
            else:
                partial_matches = [col for col in df.columns if potential_col.lower() in col.lower()]
                actual_columns.extend(partial_matches)
        
        # If no columns found, suggest some based on chart type
        if not actual_columns:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if chart_type in ["histogram", "box"] and numeric_cols:
                actual_columns = [numeric_cols[0]]
            elif chart_type in ["bar", "pie"] and categorical_cols:
                actual_columns = [categorical_cols[0]]
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                actual_columns = numeric_cols[:2]
            else:
                return {"success": False, "message": f"Cannot determine appropriate columns for {chart_type} chart"}
        
        # Limit to appropriate number of columns for chart type
        if chart_type == "scatter" and len(actual_columns) >= 2:
            actual_columns = actual_columns[:2]
        elif chart_type in ["histogram", "bar", "pie", "box"] and actual_columns:
            actual_columns = [actual_columns[0]]
        
        logger.info(f"Creating {chart_type} visualization for columns {actual_columns} on dataset {target_dataset}")
        
        # Create the visualization
        viz_result = self._viz_engine.create_visualization(
            dataset_name=target_dataset,
            chart_type=chart_type,
            columns=actual_columns,
            title=f"{chart_type.title()} - {', '.join(actual_columns)}"
        )
        
        if viz_result.get("success"):
            return {
                "success": True,
                "query": query,
                "results_count": 1,
                "visualization": viz_result,
                "insights": [
                    f"Generated {chart_type} chart for {', '.join(actual_columns)}",
                    f"Chart shows {chart_type} visualization of data from '{target_dataset}'"
                ],
                "results": [{
                    "content": f"{chart_type.title()} chart created successfully",
                    "relevance": 1.0,
                    "source": target_dataset,
                    "type": "visualization"
                }]
            }
        else:
            return {
                "success": False,
                "message": f"Failed to create visualization: {viz_result.get('error', 'Unknown error')}"
            }
    
    async def _generate_insights(self, file_name: str, result) -> List[str]:
        """Generate insights from processed dataset."""
        insights = []
        
        try:
            metrics = result.optimization_metrics
            
            if "content_analysis" in metrics:
                analysis = metrics["content_analysis"]
                insights.append(f"Dataset contains {analysis.get('row_count', 0)} rows and {analysis.get('column_count', 0)} columns")
                
                density = analysis.get('data_density', 0)
                if density > 0.8:
                    insights.append("High data density - minimal missing values")
                elif density < 0.5:
                    insights.append("Low data density - may require data cleaning")
            
            if result.processing_time < 1.0:
                insights.append("Fast processing - well-structured data")
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Dataset processed successfully")
        
        return insights
    
    async def _analyze_query_results(self, query: str, results) -> List[str]:
        """Analyze query results and generate insights."""
        insights = []
        
        try:
            insights.append(f"Found {len(results)} relevant results")
            
            # Analyze result quality
            avg_score = sum(r["similarity"] for r in results) / len(results)
            if avg_score > 0.8:
                insights.append("High relevance scores - strong matches")
            elif avg_score < 0.5:
                insights.append("Lower relevance - consider refining query")
            
            # Analyze sources
            sources = set(r["metadata"].get("source", "Unknown") for r in results)
            insights.append(f"Results span {len(sources)} data source(s)")
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            insights.append("Query processed successfully")
        
        return insights
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation session."""
        try:
            memories = await self.memory_storage.read(self.session_id)
            
            # Analyze conversation
            total_messages = len(memories)
            user_queries = len([m for m in memories if m.get("role") == "user"])
            datasets_analyzed = set()
            query_types = set()
            
            for memory in memories:
                context = memory.get("analysis_context")
                if context:
                    if context.get("type") == "dataset_analysis":
                        datasets_analyzed.add(context.get("dataset_name"))
                    elif context.get("type") == "query":
                        query_types.add(context.get("query_type"))
            
            return {
                "session_id": self.session_id,
                "total_messages": total_messages,
                "user_queries": user_queries,
                "datasets_analyzed": list(datasets_analyzed),
                "query_types": list(query_types),
                "conversation_length": len(memories)
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {
                "session_id": self.session_id,
                "error": f"Could not retrieve summary: {str(e)}"
            }
    
    async def _handle_conditional_count(self, query: str, dataset_name: str) -> Dict[str, Any]:
        """Handle conditional count queries like 'How many people are older than 30'."""
        try:
            if dataset_name not in self._dataset_cache:
                return {"success": False, "message": f"Dataset '{dataset_name}' not loaded in cache"}
            
            df = self._dataset_cache[dataset_name]
            query_lower = query.lower()
            
            # Parse different conditional patterns
            import re
            
            # Pattern: "how many X are/is [condition]"
            patterns = [
                r"how many (\w+) (?:are|is) (?:older than|greater than|above|over) (\d+)",
                r"how many (\w+) (?:are|is) (?:younger than|less than|below|under) (\d+)",
                r"how many (\w+) (?:are|is) (?:equal to|exactly) (\d+)",
                r"count (?:of )?(\w+) (?:where|with|that are) (\w+) > (\d+)",
                r"count (?:of )?(\w+) (?:where|with|that are) (\w+) < (\d+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    entity = match.group(1)  # e.g., "people", "employees"
                    
                    # Try to map entity to actual column or use age as default
                    if "older" in query_lower or "younger" in query_lower or "age" in query_lower:
                        column = "age"
                        threshold = int(match.group(2))
                        
                        if "older" in query_lower or "greater" in query_lower or "above" in query_lower or "over" in query_lower:
                            condition_result = df[df[column] > threshold]
                            condition_desc = f"{column} > {threshold}"
                        elif "younger" in query_lower or "less" in query_lower or "below" in query_lower or "under" in query_lower:
                            condition_result = df[df[column] < threshold]
                            condition_desc = f"{column} < {threshold}"
                        else:
                            condition_result = df[df[column] == threshold]
                            condition_desc = f"{column} = {threshold}"
                        
                        count = len(condition_result)
                        
                        return {
                            "success": True,
                            "query": query,
                            "results_count": 1,
                            "insights": [
                                f"Found {count} records where {condition_desc}",
                                f"Query analyzed {len(df)} total records in '{dataset_name}'"
                            ],
                            "results": [
                                {
                                    "content": f"{count} people are {condition_desc.replace(column, 'age')}",
                                    "relevance": 1.0,
                                    "source": dataset_name,
                                    "type": "statistical_count"
                                }
                            ]
                        }
            
            # If no pattern matched, fall back to general approach
            return {"success": False, "message": f"Could not parse conditional count query: '{query}'"}
            
        except Exception as e:
            logger.error(f"Error in conditional count: {e}")
            return {"success": False, "message": f"Error processing conditional count: {str(e)}"}


def create_data_analysis_agent(session_id: str = None, debug_mode: bool = False) -> DataAnalysisAgent:
    """Factory function to create a DataAnalysisAgent instance."""
    return DataAnalysisAgent(session_id=session_id, debug_mode=debug_mode) 