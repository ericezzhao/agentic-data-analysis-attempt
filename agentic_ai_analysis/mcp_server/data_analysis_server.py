"""
Data Analysis MCP Server Implementation.

This module implements a complete MCP server for data analysis operations
using our custom protocol layer and integrating with ChromaDB.
"""

import asyncio
import logging
import json
from typing import Any, Dict, List
from dataclasses import asdict

from mcp_server.protocol import (
    MCPServer, TextContent, PromptMessage, GetPromptResult
)
from mcp_server.schema_definitions import get_schema_manager
from database.enhanced_data_processor import get_enhanced_processor
from database.chroma_client import get_chroma_client
from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataAnalysisMCPServer:
    """MCP Server for data analysis operations."""
    
    def __init__(self):
        """Initialize the data analysis MCP server."""
        self.settings = get_settings()
        self.schema_manager = get_schema_manager()
        self.data_processor = get_enhanced_processor()
        self.chroma_client = get_chroma_client()
        
        # Initialize MCP server
        self.server = MCPServer("data-analysis-server", "1.0.0")
        self.server.set_capabilities(self.schema_manager.get_capabilities())
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Data Analysis MCP Server initialized")
    
    def _register_handlers(self):
        """Register all tool, resource, and prompt handlers."""
        # Override built-in handlers
        self.server._request_handlers["tools/list"] = self._handle_list_tools
        self.server._request_handlers["resources/list"] = self._handle_list_resources
        self.server._request_handlers["prompts/list"] = self._handle_list_prompts
        
        # Register tool handlers
        self.server.register_tool("upload_dataset", self._handle_upload_dataset)
        self.server.register_tool("query_data", self._handle_query_data)
        self.server.register_tool("analyze_statistics", self._handle_analyze_statistics)
        self.server.register_tool("create_visualization", self._handle_create_visualization)
        self.server.register_tool("semantic_search", self._handle_semantic_search)
        self.server.register_tool("list_datasets", self._handle_list_datasets)
        
        # Register resource handlers
        self.server.register_resource("analysis://datasets", self._handle_datasets_resource)
        
        # Register prompt handlers
        self.server.register_prompt("explore_dataset", self._handle_explore_dataset_prompt)
        self.server.register_prompt("suggest_analysis", self._handle_suggest_analysis_prompt)
        self.server.register_prompt("visualization_recommendations", self._handle_visualization_recommendations_prompt)
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list requests."""
        tools_schema = self.schema_manager.get_all_tools_schema()
        tools = []
        
        for tool_schema in tools_schema:
            tools.append({
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "inputSchema": tool_schema["inputSchema"]
            })
        
        logger.debug(f"Listed {len(tools)} tools")
        return {"tools": tools}
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list requests."""
        resources_schema = self.schema_manager.get_all_resources_schema()
        resources = []
        
        for resource_schema in resources_schema:
            resources.append({
                "uri": resource_schema["uri"],
                "name": resource_schema["name"],
                "description": resource_schema["description"],
                "mimeType": resource_schema["mimeType"]
            })
        
        logger.debug(f"Listed {len(resources)} resources")
        return {"resources": resources}
    
    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list requests."""
        prompts_schema = self.schema_manager.get_all_prompts_schema()
        prompts = []
        
        for prompt_schema in prompts_schema:
            arguments = []
            for arg in prompt_schema.get("arguments", []):
                arguments.append({
                    "name": arg["name"],
                    "description": arg["description"],
                    "required": arg.get("required", False)
                })
            
            prompts.append({
                "name": prompt_schema["name"],
                "description": prompt_schema["description"],
                "arguments": arguments
            })
        
        logger.debug(f"Listed {len(prompts)} prompts")
        return {"prompts": prompts}
    
    # Tool Handlers
    async def _handle_upload_dataset(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle dataset upload tool call."""
        file_path = arguments["file_path"]
        file_name = arguments["file_name"]
        description = arguments.get("description", "")
        
        try:
            # Process the file using enhanced processor
            result = self.data_processor.process_file_optimized(
                file_path=file_path,
                file_name=file_name,
                metadata={"description": description}
            )
            
            if result.success:
                response = f"Successfully uploaded dataset '{file_name}' with optimized processing:\n"
                response += f"- Created {result.total_chunks} total chunks\n"
                response += f"- Chunk types: {dict(result.chunk_types)}\n"
                response += f"- Processing time: {result.processing_time:.2f}s\n"
                response += f"- Optimization metrics available\n"
                
                # Add basic metadata
                opt_metrics = result.optimization_metrics
                if "content_analysis" in opt_metrics:
                    analysis = opt_metrics["content_analysis"]
                    response += f"- Data shape: {analysis.get('row_count', 0)} rows × {analysis.get('column_count', 0)} columns\n"
                    response += f"- Data density: {analysis.get('data_density', 0):.1%}"
            else:
                response = f"Failed to upload dataset: {result.error_message}"
        except Exception as e:
            response = f"Error processing file: {str(e)}"
        
        return [{"type": "text", "text": response}]
    
    async def _handle_query_data(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle data query tool call."""
        query = arguments["query"]
        dataset_name = arguments.get("dataset_name")
        limit = arguments.get("limit", 10)
        
        try:
            # Perform semantic search
            filters = {}
            if dataset_name:
                filters["source"] = dataset_name
            
            results = self.chroma_client.semantic_search(
                query=query,
                n_results=limit,
                filters=filters if filters else None
            )
            
            if results:
                response = f"Found {len(results)} relevant results for query: '{query}'\n\n"
                for i, result in enumerate(results, 1):
                    response += f"Result {i}:\n"
                    response += f"Document: {result['document'][:200]}...\n"
                    response += f"Similarity: {result['similarity']:.3f}\n"
                    response += f"Source: {result['metadata'].get('source', 'Unknown')}\n\n"
            else:
                response = f"No results found for query: '{query}'"
        except Exception as e:
            response = f"Error performing search: {str(e)}"
        
        return [{"type": "text", "text": response}]
    
    async def _handle_analyze_statistics(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle statistical analysis tool call."""
        dataset_name = arguments["dataset_name"]
        
        try:
            # Search for column and summary information
            results = self.chroma_client.semantic_search(
                query=f"statistical summary {dataset_name}",
                n_results=5,
                filters={"source": dataset_name, "chunk_type": {"$in": ["column", "summary"]}}
            )
            
            if results:
                response = f"Statistical Analysis for dataset '{dataset_name}':\n\n"
                for result in results:
                    if result['metadata'].get('chunk_type') == 'summary':
                        response += "Dataset Summary:\n"
                        response += result['document'] + "\n\n"
                    elif result['metadata'].get('chunk_type') == 'column':
                        response += f"Column Analysis:\n"
                        response += result['document'] + "\n\n"
            else:
                response = f"No statistical information found for dataset: '{dataset_name}'"
        except Exception as e:
            response = f"Error analyzing statistics: {str(e)}"
        
        return [{"type": "text", "text": response}]
    
    async def _handle_create_visualization(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle visualization creation tool call."""
        dataset_name = arguments["dataset_name"]
        chart_type = arguments["chart_type"]
        x_column = arguments.get("x_column")
        y_column = arguments.get("y_column")
        
        # For now, return instructions for creating the visualization
        response = f"Visualization request processed for dataset '{dataset_name}':\n"
        response += f"Chart Type: {chart_type}\n"
        if x_column:
            response += f"X-axis: {x_column}\n"
        if y_column:
            response += f"Y-axis: {y_column}\n"
        response += "\nNote: Visualization generation will be implemented in the UI layer."
        
        return [{"type": "text", "text": response}]
    
    async def _handle_semantic_search(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle semantic search tool call."""
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        
        try:
            # Use optimized search
            results = self.data_processor.optimized_search(
                query=query,
                n_results=max_results,
                use_optimizer=True
            )
            
            if results:
                response = f"Optimized semantic search results for: '{query}'\n\n"
                for result in results:
                    response += f"Rank {result.rank} (combined score: {result.combined_score:.3f}):\n"
                    response += f"Document: {result.document[:300]}...\n"
                    response += f"Semantic: {result.semantic_score:.3f} | Keyword: {result.keyword_score:.3f}\n"
                    response += f"Source: {result.metadata.get('source', 'Unknown')}\n\n"
            else:
                response = f"No results found for semantic search: '{query}'"
        except Exception as e:
            response = f"Error performing semantic search: {str(e)}"
        
        return [{"type": "text", "text": response}]
    
    async def _handle_list_datasets(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle list datasets tool call."""
        include_metadata = arguments.get("include_metadata", True)
        
        try:
            # Get collection info
            collection_info = self.chroma_client.get_collection_info()
            
            response = f"Available datasets:\n"
            response += f"Total documents in collection: {collection_info['count']}\n"
            response += f"Collection name: {collection_info['name']}\n\n"
            
            if include_metadata and collection_info['count'] > 0:
                # Search for summary chunks to get dataset info
                summary_results = self.chroma_client.semantic_search(
                    query="dataset summary",
                    n_results=10,
                    filters={"chunk_type": "summary"}
                )
                
                if summary_results:
                    response += "Dataset summaries:\n"
                    for result in summary_results:
                        source = result['metadata'].get('source', 'Unknown')
                        response += f"\nDataset: {source}\n"
                        response += f"{result['document'][:200]}...\n"
        except Exception as e:
            response = f"Error listing datasets: {str(e)}"
        
        return [{"type": "text", "text": response}]
    
    # Resource Handlers
    async def _handle_datasets_resource(self, params: Dict[str, Any]) -> str:
        """Handle datasets resource."""
        try:
            collection_info = self.chroma_client.get_collection_info()
            
            resource_data = {
                "collection_name": collection_info["name"],
                "total_documents": collection_info["count"],
                "description": "All available datasets for analysis"
            }
            
            return json.dumps(resource_data, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to get datasets info: {str(e)}"})
    
    # Prompt Handlers
    async def _handle_explore_dataset_prompt(self, arguments: Dict[str, str]) -> GetPromptResult:
        """Handle explore dataset prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Let's explore the dataset '{dataset_name}'. Here's a structured approach:

1. **Dataset Overview**: Use 'list_datasets' to see available datasets and 'analyze_statistics' for summaries
2. **Data Structure**: Check column names, types, and data quality
3. **Initial Analysis**: Use 'query_data' and 'semantic_search' for exploration
4. **Visualization**: Use 'create_visualization' for charts and graphs

Would you like to start with step 1?"""
                )
            )
        ]
        
        return GetPromptResult(
            description=f"Guide for exploring dataset '{dataset_name}'",
            messages=messages
        )
    
    async def _handle_suggest_analysis_prompt(self, arguments: Dict[str, str]) -> GetPromptResult:
        """Handle analysis suggestion prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Analysis suggestions for '{dataset_name}':

**Descriptive Analysis:** Basic statistics, distributions, outliers
**Exploratory Analysis:** Correlations, patterns, groups
**Visualization:** Histograms, scatter plots, bar charts

Use 'analyze_statistics' to start."""
                )
            )
        ]
        
        return GetPromptResult(
            description=f"Analysis suggestions for dataset '{dataset_name}'",
            messages=messages
        )
    
    async def _handle_visualization_recommendations_prompt(self, arguments: Dict[str, str]) -> GetPromptResult:
        """Handle visualization recommendations prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Visualization recommendations for '{dataset_name}':

**Numerical Data:** Histograms, box plots, scatter plots
**Categorical Data:** Bar charts, pie charts
**Mixed Data:** Grouped bar charts, heatmaps

Use 'create_visualization' with appropriate chart types."""
                )
            )
        ]
        
        return GetPromptResult(
            description=f"Visualization recommendations for dataset '{dataset_name}'",
            messages=messages
        )
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Data Analysis MCP Server...")
        await self.server.run_stdio()


async def main():
    """Main entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        server = DataAnalysisMCPServer()
        await server.run()
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 