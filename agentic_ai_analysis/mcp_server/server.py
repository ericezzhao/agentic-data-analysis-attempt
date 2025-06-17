"""
MCP Server implementation for data analysis operations.

This module implements the Model Context Protocol server that provides
data analysis capabilities through tools, resources, and prompts.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from .schema_definitions import get_schema_manager
from ..database.data_ingestion import DataIngestionProcessor
from ..database.chroma_client import get_chroma_client
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class DataAnalysisMCPServer:
    """MCP Server for data analysis operations."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.settings = get_settings()
        self.schema_manager = get_schema_manager()
        self.data_processor = DataIngestionProcessor()
        self.chroma_client = get_chroma_client()
        
        # Initialize MCP server
        self.server = Server("data-analysis-server")
        self._setup_handlers()
        
        logger.info("Data Analysis MCP Server initialized")
    
    def _setup_handlers(self):
        """Set up all MCP request handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """Handle tools/list requests."""
            tools_schema = self.schema_manager.get_all_tools_schema()
            tools = []
            
            for tool_schema in tools_schema:
                tools.append(
                    types.Tool(
                        name=tool_schema["name"],
                        description=tool_schema["description"],
                        inputSchema=tool_schema["inputSchema"]
                    )
                )
            
            logger.debug(f"Listed {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: dict[str, Any]
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tools/call requests."""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                if name == "upload_dataset":
                    return await self._handle_upload_dataset(arguments)
                elif name == "query_data":
                    return await self._handle_query_data(arguments)
                elif name == "analyze_statistics":
                    return await self._handle_analyze_statistics(arguments)
                elif name == "create_visualization":
                    return await self._handle_create_visualization(arguments)
                elif name == "semantic_search":
                    return await self._handle_semantic_search(arguments)
                elif name == "list_datasets":
                    return await self._handle_list_datasets(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """Handle resources/list requests."""
            resources_schema = self.schema_manager.get_all_resources_schema()
            resources = []
            
            for resource_schema in resources_schema:
                resources.append(
                    types.Resource(
                        uri=resource_schema["uri"],
                        name=resource_schema["name"],
                        description=resource_schema["description"],
                        mimeType=resource_schema["mimeType"]
                    )
                )
            
            logger.debug(f"Listed {len(resources)} resources")
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle resources/read requests."""
            try:
                logger.info(f"Reading resource: {uri}")
                
                if uri == "analysis://datasets":
                    return await self._get_datasets_resource()
                elif uri.startswith("analysis://schemas/"):
                    dataset_name = uri.split("/")[-1]
                    return await self._get_schema_resource(dataset_name)
                elif uri.startswith("analysis://results/"):
                    analysis_id = uri.split("/")[-1]
                    return await self._get_results_resource(analysis_id)
                else:
                    raise ValueError(f"Unknown resource URI: {uri}")
                    
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                raise
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[types.Prompt]:
            """Handle prompts/list requests."""
            prompts_schema = self.schema_manager.get_all_prompts_schema()
            prompts = []
            
            for prompt_schema in prompts_schema:
                arguments = []
                for arg in prompt_schema.get("arguments", []):
                    arguments.append(
                        types.PromptArgument(
                            name=arg["name"],
                            description=arg["description"],
                            required=arg.get("required", False)
                        )
                    )
                
                prompts.append(
                    types.Prompt(
                        name=prompt_schema["name"],
                        description=prompt_schema["description"],
                        arguments=arguments
                    )
                )
            
            logger.debug(f"Listed {len(prompts)} prompts")
            return prompts
        
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: dict[str, str] | None = None
        ) -> types.GetPromptResult:
            """Handle prompts/get requests."""
            try:
                logger.info(f"Getting prompt: {name} with arguments: {arguments}")
                
                if name == "explore_dataset":
                    return await self._get_explore_dataset_prompt(arguments or {})
                elif name == "suggest_analysis":
                    return await self._get_suggest_analysis_prompt(arguments or {})
                elif name == "visualization_recommendations":
                    return await self._get_visualization_recommendations_prompt(arguments or {})
                else:
                    raise ValueError(f"Unknown prompt: {name}")
                    
            except Exception as e:
                logger.error(f"Error getting prompt {name}: {e}")
                raise
    
    async def _handle_upload_dataset(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle dataset upload tool call."""
        file_path = arguments["file_path"]
        file_name = arguments["file_name"]
        description = arguments.get("description", "")
        
        # Process the file
        result = self.data_processor.process_file(
            file_path=file_path,
            file_name=file_name,
            metadata={"description": description}
        )
        
        if result["success"]:
            response = f"Successfully uploaded dataset '{file_name}':\n"
            response += f"- Created {result['chunks_created']} chunks\n"
            response += f"- Ingested {result['documents_ingested']} documents\n"
            response += f"- Rows: {result['metadata']['row_count']}\n"
            response += f"- Columns: {result['metadata']['column_count']}\n"
            response += f"- Column names: {', '.join(result['metadata']['columns'])}"
        else:
            response = f"Failed to upload dataset: {result['error']}"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_query_data(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle data query tool call."""
        query = arguments["query"]
        dataset_name = arguments.get("dataset_name")
        limit = arguments.get("limit", 10)
        
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
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_analyze_statistics(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle statistical analysis tool call."""
        dataset_name = arguments["dataset_name"]
        columns = arguments.get("columns", [])
        analysis_types = arguments.get("analysis_types", ["descriptive"])
        
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
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_create_visualization(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
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
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_semantic_search(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle semantic search tool call."""
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        
        results = self.chroma_client.semantic_search(
            query=query,
            n_results=max_results
        )
        
        if results:
            response = f"Semantic search results for: '{query}'\n\n"
            for i, result in enumerate(results, 1):
                response += f"Result {i} (similarity: {result['similarity']:.3f}):\n"
                response += f"{result['document'][:300]}...\n\n"
        else:
            response = f"No results found for semantic search: '{query}'"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_list_datasets(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle list datasets tool call."""
        include_metadata = arguments.get("include_metadata", True)
        
        # Get collection info
        collection_info = self.chroma_client.get_collection_info()
        
        response = f"Available datasets:\n"
        response += f"Total documents in collection: {collection_info['count']}\n"
        response += f"Collection name: {collection_info['name']}\n\n"
        
        if include_metadata:
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
        
        return [types.TextContent(type="text", text=response)]
    
    async def _get_datasets_resource(self) -> str:
        """Get the datasets resource."""
        collection_info = self.chroma_client.get_collection_info()
        
        resource_data = {
            "collection_name": collection_info["name"],
            "total_documents": collection_info["count"],
            "description": "All available datasets for analysis"
        }
        
        import json
        return json.dumps(resource_data, indent=2)
    
    async def _get_schema_resource(self, dataset_name: str) -> str:
        """Get schema information for a specific dataset."""
        # Search for column information
        results = self.chroma_client.semantic_search(
            query=f"columns schema {dataset_name}",
            n_results=10,
            filters={"source": dataset_name, "chunk_type": "column"}
        )
        
        schema_info = {
            "dataset_name": dataset_name,
            "columns": []
        }
        
        for result in results:
            column_name = result['metadata'].get('column_name')
            data_type = result['metadata'].get('data_type')
            if column_name and data_type:
                schema_info["columns"].append({
                    "name": column_name,
                    "type": data_type,
                    "unique_count": result['metadata'].get('unique_count', 0)
                })
        
        import json
        return json.dumps(schema_info, indent=2)
    
    async def _get_results_resource(self, analysis_id: str) -> str:
        """Get results for a specific analysis."""
        # Placeholder for analysis results
        results_data = {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": "Analysis results would be stored here"
        }
        
        import json
        return json.dumps(results_data, indent=2)
    
    async def _get_explore_dataset_prompt(self, arguments: Dict[str, str]) -> types.GetPromptResult:
        """Get the explore dataset prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"""Let's explore the dataset '{dataset_name}'. Here's a structured approach:

1. **Dataset Overview**: First, let's get basic information about the dataset:
   - Use the 'list_datasets' tool to see available datasets
   - Use the 'analyze_statistics' tool to get statistical summaries

2. **Data Structure**: Understand the data structure:
   - Check column names and types
   - Look for missing values or data quality issues
   - Identify categorical vs numerical columns

3. **Initial Analysis**: Perform initial exploratory analysis:
   - Use 'query_data' to ask specific questions about the data
   - Use 'semantic_search' to find interesting patterns

4. **Visualization**: Consider appropriate visualizations:
   - Use 'create_visualization' to generate charts
   - Start with simple plots like histograms for numerical data
   - Use bar charts for categorical data

Would you like to start with step 1 and get the dataset overview?"""
                )
            )
        ]
        
        return types.GetPromptResult(
            description=f"Guide for exploring dataset '{dataset_name}'",
            messages=messages
        )
    
    async def _get_suggest_analysis_prompt(self, arguments: Dict[str, str]) -> types.GetPromptResult:
        """Get the analysis suggestion prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"""Based on the dataset '{dataset_name}', here are some analysis suggestions:

**Descriptive Analysis:**
- Basic statistics (mean, median, mode, standard deviation)
- Data distribution analysis
- Missing value analysis
- Outlier detection

**Exploratory Analysis:**
- Correlation analysis between variables
- Group-by analysis for categorical variables
- Time series analysis (if temporal data exists)
- Pattern identification using semantic search

**Comparative Analysis:**
- Compare different groups or segments
- Before/after analysis
- Performance metrics analysis

**Visualization Recommendations:**
- Histograms for distribution analysis
- Scatter plots for correlation
- Box plots for outlier identification
- Bar charts for categorical comparisons

To get started, use the 'analyze_statistics' tool with the dataset name and specify which type of analysis you'd like to perform."""
                )
            )
        ]
        
        return types.GetPromptResult(
            description=f"Analysis suggestions for dataset '{dataset_name}'",
            messages=messages
        )
    
    async def _get_visualization_recommendations_prompt(self, arguments: Dict[str, str]) -> types.GetPromptResult:
        """Get the visualization recommendations prompt."""
        dataset_name = arguments.get("dataset_name", "your_dataset")
        
        messages = [
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"""Visualization recommendations for dataset '{dataset_name}':

**For Numerical Data:**
- **Histogram**: Show distribution of values
- **Box Plot**: Identify outliers and quartiles
- **Scatter Plot**: Show relationships between two variables
- **Line Chart**: Show trends over time (if temporal data)

**For Categorical Data:**
- **Bar Chart**: Compare categories
- **Pie Chart**: Show proportions (for few categories)

**For Mixed Data:**
- **Grouped Bar Chart**: Compare numerical values across categories
- **Heatmap**: Show correlation matrix or cross-tabulation

**Advanced Visualizations:**
- **Correlation Matrix**: Understand variable relationships
- **Distribution Plots**: Compare distributions across groups

Use the 'create_visualization' tool with the appropriate chart type and column specifications. Start by using 'analyze_statistics' to understand your data structure first."""
                )
            )
        ]
        
        return types.GetPromptResult(
            description=f"Visualization recommendations for dataset '{dataset_name}'",
            messages=messages
        )
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Data Analysis MCP Server...")
        
        # Run the server with stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="data-analysis-server",
                    server_version="1.0.0",
                    capabilities=self.schema_manager.get_capabilities()
                )
            )


async def main():
    """Main entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    
    server = DataAnalysisMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main()) 