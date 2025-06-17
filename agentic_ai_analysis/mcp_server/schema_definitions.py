"""
MCP Server Schema Definitions for Data Analysis Operations.

This module defines the schema for tools, resources, and prompts
that our MCP server will expose for data analysis functionality.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass


class ToolType(Enum):
    """Types of tools available in the data analysis server."""
    DATA_UPLOAD = "data_upload"
    DATA_QUERY = "data_query"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    DATA_EXPORT = "data_export"
    SEMANTIC_SEARCH = "semantic_search"


class ResourceType(Enum):
    """Types of resources available in the data analysis server."""
    DATASET_INFO = "dataset_info"
    SCHEMA_INFO = "schema_info"
    ANALYSIS_RESULTS = "analysis_results"
    VISUALIZATION_CONFIG = "visualization_config"


class PromptType(Enum):
    """Types of prompts available in the data analysis server."""
    DATA_EXPLORATION = "data_exploration"
    ANALYSIS_SUGGESTION = "analysis_suggestion"
    VISUALIZATION_GUIDE = "visualization_guide"


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class MCPResource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str


@dataclass
class MCPPrompt:
    """MCP Prompt definition."""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


class DataAnalysisSchemaManager:
    """Manages MCP schema definitions for data analysis operations."""
    
    def __init__(self):
        """Initialize schema manager."""
        self.tools = self._define_tools()
        self.resources = self._define_resources()
        self.prompts = self._define_prompts()
    
    def _define_tools(self) -> List[MCPTool]:
        """Define all available tools for data analysis."""
        return [
            # Data Upload and Ingestion
            MCPTool(
                name="upload_dataset",
                description="Upload and ingest a CSV or Excel file into the analysis system",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to upload"
                        },
                        "file_name": {
                            "type": "string", 
                            "description": "Name of the file being uploaded"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the dataset"
                        }
                    },
                    "required": ["file_path", "file_name"]
                },
                annotations={
                    "title": "Upload Dataset",
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False
                }
            ),
            
            # Data Querying
            MCPTool(
                name="query_data",
                description="Query ingested data using natural language",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about the data"
                        },
                        "dataset_name": {
                            "type": "string",
                            "description": "Optional: specific dataset to query"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                },
                annotations={
                    "title": "Query Data",
                    "readOnlyHint": True,
                    "openWorldHint": False
                }
            ),
            
            # Statistical Analysis
            MCPTool(
                name="analyze_statistics",
                description="Perform statistical analysis on dataset columns",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the dataset to analyze"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to analyze (empty for all)"
                        },
                        "analysis_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["descriptive", "correlation", "distribution", "outliers"]
                            },
                            "description": "Types of statistical analysis to perform"
                        }
                    },
                    "required": ["dataset_name"]
                },
                annotations={
                    "title": "Statistical Analysis",
                    "readOnlyHint": True,
                    "openWorldHint": False
                }
            ),
            
            # Data Visualization
            MCPTool(
                name="create_visualization",
                description="Create charts and visualizations from data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the dataset to visualize"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"],
                            "description": "Type of chart to create"
                        },
                        "x_column": {
                            "type": "string",
                            "description": "Column for x-axis"
                        },
                        "y_column": {
                            "type": "string",
                            "description": "Column for y-axis (if applicable)"
                        }
                    },
                    "required": ["dataset_name", "chart_type"]
                },
                annotations={
                    "title": "Create Visualization",
                    "readOnlyHint": True,
                    "openWorldHint": False
                }
            ),
            
            # Semantic Search
            MCPTool(
                name="semantic_search",
                description="Search data using semantic similarity",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                },
                annotations={
                    "title": "Semantic Search",
                    "readOnlyHint": True,
                    "openWorldHint": False
                }
            ),
            
            # List Datasets
            MCPTool(
                name="list_datasets",
                description="List all available datasets in the system",
                input_schema={
                    "type": "object",
                    "properties": {
                        "include_metadata": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include dataset metadata in response"
                        }
                    }
                },
                annotations={
                    "title": "List Datasets",
                    "readOnlyHint": True,
                    "openWorldHint": False
                }
            )
        ]
    
    def _define_resources(self) -> List[MCPResource]:
        """Define all available resources for data analysis."""
        return [
            MCPResource(
                uri="analysis://datasets",
                name="Available Datasets",
                description="List of all datasets available for analysis",
                mime_type="application/json"
            ),
            MCPResource(
                uri="analysis://schemas/{dataset_name}",
                name="Dataset Schema",
                description="Schema information for a specific dataset",
                mime_type="application/json"
            ),
            MCPResource(
                uri="analysis://results/{analysis_id}",
                name="Analysis Results",
                description="Results from a specific analysis operation",
                mime_type="application/json"
            ),
            MCPResource(
                uri="analysis://visualizations/{viz_id}",
                name="Visualization",
                description="Generated chart or visualization",
                mime_type="image/png"
            )
        ]
    
    def _define_prompts(self) -> List[MCPPrompt]:
        """Define all available prompts for data analysis."""
        return [
            MCPPrompt(
                name="explore_dataset",
                description="Guide user through initial data exploration",
                arguments=[
                    {
                        "name": "dataset_name",
                        "description": "Name of the dataset to explore",
                        "required": True
                    }
                ]
            ),
            MCPPrompt(
                name="suggest_analysis",
                description="Suggest appropriate analysis techniques for the data",
                arguments=[
                    {
                        "name": "dataset_name",
                        "description": "Name of the dataset",
                        "required": True
                    }
                ]
            ),
            MCPPrompt(
                name="visualization_recommendations",
                description="Recommend appropriate visualizations for the data",
                arguments=[
                    {
                        "name": "dataset_name",
                        "description": "Name of the dataset",
                        "required": True
                    }
                ]
            )
        ]
    
    def get_all_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schema for all tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "annotations": tool.annotations or {}
            }
            for tool in self.tools
        ]
    
    def get_all_resources_schema(self) -> List[Dict[str, Any]]:
        """Get schema for all resources."""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            }
            for resource in self.resources
        ]
    
    def get_all_prompts_schema(self) -> List[Dict[str, Any]]:
        """Get schema for all prompts."""
        return [
            {
                "name": prompt.name,
                "description": prompt.description,
                "arguments": prompt.arguments or []
            }
            for prompt in self.prompts
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for MCP negotiation."""
        return {
            "tools": {},
            "resources": {
                "subscribe": True,
                "listChanged": True
            },
            "prompts": {},
            "logging": {}
        }


# Global schema manager instance
_schema_manager = None


def get_schema_manager() -> DataAnalysisSchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = DataAnalysisSchemaManager()
    return _schema_manager 