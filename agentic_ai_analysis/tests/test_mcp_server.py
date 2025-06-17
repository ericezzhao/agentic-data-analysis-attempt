"""
Tests for the MCP Server implementation.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

import mcp.types as types
from mcp_server.server import DataAnalysisMCPServer
from mcp_server.schema_definitions import get_schema_manager


class TestDataAnalysisMCPServer:
    """Test cases for the MCP server."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch('mcp_server.server.get_schema_manager') as mock_schema, \
             patch('mcp_server.server.DataIngestionProcessor') as mock_processor, \
             patch('mcp_server.server.get_chroma_client') as mock_chroma, \
             patch('mcp_server.server.get_settings') as mock_settings:
            
            # Set up mocks
            mock_schema_instance = Mock()
            mock_schema.return_value = mock_schema_instance
            
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance
            
            mock_chroma_instance = Mock()
            mock_chroma.return_value = mock_chroma_instance
            
            mock_settings_instance = Mock()
            mock_settings.return_value = mock_settings_instance
            
            yield {
                'schema': mock_schema_instance,
                'processor': mock_processor_instance,
                'chroma': mock_chroma_instance,
                'settings': mock_settings_instance
            }
    
    @pytest.fixture
    def server(self, mock_dependencies):
        """Create a server instance with mocked dependencies."""
        return DataAnalysisMCPServer()
    
    def test_server_initialization(self, server, mock_dependencies):
        """Test server initializes correctly."""
        assert server.server.name == "data-analysis-server"
        assert server.schema_manager is not None
        assert server.data_processor is not None
        assert server.chroma_client is not None
        assert server.settings is not None
    
    @pytest.mark.asyncio
    async def test_list_tools(self, server, mock_dependencies):
        """Test tools listing functionality."""
        # Mock schema manager response
        mock_tools = [
            {
                "name": "upload_dataset",
                "description": "Upload a dataset",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {}
            },
            {
                "name": "query_data", 
                "description": "Query data",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {}
            }
        ]
        mock_dependencies['schema'].get_all_tools_schema.return_value = mock_tools
        
        # Get the handler
        handler = None
        for name, handler_func in server.server._request_handlers.items():
            if "list_tools" in str(name):
                handler = handler_func
                break
        
        assert handler is not None
        
        # Test the handler
        tools = await handler()
        
        assert len(tools) == 2
        assert tools[0].name == "upload_dataset"
        assert tools[1].name == "query_data"
    
    @pytest.mark.asyncio
    async def test_upload_dataset_tool(self, server, mock_dependencies):
        """Test dataset upload tool."""
        # Mock successful upload
        mock_result = {
            "success": True,
            "chunks_created": 5,
            "documents_ingested": 5,
            "metadata": {
                "row_count": 100,
                "column_count": 5,
                "columns": ["col1", "col2", "col3", "col4", "col5"]
            }
        }
        mock_dependencies['processor'].process_file.return_value = mock_result
        
        # Test arguments
        arguments = {
            "file_path": "/path/to/test.csv",
            "file_name": "test.csv",
            "description": "Test dataset"
        }
        
        result = await server._handle_upload_dataset(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Successfully uploaded dataset 'test.csv'" in result[0].text
        assert "100" in result[0].text  # row count
        assert "5" in result[0].text    # column count
    
    @pytest.mark.asyncio
    async def test_upload_dataset_tool_failure(self, server, mock_dependencies):
        """Test dataset upload tool failure."""
        # Mock failed upload
        mock_result = {
            "success": False,
            "error": "File not found"
        }
        mock_dependencies['processor'].process_file.return_value = mock_result
        
        arguments = {
            "file_path": "/nonexistent/file.csv",
            "file_name": "file.csv"
        }
        
        result = await server._handle_upload_dataset(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Failed to upload dataset: File not found" in result[0].text
    
    @pytest.mark.asyncio
    async def test_query_data_tool(self, server, mock_dependencies):
        """Test data query tool."""
        # Mock search results
        mock_results = [
            {
                "document": "This is a sample document with test data",
                "similarity": 0.85,
                "metadata": {"source": "test.csv"}
            },
            {
                "document": "Another document with relevant information",
                "similarity": 0.78,
                "metadata": {"source": "test.csv"}
            }
        ]
        mock_dependencies['chroma'].semantic_search.return_value = mock_results
        
        arguments = {
            "query": "test data",
            "limit": 5
        }
        
        result = await server._handle_query_data(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Found 2 relevant results" in result[0].text
        assert "test data" in result[0].text
        assert "0.85" in result[0].text  # similarity score
    
    @pytest.mark.asyncio
    async def test_semantic_search_tool(self, server, mock_dependencies):
        """Test semantic search tool."""
        # Mock search results
        mock_results = [
            {
                "document": "Document about sales performance in Q1",
                "similarity": 0.92,
                "metadata": {"chunk_type": "summary"}
            }
        ]
        mock_dependencies['chroma'].semantic_search.return_value = mock_results
        
        arguments = {
            "query": "sales performance",
            "max_results": 3
        }
        
        result = await server._handle_semantic_search(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Semantic search results for: 'sales performance'" in result[0].text
        assert "0.92" in result[0].text
    
    @pytest.mark.asyncio
    async def test_list_datasets_tool(self, server, mock_dependencies):
        """Test list datasets tool."""
        # Mock collection info
        mock_collection_info = {
            "name": "data_analysis",
            "count": 50
        }
        mock_dependencies['chroma'].get_collection_info.return_value = mock_collection_info
        
        # Mock summary search results
        mock_summary_results = [
            {
                "document": "Dataset Summary: 100 rows, 5 columns",
                "metadata": {"source": "sales_data.csv", "chunk_type": "summary"}
            }
        ]
        mock_dependencies['chroma'].semantic_search.return_value = mock_summary_results
        
        arguments = {"include_metadata": True}
        
        result = await server._handle_list_datasets(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Total documents in collection: 50" in result[0].text
        assert "sales_data.csv" in result[0].text
    
    @pytest.mark.asyncio
    async def test_analyze_statistics_tool(self, server, mock_dependencies):
        """Test statistical analysis tool."""
        # Mock search results
        mock_results = [
            {
                "document": "Dataset Summary: Shape: 100 rows, 5 columns",
                "metadata": {"chunk_type": "summary", "source": "test.csv"}
            },
            {
                "document": "Column: sales\nData type: float64\nMean: 1250.50",
                "metadata": {"chunk_type": "column", "source": "test.csv"}
            }
        ]
        mock_dependencies['chroma'].semantic_search.return_value = mock_results
        
        arguments = {
            "dataset_name": "test.csv",
            "analysis_types": ["descriptive"]
        }
        
        result = await server._handle_analyze_statistics(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Statistical Analysis for dataset 'test.csv'" in result[0].text
        assert "Dataset Summary" in result[0].text
        assert "Column Analysis" in result[0].text
    
    @pytest.mark.asyncio
    async def test_create_visualization_tool(self, server, mock_dependencies):
        """Test visualization creation tool."""
        arguments = {
            "dataset_name": "test.csv",
            "chart_type": "bar",
            "x_column": "category",
            "y_column": "sales"
        }
        
        result = await server._handle_create_visualization(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Visualization request processed for dataset 'test.csv'" in result[0].text
        assert "Chart Type: bar" in result[0].text
        assert "X-axis: category" in result[0].text
        assert "Y-axis: sales" in result[0].text
    
    @pytest.mark.asyncio
    async def test_list_resources(self, server, mock_dependencies):
        """Test resources listing."""
        mock_resources = [
            {
                "uri": "analysis://datasets",
                "name": "Available Datasets",
                "description": "List of datasets",
                "mimeType": "application/json"
            }
        ]
        mock_dependencies['schema'].get_all_resources_schema.return_value = mock_resources
        
        # Get the handler
        handler = None
        for name, handler_func in server.server._request_handlers.items():
            if "list_resources" in str(name):
                handler = handler_func
                break
        
        assert handler is not None
        
        resources = await handler()
        
        assert len(resources) == 1
        assert resources[0].uri == "analysis://datasets"
        assert resources[0].name == "Available Datasets"
    
    @pytest.mark.asyncio
    async def test_read_datasets_resource(self, server, mock_dependencies):
        """Test reading datasets resource."""
        mock_collection_info = {
            "name": "data_analysis",
            "count": 25
        }
        mock_dependencies['chroma'].get_collection_info.return_value = mock_collection_info
        
        result = await server._get_datasets_resource()
        
        data = json.loads(result)
        assert data["collection_name"] == "data_analysis"
        assert data["total_documents"] == 25
        assert "description" in data
    
    @pytest.mark.asyncio
    async def test_read_schema_resource(self, server, mock_dependencies):
        """Test reading schema resource."""
        mock_column_results = [
            {
                "metadata": {
                    "column_name": "sales",
                    "data_type": "float64",
                    "unique_count": 50
                }
            },
            {
                "metadata": {
                    "column_name": "category",
                    "data_type": "object",
                    "unique_count": 5
                }
            }
        ]
        mock_dependencies['chroma'].semantic_search.return_value = mock_column_results
        
        result = await server._get_schema_resource("test.csv")
        
        data = json.loads(result)
        assert data["dataset_name"] == "test.csv"
        assert len(data["columns"]) == 2
        assert data["columns"][0]["name"] == "sales"
        assert data["columns"][0]["type"] == "float64"
    
    @pytest.mark.asyncio
    async def test_list_prompts(self, server, mock_dependencies):
        """Test prompts listing."""
        mock_prompts = [
            {
                "name": "explore_dataset",
                "description": "Guide through data exploration",
                "arguments": [
                    {"name": "dataset_name", "description": "Dataset name", "required": True}
                ]
            }
        ]
        mock_dependencies['schema'].get_all_prompts_schema.return_value = mock_prompts
        
        # Get the handler
        handler = None
        for name, handler_func in server.server._request_handlers.items():
            if "list_prompts" in str(name):
                handler = handler_func
                break
        
        assert handler is not None
        
        prompts = await handler()
        
        assert len(prompts) == 1
        assert prompts[0].name == "explore_dataset"
        assert len(prompts[0].arguments) == 1
        assert prompts[0].arguments[0].name == "dataset_name"
        assert prompts[0].arguments[0].required is True
    
    @pytest.mark.asyncio
    async def test_get_explore_dataset_prompt(self, server, mock_dependencies):
        """Test getting explore dataset prompt."""
        arguments = {"dataset_name": "sales_data.csv"}
        
        result = await server._get_explore_dataset_prompt(arguments)
        
        assert isinstance(result, types.GetPromptResult)
        assert "sales_data.csv" in result.description
        assert len(result.messages) == 1
        assert "sales_data.csv" in result.messages[0].content.text
        assert "Dataset Overview" in result.messages[0].content.text


class TestSchemaManager:
    """Test cases for the schema manager."""
    
    def test_schema_manager_initialization(self):
        """Test schema manager initializes correctly."""
        manager = get_schema_manager()
        
        assert len(manager.tools) > 0
        assert len(manager.resources) > 0
        assert len(manager.prompts) > 0
    
    def test_get_all_tools_schema(self):
        """Test getting all tools schema."""
        manager = get_schema_manager()
        tools_schema = manager.get_all_tools_schema()
        
        assert len(tools_schema) > 0
        
        # Check required fields
        for tool in tools_schema:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "annotations" in tool
    
    def test_upload_dataset_tool_schema(self):
        """Test upload dataset tool schema."""
        manager = get_schema_manager()
        tools_schema = manager.get_all_tools_schema()
        
        upload_tool = next((t for t in tools_schema if t["name"] == "upload_dataset"), None)
        assert upload_tool is not None
        
        schema = upload_tool["inputSchema"]
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert "file_name" in schema["properties"]
        assert "file_path" in schema["required"]
        assert "file_name" in schema["required"]
    
    def test_get_capabilities(self):
        """Test getting server capabilities."""
        manager = get_schema_manager()
        capabilities = manager.get_capabilities()
        
        assert "tools" in capabilities
        assert "resources" in capabilities
        assert "prompts" in capabilities
        assert "logging" in capabilities


class TestMCPServerSetup:
    """Test cases for basic MCP server setup."""
    
    def test_schema_manager_basic_functionality(self):
        """Test basic schema manager functionality."""
        from mcp_server.schema_definitions import get_schema_manager
        
        manager = get_schema_manager()
        
        # Test tools
        tools = manager.get_all_tools_schema()
        assert len(tools) > 0
        
        # Check upload_dataset tool exists
        upload_tool = next((t for t in tools if t["name"] == "upload_dataset"), None)
        assert upload_tool is not None
        assert "file_path" in upload_tool["inputSchema"]["properties"]
        assert "file_name" in upload_tool["inputSchema"]["properties"]
        
        # Test resources
        resources = manager.get_all_resources_schema()
        assert len(resources) > 0
        
        # Test prompts
        prompts = manager.get_all_prompts_schema()
        assert len(prompts) > 0
        
        # Test capabilities
        capabilities = manager.get_capabilities()
        assert "tools" in capabilities
        assert "resources" in capabilities
        assert "prompts" in capabilities
    
    def test_tool_schemas_structure(self):
        """Test that all tool schemas have required structure."""
        from mcp_server.schema_definitions import get_schema_manager
        
        manager = get_schema_manager()
        tools = manager.get_all_tools_schema()
        
        for tool in tools:
            # Check required fields
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "annotations" in tool
            
            # Check schema structure
            schema = tool["inputSchema"]
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema
    
    def test_resource_schemas_structure(self):
        """Test that all resource schemas have required structure."""
        from mcp_server.schema_definitions import get_schema_manager
        
        manager = get_schema_manager()
        resources = manager.get_all_resources_schema()
        
        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "description" in resource
            assert "mimeType" in resource
    
    def test_prompt_schemas_structure(self):
        """Test that all prompt schemas have required structure."""
        from mcp_server.schema_definitions import get_schema_manager
        
        manager = get_schema_manager()
        prompts = manager.get_all_prompts_schema()
        
        for prompt in prompts:
            assert "name" in prompt
            assert "description" in prompt
            assert "arguments" in prompt


class TestMCPServerIntegration:
    """Integration tests for MCP server components."""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        mock_client = Mock()
        mock_client.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 10
        }
        mock_client.semantic_search.return_value = [
            {
                "document": "Test document content",
                "similarity": 0.85,
                "metadata": {"source": "test.csv", "chunk_type": "summary"}
            }
        ]
        return mock_client
    
    @pytest.fixture
    def mock_data_processor(self):
        """Mock data ingestion processor."""
        mock_processor = Mock()
        mock_processor.process_file.return_value = {
            "success": True,
            "chunks_created": 5,
            "documents_ingested": 5,
            "metadata": {
                "row_count": 100,
                "column_count": 3,
                "columns": ["col1", "col2", "col3"]
            }
        }
        return mock_processor
    
    def test_server_initialization_with_mocks(self, mock_chroma_client, mock_data_processor):
        """Test server can be initialized with mocked dependencies."""
        with patch('mcp_server.server.get_chroma_client', return_value=mock_chroma_client), \
             patch('mcp_server.server.DataIngestionProcessor', return_value=mock_data_processor), \
             patch('mcp_server.server.get_settings', return_value=Mock()):
            
            from mcp_server.server import DataAnalysisMCPServer
            
            server = DataAnalysisMCPServer()
            assert server.server.name == "data-analysis-server"
            assert server.chroma_client == mock_chroma_client
            assert server.data_processor == mock_data_processor


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 