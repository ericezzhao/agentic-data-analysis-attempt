"""
MCP Protocol Implementation compatible with Python 3.9.

This module implements the Model Context Protocol (MCP) JSON-RPC 2.0 interface
for our data analysis server, following the official MCP specification.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in MCP."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class TextContent:
    """Text content type."""
    type: str = "text"
    text: str = ""


@dataclass
class PromptArgument:
    """Prompt argument definition."""
    name: str
    description: str
    required: bool = False


@dataclass
class Prompt:
    """MCP Prompt definition."""
    name: str
    description: str
    arguments: Optional[List[PromptArgument]] = None


@dataclass
class Tool:
    """MCP Tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class Resource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mimeType: str


@dataclass
class PromptMessage:
    """Prompt message."""
    role: str
    content: TextContent


@dataclass
class GetPromptResult:
    """Result of get_prompt request."""
    description: str
    messages: List[PromptMessage]


class JSONRPCError(Exception):
    """JSON-RPC error."""
    
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")


class MCPServer:
    """
    Model Context Protocol Server implementation.
    
    This server implements the MCP specification using JSON-RPC 2.0 protocol
    for communication with clients.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize the MCP server."""
        self.name = name
        self.version = version
        self.capabilities = {}
        
        # Handler registries
        self._tool_handlers: Dict[str, Callable] = {}
        self._resource_handlers: Dict[str, Callable] = {}
        self._prompt_handlers: Dict[str, Callable] = {}
        
        # Request handlers
        self._request_handlers: Dict[str, Callable] = {}
        
        # Set up built-in handlers
        self._setup_builtin_handlers()
        
        logger.info(f"MCP Server '{name}' v{version} initialized")
    
    def _setup_builtin_handlers(self):
        """Set up built-in MCP request handlers."""
        self._request_handlers.update({
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "prompts/list": self._handle_list_prompts,
            "prompts/get": self._handle_get_prompt,
        })
    
    def set_capabilities(self, capabilities: Dict[str, Any]):
        """Set server capabilities."""
        self.capabilities = capabilities
    
    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler."""
        self._tool_handlers[name] = handler
    
    def register_resource(self, uri: str, handler: Callable):
        """Register a resource handler."""
        self._resource_handlers[uri] = handler
    
    def register_prompt(self, name: str, handler: Callable):
        """Register a prompt handler."""
        self._prompt_handlers[name] = handler
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        # This should be overridden by the actual implementation
        return {"tools": []}
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self._tool_handlers:
            raise JSONRPCError(-32601, f"Tool not found: {tool_name}")
        
        try:
            result = await self._tool_handlers[tool_name](arguments)
            return {"content": result}
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {"resources": []}
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if uri not in self._resource_handlers:
            raise JSONRPCError(-32601, f"Resource not found: {uri}")
        
        try:
            content = await self._resource_handlers[uri](params)
            return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": content}]}
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise JSONRPCError(-32603, f"Internal error reading resource: {str(e)}")
    
    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        return {"prompts": []}
    
    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self._prompt_handlers:
            raise JSONRPCError(-32601, f"Prompt not found: {prompt_name}")
        
        try:
            result = await self._prompt_handlers[prompt_name](arguments)
            return asdict(result)
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {e}")
            raise JSONRPCError(-32603, f"Internal error getting prompt: {str(e)}")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method not in self._request_handlers:
                raise JSONRPCError(-32601, f"Method not found: {method}")
            
            result = await self._request_handlers[method](params)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except JSONRPCError as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "data": e.data
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
    
    async def run_stdio(self):
        """Run the server using stdio transport."""
        logger.info("Starting MCP server with stdio transport...")
        
        try:
            while True:
                # Read JSON-RPC message from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = await self.handle_request(request)
                    
                    # Write response to stdout
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("MCP server stopped")


# Helper functions for compatibility
def tool_handler(func: Callable) -> Callable:
    """Decorator to mark a function as a tool handler."""
    func._is_tool_handler = True
    return func


def resource_handler(func: Callable) -> Callable:
    """Decorator to mark a function as a resource handler."""
    func._is_resource_handler = True
    return func


def prompt_handler(func: Callable) -> Callable:
    """Decorator to mark a function as a prompt handler."""
    func._is_prompt_handler = True
    return func 