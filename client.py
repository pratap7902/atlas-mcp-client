import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\n✅ Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        # Validate query
        if not query or not query.strip():
            return "Query cannot be empty"

        # Simplified system message as a string
        system_message = (
            "You are a helpful assistant that can use tools to query a ClickHouse database "
            "using natural language. When needed, use available tools. If you don't need a tool, respond normally."
        )

        messages = [
            {
                "role": "user",
                "content": query.strip()  # Ensure query is stripped of whitespace
            }
        ]

        try:
            # List available tools from the session
            response = await self.session.list_tools()
            available_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]

            # Initial Claude API call
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                system=system_message,
                tools=available_tools
            )

            final_text = []

            # Ensure that response.content exists and has the expected structure
            if not response.content:
                raise ValueError("Received empty or invalid content from Anthropic API")

            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call with error handling
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        final_text.append(f"[🔧 Called tool `{tool_name}` with args {tool_args}]")

                        # Continue conversation with tool results
                        if hasattr(content, 'text') and content.text:
                            messages.append({
                                "role": "assistant",
                                "content": content.text
                            })
                        messages.append({
                            "role": "user",
                            "content": result.content
                        })

                        # Continue with Claude
                        response = self.anthropic.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1000,
                            messages=messages,
                            system=system_message,
                            tools=available_tools
                        )
                        final_text.append(response.content[0].text)
                    except Exception as e:
                        final_text.append(f"[❌ Error calling tool `{tool_name}`: {str(e)}]")

            return "\n".join(final_text)

        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            return f"An error occurred: {str(e)}"

    # async def chat_loop(self):
    #     """Run an interactive chat loop"""
    #     print("\n🧠 MCP Client Started!")
    #     print("Type your queries or 'quit' to exit.")
        
    #     while True:
    #         try:
    #             query = input("\n💬 Query: ").strip()
                
    #             if query.lower() == 'quit':
    #                 break
                    
    #             response = await self.process_query(query)
    #             print("\n🪄 Response:\n" + response)
                    
    #         except Exception as e:
    #             print(f"\n❌ Error: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


# async def main():
#     if len(sys.argv) < 2:
#         print("Usage: python client.py <path_to_server_script>")
#         sys.exit(1)

#     client = MCPClient()
#     try:
#         await client.connect_to_server(sys.argv[1])
#         await client.chat_loop()
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     asyncio.run(main())

