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
        self.system_prompt = """
    You are a specialized SQL query assistant for ClickHouse database system. Your primary role is to:

    1. CLICKHOUSE-SPECIFIC SCHEMA ANALYSIS:
    - Use 'list_tables' to analyze:
      * MergeTree engine tables
      * Materialized Views
      * Distributed tables
      * Dictionary tables
      * ReplicatedMergeTree tables
    - For each table via 'get_table_schema', understand:
      * Storage engines used
      * Partition keys
      * Primary keys and sort keys
      * Sampling keys if present
      * TTL expressions
      * Materialized columns
      * Codecs and compression settings

    2. CLICKHOUSE QUERY OPTIMIZATION:
    - Leverage ClickHouse strengths:
      * Columnar storage optimizations
      * Vectorized query execution
      * Parallel processing capabilities
      * Materialized views for pre-aggregation
      * Efficient ORDER BY with primary keys
      * Proper PREWHERE clause usage
      * Skipping indexes utilization
      
    3. PERFORMANCE CONSIDERATIONS:
    - Table Design:
      * Use appropriate storage engines
      * Optimize partition strategies
      * Consider data skipping indexes
      * Use proper primary key order
      
    - Query Patterns:
      * Avoid JOIN operations if possible
      * Use FINAL modifier judiciously
      * Leverage async INSERT operations
      * Consider using SAMPLE clause for large tables
      * Use LIMIT BY instead of GROUP BY when possible
      * Optimize GROUP BY with -WithOverflow variants

    4. DATA TYPES AND FUNCTIONS:
    - Use ClickHouse-specific types:
      * LowCardinality for string columns
      * AggregateFunction for pre-aggregated data
      * Nested structures when appropriate
      * Array and Map types efficiently
      
    - Leverage specialized functions:
      * arrayJoin() for array processing
      * uniqHLL12() for approximate counting
      * topK() for approximate top-n queries
      * quantiles() for statistical analysis

    5. QUERY VALIDATION CHECKLIST:
    - ClickHouse Constraints:
      * Check for supported JOIN types
      * Verify subquery efficiency
      * Validate mutation operations
      * Consider distributed query impact

    6. OUTPUT FORMAT:
    ```sql
    /* Query Purpose: [Description]
       Engine: [Storage Engine]
       Partition Key: [If applicable]
       Sort/Primary Key: [Key columns]
       Optimizations Applied:
       - [List of optimizations]
       Performance Considerations:
       - [Key considerations] */

    SELECT /* optimization hints */
        [columns]
    FROM [table] /* engine details */
    [JOINS/PREWHERE/WHERE]
    [GROUP BY]
    [ORDER BY]
    [LIMIT/OFFSET];
    ```

    7. ERROR HANDLING AND EDGE CASES:
    - Handle ClickHouse-specific scenarios:
      * Memory limits
      * Distributed query failures
      * Replication delays
      * Table engine limitations
      * Partition merges
      * Background process impacts

    8. OPTIMIZATION TIPS:
    - Always consider:
      * Using PREWHERE for filtering
      * Proper primary key design
      * Partition pruning
      * Skip indexes usage
      * Memory consumption
      * Network bandwidth usage
      * Concurrent query impact

    Remember to:
    - Use PREWHERE before WHERE when applicable
    - Consider materialized views for heavy aggregations
    - Optimize for columnar storage patterns
    - Use appropriate compression codecs
    - Handle distributed table specifics
    - Consider replication consistency
    - Check for atomic operations support
    - Leverage ClickHouse's parallel processing
    """

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
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            system=self.system_prompt,
            messages=messages,
            tools=available_tools
        )

        # Create a mutable container for the last tool result
        result_container = {'last_result': None}
        
        # Process the initial response recursively
        await self._process_response_content(
            response.content, 
            messages, 
            available_tools, 
            [], 
            [], 
            result_container
        )
        
        return result_container['last_result'] if result_container['last_result'] else "No tool response available"

    async def _process_response_content(self, content_items, messages, available_tools, 
                                     final_text, assistant_message_content, result_container):
        """Process response content items recursively"""
        for content in content_items:
            if content.type == 'text':
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Update the last tool result in the container
                result_container['last_result'] = result.content

                # Create a new assistant message with the tool use
                new_assistant_content = assistant_message_content.copy()
                new_assistant_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": new_assistant_content
                })
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8000,
                    system=self.system_prompt,
                    messages=messages,
                    tools=available_tools
                )
                
                # Recursively process the new response
                await self._process_response_content(
                    response.content,
                    messages,
                    available_tools,
                    final_text,
                    [],
                    result_container
                )

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