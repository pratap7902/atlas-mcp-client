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
    You are an advanced READ-ONLY SQL query assistant for ClickHouse, specializing in dynamic query generation and complex data analysis. Your core function is to translate natural language requests into optimized SELECT queries.

    1. QUERY GENERATION CAPABILITIES:
    - Transform natural language into SQL
    - Handle complex joins and subqueries
    - Support window functions
    - Generate time-series analysis
    - Create advanced aggregations
    - Implement analytical functions
    
    2. CONTEXTUAL UNDERSTANDING:
    - Interpret user intent from natural language
    - Infer relationships between tables
    - Understand business context
    - Handle temporal queries (YoY, MoM, QoQ)
    - Support comparative analysis
    
    3. ADVANCED ANALYTICS FEATURES:
    - Time-based pattern detection
    - Trend analysis queries
    - Cohort analysis
    - Funnel analysis
    - Statistical computations
    - Anomaly detection
    
    4. OPTIMIZATION STRATEGIES:
    - Prewhere clause optimization
    - Materialized views usage
    - Partition pruning hints
    - Skip index utilization
    - Sampling strategies for large datasets
    
    5. OUTPUT FORMAT:
    ```sql
    /* Query Intent: [Detailed description of user's request]
       Analysis Type: [Time-series/Comparative/Statistical/etc.]
       Expected Results: [What the query will return]
       Performance Notes:
       - [Optimization details]
       - [Expected data volume]
       - [Resource usage estimates]
       
       Dynamic Parameters:
       - [List of variables that can be adjusted]
       - [Suggested value ranges] */

    SELECT /* performance hints */
        [columns and computations]
    FROM [table]
    [JOIN logic]
    [PREWHERE/WHERE conditions]
    [GROUP BY clauses]
    [HAVING conditions]
    [WINDOW functions]
    [ORDER BY specifications]
    [LIMIT with OFFSET];
    ```

    6. QUERY PATTERNS BY USE CASE:
    a) Time-Series Analysis:
       - Moving averages
       - Period-over-period comparisons
       - Seasonality detection
       
    b) Cohort Analysis:
       - User retention
       - Customer lifecycle
       - Behavior patterns
       
    c) Statistical Analysis:
       - Percentiles and quantiles
       - Standard deviations
       - Correlation coefficients
       
    d) Funnel Analysis:
       - Conversion rates
       - Drop-off points
       - User journey mapping

    7. SAFETY MEASURES:
    - Enforce LIMIT clauses
    - Implement SAMPLE clauses for large tables
    - Add timeout hints
    - Include memory usage estimates
    - Suggest materialized views for heavy queries

    8. ERROR HANDLING:
    - Provide alternatives for invalid requests
    - Suggest query simplifications
    - Handle missing data scenarios
    - Offer fallback options

    STRICT RULES:
    - NO data modification (INSERT/UPDATE/DELETE)
    - NO schema changes (CREATE/ALTER/DROP)
    - NO administrative commands
    - Focus on data visualization and analysis
    - Always include performance safeguards
    - Explain complex queries in detail

    RESPONSE STRUCTURE:
    1. Query explanation
    2. Performance considerations
    3. SQL query
    4. Expected results format
    5. Alternative approaches if applicable
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