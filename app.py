from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
from typing import Optional
from client import MCPClient

app = FastAPI(title="MCP Chat API")

class MCPClientManager:
    def __init__(self):
        self.client: Optional[MCPClient] = None

    async def get_client(self) -> MCPClient:
        if not self.client:
            self.client = MCPClient()
            # Initialize with your server script path
            await self.client.connect_to_server("/Users/chandrapratap/atlas-mcp-server/mcp-server/mcp-server.py")
        return self.client

    async def cleanup(self):
        if self.client:
            await self.client.cleanup()
            self.client = None

client_manager = MCPClientManager()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

async def get_mcp_client():
    client = await client_manager.get_client()
    return client

@app.on_event("startup")
async def startup_event():
    # Initialize the client on startup
    await client_manager.get_client()

@app.on_event("shutdown")
async def shutdown_event():
    await client_manager.cleanup()

@app.post("/chat", response_model=QueryResponse)
async def chat(
    request: QueryRequest,
    client: MCPClient = Depends(get_mcp_client)
):
    try:
        response = await client.process_query(request.query)
        print(response)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)