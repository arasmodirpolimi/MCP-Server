from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, Any
import asyncio, json, os, nest_asyncio
load_dotenv()
nest_asyncio.apply()

def _flatten_tool_content(content_list):
    parts = []
    for item in content_list:
        # TextContent (MCP) typically has .text
        text = getattr(item, 'text', None)
        if text is not None:
            parts.append(text)
        elif isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "\n".join(parts)

class MCP_ChatBot:
    def __init__(self):
        self.session: ClientSession | None = None
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.available_tools: List[dict] = []

    def _openai_tools_from_mcp(self, mcp_tools: List[types.Tool]) -> List[dict]:
        out = []
        for t in mcp_tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}}
                }
            })
        return out

    async def process_query(self, query: str, model: str = "gpt-3.5-turbo"):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]
        while True:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.available_tools or None,
                tool_choice="auto" if self.available_tools else "none",
                temperature=0.01
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
                })
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}
                    result = await self.session.call_tool(tool_name, arguments=args)
                    flattened = _flatten_tool_content(result.content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": flattened
                    })
                continue
            if msg.content:
                print(msg.content.strip())
            break

    async def chat_loop(self):
        print("MCP Chatbot Started. Type your queries or 'quit'.")
        while True:
            try:
                q = input("\nQuery: ").strip()
                if q.lower() == 'quit':
                    break
                await self.process_query(q)
            except Exception as e:
                print(f"Error: {e}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(command="uv", args=["run", "weather_server.py"], env=None)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                resp = await session.list_tools()
                self.available_tools = self._openai_tools_from_mcp(resp.tools)
                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == '__main__':
    asyncio.run(main())
