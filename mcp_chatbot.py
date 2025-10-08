from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any
from contextlib import AsyncExitStack
import asyncio, json, os

load_dotenv()


def _flatten_tool_content(content_list) -> str:
    """Flatten MCP CallToolResult.content into plain text for OpenAI 'tool' message."""
    parts = []
    if isinstance(content_list, list):
        for item in content_list:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "text":
                txt = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if txt is not None:
                    parts.append(str(txt))
            elif t in ("json", "object"):
                data = (
                    getattr(item, "data", None)
                    or (isinstance(item, dict) and (item.get("data") or item.get("value")))
                )
                try:
                    parts.append(json.dumps(data, ensure_ascii=False))
                except Exception:
                    parts.append(str(data))
            else:
                try:
                    parts.append(json.dumps(item, default=str, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
    elif content_list is not None:
        if isinstance(content_list, (dict, list)):
            parts.append(json.dumps(content_list, ensure_ascii=False))
        else:
            parts.append(str(content_list))
    return "\n".join(parts)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:
    def __init__(self):
        # Point to local proxy by default (LiteLLM on :4000); falls back to OpenAI if you set OPENAI_BASE_URL accordingly
        base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:4000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "sk-local")  # any non-empty string for local proxies
        self.model_default = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        def print_available_models(client):
            try:
                resp = client.models.list()
                print("Models from base_url:", [m.id for m in resp.data])
            except Exception as e:
                print("Failed to list models:", e)

        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[ToolDefinition] = []  # OpenAI function-calling schema

    def _openai_tools_from_mcp(self, mcp_tools: List[types.Tool]) -> List[dict]:
        tools = []
        for t in mcp_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                }
            })
        return tools

    async def connect_to_server(self, server_name: str, server_cfg: dict):
        """Connect to one MCP server (stdio)."""
        try:
            params = StdioServerParameters(**server_cfg)
            read, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # Remember the session
            self.sessions.append(session)

            # Discover tools for this server
            resp = await session.list_tools()
            tools = resp.tools or []
            print(f"Connected to {server_name} with tools:", [t.name for t in tools])

            # Map tool -> session and add to OpenAI tool list
            for t in tools:
                self.tool_to_session[t.name] = session
            self.available_tools.extend(self._openai_tools_from_mcp(tools))

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self, config_path: str = "server_config.json"):
        """Read server_config.json and connect to each server."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        servers = cfg.get("mcpServers", {})
        for name, server_cfg in servers.items():
            await self.connect_to_server(name, server_cfg)

    async def process_query(self, query: str, model: str | None = None):
        model = model or self.model_default
        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        while True:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.available_tools or None,
                tool_choice="auto" if self.available_tools else "none",
                temperature=0.1,
            )
            msg = resp.choices[0].message

            # Tool calls?
            if msg.tool_calls:
                # Keep assistant msg (with tool_calls) in history
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}

                    session = self.tool_to_session.get(tool_name)
                    if not session:
                        # Unknown tool — tell the model
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": f"Tool '{tool_name}' is not available.",
                        })
                        continue

                    print(f"Calling tool {tool_name} with args {args}")
                    try:
                        result = await session.call_tool(tool_name, arguments=args)
                        tool_content = _flatten_tool_content(result.content)
                    except Exception as e:
                        tool_content = f"Tool '{tool_name}' failed: {e}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": tool_content,
                    })

                # Let the model read tool results and continue
                continue

            # Final answer (no tool calls)
            if msg.content:
                print(msg.content.strip())
            break

    async def chat_loop(self):
        print("\nMCP Chatbot (OpenAI-compatible) — type your query, or 'quit' to exit.")
        print(f"Using model: {self.model_default}")
        while True:
            try:
                q = input("\nQuery: ").strip()
                if q.lower() == "quit":
                    break
                await self.process_query(q)  # uses self.model_default
            except Exception as e:
                print(f"Error: {e}")

    async def run(self):
        try:
            await self.connect_to_servers()
            await self.chat_loop()
        finally:
            await self.exit_stack.aclose()


async def main():
    bot = MCP_ChatBot()
    await bot.run()


# Safe runner: works in normal terminals and in environments with an active event loop
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return asyncio.create_task(coro)


if __name__ == "__main__":
    _run_async(main())
