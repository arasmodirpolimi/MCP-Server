from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any, Optional
from contextlib import AsyncExitStack
import asyncio, json, os, requests

load_dotenv()

# ---------------------------
# Ollama adapter (no OpenAI)
# ---------------------------
class OllamaAdapter:
    def __init__(self, base: str = "http://127.0.0.1:11434"):
        self.base = base.rstrip("/")

    def chat_once(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        tools: Optional[List[dict]] = None,
        tool_choice: str | dict | None = "auto",
    ):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {}) or {}
        # Normalize into an OpenAI-like shape the rest of the code expects
        return {
            "choices": [{
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            }]
        }

def _flatten_tool_content(content_list) -> str:
    parts = []
    if isinstance(content_list, list):
        for item in content_list:
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "text":
                txt = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if txt is not None:
                    parts.append(str(txt))
            elif t in ("json", "object"):
                data = getattr(item, "data", None) or (isinstance(item, dict) and (item.get("data") or item.get("value")))
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
    return "\n".join(parts).strip()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:
    def __init__(self):
        self.model_default = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
        self.client = OllamaAdapter(os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"))
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[ToolDefinition] = []

    def _openai_tools_from_mcp(self, mcp_tools: List[types.Tool]) -> List[dict]:
        out = []
        for t in mcp_tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                }
            })
        return out

    async def _call_mcp_tool(self, tool_name: str, args: dict) -> Optional[str]:
        session = self.tool_to_session.get(tool_name)
        if not session:
            return f"Tool '{tool_name}' is not registered."
        try:
            result = await session.call_tool(tool_name, arguments=args or {})
            return _flatten_tool_content(result.content) or "(empty result)"
        except Exception as e:
            return f"Tool '{tool_name}' failed: {e}"

    async def connect_to_server(self, server_name: str, server_cfg: dict):
        """Connect to one MCP server (stdio)."""
        try:
            params = StdioServerParameters(**server_cfg)
            read, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self.sessions.append(session)

            resp = await session.list_tools()
            tools = resp.tools or []
            print(f"Connected to {server_name} with tools:", [t.name for t in tools])

            for t in tools:
                self.tool_to_session[t.name] = session
            self.available_tools.extend(self._openai_tools_from_mcp(tools))
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self, config_path: str = "server_config.json"):
        """Read server_config.json and connect to each server."""
        if not os.path.exists(config_path):
            print(f"No {config_path} found — skipping MCP server connections.")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Failed to read {config_path}: {e}")
            return

        for name, server_cfg in (cfg.get("mcpServers", {}) or {}).items():
            await self.connect_to_server(name, server_cfg)

    async def process_query(self, query: str, model: Optional[str] = None):
        model = model or self.model_default

        system_preamble = None
        if self.available_tools:
            tool_names = [t['function']['name'] for t in self.available_tools]
            system_preamble = {
                'role': 'system',
                'content': 'You can call functions to satisfy user requests. Available tools: ' + ', '.join(tool_names)
            }

        messages: List[Dict[str, Any]] = []
        if system_preamble:
            messages.append(system_preamble)
        messages.append({'role': 'user', 'content': query})

        # Run a full tool loop until the model stops calling tools
        while True:
            resp = self.client.chat_once(
                model=model,
                messages=messages,
                temperature=0.1,
                tools=self.available_tools or None,
                tool_choice='auto' if self.available_tools else 'none',
            )
            msg = resp['choices'][0]['message']
            tool_calls = msg.get('tool_calls') or []

            # If the model wants to call tools, do them and continue
            if tool_calls:
                # Keep the assistant step (with tool_calls) in the transcript
                messages.append({
                    'role': msg.get('role', 'assistant'),
                    'content': msg.get('content', '') or "",
                    'tool_calls': tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get('function', {}) or {}
                    name = fn.get('name')
                    if not name:
                        continue
                    raw_args = fn.get('arguments') or '{}'
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}

                    print(f"Calling tool {name} with args {args}")
                    result_text = await self._call_mcp_tool(name, args)

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.get('id') or name,  # Ollama may omit id
                        'name': name,
                        'content': result_text,
                    })

                # Let the model see tool outputs and possibly call more tools
                continue

            # No more tool calls — print final content if present and exit
            if msg.get('content'):
                print(msg['content'].strip())
            break

    async def chat_loop(self):
        print("\nLocal MCP Chatbot (Ollama) — type your query, or 'quit' to exit.")
        while True:
            try:
                q = input("\nQuery: ").strip()
                if not q:
                    continue
                if q.lower() in ("quit", "exit"):
                    break
                await self.process_query(q)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        # Ensure all sessions and the stdio pipes are closed
        try:
            await self.exit_stack.aclose()
        except Exception:
            pass

async def main():
    bot = MCP_ChatBot()
    try:
        await bot.connect_to_servers()  # discover and register tools
        await bot.chat_loop()
    finally:
        await bot.cleanup()

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
