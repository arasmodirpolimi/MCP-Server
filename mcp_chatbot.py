from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any, Optional, Union
from contextlib import AsyncExitStack
import asyncio, json, os, requests

# ===== LangChain imports (local summarizer + persistent history) =====
from langchain_ollama import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory   # <- use this
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
        # ===== Models & endpoints =====
        self.model_default = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
        self.client = OllamaAdapter(os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"))

        # ===== MCP infra =====
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[ToolDefinition] = []

        # ===== LangChain Memory (NEW) =====
        # 1) Persistent chat history file (override with LC_HISTORY_PATH)
        history_path = os.getenv("LC_HISTORY_PATH", ".mcp_chat_history.json")
        self.history = FileChatMessageHistory(history_path)

        # 2) Local summarizer model via Ollama (override with LC_SUMMARY_MODEL)
        #    Pick a small/fast model installed in Ollama (e.g., llama3.2:3b, qwen2.5:1.5b, phi4:latest).
        summary_model = os.getenv("LC_SUMMARY_MODEL", "qwen2.5:1.5b")
        self.summary_llm = ChatOllama(
            model=summary_model,
            base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0.0
        )

        # 3) Summary buffer to keep memory compact
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm,
            chat_memory=self.history,          
            return_messages=True,
            input_key="input",
            output_key="output",
            max_token_limit=int(os.getenv("LC_MAX_TOKENS", "3000")),
        )

    # ---------------------- MCP wiring ----------------------
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

    # ---------------------- Memory helpers (NEW) ----------------------
    def _lc_history_to_openai_messages(self, history: Union[str, List[Any]]) -> List[Dict[str, str]]:
        """Convert LangChain memory history into Ollama/OpenAI-style messages."""
        msgs: List[Dict[str, str]] = []
        # If ConversationSummaryBufferMemory returns a summary string
        if isinstance(history, str):
            if history.strip():
                msgs.append({"role": "system", "content": f"Conversation summary so far:\n{history}"})
            return msgs

        # Otherwise it's a list[BaseMessage]
        for m in history:
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msgs.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                msgs.append({"role": "system", "content": m.content})
            # Skip tool messages here; tool outputs are re-fed within the same turn
        return msgs

    def clear_memory(self):
        self.history.clear()
        print("✅ Memory cleared.")

    # ---------------------- Chat loop with memory ----------------------
    async def process_query(self, query: str, model: Optional[str] = None):
        model = model or self.model_default

        # 1) Pull prior memory and convert to messages
        prior = self.memory.load_memory_variables({}).get("history", [])
        memory_msgs = self._lc_history_to_openai_messages(prior)

        # 2) Optional system preamble advertising tools
        system_preamble = None
        if self.available_tools:
            tool_names = [t['function']['name'] for t in self.available_tools]
            system_preamble = {
                'role': 'system',
                'content': 'You can call functions to satisfy user requests. Available tools: ' + ', '.join(tool_names)
            }

        # 3) Build the conversation to send to the model
        messages: List[Dict[str, Any]] = []
        messages.extend(memory_msgs)
        if system_preamble:
            messages.append(system_preamble)
        messages.append({'role': 'user', 'content': query})

        # 4) Full tool loop until model stops calling tools
        final_text: str = ""
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

            if tool_calls:
                # Keep the assistant step (with tool_calls) in the transcript for the model
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
                # loop to let the model see tool outputs
                continue

            # No more tool calls — print final content if present and exit
            final_text = (msg.get('content') or "").strip()
            if final_text:
                print(final_text)
            break

        # 5) Save this turn into memory (triggers summarization as needed)
        try:
            self.memory.save_context({"input": query}, {"output": final_text or "[no text]"})
        except Exception as mem_err:
            print(f"(Memory error — continuing without memory this turn): {mem_err}")

    async def chat_loop(self):
        print(f"\nLocal MCP Chatbot (Ollama {self.model_default} + LangChain Memory) — type your query, 'forget' to clear memory, or 'quit' to exit.")
        while True:
            try:
                q = input("\nQuery: ").strip()
                if not q:
                    continue
                if q.lower() in ("quit", "exit"):
                    break
                if q.lower() == "forget":
                    self.clear_memory()
                    continue
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
