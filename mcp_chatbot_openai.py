from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any, Union
from contextlib import AsyncExitStack
import asyncio, json, os

# ===== LangChain imports (with fallbacks for version differences) =====
try:
    from langchain_openai import ChatOpenAI  # >= 0.2 style
except Exception:
    from langchain.chat_models import ChatOpenAI  # legacy fallback

try:
    from langchain_community.chat_message_histories import FileChatMessageHistory
except Exception:
    # older versions
    from langchain.memory.chat_message_histories import FileChatMessageHistory  # type: ignore

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    return "\n".join(parts)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class MCP_ChatBot:
    def __init__(self):
        # --- OpenAI client for actual chat/tool-calling loop ---
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # --- MCP infra ---
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[ToolDefinition] = []  # OpenAI function-calling schema

        # --- LangChain Memory (NEW) ---
        # Persistent history file (JSON). Change path via LC_HISTORY_PATH env var if you like.
        history_path = os.getenv("LC_HISTORY_PATH", ".mcp_chat_history.json")
        self.history = FileChatMessageHistory(history_path)

        # LLM used only for summarizing memory (cheap/small recommended)
        summary_model = os.getenv("LC_SUMMARY_MODEL", "gpt-3.5-turbo")
        self.summary_llm = ChatOpenAI(
            model=summary_model,
            temperature=0.0,
            # LangChain uses OPENAI_API_KEY env under the hood
        )

        # ConversationSummaryBufferMemory will auto-summarize older turns.
        # Increase/decrease max_token_limit based on how much context you want retained.
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm,
            chat_memory=self.history,
            return_messages=True,           # we’ll convert messages to OpenAI format
            input_key="input",
            output_key="output",
            max_token_limit=int(os.getenv("LC_MAX_TOKENS", "3000")),
        )

    # ---------------------- MCP wiring ----------------------
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
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        servers = cfg.get("mcpServers", {})
        for name, server_cfg in servers.items():
            await self.connect_to_server(name, server_cfg)

    # ---------------------- Memory helpers (NEW) ----------------------
    def _lc_history_to_openai_messages(self, history: Union[str, List[Any]]) -> List[Dict[str, str]]:
        """Convert LangChain memory 'history' to OpenAI Chat Completions messages."""
        msgs: List[Dict[str, str]] = []
        if isinstance(history, str):
            if history.strip():
                msgs.append({
                    "role": "system",
                    "content": f"Conversation summary so far:\n{history}"
                })
            return msgs

        # history is a list of BaseMessage
        for m in history:
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                # We store only the assistant's final content between turns.
                msgs.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                msgs.append({"role": "system", "content": m.content})
            else:
                # Skip tool/other message types to keep things simple
                continue
        return msgs

    def _remember_turn(self, user_text: str, ai_text: str):
        """Append the latest user/assistant messages to LangChain memory."""
        self.memory.chat_memory.add_user_message(user_text)
        self.memory.chat_memory.add_ai_message(ai_text)

    def clear_memory(self):
        """Erase persisted memory (type 'forget' in the REPL)."""
        self.history.clear()
        print("✅ Memory cleared.")

    # ---------------------- Chat loop with memory ----------------------
    async def process_query(self, query: str, model: str = "gpt-3.5-turbo"):
        # 1) Pull prior memory (summarized or raw messages) and convert to OpenAI format
        prior = self.memory.load_memory_variables({}).get("history", [])
        messages: List[Dict[str, Any]] = self._lc_history_to_openai_messages(prior)

        # 2) Add the new user input
        messages.append({"role": "user", "content": query})

        # 3) Chat/tool loop
        while True:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.available_tools or None,
                tool_choice="auto" if self.available_tools else "none",
                temperature=0.1,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                # Keep assistant msg (with tool_calls) in history for the model context
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
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": f"Tool '{tool_name}' is not available.",
                        })
                        continue

                    print(f"Calling tool {tool_name} with args {args}")
                    result = await session.call_tool(tool_name, arguments=args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": _flatten_tool_content(result.content),
                    })
                continue  # let the model read tool results and continue

            # Final answer (no tool calls)
            final_text = (msg.content or "").strip()
            if final_text:
                print(final_text)

            # 4) Store this turn in memory (NEW)
            #    We store only user query and final assistant message for compactness.
            try:
                self._remember_turn(query, final_text if final_text else "[no text]")
            except Exception as mem_err:
                # Don’t crash chat if memory write fails
                print(f"(Memory error — continuing without memory this turn): {mem_err}")

            break

    async def chat_loop(self):
        print("\nMCP Chatbot (OpenAI + LangChain Memory) — type your query, 'forget' to clear memory, or 'quit' to exit.")
        while True:
            try:
                q = input("\nQuery: ").strip()
                if q.lower() == "quit":
                    break
                if q.lower() == "forget":
                    self.clear_memory()
                    continue
                await self.process_query(q)
            except Exception as e:
                print(f"Error: {e}")

    async def run(self):
        try:
            await self.connect_to_servers()
            await self.chat_loop()
        finally:
            await self.exit_stack.aclose()

    async def cleanup(self):  # existing
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()  # init MCP servers
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
