from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any, Optional
from contextlib import AsyncExitStack
import asyncio, json, os, requests, re, urllib.parse

load_dotenv()

# ---------------------------
# Ollama adapter (no OpenAI)
# ---------------------------
# --- in OllamaAdapter ---
class OllamaAdapter:
    def __init__(self, base: str = "http://127.0.0.1:11434"):
        self.base = base.rstrip("/")

    def chat_once(
        self,
        model: str,
        messages,
        temperature: float = 0.1,
        tools=None,                # NEW
        tool_choice: str | dict | None = "auto",  # "auto" | "none" | {"type":"function","function":{"name":"..."}}
    ):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:        payload["tools"] = tools
        if tool_choice:  payload["tool_choice"] = tool_choice
        r = requests.post(f"{self.base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Preserve tool_calls if present
        msg = data.get("message", {}) or {}
        return {
            "choices": [{
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls"),  # <-- keep it
                }
            }]
        }



def _flatten_tool_content(content_list) -> str:
    """Flatten MCP CallToolResult.content into plain text for display."""
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
        # Point directly at Ollama by default; override via env if needed
        #   LOCAL_BASE_URL=http://127.0.0.1:11434
        #   LOCAL_MODEL=qwen2.5:1.5b
        self.model_default = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
        self.client = OllamaAdapter(os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434"))

        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[ToolDefinition] = []  # schemas we discovered

    # ---- MCP tooling helpers ----
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

    def _tools_summary_text(self) -> str:
        if not self.available_tools:
            return "No MCP tools are currently connected."
        lines = []
        for t in self.available_tools:
            fn = t.get("function", {})
            name = fn.get("name", "unknown")
            desc = fn.get("description", "") or ""
            params = fn.get("parameters", {}) or {}
            lines.append(f"- {name}: {desc} | params: {json.dumps(params, ensure_ascii=False)}")
        return "\n".join(lines)

    def _system_preamble(self) -> str:
        # Keep the model from hallucinating about what MCP is.
        summary = self._tools_summary_text()
        return (
            "You are chatting in a local environment using a model served by Ollama. "
            "MCP stands for Model Context Protocol. This runtime discovers MCP tools but "
            "does not use OpenAI-style function calling automatically. "
            "If the user asks about tools, show the list below.\n\n"
            "Connected MCP tools:\n" + (summary if summary else "None")
        )

    # Convenience: call an MCP tool by name with JSON args
    async def _call_mcp_tool(self, tool_name: str, args: dict) -> Optional[str]:
        session = self.tool_to_session.get(tool_name)
        if not session:
            return None
        try:
            result = await session.call_tool(tool_name, arguments=args)
            return _flatten_tool_content(result.content).strip()
        except Exception as e:
            return f"Tool '{tool_name}' failed: {e}"

    # NEW: Let the model refine tool output into a nice human summary
    async def _refine_with_model(
        self,
        tool_name: str,
        tool_args: dict,
        tool_output_text: str,
        model: Optional[str] = None,
    ) -> str:
        """Pass tool output to the LLM for natural-language refinement."""
        model = model or self.model_default

        system = (
            "You are a precise assistant. You will be given FRESH data from a tool. "
            "Write a concise, helpful answer for a general audience. "
            "Do NOT invent numbers or facts; only use the tool output. "
            "Prefer Celsius if unit=celsius, Fahrenheit if unit=fahrenheit. "
            "If appropriate, add one short tip (e.g., umbrella/sunglasses) based strictly on conditions."
        )

        user = (
            f"Tool: {tool_name}\n"
            f"Args: {json.dumps(tool_args, ensure_ascii=False)}\n"
            "Raw tool output (JSON or text):\n"
            "```\n"
            f"{tool_output_text}\n"
            "```\n\n"
            "Task: Summarize the current weather succinctly (1-3 sentences). "
            "Include temperature with unit and the main condition. "
            "If the tool data lacks a value, omit it rather than guessing."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        resp = self.client.chat_once(model, messages, temperature=0.1)
        msg = resp["choices"][0]["message"]
        return (msg.get("content") or "").strip()

    # ---- MCP discovery ----
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

            # Map tool -> session and add to tool list
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

        servers = (cfg or {}).get("mcpServers", {})
        for name, server_cfg in servers.items():
            await self.connect_to_server(name, server_cfg)

        # After connecting, show a neat summary
        print("\n=== MCP tools summary ===")
        print(self._tools_summary_text())
        print("=========================\n")

    # ---- Intent routing helpers ----
    _url_re = re.compile(r"https?://\S+", re.I)

    def _extract_url(self, text: str) -> Optional[str]:
        m = self._url_re.search(text or "")
        return m.group(0) if m else None

    def _parse_weather(self, text: str) -> Optional[dict]:
        """
        Naive weather intent parser.
        Returns dict like {"location": "Milan", "unit": "celsius"} or None if not detected.
        """
        if not text:
            return None
        lower = text.lower()
        if "weather" not in lower:
            return None

        # try to extract location after "in ..."
        loc = None
        m = re.search(r"\b(?:in|at|for)\s+([a-zA-Z\u00C0-\u017F\s\-']+)", text)
        if m:
            loc = m.group(1).strip(" .,!?:;")

        # Default to Celsius; switch if Fahrenheit mentioned
        unit = "celsius"
        if "fahrenheit" in lower or "°f" in lower:
            unit = "fahrenheit"

        # Also support prompts like "Milan celsius"
        if not loc:
            m2 = re.search(r"\b([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s\-']+)\s+(celsius|fahrenheit)\b", lower)
            if m2:
                loc = m2.group(1).strip()
                unit = m2.group(2)

        if not loc:
            # last resort
            m3 = re.search(r"weather\s+(?:in|at|for)?\s*([A-Za-z\u00C0-\u017F][A-Za-z\u00C0-\u017F\s\-']*)", lower)
            if m3:
                loc = m3.group(1).strip()

        if not loc:
            return None
        return {"location": loc, "unit": unit}

    # ---- Chat / routing ----
    def _looks_like_tools_query(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"\b(mcp\s+tools?|tools?|what.*tools|list.*tools)\b", text, re.I))

    async def process_query(self, query: str, model: Optional[str] = None):
        model = model or self.model_default

        # 1) Deterministic tools listing
        if self._looks_like_tools_query(query):
            print("MCP tools available:")
            print(self._tools_summary_text())
            return

        # 2) Direct URL → use fetch tool if available
        url = self._extract_url(query)
        if url and "fetch" in self.tool_to_session:
            print(f"Fetching: {url}")
            content = await self._call_mcp_tool("fetch", {
                "url": url,
                "max_length": 5000,
                "raw": False
            })
            print(content or "(no content)")
            return

        # 3) Weather intent → call weather tool if present; else fallback to fetch wttr.in
        w = self._parse_weather(query)
        if w:
            if "get_current_weather" in self.tool_to_session:
                args = {"location": w["location"], "unit": w["unit"]}
                print(f"Calling MCP tool get_current_weather with {args}")
                content = await self._call_mcp_tool("get_current_weather", args)

                # Refine with the model for a human-friendly answer
                refined = await self._refine_with_model(
                    tool_name="get_current_weather",
                    tool_args=args,
                    tool_output_text=content or "",
                    model=model,
                )
                print(refined or (content or "(no result)"))
                return
            else:
                print("No weather-capable MCP tools connected (get_current_weather/fetch).")
                return

        # 4) Normal chat (with preamble so the model knows it's offline for tools)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_preamble()},
            {"role": "user", "content": query},
        ]
        resp = self.client.chat_once(model, messages, temperature=0.1)
        msg = resp["choices"][0]["message"]
        if msg.get("content"):
            print(msg["content"].strip())

    async def chat_loop(self):
        print("\nMCP Chatbot (Ollama) — type your query, or 'quit' to exit.")
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
