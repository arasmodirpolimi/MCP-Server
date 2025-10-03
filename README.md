# MCP-Server

> Run your own **MCP tools** (weather, filesystem, fetch, etc.) and chat with them using an LLM that supports **function-calling**—either a hosted model (OpenAI) or a **local** model behind an OpenAI-compatible proxy.  
> The chatbot discovers tools from one or more MCP servers and routes tool calls to the right server automatically.

## Contents

* `weather_server.py` – a minimal MCP server (via `FastMCP`) exposing `get_current_weather`.
* `mcp_chatbot.py` – a chat client that:
  * connects to one or more MCP servers over **stdio**,
  * **discovers** their tools,
  * exposes those tools to the model using **OpenAI function-calling**,
  * executes tool calls via MCP and returns results to the model.
* `server_config.json` – lists the MCP servers to spawn (e.g., `fetch`, `filesystem`, `weather`).

---

## Quick Start

### 0) Prereqs
* Python 3.10+
* (Windows/macOS/Linux terminals are fine; avoid running the chat from inside notebook/interactive consoles.)

### 1) Install deps

Using **uv** (recommended):
```bash
uv pip install mcp python-dotenv openai requests
```

Or with pip:
```bash
pip install mcp python-dotenv openai requests
```

### 2) Set environment variables

Create a `.env` file (or export in your shell):
```
# For hosted OpenAI usage (optional if you use a local model + proxy)
OPENAI_API_KEY=sk-...

# For the weather MCP tool
OPENWEATHER_API_KEY=your_openweather_api_key
```

### 3) Configure which MCP servers to run

`server_config.json` example with three servers:
```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "--root", "."]
    },
    "weather": {
      "command": "uv",
      "args": ["run", "weather_server.py"]
    }
  }
}
```
> The `fetch` and `filesystem` servers require Node.js for `npx`.  
> Adjust `--root` to the directory you want exposed to tools.

### 4) Run the chatbot
```bash
python mcp_chatbot.py
```

Expected output:
```
Connected to fetch with tools: ['fetch']
Connected to filesystem with tools: ['read_file', 'write_file', ...]
Connected to weather with tools: ['get_current_weather']

MCP Chatbot (OpenAI) — type your query, or 'quit' to exit.
```

Try:
* What’s the weather in Milan in celsius?
* Fetch https://example.com and summarize.
* Read README.md and show the first 20 lines.

---

## How it Works

### 1) MCP server (`weather_server.py`)
```python
from mcp.server.fastmcp import FastMCP
import os, requests
from typing import Dict, Any

mcp = FastMCP("weather")

@mcp.tool()
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    key = os.getenv("OPENWEATHER_API_KEY") or ...
    base = "https://api.openweathermap.org/data/2.5/weather"
    units = {"celsius": "metric", "fahrenheit": "imperial"}[unit.lower()]
    resp = requests.get(base, params={"q": location, "units": units, "appid": key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
      "location": data.get("name") or location,
      "temperature": (data.get("main") or {}).get("temp"),
      "unit": "celsius" if units == "metric" else "fahrenheit",
      "forecast": [w.get("description") for w in (data.get("weather") or []) if isinstance(w, dict)]
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
```
* `@mcp.tool()` registers a tool with JSON Schema.
* Runs over stdio so clients can spawn it as a subprocess.

### 2) Chatbot (`mcp_chatbot.py`)
Responsibilities:
1. Spawn & connect to each server via stdio.
2. Call `list_tools()` to discover tools.
3. Convert tools to OpenAI function-calling schema.
4. Loop: send messages → model may request tool calls → execute via MCP → return results → model finalizes.

Flatten helper:
```python
def _flatten_tool_content(content_list) -> str:
    # Joins text parts and dumps JSON/object parts
```

> Any MCP server/tool can plug in (fetch, filesystem, custom, etc.).

---

## Use a local LLM (optional)

Replace hosted OpenAI with a local backend.

1. Pull models (Ollama):
```
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull deepseek-r1:7b
ollama pull qwen2-vl:2b
```
2. Run LiteLLM proxy:
```bash
pip install "litellm[proxy]"
```
`litellm.yaml`:
```yaml
model_list:
  - model_name: qwen2.5-7b-instruct-q4
    litellm_params:
      model: ollama/qwen2.5:7b-instruct-q4_K_M
  - model_name: deepseek-r1-7b
    litellm_params:
      model: ollama/deepseek-r1:7b
  - model_name: qwen2-vl-2b
    litellm_params:
      model: ollama/qwen2-vl:2b
```
Run:
```bash
litellm --config litellm.yaml --host 127.0.0.1 --port 4000
```

Client change:
```python
self.client = OpenAI(
  base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:4000/v1"),
  api_key=os.getenv("OPENAI_API_KEY", "sk-local")
)
```

Then:
```python
await self.process_query(q, model="qwen2.5-7b-instruct-q4")
```

Hardware note: i7-8565U / 20GB RAM / GTX 1650 Max-Q → best with 7B Q4; 3–8 tok/s.

---

## Example Interaction

User:
```
What’s the weather in Milan in celsius?
```
Flow:
1. Model calls tool.
2. Chatbot invokes MCP tool.
3. Weather server queries OpenWeather.
4. Result flattened & returned.
5. Model replies naturally.

---

## Troubleshooting

* ModuleNotFoundError: No module named 'requests'  
  `uv pip install requests`
* Object of type TextContent is not JSON serializable  
  Use `_flatten_tool_content(...)` instead of dumping raw objects.
* UnsupportedOperation: fileno (Windows / notebooks)  
  Run from a terminal, not inside Jupyter.
* Empty Authorization warning (Inspector)  
  Disable or supply a token.
* asyncio.run() cannot be called from a running event loop  
  Avoid running inside notebook; or adapt to existing loop.

---

## Security

* Never commit API keys.
* Limit filesystem server `--root` scope.

---

## Project Structure
```
.
├── mcp_chatbot.py        # Chat client (multi-server MCP + OpenAI tools)
├── weather_server.py     # FastMCP weather tool
├── server_config.json    # MCP server process definitions
├── README.md
└── .env.example          # (optional) example env file
```

---

## Contributing

Ideas:
* PDF / CSV parsing MCP tool
* Vision / image analysis tool
* CLI flags for model/base_url selection

PRs welcome.

---

