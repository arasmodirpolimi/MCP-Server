# mcp_chatbot_api.py
# ---------------------------------------------------------
# FastAPI wrapper around your MCP_ChatBot (Ollama + LangChain memory)
# Endpoints:
#   - GET  /health
#   - POST /chat                      -> {"query": "...", "model": "...?"}
#   - POST /v1/chat/completions       -> OpenAI-compatible subset
#
# Env:
#   LOCAL_BASE_URL (default http://127.0.0.1:11434)
#   LOCAL_MODEL    (default qwen2.5:1.5b)
#   LC_HISTORY_PATH, LC_SUMMARY_MODEL, LC_MAX_TOKENS (optional)
# ---------------------------------------------------------

import os
import json
import asyncio
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- import your existing class directly from your script ---
# If your file name is different, adjust the import path accordingly.
from mcp_chatbot import MCP_ChatBot  # uses your exact implementation


# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI(title="Local MCP Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared bot instance + a request lock to keep MCP calls tidy
bot: Optional[MCP_ChatBot] = None
bot_lock = asyncio.Lock()


@app.on_event("startup")
async def on_startup():
    global bot
    bot = MCP_ChatBot()
    # Connect to MCP servers per your server_config.json
    await bot.connect_to_servers("server_config.json")
    print("✅ MCP Chatbot API started and tools registered.")


@app.on_event("shutdown")
async def on_shutdown():
    if bot:
        await bot.cleanup()
    print("🛑 MCP Chatbot API stopped.")


@app.get("/health")
async def health():
    return {"ok": True, "tools": [t["function"]["name"] for t in (bot.available_tools if bot else [])]}


# ---------------------------
# Simple JSON endpoint
# ---------------------------
class ChatIn(BaseModel):
    query: str = Field(..., description="User question/prompt")
    model: Optional[str] = Field(None, description="Override model (defaults to LOCAL_MODEL)")

class ChatOut(BaseModel):
    output: str
    used_model: str

@app.post("/chat", response_model=ChatOut)
async def chat_simple(req: ChatIn):
    if not bot:
        raise HTTPException(503, "Bot not ready.")
    async with bot_lock:
        # Use the same internal pipeline you have (with memory + MCP loop)
        try:
            await bot.process_query(req.query, model=req.model)
            # The final text is printed to console in your code; we need it here too.
            # Easiest: lightly patch process_query to return the final text.
            # To avoid touching your file, we re-run a thin helper:
        except Exception as e:
            raise HTTPException(500, f"Processing error: {e}")

    # Mini helper to re-ask once and capture answer without tool loop duplication:
    # (We’ll call the model one more time, but now with memory already updated so it’s cheap.)
    async with bot_lock:
        from copy import deepcopy
        prior = bot.memory.load_memory_variables({}).get("history", [])
        memory_msgs = bot._lc_history_to_openai_messages(prior)
        messages: List[Dict[str, Any]] = []
        messages.extend(memory_msgs)
        messages.append({"role": "user", "content": f"Repeat your last final answer to the previous query exactly, without extra commentary."})

        resp = bot.client.chat_once(
            model=req.model or bot.model_default,
            messages=messages,
            temperature=0.0,
            tools=None,
            tool_choice="none",
        )
        msg = resp["choices"][0]["message"]
        final_text = (msg.get("content") or "").strip()

    return ChatOut(output=final_text or "[no text]", used_model=req.model or bot.model_default)


# ---------------------------
# OpenAI-compatible subset
# ---------------------------
class OAIMessage(BaseModel):
    role: str
    content: str

class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAIMessage]
    tools: Optional[list] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512

@app.post("/v1/chat/completions")
async def oai_chat(req: OAIChatRequest):
    if not bot:
        raise HTTPException(503, "Bot not ready.")
    # We’ll perform a single-turn answer that respects tools if requested.
    # To keep parity with your terminal flow, we implement the same loop once.
    messages = [m.dict() for m in req.messages]
    temperature = req.temperature or 0.2
    model = req.model or bot.model_default

    # Ensure tool list is our MCP tool inventory if caller passes "auto"
    tools = req.tools if req.tools is not None else (bot.available_tools or None)

    # Tool loop (mirrors your process_query but with provided messages)
    async with bot_lock:
        while True:
            resp = bot.client.chat_once(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice="auto" if tools else "none",
            )
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # Keep assistant message (with tool_calls)
                messages.append({
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", "") or "",
                    "tool_calls": tool_calls,
                })

                # Execute tools via MCP
                for tc in tool_calls:
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}
                    result_text = await bot._call_mcp_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or name,
                        "name": name,
                        "content": result_text,
                    })
                # loop again so model “sees” tool outputs
                continue

            # Done
            return {
                "id": "local-oai",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Local MCP Chatbot</title>
<style>
  :root { --bg:#0b0f14; --panel:#121821; --text:#e8edf3; --muted:#a7b0ba; --accent:#6aa3ff; --accent-2:#22c55e; }
  * { box-sizing: border-box }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; height:100vh; display:flex }
  .sidebar { width:260px; background:var(--panel); border-right:1px solid #1f2732; display:flex; flex-direction:column }
  .side-top { padding:16px; border-bottom:1px solid #1f2732; display:flex; gap:8px; align-items:center }
  .new { background:#1b2430; color:#fff; border:1px solid #2a3544; border-radius:10px; padding:8px 10px; cursor:pointer }
  .rooms { overflow:auto; padding:8px }
  .room { padding:10px 12px; border-radius:10px; color:var(--muted); cursor:pointer }
  .room.active, .room:hover { background:#1b2430; color:#fff }
  .main { flex:1; display:flex; flex-direction:column; }
  header { padding:12px 16px; border-bottom:1px solid #1f2732; display:flex; gap:12px; align-items:center }
  .pill { background:#1b2430; color:#c9d6e2; border:1px solid #2a3544; border-radius:999px; padding:6px 10px; display:inline-flex; gap:8px; align-items:center }
  .model { width:170px; background:transparent; color:#c9d6e2; border:none; outline:none }
  .chat { flex:1; overflow:auto; padding:16px; }
  .bubble { max-width:880px; padding:12px 14px; margin:12px 0; border-radius:14px; white-space:pre-wrap }
  .user   { background:#1b2430; border:1px solid #2a3544; margin-left:auto }
  .assistant { background:#0f1420; border:1px solid #1e2633; }
  .assistant .tag { color:#a0e7b2; font-size:12px; margin-bottom:6px }
  footer { padding:12px; border-top:1px solid #1f2732; }
  textarea { width:100%; resize:vertical; min-height:72px; max-height:240px; background:#0f1420; color:#e8edf3; border:1px solid #2a3544; border-radius:12px; padding:12px; outline:none }
  .bar { margin-top:8px; display:flex; gap:8px; align-items:center; }
  button.send { background:var(--accent); border:none; color:#091018; padding:10px 14px; border-radius:10px; cursor:pointer; }
  .muted { color:var(--muted) }
  .spinner { width:16px; height:16px; border:2px solid #fff3; border-top-color:#fff9; border-radius:50%; animation:spin 1s linear infinite; display:none }
  @keyframes spin { to { transform: rotate(360deg) } }
</style>
</head>
<body>
  <div class="sidebar">
    <div class="side-top">
      <button class="new" id="newChat">+ New chat</button>
      <span class="muted">Local MCP</span>
    </div>
    <div class="rooms" id="rooms"></div>
  </div>
  <div class="main">
    <header>
      <div class="pill">
        Model:
        <select id="model" class="model">
          <option>qwen2.5:1.5b</option>
        </select>
      </div>
      <div class="pill"><span id="toolBadge">MCP Tools Enabled</span></div>
      <div class="spinner" id="spin"></div>
      <span class="muted" id="status"></span>
    </header>
    <div class="chat" id="chat"></div>
    <footer>
      <textarea id="box" placeholder="Message (Ctrl+Enter to send)…"></textarea>
      <div class="bar">
        <button class="send" id="send">Send</button>
        <span class="muted">History is local to your browser.</span>
      </div>
    </footer>
  </div>

<script>
  const API_BASE = "/v1/chat/completions"; // uses your OpenAI-compatible route
  const chatEl = document.getElementById('chat');
  const box = document.getElementById('box');
  const sendBtn = document.getElementById('send');
  const modelSel = document.getElementById('model');
  const roomsEl = document.getElementById('rooms');
  const statusEl = document.getElementById('status');
  const spin = document.getElementById('spin');

  // simple multi-chat in localStorage
  const store = {
    loadAll(){ try { return JSON.parse(localStorage.getItem('mcp_chats')||'{}'); } catch { return {}; } },
    saveAll(obj){ localStorage.setItem('mcp_chats', JSON.stringify(obj)); },
    create(){ const id = Date.now().toString(36); const all=this.loadAll(); all[id]=[]; this.saveAll(all); return id; },
    list(){ return Object.entries(this.loadAll()).sort((a,b)=>a[0]<b[0]?1:-1); },
    get(id){ return this.loadAll()[id] || []; },
    set(id, msgs){ const all=this.loadAll(); all[id]=msgs; this.saveAll(all); },
    remove(id){ const all=this.loadAll(); delete all[id]; this.saveAll(all); }
  };

  let current = null;
  function renderRooms(){
    roomsEl.innerHTML='';
    for (const [id, msgs] of store.list()){
      const div = document.createElement('div');
      div.className = 'room' + (id===current?' active':'');
      const title = (msgs.find(m=>m.role==='user')?.content || 'New chat').slice(0,40);
      div.textContent = title || 'New chat';
      div.onclick = ()=>{ current=id; renderRooms(); renderChat(); };
      roomsEl.appendChild(div);
    }
  }
  function renderChat(){
    chatEl.innerHTML='';
    const msgs = store.get(current);
    for (const m of msgs){
      const b = document.createElement('div');
      b.className = 'bubble ' + (m.role==='user'?'user':'assistant');
      if(m.role!=='user'){
        const tag = document.createElement('div'); tag.className='tag'; tag.textContent='Assistant'; b.appendChild(tag);
      }
      b.appendChild(document.createTextNode(m.content));
      chatEl.appendChild(b);
    }
    chatEl.scrollTop = chatEl.scrollHeight;
  }
  function append(role, content){
    const msgs = store.get(current);
    msgs.push({role, content});
    store.set(current, msgs);
    renderChat();
  }

  async function send(){
    const text = box.value.trim();
    if(!text) return;
    append('user', text);
    box.value = '';
    spin.style.display='inline-block'; statusEl.textContent='Thinking…';

    try {
      const body = {
        model: modelSel.value,
        messages: store.get(current).map(m => ({ role: m.role, content: m.content })),
        temperature: 0.2
      };
      const r = await fetch(API_BASE, {
        method: 'POST',
        headers: {'content-type':'application/json'},
        body: JSON.stringify(body)
      });
      if(!r.ok){
        const t = await r.text();
        append('assistant', 'Error: '+t);
      } else {
        const data = await r.json();
        const msg = data.choices?.[0]?.message?.content || '[no text]';
        append('assistant', msg);
      }
    } catch (e){
      append('assistant', 'Network error: '+e);
    } finally {
      spin.style.display='none'; statusEl.textContent='';
    }
  }

  sendBtn.onclick = send;
  box.addEventListener('keydown', e=>{
    if(e.key==='Enter' && (e.ctrlKey||e.metaKey)) send();
  });
  document.getElementById('newChat').onclick = ()=>{
    current = store.create();
    renderRooms(); renderChat(); box.focus();
  };

  // bootstrap
  if (store.list().length===0) current = store.create();
  else current = store.list()[0][0];
  renderRooms(); renderChat();
</script>
</body>
</html>
"""
