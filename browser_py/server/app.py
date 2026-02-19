"""FastAPI server — chat API, SSE streaming, cron management, and web UI.

Start with: bpy serve [--port 8321]
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from browser_py.agent.config import get_agent_config, get_workspace, is_configured
from browser_py.agent.loop import Agent

app = FastAPI(title="browser-py", docs_url=None, redoc_url=None)

# Global agent instance (per server process)
_agent: Agent | None = None
_agent_lock = threading.Lock()
_ws_clients: list[WebSocket] = []


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        cfg = get_agent_config()
        _agent = Agent(
            browser_profile=cfg.get("browser_profile"),
            on_tool_call=_on_tool_call,
            on_message=_on_message,
        )
        # Disable shell if configured
        if not cfg.get("shell_enabled", True):
            _agent._shell.enabled = False
    return _agent


def _on_tool_call(name: str, params: dict, result: str) -> None:
    """Broadcast tool calls to connected WebSocket clients."""
    msg = json.dumps({
        "type": "tool_call",
        "tool": name,
        "params": params,
        "result": result[:2000],  # Cap for WS
    })
    _broadcast(msg)


def _on_message(text: str) -> None:
    """Broadcast agent messages to WebSocket clients."""
    msg = json.dumps({"type": "message", "content": text})
    _broadcast(msg)


def _broadcast(msg: str) -> None:
    """Send to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


_loop: asyncio.AbstractEventLoop = None  # type: ignore


# ── REST API ──


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the chat UI."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse(_FALLBACK_HTML)


@app.post("/api/chat")
async def chat(body: dict) -> JSONResponse:
    """Send a message and get a response (blocking)."""
    message = body.get("message", "")
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    agent = _get_agent()

    # Run in thread to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, agent.chat, message)

    return JSONResponse({
        "response": result,
        "history_length": len(agent.messages),
    })


@app.post("/api/reset")
async def reset() -> JSONResponse:
    """Clear conversation history."""
    agent = _get_agent()
    agent.reset()
    return JSONResponse({"ok": True})


@app.get("/api/history")
async def history() -> JSONResponse:
    """Get conversation history."""
    agent = _get_agent()
    return JSONResponse({"messages": agent.get_history()})


@app.get("/api/config")
async def config() -> JSONResponse:
    """Get agent configuration (no secrets)."""
    cfg = get_agent_config()
    safe = {
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "workspace": cfg.get("workspace"),
        "browser_profile": cfg.get("browser_profile"),
        "shell_enabled": cfg.get("shell_enabled", True),
        "configured": is_configured(),
    }
    return JSONResponse(safe)


@app.get("/api/jobs")
async def list_jobs() -> JSONResponse:
    """List cron jobs."""
    agent = _get_agent()
    result = agent._cron.execute(action="list")
    return JSONResponse({"jobs": result})


# ── WebSocket for live updates ──


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "chat":
                agent = _get_agent()
                # Send thinking indicator
                await ws.send_text(json.dumps({"type": "thinking"}))

                # Run agent in thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, agent.chat, msg.get("message", "")
                )

                await ws.send_text(json.dumps({
                    "type": "response",
                    "content": result,
                }))

            elif msg.get("type") == "reset":
                agent = _get_agent()
                agent.reset()
                await ws.send_text(json.dumps({"type": "reset_ok"}))

    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Scheduler ──


def _start_scheduler() -> None:
    """Start APScheduler for cron jobs."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.date import DateTrigger
    except ImportError:
        return  # APScheduler not installed — cron disabled

    from browser_py.agent.tools.cron import _load_jobs

    scheduler = BackgroundScheduler()
    jobs = _load_jobs()

    for jid, job in jobs.items():
        if job.get("paused"):
            continue

        task_text = job.get("task", "")

        if job.get("schedule_type") == "cron":
            parts = job["cron"].split()
            if len(parts) == 5:
                trigger = CronTrigger(
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                    timezone=job.get("timezone") or None,
                )
                scheduler.add_job(
                    _run_scheduled_task, trigger, args=[task_text], id=jid
                )
        elif job.get("schedule_type") == "interval":
            minutes = job.get("interval_minutes", 60)
            scheduler.add_job(
                _run_scheduled_task,
                IntervalTrigger(minutes=minutes),
                args=[task_text],
                id=jid,
            )
        elif job.get("schedule_type") == "date":
            scheduler.add_job(
                _run_scheduled_task,
                DateTrigger(run_date=job["run_at"]),
                args=[task_text],
                id=jid,
            )

    scheduler.start()


def _run_scheduled_task(task: str) -> None:
    """Execute a scheduled task in a fresh agent context."""
    cfg = get_agent_config()
    agent = Agent(
        browser_profile=cfg.get("browser_profile"),
    )
    if not cfg.get("shell_enabled", True):
        agent._shell.enabled = False

    try:
        result = agent.chat(task)
        # Log result
        log_dir = get_workspace() / ".cron_logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{int(time.time())}.log"
        log_file.write_text(f"Task: {task}\n\nResult:\n{result}\n")
    except Exception as e:
        log_dir = get_workspace() / ".cron_logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{int(time.time())}_error.log"
        log_file.write_text(f"Task: {task}\n\nError:\n{e}\n")


# ── Server entry point ──


def start_server(host: str = "127.0.0.1", port: int = 8321) -> None:
    """Start the web server."""
    global _loop
    import uvicorn

    print(f"\n🌐 browser-py agent running at http://{host}:{port}\n")

    _start_scheduler()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(server.serve())


# ── Fallback HTML (embedded chat UI) ──

_FALLBACK_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>browser-py</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
    --tool-bg: #1c2128; --user-bg: #1f3a5f; --agent-bg: #1c2128;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); height: 100vh; display: flex;
    flex-direction: column; }
  header { padding: 12px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 16px; font-weight: 600; }
  header .status { font-size: 12px; color: var(--text-dim); margin-left: auto; }
  #chat { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex;
    flex-direction: column; gap: 12px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px;
    font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .msg.user { background: var(--user-bg); align-self: flex-end;
    border-bottom-right-radius: 4px; }
  .msg.agent { background: var(--agent-bg); align-self: flex-start;
    border-bottom-left-radius: 4px; border: 1px solid var(--border); }
  .msg.tool { background: var(--tool-bg); align-self: flex-start; font-size: 12px;
    font-family: 'SF Mono', Monaco, monospace; color: var(--text-dim);
    border-left: 3px solid var(--accent); max-width: 90%; }
  .msg.tool .tool-name { color: var(--accent); font-weight: 600; }
  .msg.thinking { color: var(--text-dim); font-style: italic; }
  #input-area { padding: 12px 20px; border-top: 1px solid var(--border);
    display: flex; gap: 8px; }
  #input { flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; color: var(--text); font-size: 14px;
    outline: none; resize: none; min-height: 44px; max-height: 120px;
    font-family: inherit; }
  #input:focus { border-color: var(--accent); }
  #input::placeholder { color: var(--text-dim); }
  #send { background: var(--accent); border: none; border-radius: 8px;
    padding: 10px 20px; color: #fff; font-size: 14px; font-weight: 600;
    cursor: pointer; }
  #send:hover { opacity: 0.9; }
  #send:disabled { opacity: 0.4; cursor: default; }
  .controls { display: flex; gap: 8px; }
  .controls button { background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; color: var(--text-dim); font-size: 12px;
    cursor: pointer; }
  .controls button:hover { color: var(--text); border-color: var(--text-dim); }
</style>
</head>
<body>
<header>
  <h1>🌐 browser-py</h1>
  <div class="controls">
    <button onclick="resetChat()">New Chat</button>
  </div>
  <div class="status" id="status">Ready</div>
</header>
<div id="chat"></div>
<div id="input-area">
  <textarea id="input" placeholder="What should I do?" rows="1"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const status = document.getElementById('status');
let ws;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => { status.textContent = 'Connected'; };
  ws.onclose = () => { status.textContent = 'Disconnected'; setTimeout(connect, 2000); };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'thinking') {
      removeThinking();
      addMsg('Thinking...', 'agent thinking');
    } else if (msg.type === 'tool_call') {
      removeThinking();
      const action = msg.params?.action || '';
      const detail = action ? ` → ${action}` : '';
      let text = `🔧 ${msg.tool}${detail}`;
      if (msg.result) text += '\\n' + msg.result.slice(0, 500);
      addMsg(text, 'tool');
    } else if (msg.type === 'response') {
      removeThinking();
      addMsg(msg.content, 'agent');
      sendBtn.disabled = false;
      input.focus();
    } else if (msg.type === 'reset_ok') {
      chat.innerHTML = '';
      addMsg('Chat cleared. How can I help?', 'agent');
    }
  };
}

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  if (cls === 'tool') {
    const parts = text.split('\\n');
    const nameSpan = document.createElement('span');
    nameSpan.className = 'tool-name';
    nameSpan.textContent = parts[0];
    div.appendChild(nameSpan);
    if (parts.length > 1) {
      div.appendChild(document.createTextNode('\\n' + parts.slice(1).join('\\n')));
    }
  } else {
    div.textContent = text;
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeThinking() {
  const thinking = chat.querySelector('.thinking');
  if (thinking) thinking.remove();
}

function send() {
  const text = input.value.trim();
  if (!text || sendBtn.disabled) return;
  addMsg(text, 'user');
  ws.send(JSON.stringify({ type: 'chat', message: text }));
  input.value = '';
  sendBtn.disabled = true;
}

function resetChat() {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
}

// Auto-resize textarea
input.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

connect();
</script>
</body>
</html>
"""
