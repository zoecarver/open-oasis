#!/usr/bin/env python3
"""
Oasis interactive server. Reads frames from /tmp/oasis_live_frame.png (written
by play.py), serves them to the browser, and writes actions to
/tmp/oasis_action.json for play.py to read.

Usage: python play_server.py
"""
import os
import sys
import json
import time
import base64
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PORT = 8080
FRAME_PATH = "/tmp/oasis_live_frame.png"
ACTION_PATH = "/tmp/oasis_action.json"
STATUS_PATH = "/tmp/oasis_status.json"
CONFIG_PATH = "/tmp/oasis_config.json"
DEFAULT_DDIM_STEPS = 4
MIN_DDIM_STEPS = 1
MAX_DDIM_STEPS = 12

# Action key names (indices 0-24)
ACTION_KEYS = [
    "inventory", "ESC",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right",
    "cameraX", "cameraY",
    "jump", "sneak", "sprint", "swapHands",
    "attack", "use", "pickItem", "drop",
]

KEY_TO_ACTION = {
    "w": 11, "s": 12, "a": 13, "d": 14,
    "arrowup": -1, "arrowdown": -2, "arrowleft": -3, "arrowright": -4,
}

HTML_PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Oasis</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin:0; height:100%%; background:#111; color:#eee; font-family: system-ui, sans-serif; }
  #app { padding: 16px; }
  #frame { image-rendering: pixelated; width: 640px; height: 360px; background:#000; display:block; margin-top:12px;
           border: 1px solid #2a2a2a; border-radius: 10px; padding: 4px;
           box-shadow: 0 0 0 1px #1a1a1a, 0 8px 32px rgba(0,150,255,0.15), inset 0 0 0 1px #444; }
  .controls { margin-top: 16px; display: flex; gap: 24px; align-items: flex-start; }
  .key-grid { display: grid; grid-template-columns: repeat(3, 48px); gap: 4px; }
  .key { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;
         background: #222; border: 1px solid #444; border-radius: 6px; font-size: 14px;
         font-family: monospace; transition: background 0.05s; }
  .key.active { background: #09f; border-color: #0bf; color: #fff; }
  .key.empty { visibility: hidden; }
  #status { margin-top: 12px; font-family: monospace; font-size: 13px; color: #888; }
</style>
</head>
<body>
<div id="app">
  <h1>Oasis</h1>
  <img id="frame" alt="Waiting for frames..." />

  <div class="controls">
    <div>
      <div style="margin-bottom: 8px; font-size: 13px; color: #888;">Movement</div>
      <div class="key-grid">
        <div class="key empty"></div>
        <div class="key" data-key="w" id="k-w">W</div>
        <div class="key empty"></div>
        <div class="key" data-key="a" id="k-a">A</div>
        <div class="key" data-key="s" id="k-s">S</div>
        <div class="key" data-key="d" id="k-d">D</div>
      </div>
    </div>

    <div>
      <div style="margin-bottom: 8px; font-size: 13px; color: #888;">Camera (arrows)</div>
      <div class="key-grid">
        <div class="key empty"></div>
        <div class="key" data-key="arrowup" id="k-arrowup">&uarr;</div>
        <div class="key empty"></div>
        <div class="key" data-key="arrowleft" id="k-arrowleft">&larr;</div>
        <div class="key" data-key="arrowdown" id="k-arrowdown">&darr;</div>
        <div class="key" data-key="arrowright" id="k-arrowright">&rarr;</div>
      </div>
    </div>

  </div>

  <div style="margin-top: 20px;">
    <label style="font-size: 13px; color: #888;">DDIM steps: <span id="ddimVal">%(ddim_default)d</span></label>
    <input type="range" id="ddimSlider" min="%(ddim_min)d" max="%(ddim_max)d" step="1"
           value="%(ddim_default)d" style="width: 320px; vertical-align: middle; margin-left: 8px;">
    <div style="font-size: 11px; color: #666; margin-top: 4px;">
      Lower = faster but rougher. Generation pauses while the trace rebuilds.
    </div>
  </div>

  <div id="status">Waiting for generator...</div>
</div>

<script>
const ACTION_KEYS = %(action_keys_json)s;
const KEY_MAP = %(key_map_json)s;

let pressed = new Set();

function buildActionVec() {
  let vec = new Array(25).fill(0);
  for (const key of pressed) {
    const idx = KEY_MAP[key];
    if (idx !== undefined && idx >= 0) vec[idx] = 1;
  }
  if (pressed.has('arrowup'))    vec[15] = -0.25;
  if (pressed.has('arrowdown'))  vec[15] = 0.25;
  if (pressed.has('arrowleft'))  vec[16] = -0.25;
  if (pressed.has('arrowright')) vec[16] = 0.25;
  return vec;
}

function updateUI() {
  document.querySelectorAll('.key').forEach(el => {
    el.classList.toggle('active', pressed.has(el.dataset.key));
  });
}

document.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (KEY_MAP[k] !== undefined) {
    pressed.add(k);
    updateUI();
    sendAction();
    e.preventDefault();
  }
});
document.addEventListener('keyup', (e) => {
  pressed.delete(e.key.toLowerCase());
  updateUI();
  sendAction();
});

let actionPending = false;
let actionDirty = false;
function sendAction() {
  actionDirty = true;
  if (actionPending) return;
  actionPending = true;
  actionDirty = false;
  fetch('/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: buildActionVec()})
  }).finally(() => {
    actionPending = false;
    if (actionDirty) sendAction();
  });
}

// DDIM slider: debounce + post on change
const ddimSlider = document.getElementById('ddimSlider');
const ddimValEl = document.getElementById('ddimVal');
let ddimTimer = null;
ddimSlider.addEventListener('input', () => {
  ddimValEl.textContent = ddimSlider.value;
  if (ddimTimer) clearTimeout(ddimTimer);
  ddimTimer = setTimeout(() => {
    fetch('/config', {method: 'POST',
                       headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify({ddim_steps: parseInt(ddimSlider.value, 10)})});
  }, 200);
});

// Poll for frames
async function pollFrames() {
  const frameImg = document.getElementById('frame');
  const statusEl = document.getElementById('status');
  while (true) {
    try {
      const resp = await fetch('/frame?t=' + Date.now());
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        frameImg.onload = () => URL.revokeObjectURL(url);
        frameImg.src = url;
        const fps = resp.headers.get('X-FPS') || '?';
        const idx = resp.headers.get('X-Frame-Index') || '?';
        const ddim = resp.headers.get('X-DDIM-Steps') || '?';
        statusEl.textContent = 'Frame: ' + idx + '  FPS: ' + fps + '  DDIM: ' + ddim;
      }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 21));
  }
}
pollFrames();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            current_ddim = DEFAULT_DDIM_STEPS
            try:
                with open(CONFIG_PATH) as f:
                    current_ddim = int(json.load(f).get("ddim_steps", DEFAULT_DDIM_STEPS))
            except Exception:
                pass
            page = HTML_PAGE % {
                "action_keys_json": json.dumps(ACTION_KEYS),
                "key_map_json": json.dumps(KEY_TO_ACTION),
                "ddim_default": current_ddim,
                "ddim_min": MIN_DDIM_STEPS,
                "ddim_max": MAX_DDIM_STEPS,
            }
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/frame"):
            try:
                data = open(FRAME_PATH, "rb").read()
                mtime = str(os.path.getmtime(FRAME_PATH))
                fps, idx, ddim = "0", "0", "?"
                try:
                    st = json.load(open(STATUS_PATH))
                    fps = "%.1f" % st.get("fps", 0)
                    idx = str(st.get("frame_index", 0))
                    if "ddim_steps" in st:
                        ddim = str(st["ddim_steps"])
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "image/bmp")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Frame-Mtime", mtime)
                self.send_header("X-FPS", fps)
                self.send_header("X-Frame-Index", idx)
                self.send_header("X-DDIM-Steps", ddim)
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_error(404, "No frame yet")

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/action":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            tmp = ACTION_PATH + ".tmp"
            with open(tmp, "w") as f:
                f.write(body.decode())
            os.rename(tmp, ACTION_PATH)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = b'{"ok":true}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/config":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body.decode())
                ddim_steps = int(req.get("ddim_steps", DEFAULT_DDIM_STEPS))
                ddim_steps = max(MIN_DDIM_STEPS, min(MAX_DDIM_STEPS, ddim_steps))
            except Exception:
                self.send_error(400, "Bad config")
                return
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"ddim_steps": ddim_steps}, f)
            os.rename(tmp, CONFIG_PATH)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            resp = ('{"ok":true,"ddim_steps":%d}' % ddim_steps).encode()
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)


if __name__ == "__main__":
    with open(ACTION_PATH, "w") as f:
        json.dump({"action": [0] * 25}, f)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump({"ddim_steps": DEFAULT_DDIM_STEPS}, f)
    print("Oasis server on http://0.0.0.0:%d" % PORT)
    print("Waiting for play.py to generate frames at %s" % FRAME_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()
