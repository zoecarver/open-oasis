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
    " ": 17, "shift": 18, "control": 19,
    "q": 24, "e": 0,
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
  #frame { image-rendering: pixelated; width: 640px; height: 360px; background:#222; display:block; margin-top:12px; border: 1px solid #333; }
  .controls { margin-top: 16px; display: flex; gap: 24px; align-items: flex-start; }
  .key-grid { display: grid; grid-template-columns: repeat(3, 48px); gap: 4px; }
  .key { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;
         background: #222; border: 1px solid #444; border-radius: 6px; font-size: 14px;
         font-family: monospace; transition: background 0.05s; }
  .key.active { background: #09f; border-color: #0bf; color: #fff; }
  .key.empty { visibility: hidden; }
  .action-list { font-family: monospace; font-size: 13px; line-height: 1.8; }
  .action-item { padding: 2px 8px; border-radius: 4px; transition: background 0.05s; }
  .action-item.active { background: #09f; color: #fff; }
  .mouse-area { width: 120px; height: 80px; background: #222; border: 1px solid #444;
                border-radius: 6px; display: flex; align-items: center; justify-content: center;
                cursor: crosshair; font-size: 12px; color: #666; position: relative; user-select: none; }
  .mouse-area.locked { border-color: #09f; }
  .mouse-dot { width: 8px; height: 8px; background: #09f; border-radius: 50%%;
               position: absolute; transform: translate(-50%%, -50%%); display: none; }
  .mouse-area.locked .mouse-dot { display: block; }
  #status { margin-top: 12px; font-family: monospace; font-size: 13px; color: #888; }
  .mouse-btns { display: flex; gap: 4px; margin-top: 8px; }
  .mouse-btn { width: 56px; height: 32px; display: flex; align-items: center; justify-content: center;
               background: #222; border: 1px solid #444; border-radius: 4px; font-size: 11px;
               font-family: monospace; transition: background 0.05s; }
  .mouse-btn.active { background: #c33; border-color: #e55; color: #fff; }
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
      <div style="margin-top: 8px;">
        <div class="key" data-key=" " id="k-space" style="width: 152px;">Space</div>
      </div>
      <div style="margin-top: 4px; display: flex; gap: 4px;">
        <div class="key" data-key="shift" id="k-shift" style="width: 74px; font-size: 11px;">Shift</div>
        <div class="key" data-key="control" id="k-ctrl" style="width: 74px; font-size: 11px;">Ctrl</div>
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
      <div class="mouse-btns" style="margin-top: 8px;">
        <div class="mouse-btn" id="mb-attack">LMB</div>
        <div class="mouse-btn" id="mb-use">RMB</div>
      </div>
    </div>

    <div>
      <div style="margin-bottom: 8px; font-size: 13px; color: #888;">Active actions</div>
      <div class="action-list" id="actionList"></div>
    </div>
  </div>

  <div id="status">Waiting for generator...</div>
</div>

<script>
const ACTION_KEYS = %(action_keys_json)s;
const KEY_MAP = %(key_map_json)s;

let pressed = new Set();
let cameraX = 0, cameraY = 0;
let mouseDown = {left: false, right: false};

const actionListEl = document.getElementById('actionList');
ACTION_KEYS.forEach((name, i) => {
  const el = document.createElement('div');
  el.className = 'action-item';
  el.id = 'act-' + i;
  el.textContent = i + ': ' + name;
  actionListEl.appendChild(el);
});

function buildActionVec() {
  let vec = new Array(25).fill(0);
  for (const key of pressed) {
    const idx = KEY_MAP[key];
    if (idx !== undefined && idx >= 0) vec[idx] = 1;
  }
  // Arrow keys control camera (binary, like WASD)
  if (pressed.has('arrowup'))    vec[15] = -0.08;
  if (pressed.has('arrowdown'))  vec[15] = 0.08;
  if (pressed.has('arrowleft'))  vec[16] = -0.08;
  if (pressed.has('arrowright')) vec[16] = 0.08;
  if (mouseDown.left) vec[21] = 1;
  if (mouseDown.right) vec[22] = 1;
  return vec;
}

function updateUI() {
  const vec = buildActionVec();
  document.querySelectorAll('.key').forEach(el => {
    el.classList.toggle('active', pressed.has(el.dataset.key));
  });
  ACTION_KEYS.forEach((_, i) => {
    document.getElementById('act-' + i).classList.toggle('active', vec[i] !== 0);
  });
  document.getElementById('mb-attack').classList.toggle('active', mouseDown.left);
  document.getElementById('mb-use').classList.toggle('active', mouseDown.right);
}

document.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (KEY_MAP[k] !== undefined || k === 'shift' || k === 'control') {
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

document.addEventListener('mousedown', (e) => {
  if (e.button === 0) mouseDown.left = true;
  if (e.button === 2) mouseDown.right = true;
  updateUI(); sendAction();
});
document.addEventListener('mouseup', (e) => {
  if (e.button === 0) mouseDown.left = false;
  if (e.button === 2) mouseDown.right = false;
  updateUI(); sendAction();
});
document.addEventListener('contextmenu', (e) => e.preventDefault());

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

// Poll for frames
async function pollFrames() {
  const frameImg = document.getElementById('frame');
  const statusEl = document.getElementById('status');
  let lastMod = '';
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
        statusEl.textContent = 'Frame: ' + idx + '  FPS: ' + fps;
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
            page = HTML_PAGE % {
                "action_keys_json": json.dumps(ACTION_KEYS),
                "key_map_json": json.dumps(KEY_TO_ACTION),
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
                # Read status for FPS/frame index
                fps, idx = "0", "0"
                try:
                    st = json.load(open(STATUS_PATH))
                    fps = "%.1f" % st.get("fps", 0)
                    idx = str(st.get("frame_index", 0))
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "image/bmp")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Frame-Mtime", mtime)
                self.send_header("X-FPS", fps)
                self.send_header("X-Frame-Index", idx)
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
            # Atomic write
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
        else:
            self.send_error(404)


if __name__ == "__main__":
    # Write initial zero action
    with open(ACTION_PATH, "w") as f:
        json.dump({"action": [0] * 25}, f)
    print("Oasis server on http://0.0.0.0:%d" % PORT)
    print("Waiting for play.py to generate frames at %s" % FRAME_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()
