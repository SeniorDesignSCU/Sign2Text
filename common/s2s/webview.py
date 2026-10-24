"""Live caption page served by the Pi, viewed in a laptop (or phone) browser.

The Pi does all recognition. The laptop is only a screen: open http://s2s.local:8000 on any
device on the same network. No display hardware, no extra installs (Python standard library only).

Endpoints
    GET  /         the caption page (big text, history, live status, optional camera view)
    GET  /events   Server-Sent Events stream of the caption state (JSON), pushed on change
    GET  /video    MJPEG stream of the camera preview with landmarks (only encoded while watched)
    GET  /state    current state as JSON (debugging)
    POST /cmd?c=clear|finish   buttons on the page: clear captions / finish the current line

Anyone on the same network can open the page (it shows the camera), so for demos prefer a phone
hotspot or the Pi's own hotspot over shared campus Wi-Fi, or pass --host 127.0.0.1 to disable it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from . import vocab


def local_urls(port: int) -> list[str]:
    urls = [f"http://{socket.gethostname()}.local:{port}"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outbound interface
            ip = s.getsockname()[0]
        if not ip.startswith("127."):
            urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


class WebView:
    def __init__(self, port: int = 8000, host: str = "0.0.0.0", jpeg_quality: int = 70, video_fps: float = 12):
        self.port = port
        self._cond = threading.Condition()
        self._state: dict = {"current": [], "last_word": "", "history": [], "label": "", "conf": 0.0,
                             "fps": 0.0, "status": "starting", "has_video": False}
        self._ver = 0
        self._jpeg: bytes | None = None
        self._jpeg_ver = 0
        self._jpeg_t = 0.0
        self._video_viewers = 0
        self._cmds: list[str] = []
        self.jpeg_quality = jpeg_quality
        self.video_period = 1.0 / video_fps
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    # ---- called from the main loop ------------------------------------------------------
    def update(self, transcript=None, label: str = "", conf: float = 0.0, fps: float = 0.0,
               status: str = "", frame: np.ndarray | None = None):
        shown = "" if not label or label == vocab.NO_SIGN else label.replace("_", " ")
        new = {
            "current": list(transcript.current) if transcript else [],
            "last_word": transcript.last_word if transcript else "",
            "history": list(transcript.history) if transcript else [],
            "label": shown,
            "conf": round(float(conf), 2),
            "fps": round(float(fps), 1),
            "status": status,
            "has_video": frame is not None,
        }
        with self._cond:
            if new != self._state:
                self._state = new
                self._ver += 1
                self._cond.notify_all()
        now = time.time()
        if frame is not None and self._video_viewers > 0 and now - self._jpeg_t >= self.video_period:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self._cond:
                    self._jpeg, self._jpeg_t = buf.tobytes(), now
                    self._jpeg_ver += 1
                    self._cond.notify_all()

    def pop_commands(self) -> list[str]:
        with self._cond:
            cmds, self._cmds = self._cmds, []
        return cmds

    def urls(self) -> list[str]:
        return local_urls(self.port)

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    # ---- HTTP ---------------------------------------------------------------------------
    def _handler_class(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for recognition output
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/state":
                    with view._cond:
                        body = json.dumps(view._state).encode()
                    self._send(200, body, "application/json")
                elif path == "/events":
                    self._events()
                elif path == "/video":
                    self._video()
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                u = urlparse(self.path)
                c = parse_qs(u.query).get("c", [""])[0]
                if u.path == "/cmd" and c in ("clear", "finish"):
                    with view._cond:
                        view._cmds.append(c)
                    self._send(204, b"", "text/plain")
                else:
                    self._send(400, b"bad command", "text/plain")

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with view._cond:
                            view._cond.wait_for(lambda: view._ver != seen, timeout=15)
                            changed = view._ver != seen
                            seen, state = view._ver, view._state
                        msg = f"data: {json.dumps(state)}\n\n" if changed else ": keepalive\n\n"
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _video(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with view._cond:
                    view._video_viewers += 1
                    seen = 0 if view._jpeg is None else view._jpeg_ver - 1  # wait for a real frame
                try:
                    while True:
                        with view._cond:
                            if not view._cond.wait_for(lambda: view._jpeg_ver != seen, timeout=5):
                                continue
                            seen, jpg = view._jpeg_ver, view._jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with view._cond:
                        view._video_viewers -= 1

        return Handler


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signs to Speech</title>
<style>
  :root { --bg:#111; --panel:#1d1d1d; --text:#f4f4f4; --dim:#8a8a8a; --hi:#ffd23f; --ok:#5cc85c; --warn:#e5a50a; --bad:#e05252; }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  body { display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:16px; padding:10px 16px; background:var(--panel); flex-wrap:wrap; }
  header h1 { font-size:16px; font-weight:600; margin:0; margin-right:auto; letter-spacing:.02em; }
  .pill { font-size:13px; color:var(--dim); }
  #conn::before { content:"\\25CF  "; color:var(--bad); }
  #conn.live::before { color:var(--ok); }
  button { background:#2c2c2c; color:var(--text); border:1px solid #3a3a3a; border-radius:6px;
    padding:6px 12px; font-size:13px; cursor:pointer; }
  button:hover { background:#383838; }
  main { flex:1; display:flex; min-height:0; }
  #videoBox { flex:0 0 auto; width:min(46vw, 640px); padding:16px; display:none; flex-direction:column; gap:10px; }
  #videoBox.on { display:flex; }
  #video { width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; object-fit:contain; }
  #seeing { font-size:14px; color:var(--dim); }
  #seeing b { color:var(--text); font-size:20px; }
  .bar { height:8px; background:#333; border-radius:4px; overflow:hidden; }
  .bar i { display:block; height:100%; width:0; background:var(--ok); transition:width .15s; }
  #captions { flex:1; display:flex; flex-direction:column; justify-content:flex-end; padding:24px 32px 36px; overflow:hidden; }
  #history div { color:var(--dim); font-size:clamp(20px, 3vw, 38px); line-height:1.3; margin-bottom:8px; }
  #current { font-size:clamp(36px, 6.5vw, 96px); font-weight:650; line-height:1.15; min-height:1.2em; word-wrap:break-word; }
  #current .last { color:var(--hi); }
  #current .empty { color:#444; }
  @media (max-width: 700px) { main { flex-direction:column; } #videoBox { width:100%; } }
</style>
</head>
<body>
<header>
  <h1>Signs to Speech</h1>
  <span class="pill" id="conn">connecting</span>
  <span class="pill" id="fps"></span>
  <span class="pill" id="status"></span>
  <button id="btnVideo" title="v">Show camera</button>
  <button id="btnFinish" title="space">New line</button>
  <button id="btnClear" title="c">Clear</button>
  <button id="btnFull" title="f">Fullscreen</button>
</header>
<main>
  <section id="videoBox">
    <img id="video" alt="camera preview">
    <div id="seeing">seeing <b id="label">-</b> <span id="confTxt"></span></div>
    <div class="bar"><i id="confBar"></i></div>
  </section>
  <section id="captions">
    <div id="history"></div>
    <div id="current"><span class="empty">...</span></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let videoOn = false;
function esc(s) { return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function render(s) {
  $('history').innerHTML = s.history.map(h => '<div>' + esc(h) + '</div>').join('');
  if (!s.current.length) { $('current').innerHTML = '<span class="empty">...</span>'; }
  else {
    $('current').innerHTML = s.current.map((w, i) =>
      (i === s.current.length - 1 && w === s.last_word) ? '<span class="last">' + esc(w) + '</span>' : esc(w)).join(' ');
  }
  $('label').textContent = s.label || '-';
  $('confTxt').textContent = s.label ? Math.round(s.conf * 100) + '%' : '';
  const bar = $('confBar');
  bar.style.width = (s.label ? s.conf * 100 : 0) + '%';
  bar.style.background = s.conf >= 0.8 ? 'var(--ok)' : s.conf >= 0.5 ? 'var(--warn)' : 'var(--bad)';
  $('fps').textContent = s.fps ? s.fps.toFixed(0) + ' FPS' : '';
  $('fps').style.color = s.fps >= 10 ? 'var(--ok)' : 'var(--bad)';
  $('status').textContent = s.status || '';
  $('btnVideo').disabled = !s.has_video;
}
function connect() {
  const es = new EventSource('/events');
  es.onopen = () => { $('conn').textContent = 'live'; $('conn').classList.add('live'); };
  es.onmessage = e => render(JSON.parse(e.data));
  es.onerror = () => { $('conn').textContent = 'reconnecting'; $('conn').classList.remove('live'); };
}
function toggleVideo() {
  videoOn = !videoOn;
  $('videoBox').classList.toggle('on', videoOn);
  $('video').src = videoOn ? '/video?t=' + Date.now() : '';
  $('btnVideo').textContent = videoOn ? 'Hide camera' : 'Show camera';
}
function cmd(c) { fetch('/cmd?c=' + c, {method: 'POST'}).catch(() => {}); }
function full() { if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {}); else document.exitFullscreen(); }
$('btnVideo').onclick = toggleVideo;
$('btnClear').onclick = () => cmd('clear');
$('btnFinish').onclick = () => cmd('finish');
$('btnFull').onclick = full;
document.addEventListener('keydown', e => {
  if (e.key === 'v') toggleVideo();
  else if (e.key === 'c') cmd('clear');
  else if (e.key === ' ') { e.preventDefault(); cmd('finish'); }
  else if (e.key === 'f') full();
});
connect();
</script>
</body>
</html>
"""
