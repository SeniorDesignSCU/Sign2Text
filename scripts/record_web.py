#!/usr/bin/env python3
"""Record training clips on the Pi, controlled from a laptop or phone browser.

    python scripts/record_web.py --signer aditya
    python scripts/record_web.py --signer karan --signs HELLO,THANK_YOU,PLEASE,YES,NO
    then open http://s2s.local:8000 on the laptop

The page shows the Pi camera with landmarks, a list of signs with clip counts, and
buttons (or keys) to record. Clips are saved exactly like scripts/collect.py does
(data/clips/<SIGN>/<signer>_<session>_<nnn>.npz, 640x480, source "picam"), so
training/train.py uses them unchanged. Works without a monitor on the Pi; this is the
recording tool for the device itself. collect.py stays the laptop-window version.

Keys on the page:  space record 1 · r record 5 · s stop · u delete last · n / p next / previous
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("GLOG_minloglevel", "2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from s2s import vocab  # noqa: E402
from s2s.camera import add_camera_args, open_camera  # noqa: E402
from s2s.clips import CLIP_DIR, ClipMeta, save_clip  # noqa: E402
from s2s.landmarks import LandmarkExtractor, draw  # noqa: E402

LIGHTING = ["bright", "office", "dim"]
SIGNER_RE = re.compile(r"^[a-z0-9]{1,20}$")


def count_clips(classes: list[str]) -> dict[str, dict[str, int]]:
    """{sign: {signer: n}} from file names only (fast; no clip loading)."""
    out: dict[str, dict[str, int]] = {c: {} for c in classes}
    if CLIP_DIR.exists():
        for d in CLIP_DIR.iterdir():
            if d.is_dir() and d.name in out:
                for p in d.glob("*.npz"):
                    signer = p.name.split("_", 1)[0]
                    out[d.name][signer] = out[d.name].get(signer, 0) + 1
    return out


def local_urls(port: int) -> list[str]:
    urls = [f"http://{socket.gethostname()}.local:{port}"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        if not ip.startswith("127."):
            urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


class Recorder:
    """Owns the state machine. The camera loop calls tick(); HTTP handlers call command()."""

    def __init__(self, args, classes: list[str], frame_w: int, frame_h: int, source: str):
        self.args = args
        self.classes = classes
        self.ci = 1 if len(classes) > 1 else 0          # start on the first real sign
        self.signer = args.signer
        self.session = args.session
        self.lighting_i = 1
        self.frame_w, self.frame_h, self.source = frame_w, frame_h, source
        self.state = "idle"                              # idle | countdown | recording
        self.t0 = 0.0
        self.burst_left = 0
        self.buf_raw: list[np.ndarray] = []
        self.buf_t: list[int] = []
        self.saved: list[Path] = []                       # this run, for undo
        self.message = "Ready. Pick a sign and press Record."
        self.counts = count_clips(classes)
        self.cmds: "queue.Queue[tuple[str, str]]" = queue.Queue()

    # --- called from HTTP threads ---
    def command(self, c: str, v: str = "") -> None:
        self.cmds.put((c, v))

    # --- called from the camera loop ---
    def _apply(self, c: str, v: str, now: float) -> None:
        idle = self.state == "idle"
        if c == "rec1" and idle:
            self.burst_left, self.state, self.t0 = 0, "countdown", now
        elif c == "rec5" and idle:
            self.burst_left, self.state, self.t0 = 4, "countdown", now
        elif c == "stop":
            if self.state == "recording":
                self._save(now)
            self.burst_left, self.state = 0, "idle"
            self.message = "Stopped."
        elif c == "undo" and idle:
            if self.saved:
                p = self.saved.pop()
                sign = p.parent.name
                p.unlink(missing_ok=True)
                self.counts[sign][self.signer] = max(0, self.counts[sign].get(self.signer, 0) - 1)
                self.message = f"Deleted {sign}/{p.name}"
            else:
                self.message = "Nothing to delete from this run."
        elif c in ("next", "prev") and idle:
            self.ci = (self.ci + (1 if c == "next" else -1)) % len(self.classes)
        elif c == "select" and idle and v in self.classes:
            self.ci = self.classes.index(v)
        elif c == "light":
            self.lighting_i = (self.lighting_i + 1) % len(LIGHTING)
        elif c == "signer" and idle:
            v = v.strip().lower()
            if SIGNER_RE.match(v):
                self.signer, self.saved = v, []
                self.message = f"Signer is now '{v}'."
            else:
                self.message = "Signer name: lowercase letters and numbers only."

    def _save(self, now: float) -> None:
        sign = self.classes[self.ci]
        if len(self.buf_raw) >= 5:
            raw = np.stack(self.buf_raw)
            dur = (self.buf_t[-1] - self.buf_t[0]) / 1000.0 or 1e-3
            meta = ClipMeta(sign=sign, signer=self.signer, session=self.session,
                            fps=len(self.buf_raw) / dur, n_frames=len(self.buf_raw),
                            lighting=LIGHTING[self.lighting_i], handedness=self.args.handedness,
                            source=self.source, width=self.frame_w, height=self.frame_h)
            p = save_clip(raw, np.array(self.buf_t), meta)
            self.saved.append(p)
            self.counts[sign][self.signer] = self.counts[sign].get(self.signer, 0) + 1
            hands = sum(1 for r in self.buf_raw if r[-3] > 0.5 or r[-2] > 0.5)
            warn = "  (no hands seen — check framing)" if sign != vocab.NO_SIGN and hands == 0 else ""
            self.message = f"Saved {sign} #{self.counts[sign][self.signer]} ({len(self.buf_raw)} frames){warn}"
        else:
            self.message = "Clip too short, discarded."
        self.buf_raw, self.buf_t = [], []

    def tick(self, lm, now: float) -> None:
        while not self.cmds.empty():
            self._apply(*self.cmds.get_nowait(), now)
        if self.state == "countdown" and now - self.t0 >= self.args.countdown:
            self.state, self.t0 = "recording", now
            self.buf_raw, self.buf_t = [], []
        if self.state == "recording":
            self.buf_raw.append(lm.as_flat())
            self.buf_t.append(lm.t_ms)
            if now - self.t0 >= self.args.max_seconds:
                self._save(now)
                if self.burst_left > 0:
                    self.burst_left -= 1
                    self.state, self.t0 = "countdown", now
                else:
                    self.state = "idle"

    def snapshot(self, now: float, fps: float, lm) -> dict:
        sign = self.classes[self.ci]
        if self.state == "countdown":
            phase, remain = "countdown", max(0.0, self.args.countdown - (now - self.t0))
        elif self.state == "recording":
            phase, remain = "recording", max(0.0, self.args.max_seconds - (now - self.t0))
        else:
            phase, remain = "idle", 0.0
        return {
            "sign": sign, "phase": phase, "remain": round(remain, 1), "burst_left": self.burst_left,
            "signer": self.signer, "session": self.session, "lighting": LIGHTING[self.lighting_i],
            "message": self.message, "fps": round(fps, 1), "target": self.args.target,
            "hands": ("L" if lm.left_present else "") + ("R" if lm.right_present else ""),
            "pose": bool(lm.pose_present),
            "classes": [{"name": c, "mine": self.counts[c].get(self.signer, 0),
                         "total": sum(self.counts[c].values())} for c in self.classes],
            "can_undo": bool(self.saved),
        }

    def overlay(self, frame: np.ndarray, now: float) -> None:
        h, w = frame.shape[:2]
        sign = self.classes[self.ci]
        if self.state == "countdown":
            r = max(0.0, self.args.countdown - (now - self.t0))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 220, 255), 8)
            txt = f"GET READY  {sign}  {r:.1f}"
            col = (0, 220, 255)
        elif self.state == "recording":
            r = max(0.0, self.args.max_seconds - (now - self.t0))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            txt = f"REC  {sign}  {r:.1f}"
            col = (0, 0, 255)
        else:
            return
        cv2.putText(frame, txt, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(frame, txt, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)


class Hub:
    """Shared state between the camera loop and HTTP threads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.state: dict = {}
        self.ver = 0
        self.jpeg: bytes | None = None
        self.jpeg_ver = 0
        self.viewers = 0


def make_handler(hub: Hub, rec: Recorder):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype):
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
                with hub.cond:
                    self._send(200, json.dumps(hub.state).encode(), "application/json")
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with hub.cond:
                            hub.cond.wait_for(lambda: hub.ver != seen, timeout=15)
                            changed, seen, st = hub.ver != seen, hub.ver, hub.state
                        self.wfile.write((f"data: {json.dumps(st)}\n\n" if changed else ": ka\n\n").encode())
                        self.wfile.flush()
                except OSError:
                    pass
            elif path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with hub.cond:
                    hub.viewers += 1
                    seen = 0 if hub.jpeg is None else hub.jpeg_ver - 1
                try:
                    while True:
                        with hub.cond:
                            if not hub.cond.wait_for(lambda: hub.jpeg_ver != seen, timeout=5):
                                continue
                            seen, jpg = hub.jpeg_ver, hub.jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        self.wfile.flush()
                except OSError:
                    pass
                finally:
                    with hub.cond:
                        hub.viewers -= 1
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            c, v = q.get("c", [""])[0], q.get("v", [""])[0]
            if u.path == "/cmd" and c in ("rec1", "rec5", "stop", "undo", "next", "prev", "select", "light", "signer"):
                rec.command(c, v)
                self._send(204, b"", "text/plain")
            else:
                self._send(400, b"bad command", "text/plain")

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True, help="short id, lowercase letters/numbers, e.g. aditya")
    ap.add_argument("--session", default="pi" + time.strftime("%Y%m%d"), help="session tag")
    ap.add_argument("--signs", default="", help="comma-separated subset to record, e.g. HELLO,YES,NO "
                                                "(NO_SIGN is always included)")
    ap.add_argument("--target", type=int, default=40, help="clips per sign to aim for (progress bars)")
    add_camera_args(ap)
    ap.add_argument("--max-seconds", type=float, default=2.5, help="clip length (keep equal for all clips)")
    ap.add_argument("--countdown", type=float, default=1.5)
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    args.signer = args.signer.strip().lower()
    if not SIGNER_RE.match(args.signer):
        sys.exit("--signer must be lowercase letters/numbers only (it becomes part of file names)")
    if args.signs:
        wanted = [s.strip().upper() for s in args.signs.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in vocab.CLASS_TO_ID]
        if unknown:
            sys.exit(f"not in common/s2s/vocab.py: {unknown}. Known signs: {', '.join(vocab.ALL_CLASSES)}")
        classes = [vocab.NO_SIGN] + [s for s in wanted if s != vocab.NO_SIGN]
    else:
        classes = list(vocab.ALL_CLASSES)

    try:
        cam = open_camera(args.camera, args.width, args.height)
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    ok, first = cam.read()
    if not ok:
        sys.exit("camera opened but returned no frame")
    fh, fw = first.shape[:2]
    if abs(fw / fh - 4 / 3) > 0.02:
        cam.release()
        sys.exit(f"camera gives {fw}x{fh}, not 4:3; clips must be 640x480 to match the Pi runtime")

    ext = LandmarkExtractor()
    rec = Recorder(args, classes, fw, fh, cam.source)
    hub = Hub()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(hub, rec))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"camera: {cam.description}")
    print(f"signer: {rec.signer}   session: {rec.session}   signs: {len(classes) - 1} + NO_SIGN")
    print("open  " + "  or  ".join(local_urls(args.port)) + "  in a browser on the same network")
    print("Ctrl+C here to stop. Clips go to data/clips/")

    fps, fps_n, fps_t, last_jpeg = 0.0, 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("camera stopped returning frames")
                break
            lm = ext.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            now = time.time()
            rec.tick(lm, now)
            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            st = rec.snapshot(now, fps, lm)
            with hub.cond:
                if st != hub.state:
                    hub.state, hub.ver = st, hub.ver + 1
                    hub.cond.notify_all()
            if hub.viewers and now - last_jpeg >= 1 / 12:
                draw(frame, lm)
                rec.overlay(frame, now)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok2:
                    with hub.cond:
                        hub.jpeg, hub.jpeg_ver = buf.tobytes(), hub.jpeg_ver + 1
                        hub.cond.notify_all()
                last_jpeg = now
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        ext.close()
        server.shutdown()
        total = sum(sum(v.values()) for v in rec.counts.values())
        print(f"\nsaved this run: {len(rec.saved)} clips   all clips for these signs: {total}")
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signs to Speech — Record</title>
<style>
:root{--bg:#111;--panel:#1c1c1c;--line:#2e2e2e;--text:#f2f2f2;--dim:#8d8d8d;--ok:#4cc36b;--warn:#f0b429;--rec:#e5484d;--sel:#2f5bd3}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:10px 16px;background:var(--panel);flex-wrap:wrap}
header h1{font-size:16px;margin:0 auto 0 0}.pill{font-size:13px;color:var(--dim)}
#conn::before{content:"\25CF  ";color:var(--rec)}#conn.live::before{color:var(--ok)}
main{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
#video{width:100%;aspect-ratio:4/3;background:#000;border-radius:8px;object-fit:contain;display:block}
#status{margin-top:12px;padding:14px;border-radius:8px;background:var(--panel);font-size:22px;font-weight:650;min-height:58px}
#status.countdown{background:#4a3a06;color:#ffd666}#status.recording{background:#4a1214;color:#ff9a9d}
#msg{margin-top:8px;color:var(--dim);font-size:14px;min-height:20px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
button{background:#2a2a2a;color:var(--text);border:1px solid #3a3a3a;border-radius:6px;padding:9px 14px;font-size:14px;cursor:pointer}
button:hover{background:#353535}button.primary{background:var(--rec);border-color:var(--rec)}button:disabled{opacity:.4;cursor:default}
.k{color:var(--dim);font-size:12px;margin-left:4px}
aside{background:var(--panel);border-radius:8px;padding:14px;max-height:calc(100vh - 110px);overflow:auto}
aside h2{font-size:14px;margin:0 0 8px;color:var(--dim);font-weight:600}
.row{display:grid;grid-template-columns:1fr 70px 60px;gap:8px;align-items:center;padding:7px 8px;border-radius:6px;cursor:pointer}
.row:hover{background:#262626}.row.sel{background:var(--sel)}.row .n{font-size:13px;text-align:right;color:var(--dim)}.row.sel .n{color:#dfe6ff}
.bar{height:6px;background:#333;border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.signer{display:flex;gap:8px;margin-bottom:12px}.signer input{flex:1;background:#111;color:var(--text);border:1px solid #3a3a3a;border-radius:6px;padding:8px}
.tips{font-size:13px;color:var(--dim);line-height:1.5;margin-top:12px}
</style></head><body>
<header><h1>Signs to Speech — Record clips</h1>
<span class="pill" id="conn">connecting</span><span class="pill" id="who"></span><span class="pill" id="fps"></span><span class="pill" id="det"></span></header>
<main>
<section>
<img id="video" src="/video" alt="Pi camera">
<div id="status">Ready</div><div id="msg"></div>
<div class="btns">
<button class="primary" id="b1">Record 1<span class="k">space</span></button>
<button id="b5">Record 5<span class="k">r</span></button>
<button id="bs">Stop<span class="k">s</span></button>
<button id="bp">Prev sign<span class="k">p</span></button>
<button id="bn">Next sign<span class="k">n</span></button>
<button id="bu">Delete last<span class="k">u</span></button>
<button id="bl">Lighting</button>
</div>
<div class="tips">Yellow = get ready. Start the sign when the frame turns <b style="color:var(--rec)">red</b> and finish before it stops.
Sign the same way each time, but vary position a little and change lighting/clothes across sessions.
For <b>NO_SIGN</b>: rest your hands, scratch your face, adjust your hair, look away. These clips stop random words.</div>
</section>
<aside>
<div class="signer"><input id="signer" placeholder="signer name"><button id="bsg">Set signer</button></div>
<h2>Signs · your clips / target</h2><div id="list"></div>
</aside>
</main>
<script>
const $=id=>document.getElementById(id);let S=null;
function cmd(c,v){fetch('/cmd?c='+c+(v!==undefined?'&v='+encodeURIComponent(v):''),{method:'POST'}).catch(()=>{});}
function render(s){S=s;
 $('who').textContent='signer: '+s.signer+' · session: '+s.session+' · light: '+s.lighting;
 $('fps').textContent=s.fps.toFixed(0)+' FPS';
 $('det').textContent='hands: '+(s.hands||'none')+(s.pose?' · body ✓':' · body ✗');
 const st=$('status');st.className=s.phase;
 st.textContent=s.phase==='countdown'?('Get ready: '+s.sign+'  '+s.remain.toFixed(1)):
   s.phase==='recording'?('● REC '+s.sign+'  '+s.remain.toFixed(1)+(s.burst_left?'  ('+s.burst_left+' more)':'')):
   ('Ready: '+s.sign);
 $('msg').textContent=s.message;
 const busy=s.phase!=='idle';
 ['b1','b5','bp','bn','bsg'].forEach(i=>$(i).disabled=busy);$('bu').disabled=busy||!s.can_undo;
 if(document.activeElement!==$('signer'))$('signer').value=s.signer;
 $('list').innerHTML=s.classes.map(c=>{const pct=Math.min(100,100*c.mine/s.target);
  return '<div class="row'+(c.name===s.sign?' sel':'')+'" data-n="'+c.name+'"><span>'+c.name.replace('_',' ')+'</span>'+
  '<div class="bar"><i style="width:'+pct+'%"></i></div><span class="n">'+c.mine+' / '+s.target+'</span></div>';}).join('');
 document.querySelectorAll('.row').forEach(r=>r.onclick=()=>{if(S.phase==='idle')cmd('select',r.dataset.n);});
}
function connect(){const es=new EventSource('/events');
 es.onopen=()=>{$('conn').textContent='live';$('conn').classList.add('live');};
 es.onmessage=e=>render(JSON.parse(e.data));
 es.onerror=()=>{$('conn').textContent='reconnecting';$('conn').classList.remove('live');setTimeout(()=>{$('video').src='/video?t='+Date.now();},1500);};}
$('b1').onclick=()=>cmd('rec1');$('b5').onclick=()=>cmd('rec5');$('bs').onclick=()=>cmd('stop');
$('bp').onclick=()=>cmd('prev');$('bn').onclick=()=>cmd('next');$('bu').onclick=()=>cmd('undo');$('bl').onclick=()=>cmd('light');
$('bsg').onclick=()=>cmd('signer',$('signer').value);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 const m={' ':'rec1','r':'rec5','s':'stop','u':'undo','n':'next','p':'prev'};if(m[e.key]){e.preventDefault();cmd(m[e.key]);}});
connect();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
