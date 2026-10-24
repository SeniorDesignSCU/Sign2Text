#!/usr/bin/env python3
"""Device runtime: camera -> landmarks -> sliding window -> TFLite -> debounce -> captions (+ speech).

    python pi/run.py                  # captions served to a laptop browser: http://s2s.local:8000
    python pi/run.py --camera 0       # use the USB webcam (C270)
    python pi/run.py --speak          # also speak words (needs speaker + Piper, see pi/SETUP.md)
    python pi/run.py --no-web         # no browser page; window (if a monitor is attached) / console

The output today is on-screen text in a laptop browser; the laptop is only a screen, all
recognition runs on the Pi. Speech is opt-in with --speak until the speaker hardware is added.
If a monitor session exists (e.g. on a laptop during development) a local window opens too;
keys there: q quit, c clear, space finish the line, f fullscreen. --no-preview / --fullscreen
change that window.

Needs a trained model in data/models/ (training/train.py). Before that exists, use
scripts/check_hw.py to test the camera, landmarks, FPS and speaker.

Speech uses Piper (offline). The runtime shells out to the `piper` command and plays
through ALSA's `aplay`, so the same code drives the MAX98357A I2S amp, a USB speaker,
or HDMI audio; pick the output with --audio-device (see pi/SETUP.md).
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from s2s import vocab  # noqa: E402
from s2s.camera import add_camera_args, open_camera  # noqa: E402
from s2s.display import CaptionView, Transcript, has_display  # noqa: E402
from s2s.webview import WebView  # noqa: E402
from s2s.features import FEAT_DIM, clip_to_features, resample  # noqa: E402
from s2s.landmarks import LandmarkExtractor, draw  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"


class Speaker:
    """Non-blocking Piper TTS. One utterance at a time; later ones queue."""

    def __init__(self, voice: Path, device: str = "default", volume: float = 1.0):
        self.voice, self.device, self.volume = voice, device, volume
        self.rate = 22050
        cfg = Path(str(voice) + ".json")
        if cfg.exists():
            self.rate = int(json.loads(cfg.read_text()).get("audio", {}).get("sample_rate", 22050))
        self.q: collections.deque[str] = collections.deque()
        self.lock = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str):
        with self.lock:
            self.q.append(text)
            self.lock.notify()

    def _loop(self):
        while True:
            with self.lock:
                while not self.q:
                    self.lock.wait()
                text = self.q.popleft()
            try:
                p = subprocess.Popen(["piper", "--model", str(self.voice), "--output-raw",
                                      "--volume", str(self.volume)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
                a = subprocess.Popen(["aplay", "-D", self.device, "-r", str(self.rate),
                                      "-f", "S16_LE", "-c", "1", "-t", "raw", "-q"],
                                     stdin=p.stdout)
                p.stdin.write(text.encode()); p.stdin.close()
                a.wait()
            except FileNotFoundError:
                print("[say] piper/aplay not installed; see pi/SETUP.md")


class Recognizer:
    """Sliding-window TFLite inference with confidence + stability gating."""

    def __init__(self, model_dir: Path, window: int, min_conf: float, stable_frames: int):
        try:
            from ai_edge_litert.interpreter import Interpreter  # current name of TFLite runtime
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter  # older installs
            except ImportError:
                import tensorflow as tf  # laptop fallback
                Interpreter = tf.lite.Interpreter
        self.interp = Interpreter(model_path=str(model_dir / "sign_classifier.tflite"))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.labels = (model_dir / "labels.txt").read_text().split()
        self.window = window
        self.min_conf = min_conf
        self.stable_frames = stable_frames
        self.raw_buf: collections.deque[np.ndarray] = collections.deque(maxlen=int(window * 1.5))
        self.history: collections.deque[int] = collections.deque(maxlen=stable_frames)
        self.last_emitted: str | None = None
        self.idle_since = time.time()

    def push(self, raw_frame: np.ndarray) -> tuple[str, float, str | None]:
        """Returns (top_label, confidence, emitted_label_or_None)."""
        self.raw_buf.append(raw_frame)
        if len(self.raw_buf) < self.window // 2:
            return vocab.NO_SIGN, 0.0, None
        feats = resample(clip_to_features(np.stack(self.raw_buf)), self.window)
        self.interp.set_tensor(self.inp["index"], feats[None].astype(np.float32))
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out["index"])[0]
        k = int(probs.argmax()); conf = float(probs[k]); label = self.labels[k]

        self.history.append(k if conf >= self.min_conf else -1)
        emitted = None
        if label == vocab.NO_SIGN:
            # Idle gap resets "already said this" so a repeated sign can be re-emitted.
            if time.time() - self.idle_since > 0.6:
                self.last_emitted = None
        else:
            self.idle_since = time.time()
            stable = len(self.history) == self.stable_frames and all(h == k for h in self.history)
            if stable and label != self.last_emitted:
                emitted = label
                self.last_emitted = label
        return label, conf, emitted


def main():
    ap = argparse.ArgumentParser()
    add_camera_args(ap)
    ap.add_argument("--min-conf", type=float, default=0.80)
    ap.add_argument("--stable-frames", type=int, default=6)
    ap.add_argument("--sentence-timeout", type=float, default=5.0,
                    help="seconds without a new word before the line is finished (0 = never)")
    ap.add_argument("--fullscreen", action="store_true", help="fill the attached display")
    ap.add_argument("--no-preview", action="store_true", help="captions only, no camera image")
    ap.add_argument("--headless", action="store_true", help="no local window (the web page still works)")
    ap.add_argument("--no-web", action="store_true", help="don't serve the caption page")
    ap.add_argument("--port", type=int, default=8000, help="caption page port")
    ap.add_argument("--host", default="0.0.0.0", help="127.0.0.1 = only viewable on this machine")
    ap.add_argument("--speak", action="store_true", help="also speak words through the speaker")
    ap.add_argument("--voice", default=str(ROOT / "data" / "models" / "en_US-lessac-medium.onnx"))
    ap.add_argument("--audio-device", default="default",
                    help="ALSA output for aplay, e.g. default or plughw:CARD=MAX98357A,DEV=0")
    ap.add_argument("--volume", type=float, default=1.0, help="speech volume multiplier")
    args = ap.parse_args()

    cfg_path = MODEL_DIR / "model_config.json"
    if not cfg_path.exists():
        sys.exit("No trained model yet (data/models/model_config.json missing). Collect clips, run "
                 "training/train.py, copy data/models/ to the Pi. To test hardware now, run "
                 "scripts/check_hw.py instead.")
    cfg = json.loads(cfg_path.read_text())
    rec = Recognizer(MODEL_DIR, cfg["window"], args.min_conf, args.stable_frames)
    spk = Speaker(Path(args.voice), device=args.audio_device, volume=args.volume) if args.speak else None
    ext = LandmarkExtractor()
    cam = open_camera(args.camera, args.width, args.height)
    print(f"camera: {cam.description}")

    transcript = Transcript(sentence_timeout=args.sentence_timeout)
    window_ok = not args.headless and has_display()
    view = CaptionView(show_preview=not args.no_preview, fullscreen=args.fullscreen) if window_ok else None
    web = None if args.no_web else WebView(port=args.port, host=args.host)
    if web:
        print("captions: open " + "  or  ".join(web.urls()) + "  in a browser on the same network")
    if not args.headless and not window_ok:
        print("no monitor session (SSH?), so no local window")
    print("speech: " + ("on" if spk else "off (use --speak once the speaker is wired)"))
    status = "speech on" if spk else "speech off"
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            lm = ext.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, conf, emitted = rec.push(lm.as_flat())

            n_words = len(transcript.current)
            finished = transcript.push(emitted)
            if len(transcript.current) > n_words:          # a new word was added
                print(f"[word] {transcript.last_word}")
                if spk:
                    spk.say(transcript.last_word)
            if finished:
                print(f"[line] {finished}")
                if spk and emitted == "SPEAK":
                    spk.say(finished)

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

            if web:
                for c in web.pop_commands():
                    if c == "clear":
                        transcript.clear_all()
                    elif c == "finish" and (done := transcript.finish()):
                        print(f"[line] {done}")
            if web or (view and view.show_preview):
                draw(frame, lm)
            if web:
                web.update(transcript, label, conf, fps, status, frame)
            if view:
                canvas = view.render(transcript, frame, label, conf, fps, status=status)
                key = view.show(canvas)
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    transcript.clear_all()
                elif key == ord(" "):
                    done = transcript.finish()
                    if done:
                        print(f"[line] {done}")
                elif key == ord("f"):
                    view.toggle_fullscreen()
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        ext.close()
        if view:
            view.close()
        if web:
            web.close()


if __name__ == "__main__":
    main()
