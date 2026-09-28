#!/usr/bin/env python3
"""Device runtime: camera -> landmarks -> sliding window -> TFLite -> debounce -> TTS.

    python pi/run.py                 # Pi Camera via picamera2 if available, else /dev/video0
    python pi/run.py --camera 0      # force a USB webcam
    python pi/run.py --no-tts        # print only

Speech uses Piper (offline). Install per pi/SETUP.md; the runtime shells out to
the `piper` binary and plays through ALSA, so it works with a USB speaker or an
I2S amp without code changes.
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
from s2s.features import FEAT_DIM, clip_to_features, resample  # noqa: E402
from s2s.landmarks import LandmarkExtractor, draw  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"


class Speaker:
    """Non-blocking Piper TTS. One utterance at a time; later ones queue."""

    def __init__(self, voice: Path, enabled: bool = True):
        self.voice, self.enabled = voice, enabled
        self.q: collections.deque[str] = collections.deque()
        self.lock = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str):
        print(f"[say] {text}")
        if not self.enabled:
            return
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
                p = subprocess.Popen(["piper", "--model", str(self.voice), "--output-raw"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                a = subprocess.Popen(["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                                     stdin=p.stdout)
                p.stdin.write(text.encode()); p.stdin.close()
                a.wait()
            except FileNotFoundError:
                print("[say] piper/aplay not installed; see pi/SETUP.md")


class Recognizer:
    """Sliding-window TFLite inference with confidence + stability gating."""

    def __init__(self, model_dir: Path, window: int, min_conf: float, stable_frames: int):
        try:
            from tflite_runtime.interpreter import Interpreter
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


def open_camera(index: int | None, w: int, h: int):
    if index is None:
        try:
            from picamera2 import Picamera2  # type: ignore

            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (w, h), "format": "RGB888"}))
            cam.start()
            return lambda: (True, cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)), cam.stop
        except ImportError:
            index = 0
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap.read, cap.release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="V4L2 index; default = Pi camera")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-conf", type=float, default=0.80)
    ap.add_argument("--stable-frames", type=int, default=6)
    ap.add_argument("--voice", default=str(ROOT / "data" / "models" / "en_US-lessac-medium.onnx"))
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no preview window (systemd)")
    args = ap.parse_args()

    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    rec = Recognizer(MODEL_DIR, cfg["window"], args.min_conf, args.stable_frames)
    spk = Speaker(Path(args.voice), enabled=not args.no_tts)
    ext = LandmarkExtractor()
    read, release = open_camera(args.camera, args.width, args.height)

    sentence: list[str] = []
    fps_t, fps_n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = read()
            if not ok:
                break
            lm = ext.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, conf, emitted = rec.push(lm.as_flat())

            if emitted == "CLEAR":
                sentence.clear()
            elif emitted == "SPEAK":
                if sentence:
                    spk.say(" ".join(sentence)); sentence.clear()
            elif emitted:
                word = vocab.SPOKEN.get(emitted, emitted.lower())
                sentence.append(word)
                spk.say(word)

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

            if not args.headless:
                draw(frame, lm)
                cv2.putText(frame, f"{label} {conf:.2f}  fps {fps:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, " ".join(sentence), (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Signs to Speech", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        release(); ext.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
