#!/usr/bin/env python3
"""Laptop / Pi data-collection tool.

Shows the webcam with landmarks drawn, and records labeled clips of one sign
at a time. Runs anywhere MediaPipe runs (laptop with a webcam, or the Pi).

    python scripts/collect.py --signer aditya --session s01 [--camera 0]

Keys:
    SPACE   start a countdown, then record one clip of the current sign
    n / p   next / previous sign in the vocabulary
    r       repeat: record 5 clips of the current sign back to back
    l       cycle lighting tag (bright / office / dim)
    q       quit

Each clip is saved under data/clips/<SIGN>/ as an .npz (see common/s2s/clips.py).
Recording auto-stops after --max-seconds, or press SPACE again to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from s2s import vocab  # noqa: E402
from s2s.clips import ClipMeta, dataset_summary, save_clip  # noqa: E402
from s2s.landmarks import LandmarkExtractor, draw  # noqa: E402

LIGHTING = ["bright", "office", "dim"]


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True, help="short id, e.g. aditya")
    ap.add_argument("--session", default=time.strftime("%Y%m%d"), help="session tag")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-seconds", type=float, default=2.5)
    ap.add_argument("--countdown", type=float, default=1.5)
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--no-pose", action="store_true", help="hands only (faster)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    ext = LandmarkExtractor(use_pose=not args.no_pose)
    classes = vocab.ALL_CLASSES
    ci = 1  # start on first real sign, not NO_SIGN
    lighting_i = 1

    state = "idle"          # idle | countdown | recording
    state_t0 = 0.0
    buf_raw: list[np.ndarray] = []
    buf_t: list[int] = []
    queue = 0               # clips left in a repeat burst
    last_saved = ""
    fps_t, fps_n, fps = time.time(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = ext.process(rgb)
        draw(frame, lm)

        now = time.time()
        fps_n += 1
        if now - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

        sign = classes[ci]
        counts = dataset_summary().get(sign, {})
        n_mine = counts.get(args.signer, 0)

        if state == "countdown":
            remaining = args.countdown - (now - state_t0)
            put(frame, f"GET READY: {sign}  {remaining:.1f}", 80, (0, 255, 255), 1.0)
            if remaining <= 0:
                state, state_t0 = "recording", now
                buf_raw, buf_t = [], []
        elif state == "recording":
            buf_raw.append(lm.as_flat())
            buf_t.append(lm.t_ms)
            elapsed = now - state_t0
            put(frame, f"REC {sign}  {elapsed:.1f}s", 80, (0, 0, 255), 1.0)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            if elapsed >= args.max_seconds:
                state = "save"

        if state == "save":
            if len(buf_raw) >= 5:
                raw = np.stack(buf_raw)
                dur = (buf_t[-1] - buf_t[0]) / 1000.0 or 1e-3
                meta = ClipMeta(sign=sign, signer=args.signer, session=args.session,
                                fps=len(buf_raw) / dur, n_frames=len(buf_raw),
                                lighting=LIGHTING[lighting_i], handedness=args.handedness)
                p = save_clip(raw, np.array(buf_t), meta)
                last_saved = f"saved {p.name} ({len(buf_raw)} frames)"
            else:
                last_saved = "clip too short, discarded"
            if queue > 0:
                queue -= 1
                state, state_t0 = "countdown", now
            else:
                state = "idle"

        # HUD
        put(frame, f"[{ci}/{len(classes)-1}] {sign}   mine:{n_mine}  total:{sum(counts.values())}", 30)
        put(frame, f"signer:{args.signer} session:{args.session} light:{LIGHTING[lighting_i]} fps:{fps:.0f}", 55, (200, 200, 200), 0.55)
        put(frame, "SPACE rec | n/p sign | r x5 | l light | q quit", frame.shape[0] - 40, (200, 200, 200), 0.55)
        if last_saved:
            put(frame, last_saved, frame.shape[0] - 15, (0, 255, 0), 0.55)
        hands = ("L" if lm.left_present else "-") + ("R" if lm.right_present else "-") + ("P" if lm.pose_present else "-")
        put(frame, hands, frame.shape[1] - 80, (255, 255, 0))

        cv2.imshow("Signs to Speech - collect", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            if state == "idle":
                state, state_t0 = "countdown", now
            elif state == "recording":
                state = "save"
        elif k == ord("r") and state == "idle":
            queue, state, state_t0 = 4, "countdown", now
        elif k == ord("n") and state == "idle":
            ci = (ci + 1) % len(classes)
        elif k == ord("p") and state == "idle":
            ci = (ci - 1) % len(classes)
        elif k == ord("l"):
            lighting_i = (lighting_i + 1) % len(LIGHTING)

    cap.release()
    ext.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
