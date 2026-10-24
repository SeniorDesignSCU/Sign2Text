#!/usr/bin/env python3
"""Hardware bring-up check. Run this before anything else on a new Pi (or laptop).

    python scripts/check_hw.py --web            # over SSH: watch camera + landmarks in a laptop browser
    python scripts/check_hw.py                  # over SSH with no --web: prints FPS / detections for 20 s
    python scripts/check_hw.py --hands-only     # compare FPS without the pose model
    python scripts/check_hw.py --camera 0       # test the USB webcam (C270) instead of the Pi camera
    python scripts/check_hw.py --web --captions # the caption page with demo words, before any model exists
    python scripts/check_hw.py --speaker        # say a test sentence (later, once a speaker is wired)
    python scripts/check_hw.py --speaker --audio-device plughw:CARD=MAX98357A,DEV=0

What "pass" means (targets from docs/PROJECT_OBJECTIVES.md):
    landmark throughput >= 10 FPS with hand + pose on, and hands/pose detected when you
    stand about an arm's length from the camera with your hands up.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

FPS_TARGET = 10.0
DEFAULT_VOICE = ROOT / "data" / "models" / "en_US-lessac-medium.onnx"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def system_info() -> None:
    print("== system")
    model = Path("/proc/device-tree/model")
    print("  board      :", model.read_text().strip("\x00\n") if model.exists() else platform.machine())
    print("  python     :", platform.python_version())
    try:
        import mediapipe as mp
        print("  mediapipe  :", mp.__version__)
    except ImportError:
        print("  mediapipe  : NOT INSTALLED (pip install -r requirements-pi.txt)")
    if shutil.which("vcgencmd"):
        print("  temperature:", sh(["vcgencmd", "measure_temp"]).replace("temp=", ""))
        thr = sh(["vcgencmd", "get_throttled"]).replace("throttled=", "")
        note = "ok" if thr in ("0x0", "") else "see pi/SETUP.md > Troubleshooting (power or heat)"
        print("  throttled  :", thr or "?", f"({note})")


def check_camera(args) -> int:
    import cv2

    from s2s.camera import open_camera
    from s2s.display import CaptionView, Transcript, has_display
    from s2s.landmarks import HAND_MODEL, POSE_MODEL, LandmarkExtractor, draw

    print("== camera + landmarks")
    missing = [p.name for p in (HAND_MODEL, POSE_MODEL) if not p.exists()]
    if missing:
        print(f"  model files missing: {missing}. Run: bash scripts/download_models.sh")
        return 1

    try:
        cam = open_camera(args.camera, args.width, args.height)
    except (RuntimeError, ValueError) as e:
        print("  FAIL:", e)
        return 1
    print("  camera     :", cam.description)
    ext = LandmarkExtractor(use_pose=not args.hands_only)
    print("  models     :", "hands only" if args.hands_only else "hands + pose")

    # Where the picture goes: a local window if a monitor session exists, a browser page with --web.
    window = not args.headless and has_display()
    if not args.headless and not window and not args.web:
        print("  no monitor session (plain SSH), so no window. Add --web to watch from a laptop browser.")
    web = None
    if args.web:
        from s2s.webview import WebView
        web = WebView(port=args.port)
        print("  browser    : open " + "  or  ".join(web.urls()) + "  (press Show camera)")
    timed = not window and not web          # nothing to look at -> run for --seconds and report
    if args.seconds_set or timed:
        print(f"  running for {args.seconds:.0f} s" + ("" if timed else " (or until q / Ctrl+C)"))
    else:
        print("  stand ~1 m back, raise both hands. Stop with " + ("q in the window or " if window else "") + "Ctrl+C.")

    transcript = Transcript(sentence_timeout=0)
    view = CaptionView(show_preview=not args.no_preview, fullscreen=args.fullscreen) if window and args.captions else None
    demo_words = ["HELLO", "ME", "NAME", "SPEAK", "THANK_YOU", "FRIEND", "SPEAK",
                  "WHERE", "BATHROOM", "PLEASE", "SPEAK", "CLEAR"]
    demo_i, next_demo = 0, time.time() + 1.2
    if args.captions:
        print("  captions   : demo words cycling every 1.2 s (no model involved)")
    status = "hardware check" + (", demo words" if args.captions else "")

    frames = hands = poses = 0
    fps_samples: list[float] = []
    t_start = t_win = time.time()
    n_win, shape = 0, None
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("  camera stopped returning frames")
                break
            shape = frame.shape
            lm = ext.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames += 1
            n_win += 1
            hands += int(lm.left_present or lm.right_present)
            poses += int(lm.pose_present)

            now = time.time()
            if now - t_win >= 1.0:
                fps_samples.append(n_win / (now - t_win))
                n_win, t_win = 0, now
                if not window:
                    flags = ("L" if lm.left_present else "-") + ("R" if lm.right_present else "-") + \
                            ("P" if lm.pose_present else "-")
                    print(f"  {fps_samples[-1]:5.1f} FPS   detected now: {flags}")
            cur = fps_samples[-1] if fps_samples else 0.0

            if args.captions and now >= next_demo:
                transcript.push(demo_words[demo_i % len(demo_words)], now)
                demo_i, next_demo = demo_i + 1, now + 1.2
            demo_label = transcript.last_word.upper() if args.captions and transcript.last_word else ""

            if window or web:
                draw(frame, lm)
            if web:
                if args.captions:
                    for c in web.pop_commands():
                        transcript.clear_all() if c == "clear" else transcript.finish()
                web.update(transcript, demo_label, 0.9 if demo_label else 0.0, cur, status, frame)
            if view:
                key = view.show(view.render(transcript, frame, demo_label, 0.9 if demo_label else 0.0,
                                            cur, status=status))
            elif window:
                color = (0, 200, 0) if cur >= FPS_TARGET else (0, 0, 255)
                cv2.putText(frame, f"{cur:.1f} FPS (target {FPS_TARGET:.0f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("check_hw", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
            if key == ord("q"):
                break
            if key == ord("f") and view:
                view.toggle_fullscreen()
            if (args.seconds_set or timed) and now - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        ext.close()
        if view:
            view.close()
        elif window:
            cv2.destroyAllWindows()
        if web:
            web.close()

    if not frames:
        return 1
    steady = fps_samples[1:] or fps_samples  # skip the first second (model warm-up)
    avg = sum(steady) / len(steady) if steady else 0.0
    h, w = shape[:2]
    print("== result")
    print(f"  frames {frames} at {w}x{h}, average {avg:.1f} FPS -> "
          f"{'PASS' if avg >= FPS_TARGET else 'BELOW TARGET (see SETUP.md > Troubleshooting)'}")
    print(f"  hands seen in {100 * hands / frames:.0f}% of frames, "
          + ("pose off (--hands-only)" if args.hands_only else f"pose in {100 * poses / frames:.0f}%"))
    if abs(w / h - 4 / 3) > 0.02:
        print("  NOTE: frame is not 4:3; record and run at 640x480 so training data matches the Pi")
    if hands == 0:
        print("  NOTE: no hands detected at all. Check lighting, distance, and that the lens film is off.")
    if shutil.which("vcgencmd"):
        print("  temperature:", sh(["vcgencmd", "measure_temp"]).replace("temp=", ""),
              "| throttled:", sh(["vcgencmd", "get_throttled"]).replace("throttled=", ""))
    return 0


def check_speaker(args) -> int:
    print("== speaker")
    if shutil.which("aplay"):
        cards = [l for l in sh(["aplay", "-l"]).splitlines() if l.startswith("card")]
        print("  ALSA cards :")
        for c in cards or ["  (none found)"]:
            print("   ", c)
    else:
        print("  aplay not found: sudo apt install alsa-utils")
        return 1

    voice = Path(args.voice)
    text = "Hello. This is Signs to Speech. If you can hear this, the speaker works."
    if shutil.which("piper") and voice.exists():
        print(f"  speaking through '{args.audio_device}' with Piper ({voice.name}) ...")
        rate = "22050"
        cfg = Path(str(voice) + ".json")
        if cfg.exists():
            import json
            rate = str(json.loads(cfg.read_text()).get("audio", {}).get("sample_rate", 22050))
        p = subprocess.Popen(["piper", "--model", str(voice), "--output-raw", "--volume", str(args.volume)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        a = subprocess.Popen(["aplay", "-D", args.audio_device, "-r", rate, "-f", "S16_LE", "-c", "1",
                              "-t", "raw"], stdin=p.stdout)
        p.stdin.write(text.encode())
        p.stdin.close()
        rc = a.wait()
        print("  done" if rc == 0 else f"  aplay failed (exit {rc}); check --audio-device against the card list above")
        return rc
    print("  Piper or its voice is not installed yet; playing ALSA's built-in test voice instead.")
    print("  (Install Piper per pi/SETUP.md step 6 to test real speech.)")
    return subprocess.call(["speaker-test", "-D", args.audio_device, "-c", "2", "-t", "wav", "-l", "1"])


def main() -> int:
    from s2s.camera import add_camera_args

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(ap)
    ap.add_argument("--headless", action="store_true", help="never open a local window")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this many seconds (default 20 when there's nothing to watch)")
    ap.add_argument("--web", action="store_true", help="serve the live view to a laptop browser")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hands-only", action="store_true", help="skip the pose model")
    ap.add_argument("--captions", action="store_true", help="show the caption screen with demo words")
    ap.add_argument("--fullscreen", action="store_true", help="local window with --captions: fill the display")
    ap.add_argument("--no-preview", action="store_true", help="local window with --captions: text only")
    ap.add_argument("--speaker", action="store_true", help="test audio output instead of the camera")
    ap.add_argument("--audio-device", default="default", help="ALSA device for aplay")
    ap.add_argument("--voice", default=str(DEFAULT_VOICE))
    ap.add_argument("--volume", type=float, default=1.0)
    args = ap.parse_args()
    args.seconds_set = args.seconds is not None
    if not args.seconds_set:
        args.seconds = 20.0

    system_info()
    return check_speaker(args) if args.speaker else check_camera(args)


if __name__ == "__main__":
    sys.exit(main())
