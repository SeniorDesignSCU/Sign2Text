"""One way to open a camera for every script (collect, run, check_hw).

    --camera auto   Pi Camera Module via picamera2 if one is connected, else USB webcam 0
    --camera pi     Pi Camera Module (CSI ribbon cable) via picamera2
    --camera 0      USB webcam by index (the Logitech C270, a laptop's built-in camera)
    --camera x.mp4  a recorded video file (for replaying a test session)

The Pi Camera Module 3 is a libcamera device, so plain cv2.VideoCapture(0) does not
open it on a Pi 5; picamera2 is required. That is the main reason this module exists.

open_camera() returns a Camera whose read() gives (ok, frame_bgr), the same contract
as cv2.VideoCapture.read(), so callers do not care which source they got.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Camera:
    read: Callable[[], tuple[bool, np.ndarray | None]]
    release: Callable[[], None]
    source: str        # "picam" | "webcam" | "file"
    description: str   # human-readable, printed at startup


def _pi_camera_available() -> bool:
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError:
        return False
    try:
        return len(Picamera2.global_camera_info()) > 0
    except Exception:
        return False


def _open_picam(width: int, height: int) -> Camera:
    from picamera2 import Picamera2  # type: ignore

    cam = Picamera2()
    # "RGB888" in picamera2/libcamera naming is stored B,G,R in memory, i.e. it is
    # already what OpenCV calls BGR. No colour conversion needed.
    cfg = cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    info = Picamera2.global_camera_info()[0]

    def read():
        return True, cam.capture_array()

    def release():
        cam.stop()
        cam.close()

    return Camera(read, release, "picam", f"Pi camera ({info.get('Model', 'unknown')}) {width}x{height}")


def _open_cv(spec: int | str, width: int, height: int, source: str) -> Camera:
    import cv2

    cap = cv2.VideoCapture(spec)
    if source == "webcam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        hint = " (is the webcam plugged in? try --camera 1)" if source == "webcam" else ""
        raise RuntimeError(f"could not open camera {spec!r}{hint}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    label = f"USB webcam {spec}" if source == "webcam" else f"video file {spec}"
    return Camera(cap.read, cap.release, source, f"{label} {w}x{h}")


def open_camera(spec: str = "auto", width: int = 640, height: int = 480) -> Camera:
    spec = str(spec).strip()
    if spec == "auto":
        spec = "pi" if _pi_camera_available() else "0"
    if spec == "pi":
        if not _pi_camera_available():
            raise RuntimeError(
                "no Pi camera found. Check the ribbon cable (power off, reseat, flip it at the "
                "Pi end if needed) and run `rpicam-hello --list-cameras`."
            )
        return _open_picam(width, height)
    if spec.isdigit():
        return _open_cv(int(spec), width, height, "webcam")
    if Path(spec).exists():
        return _open_cv(spec, width, height, "file")
    raise ValueError(f"--camera must be auto, pi, a webcam index, or a video file path (got {spec!r})")


def add_camera_args(ap, width: int = 640, height: int = 480) -> None:
    """Shared argparse flags so every script takes the same camera options."""
    ap.add_argument("--camera", default="auto",
                    help="auto (default) | pi | webcam index like 0 | path to a video file")
    ap.add_argument("--width", type=int, default=width)
    ap.add_argument("--height", type=int, default=height)
