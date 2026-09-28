"""MediaPipe Tasks wrapper: one frame in -> hand + upper-body pose landmarks out.

Uses the MediaPipe Tasks API (mediapipe >= 0.10), which is what Google's
Raspberry Pi examples use, so the same code runs on a laptop and on the Pi 5.

Model files are downloaded by scripts/download_models.sh into data/models/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Repo root = two levels above this file (common/s2s/landmarks.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "data" / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"

N_HAND_POINTS = 21
# Upper-body subset of MediaPipe's 33 pose points: nose, shoulders, elbows, wrists.
POSE_SUBSET = [0, 11, 12, 13, 14, 15, 16]
N_POSE_POINTS = len(POSE_SUBSET)

# Indices within POSE_SUBSET (used by features.py for normalization)
POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER = 0, 1, 2


@dataclass
class FrameLandmarks:
    """Landmarks for one frame. Missing hands are all-zero with present=False.

    left/right: (21, 3) normalized image coords (x, y in [0,1], z relative).
    pose:       (7, 3) upper-body subset, same coordinate convention.
    """

    t_ms: int
    left: np.ndarray = field(default_factory=lambda: np.zeros((N_HAND_POINTS, 3), np.float32))
    right: np.ndarray = field(default_factory=lambda: np.zeros((N_HAND_POINTS, 3), np.float32))
    pose: np.ndarray = field(default_factory=lambda: np.zeros((N_POSE_POINTS, 3), np.float32))
    left_present: bool = False
    right_present: bool = False
    pose_present: bool = False

    def as_flat(self) -> np.ndarray:
        """Raw (not yet normalized) flat vector, used for storage."""
        return np.concatenate(
            [
                self.left.ravel(),
                self.right.ravel(),
                self.pose.ravel(),
                np.array([self.left_present, self.right_present, self.pose_present], np.float32),
            ]
        )


RAW_DIM = N_HAND_POINTS * 3 * 2 + N_POSE_POINTS * 3 + 3  # 150


class LandmarkExtractor:
    """Runs Hand Landmarker + Pose Landmarker in VIDEO mode."""

    def __init__(self, num_hands: int = 2, use_pose: bool = True):
        if not HAND_MODEL.exists() or (use_pose and not POSE_MODEL.exists()):
            raise FileNotFoundError(
                f"Model files missing in {MODEL_DIR}. Run scripts/download_models.sh first."
            )

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

        self.pose = None
        if use_pose:
            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        self._t0 = time.monotonic()
        self._last_ts = -1

    def _timestamp_ms(self) -> int:
        ts = int((time.monotonic() - self._t0) * 1000)
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, frame_rgb: np.ndarray) -> FrameLandmarks:
        """frame_rgb: HxWx3 uint8 RGB image."""
        ts = self._timestamp_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        out = FrameLandmarks(t_ms=ts)

        hres = self.hands.detect_for_video(mp_image, ts)
        for lms, handedness in zip(hres.hand_landmarks, hres.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            # MediaPipe labels handedness assuming a mirrored (selfie) image.
            # We do NOT flip frames, so "Left" from MediaPipe is the signer's right hand.
            label = handedness[0].category_name
            if label == "Left":
                out.right, out.right_present = arr, True
            else:
                out.left, out.left_present = arr, True

        if self.pose is not None:
            pres = self.pose.detect_for_video(mp_image, ts)
            if pres.pose_landmarks:
                full = pres.pose_landmarks[0]
                out.pose = np.array([[full[i].x, full[i].y, full[i].z] for i in POSE_SUBSET], np.float32)
                out.pose_present = True

        return out

    def close(self):
        self.hands.close()
        if self.pose is not None:
            self.pose.close()


# Hand connections for drawing (subset of mediapipe's HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_CONNECTIONS = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]  # indices into POSE_SUBSET


def draw(frame_bgr: np.ndarray, lm: FrameLandmarks) -> np.ndarray:
    """Draw landmarks on a BGR frame in place (debug/preview only)."""
    import cv2

    h, w = frame_bgr.shape[:2]

    def pt(p):
        return int(p[0] * w), int(p[1] * h)

    for hand, present, color in ((lm.left, lm.left_present, (0, 200, 255)),
                                 (lm.right, lm.right_present, (255, 120, 0))):
        if not present:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pt(hand[a]), pt(hand[b]), color, 2)
        for p in hand:
            cv2.circle(frame_bgr, pt(p), 3, (255, 255, 255), -1)

    if lm.pose_present:
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame_bgr, pt(lm.pose[a]), pt(lm.pose[b]), (0, 255, 0), 2)
        for p in lm.pose:
            cv2.circle(frame_bgr, pt(p), 4, (0, 255, 0), -1)
    return frame_bgr
