"""Feature engineering: raw landmark sequences -> model input.

This is the project's main departure from single-frame fingerspelling systems.
Each frame is normalized relative to the signer's body (not the hand's own
bounding box) so that *where* a sign happens is preserved, and per-frame
velocities are appended so that *how* it moves is preserved.

Frame feature layout (FEAT_DIM = 2*63 + 21 + 3 = 150 positional + 150 velocity = 300):
    left hand   21 x 3  body-normalized coords
    right hand  21 x 3
    pose         7 x 3
    presence     3       (left, right, pose)
    velocity     same 150 values, first difference in time
"""

from __future__ import annotations

import numpy as np

from .landmarks import (
    N_HAND_POINTS,
    N_POSE_POINTS,
    POSE_L_SHOULDER,
    POSE_R_SHOULDER,
    RAW_DIM,
)

POS_DIM = RAW_DIM  # 150
FEAT_DIM = POS_DIM * 2  # 300

_EPS = 1e-6


def _split_raw(raw: np.ndarray):
    """raw: (T, RAW_DIM) -> left (T,21,3), right (T,21,3), pose (T,7,3), presence (T,3)"""
    i = 0
    left = raw[:, i:i + N_HAND_POINTS * 3].reshape(-1, N_HAND_POINTS, 3); i += N_HAND_POINTS * 3
    right = raw[:, i:i + N_HAND_POINTS * 3].reshape(-1, N_HAND_POINTS, 3); i += N_HAND_POINTS * 3
    pose = raw[:, i:i + N_POSE_POINTS * 3].reshape(-1, N_POSE_POINTS, 3); i += N_POSE_POINTS * 3
    presence = raw[:, i:i + 3]
    return left, right, pose, presence


def body_normalize(raw: np.ndarray) -> np.ndarray:
    """Translate to shoulder midpoint and scale by shoulder width, per frame.

    Falls back to the median over the clip when pose is missing in a frame,
    and to identity (no normalization) if pose is never seen.
    Input/output: (T, RAW_DIM).
    """
    raw = raw.astype(np.float32)
    left, right, pose, presence = _split_raw(raw)
    T = raw.shape[0]

    ls = pose[:, POSE_L_SHOULDER, :2]
    rs = pose[:, POSE_R_SHOULDER, :2]
    center = (ls + rs) / 2.0                       # (T, 2)
    width = np.linalg.norm(ls - rs, axis=1)        # (T,)
    pose_ok = presence[:, 2] > 0.5

    if pose_ok.any():
        med_center = np.median(center[pose_ok], axis=0)
        med_width = float(np.median(width[pose_ok]))
        center = np.where(pose_ok[:, None], center, med_center)
        width = np.where(pose_ok, width, med_width)
    else:
        center = np.full((T, 2), 0.5, np.float32)
        width = np.ones(T, np.float32)
    width = np.maximum(width, _EPS)

    out_parts = []
    for block, pres in ((left, presence[:, 0]), (right, presence[:, 1]), (pose, presence[:, 2])):
        b = block.copy()
        b[:, :, :2] = (b[:, :, :2] - center[:, None, :]) / width[:, None, None]
        b[:, :, 2] = b[:, :, 2] / width[:, None]
        b[pres < 0.5] = 0.0  # keep missing parts at exactly zero
        out_parts.append(b.reshape(T, -1))
    out_parts.append(presence)
    return np.concatenate(out_parts, axis=1).astype(np.float32)


def add_velocity(pos: np.ndarray) -> np.ndarray:
    """(T, POS_DIM) -> (T, FEAT_DIM) by appending first differences."""
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel], axis=1)


def clip_to_features(raw: np.ndarray) -> np.ndarray:
    """Full pipeline for one clip: (T, RAW_DIM) -> (T, FEAT_DIM)."""
    return add_velocity(body_normalize(raw))


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (length, D)."""
    T = seq.shape[0]
    if T == length:
        return seq
    if T == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0, T - 1, num=length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def mirror_lr(raw: np.ndarray) -> np.ndarray:
    """Augmentation: swap left/right hands and flip x. Turns a right-handed
    signer's clip into a plausible left-handed one."""
    left, right, pose, presence = _split_raw(raw.copy())
    for blk in (left, right, pose):
        blk[:, :, 0] = 1.0 - blk[:, :, 0]
    # swap shoulders/elbows/wrists in the pose subset: [nose, Ls, Rs, Le, Re, Lw, Rw]
    pose = pose[:, [0, 2, 1, 4, 3, 6, 5], :]
    T = raw.shape[0]
    pres = presence[:, [1, 0, 2]]
    return np.concatenate([right.reshape(T, -1), left.reshape(T, -1), pose.reshape(T, -1), pres], axis=1)
