"""Clip storage. One .npz per recorded sign, plus a metadata index.

data/clips/<sign>/<signer>_<session>_<nnn>.npz
    raw      : (T, RAW_DIM) float32   -- see landmarks.FrameLandmarks.as_flat
    t_ms     : (T,) int64             -- capture timestamps
    meta     : JSON string            -- sign, signer, session, fps, lighting, notes

The raw (un-normalized) landmarks are what we store, so feature engineering
can change later without re-recording anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIP_DIR = REPO_ROOT / "data" / "clips"


@dataclass
class ClipMeta:
    sign: str
    signer: str
    session: str
    fps: float
    n_frames: int
    lighting: str = ""
    handedness: str = "right"  # signer's dominant hand
    source: str = "webcam"     # webcam | picam | wlasl | ...
    notes: str = ""


def next_index(sign: str, signer: str, session: str) -> int:
    d = CLIP_DIR / sign
    if not d.exists():
        return 0
    prefix = f"{signer}_{session}_"
    return len([p for p in d.glob("*.npz") if p.name.startswith(prefix)])


def save_clip(raw: np.ndarray, t_ms: np.ndarray, meta: ClipMeta) -> Path:
    d = CLIP_DIR / meta.sign
    d.mkdir(parents=True, exist_ok=True)
    idx = next_index(meta.sign, meta.signer, meta.session)
    path = d / f"{meta.signer}_{meta.session}_{idx:03d}.npz"
    np.savez_compressed(path, raw=raw.astype(np.float32), t_ms=np.asarray(t_ms, np.int64),
                        meta=json.dumps(asdict(meta)))
    return path


def load_clip(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["raw"], z["t_ms"], ClipMeta(**json.loads(str(z["meta"])))


def iter_clips(signs: list[str] | None = None):
    """Yield (path, raw, t_ms, meta) for every clip, optionally filtered by sign."""
    for sign_dir in sorted(CLIP_DIR.iterdir()) if CLIP_DIR.exists() else []:
        if not sign_dir.is_dir():
            continue
        if signs is not None and sign_dir.name not in signs:
            continue
        for p in sorted(sign_dir.glob("*.npz")):
            raw, t, meta = load_clip(p)
            yield p, raw, t, meta


def dataset_summary() -> dict[str, dict[str, int]]:
    """{sign: {signer: count}} for a quick look at class balance."""
    out: dict[str, dict[str, int]] = {}
    for _, _, _, meta in iter_clips():
        out.setdefault(meta.sign, {}).setdefault(meta.signer, 0)
        out[meta.sign][meta.signer] += 1
    return out
