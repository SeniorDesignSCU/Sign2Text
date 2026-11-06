"""From a live landmark stream to words. Shared by training/train.py and pi/run.py.

On the Pi, the model re-scores the last `clip_seconds` of landmarks on every frame. Most
of those windows are not a clean, centred sign like the recorded clips: the next sign is
only starting, the last one is leaving, or the hands are resting. Two pieces deal with that:

- `splice_windows()` builds training windows that look like those in-between moments, by
  splicing idle (NO_SIGN) footage onto part of a sign clip. A window that shows only a
  small part of the sign is labelled NO_SIGN; one that shows most of it keeps the sign's
  label. Without these, the model gives confident nonsense on half-seen signs.
- `WordDecoder` turns the per-frame guesses into words. train.py replays held-out clips
  through it to report a live-style score, so the rules there and on the Pi are the same.
"""

from __future__ import annotations

import collections

import numpy as np

from . import vocab
from .landmarks import RAW_DIM

# Presence flags sit at the end of each raw frame: left hand, right hand, pose.
_PRES_L, _PRES_R = RAW_DIM - 3, RAW_DIM - 2


def active_span(raw: np.ndarray) -> tuple[int, int]:
    """Frames [a, b) from the first to the last frame with a hand in view.
    Falls back to the whole clip if hands are seen in fewer than 3 frames."""
    hands = (raw[:, _PRES_L] > 0.5) | (raw[:, _PRES_R] > 0.5)
    idx = np.flatnonzero(hands)
    if len(idx) < 3:
        return 0, len(raw)
    return int(idx[0]), int(idx[-1]) + 1


COOLDOWN_FRACTION = 0.5  # default cooldown = this x clip_seconds


def splice_windows(sign_raw: np.ndarray, idle_raw: np.ndarray, rng: np.random.Generator,
                   n_pos: int = 1, pos_min: float = 0.8,
                   neg_max: float = 0.3) -> list[tuple[np.ndarray, bool]]:
    """Windows of len(sign_raw) frames cut from  idle | sign | idle.

    idle_raw must have at least 2 * len(sign_raw) frames. Returns (window, is_sign) pairs:
    - n_pos windows with >= pos_min of the sign's active part, shifted off-centre (is_sign);
    - one window showing only the start of the sign and one showing only its end, each
      <= neg_max of the active part (not is_sign: these should read as NO_SIGN).
    """
    L = len(sign_raw)
    a, b = active_span(sign_raw)
    stream = np.concatenate([idle_raw[:L], sign_raw, idle_raw[L:2 * L]])
    starts = np.arange(0, len(stream) - L + 1)
    overlap = np.clip(np.minimum(starts + L, L + b) - np.maximum(starts, L + a), 0, None) / (b - a)
    sliver = (overlap > 0.05) & (overlap <= neg_max)
    groups = [
        (starts[(overlap >= pos_min) & (starts != L)], n_pos, True),
        (starts[sliver & (starts < L)], 1, False),   # sign just coming into the window
        (starts[sliver & (starts > L)], 1, False),   # sign's tail leaving the window
    ]
    out: list[tuple[np.ndarray, bool]] = []
    for pool, n, is_sign in groups:
        if len(pool):
            for s in rng.choice(pool, size=min(n, len(pool)), replace=False):
                out.append((stream[s:s + L], is_sign))
    return out


class WordDecoder:
    """Per-frame (class index, confidence) -> emitted words.

    A word is emitted when the same sign has been the top guess, at >= min_conf, for
    `stable_frames` frames in a row. The same word is not repeated until NO_SIGN has been
    the top guess for `rearm` seconds. After a word, a different word must wait
    `cooldown` seconds, because the window still holds the tail of the sign just emitted.
    """

    def __init__(self, labels: list[str], min_conf: float = 0.8, stable_frames: int = 6,
                 cooldown: float = 1.2, rearm: float = 0.6):
        self.labels = labels
        self.min_conf = min_conf
        self.cooldown = cooldown
        self.rearm = rearm
        self.history: collections.deque[int] = collections.deque(maxlen=stable_frames)
        self.last_word: str | None = None
        self.last_emit_t = -1e9
        self.idle_start: float | None = None

    def update(self, k: int, conf: float, now: float) -> str | None:
        label = self.labels[k]
        self.history.append(k if conf >= self.min_conf else -1)
        if label == vocab.NO_SIGN:
            if self.idle_start is None:
                self.idle_start = now
            if now - self.idle_start >= self.rearm:
                self.last_word = None
            return None
        self.idle_start = None
        stable = len(self.history) == self.history.maxlen and all(h == k for h in self.history)
        if not stable or label == self.last_word or now - self.last_emit_t < self.cooldown:
            return None
        self.last_word = label
        self.last_emit_t = now
        return label


def align_words(truth: list[str], said: list[str]) -> dict[str, int]:
    """Edit-distance alignment of the signed words vs the emitted words."""
    n, m = len(truth), len(said)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (truth[i - 1] != said[j - 1]))
    res = {"right": 0, "wrong": 0, "missed": 0, "extra": 0}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (truth[i - 1] != said[j - 1]):
            res["right" if truth[i - 1] == said[j - 1] else "wrong"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            res["missed"] += 1
            i -= 1
        else:
            res["extra"] += 1
            j -= 1
    return res
