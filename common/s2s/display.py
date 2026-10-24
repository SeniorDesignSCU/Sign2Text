"""On-screen output for a window on an attached display (laptop dev runs, or a Pi desktop).

The demo screen is a laptop browser (see webview.py); this module also holds the Transcript
(sentence logic) that both use. Two pieces:
    Transcript   - turns recognized signs into sentences (CLEAR, SPEAK, auto line-break on pause)
    CaptionView  - draws a caption screen: camera preview + big text, or text only

The screen is rendered at a fixed canvas size (default 1280x720) and the window scales it to
whatever display is attached (HDMI monitor, small HDMI screen, Touch Display), keeping the
aspect ratio. Pure OpenCV, so it runs the same on a laptop and on the Pi.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import vocab

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, GREY, DIM = (255, 255, 255), (170, 170, 170), (110, 110, 110)
YELLOW, GREEN, RED = (0, 220, 255), (90, 200, 90), (80, 80, 230)
BG, PANEL = (18, 18, 18), (34, 34, 34)


def has_display() -> bool:
    """False on a Pi reached over plain SSH (no monitor session), where a window would crash."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


@dataclass
class Transcript:
    """Sentence buffer driven by recognized sign labels.

    - a word sign appends its text to the current line
    - CLEAR empties the current line
    - SPEAK (or a pause longer than sentence_timeout) finishes the line and moves it to history
    """

    sentence_timeout: float = 5.0   # seconds without a new word -> finish the line (0 = never)
    max_history: int = 3
    current: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    last_word: str = ""
    last_word_t: float = 0.0

    def push(self, emitted: str | None, now: float | None = None) -> str | None:
        """Feed one emitted label (or None). Returns a finished sentence when one completes."""
        now = time.time() if now is None else now
        finished = None
        if emitted == "CLEAR":
            self.current.clear()
            self.last_word = ""
        elif emitted == "SPEAK":
            finished = self.finish()
        elif emitted and emitted != vocab.NO_SIGN:
            word = vocab.SPOKEN.get(emitted, emitted.replace("_", " ").lower())
            self.current.append(word)
            self.last_word, self.last_word_t = word, now
        if (not emitted and self.current and self.sentence_timeout > 0
                and now - self.last_word_t > self.sentence_timeout):
            finished = self.finish()
        return finished

    def finish(self) -> str | None:
        if not self.current:
            return None
        text = " ".join(self.current)
        text = text[0].upper() + text[1:]
        self.history.append(text)
        self.history = self.history[-self.max_history:]
        self.current.clear()
        self.last_word = ""
        return text

    def clear_all(self):
        self.current.clear()
        self.history.clear()
        self.last_word = ""


def _wrap(words: list[str], scale: float, thick: int, max_w: int) -> list[list[str]]:
    lines: list[list[str]] = [[]]
    for w in words:
        trial = " ".join(lines[-1] + [w])
        if lines[-1] and cv2.getTextSize(trial, FONT, scale, thick)[0][0] > max_w:
            lines.append([w])
        else:
            lines[-1].append(w)
    return [l for l in lines if l]


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize keeping aspect ratio, letterboxed onto a w x h background."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.full((h, w, 3), BG, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


class CaptionView:
    WINDOW = "Signs to Speech"

    def __init__(self, show_preview: bool = True, fullscreen: bool = False,
                 size: tuple[int, int] = (1280, 720)):
        self.show_preview = show_preview
        self.fullscreen = fullscreen
        self.W, self.H = size
        self._window_ready = False

    # ---- window -------------------------------------------------------------------------
    def _ensure_window(self):
        if self._window_ready:
            return
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.WINDOW, self.W, self.H)
        self._apply_fullscreen()
        self._window_ready = True

    def _apply_fullscreen(self):
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self._window_ready:
            self._apply_fullscreen()

    def show(self, canvas: np.ndarray) -> int:
        """Display the canvas; returns the key pressed (or -1)."""
        self._ensure_window()
        cv2.imshow(self.WINDOW, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self._window_ready:
            cv2.destroyWindow(self.WINDOW)

    # ---- drawing ------------------------------------------------------------------------
    def render(self, transcript: Transcript, frame: np.ndarray | None = None,
               label: str = "", conf: float = 0.0, fps: float = 0.0, status: str = "") -> np.ndarray:
        W, H = self.W, self.H
        canvas = np.full((H, W, 3), BG, np.uint8)

        if self.show_preview and frame is not None:
            # camera on top-left, live-recognition panel top-right, captions along the bottom
            top_h = int(H * 0.62)
            cam_w = int(top_h * 4 / 3)
            canvas[:top_h, :cam_w] = _fit(frame, cam_w, top_h)
            self._status_panel(canvas, cam_w, 0, W - cam_w, top_h, label, conf, fps, status)
            self._captions(canvas, 0, top_h, W, H - top_h, transcript, big=1.6, hist=1.0)
        else:
            bar_h = 46
            self._status_bar(canvas, 0, H - bar_h, W, bar_h, label, conf, fps, status)
            self._captions(canvas, 0, 0, W, H - bar_h, transcript, big=2.4, hist=1.3)
        return canvas

    def _status_panel(self, c, x, y, w, h, label, conf, fps, status):
        cv2.rectangle(c, (x, y), (x + w, y + h), PANEL, -1)
        pad = 24
        cv2.putText(c, "SEEING", (x + pad, y + 50), FONT, 0.7, GREY, 1, cv2.LINE_AA)
        shown = "-" if not label or label == vocab.NO_SIGN else label.replace("_", " ")
        scale = 1.6
        while scale > 0.6 and cv2.getTextSize(shown, FONT, scale, 2)[0][0] > w - 2 * pad:
            scale -= 0.1
        cv2.putText(c, shown, (x + pad, y + 110), FONT, scale, WHITE, 2, cv2.LINE_AA)
        # confidence bar
        bx, by, bw, bh = x + pad, y + 135, w - 2 * pad, 16
        cv2.rectangle(c, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        color = GREEN if conf >= 0.8 else YELLOW if conf >= 0.5 else RED
        cv2.rectangle(c, (bx, by), (bx + int(bw * max(0.0, min(conf, 1.0))), by + bh), color, -1)
        cv2.putText(c, f"confidence {conf:.0%}", (bx, by + 45), FONT, 0.6, GREY, 1, cv2.LINE_AA)
        fps_color = GREEN if fps >= 10 else RED
        cv2.putText(c, f"{fps:.0f} FPS", (bx, y + h - 50), FONT, 0.6, fps_color, 1, cv2.LINE_AA)
        if status:
            cv2.putText(c, status, (bx, y + h - 22), FONT, 0.55, DIM, 1, cv2.LINE_AA)

    def _status_bar(self, c, x, y, w, h, label, conf, fps, status):
        cv2.rectangle(c, (x, y), (x + w, y + h), PANEL, -1)
        shown = "-" if not label or label == vocab.NO_SIGN else label.replace("_", " ")
        txt = f"seeing: {shown} ({conf:.0%})    {fps:.0f} FPS"
        if status:
            txt += f"    {status}"
        cv2.putText(c, txt, (x + 16, y + 30), FONT, 0.65, GREY, 1, cv2.LINE_AA)

    def _captions(self, c, x, y, w, h, t: Transcript, big: float, hist: float):
        pad = 28
        max_w = w - 2 * pad
        line_h = int(48 * big)
        # current sentence, wrapped; newest line at the bottom, last word highlighted
        lines = _wrap(t.current, big, 2, max_w) if t.current else []
        max_lines = max(1, (h - pad - int(40 * hist)) // line_h)
        lines = lines[-max_lines:]
        cy = y + h - pad
        if not lines:
            cv2.putText(c, "...", (x + pad, cy), FONT, big, DIM, 2, cv2.LINE_AA)
            cy -= line_h
        for i, line in enumerate(reversed(lines)):
            words = line
            xx = x + pad
            for j, word in enumerate(words):
                is_last = i == 0 and j == len(words) - 1 and word == t.last_word
                cv2.putText(c, word, (xx, cy), FONT, big, YELLOW if is_last else WHITE, 2, cv2.LINE_AA)
                xx += cv2.getTextSize(word + " ", FONT, big, 2)[0][0]
            cy -= line_h
        # finished sentences above, dimmer and smaller
        hist_h = int(44 * hist)
        for s in reversed(t.history):
            if cy - hist_h < y + 10:
                break
            for line in reversed(_wrap(s.split(), hist, 1, max_w)):
                if cy < y + hist_h:
                    break
                cv2.putText(c, " ".join(line), (x + pad, cy), FONT, hist, GREY, 1, cv2.LINE_AA)
                cy -= hist_h
