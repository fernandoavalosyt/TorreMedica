"""Overlay del video: cuadro por persona con su ID, esqueleto de 17 puntos y estado."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .heuristics import TrackState
from .vision import KP_CONF, SKELETON

COLORS = {"ok": (80, 200, 80), "somnolencia": (0, 170, 255), "dormido": (60, 60, 255)}
LEVEL_TEXT = {"ok": "despierta", "somnolencia": "SOMNOLIENTA", "dormido": "DORMIDA"}


def draw_tracks(frame: np.ndarray, tracks: Iterable[TrackState], camera_name: str, fps: float,
                roi: Optional[list] = None, dark: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if roi:
        pts = np.array([[int(x * w), int(y * h)] for x, y in roi], np.int32)
        cv2.polylines(out, [pts], True, (200, 200, 0), 1, cv2.LINE_AA)
    for ts in tracks:
        color = COLORS.get(ts.level, COLORS["ok"])
        x1, y1, x2, y2 = (int(v) for v in ts.bbox)
        for a, b in SKELETON:
            if ts.kpts[a, 2] >= KP_CONF and ts.kpts[b, 2] >= KP_CONF:
                cv2.line(out, (int(ts.kpts[a, 0]), int(ts.kpts[a, 1])), (int(ts.kpts[b, 0]), int(ts.kpts[b, 1])),
                         (255, 255, 0), 2, cv2.LINE_AA)
        for x, y, c in ts.kpts:
            if c >= KP_CONF:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {ts.track_id} {LEVEL_TEXT.get(ts.level, ts.level)} {ts.score_ema:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    header = f"{camera_name} | {fps:.1f} fps" + (" | modo nocturno" if dark else "")
    cv2.putText(out, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def placeholder(text: str, size=(640, 360)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), 30, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, text, (max(8, (size[0] - tw) // 2), size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA)
    return img
