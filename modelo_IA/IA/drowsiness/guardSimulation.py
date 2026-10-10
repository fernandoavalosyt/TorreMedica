"""
guardSimulation.py

Guardia sintetico para probar y demostrar la deteccion de somnolencia sin
camara ni modelos. Genera keypoints COCO-17 geometricamente coherentes (mismo
formato que S5) para un guardia sentado de frente, siguiendo un guion de 60 s:

    0-12 s   despierto: escribe (manos en movimiento) y mueve la cabeza
   12-22 s   somnoliento: manos quietas y 3 cabeceos (caida rapida + recuperacion)
   22-24 s   la cabeza cae sobre el pecho y los hombros se encorvan
   24-50 s   dormido: inmovil
   50-60 s   despierta y vuelve a escribir

Tambien dibuja la figura del guardia sobre un escritorio para el visualizador.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

CYCLE_SECONDS = 60.0
_BOBS = (13.5, 16.5, 19.5)


def _smooth(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _head_drop(t: float) -> float:
    """0 = cabeza erguida, 1 = cabeza caida sobre el pecho."""
    if t < 12.0:
        return 0.05 + 0.05 * math.sin(t * 1.3)
    if t < 22.0:
        for start in _BOBS:  # caida 0.4 s, sostiene 0.3 s, recupera 0.5 s
            dt = t - start
            if 0 <= dt < 0.4:
                return 0.15 + 0.65 * _smooth(dt / 0.4)
            if 0.4 <= dt < 0.7:
                return 0.8
            if 0.7 <= dt < 1.2:
                return 0.8 - 0.65 * _smooth((dt - 0.7) / 0.5)
        return 0.15
    if t < 24.0:
        return 0.15 + 0.85 * _smooth((t - 22.0) / 2.0)
    if t < 50.0:
        return 1.0
    if t < 52.0:
        return 1.0 - 0.95 * _smooth((t - 50.0) / 2.0)
    return 0.05 + 0.05 * math.sin(t * 1.3)


def _slump(t: float) -> float:
    if 22.0 <= t < 50.0:
        return _smooth((t - 22.0) / 2.0)
    if 50.0 <= t < 52.0:
        return 1.0 - _smooth((t - 50.0) / 2.0)
    return 0.0


def _writing(t: float) -> float:
    if t < 12.0 or t >= 52.0:
        return 1.0
    if 50.0 <= t < 52.0:
        return _smooth((t - 50.0) / 2.0)
    return 0.0


def guard_pose(t: float, center_x: float = 640.0, shoulder_y: float = 330.0, width: float = 150.0,
               rng: np.random.Generator | None = None) -> Tuple[Dict[str, Dict[str, Any]], List[float]]:
    """Keypoints (formato S5) y bbox [x1, y1, x2, y2] del guardia en el segundo t del guion."""
    t = t % CYCLE_SECONDS
    rng = rng or np.random.default_rng()
    w = width
    d = _head_drop(t)
    write = _writing(t)
    sway = 0.03 * w * math.sin(t * 0.9) * (1.0 if write > 0 else 0.0)

    sy = shoulder_y + 0.2 * w * _slump(t)
    pts: Dict[str, Tuple[float, float]] = {
        "left_shoulder": (center_x - w / 2 + sway, sy),
        "right_shoulder": (center_x + w / 2 + sway, sy),
        "left_hip": (center_x - 0.32 * w, shoulder_y + 1.1 * w),
        "right_hip": (center_x + 0.32 * w, shoulder_y + 1.1 * w),
    }
    ear_y = sy - 0.75 * w + 0.25 * d * w
    ear_x = center_x + sway * 1.5
    pts["left_ear"] = (ear_x - 0.28 * w, ear_y)
    pts["right_ear"] = (ear_x + 0.28 * w, ear_y)
    nose_y = ear_y + (0.08 + 0.45 * d) * w
    pts["nose"] = (ear_x, nose_y)
    pts["left_eye"] = (ear_x - 0.1 * w, nose_y - 0.12 * w)
    pts["right_eye"] = (ear_x + 0.1 * w, nose_y - 0.12 * w)

    hand_y = shoulder_y + 0.85 * w
    pts["left_elbow"] = (center_x - 0.62 * w, shoulder_y + 0.55 * w)
    pts["right_elbow"] = (center_x + 0.62 * w, shoulder_y + 0.55 * w)
    pts["left_wrist"] = (center_x - 0.3 * w + write * 0.15 * w * math.sin(t * 9.0), hand_y)
    pts["right_wrist"] = (center_x + 0.3 * w + write * 0.15 * w * math.sin(t * 7.0 + 1.0),
                          hand_y + write * 0.06 * w * math.cos(t * 11.0))

    keypoints = {}
    for name, (x, y) in pts.items():
        jitter = rng.normal(0.0, 0.004 * w, 2)  # ruido del estimador de pose
        keypoints[name] = {"x": float(x + jitter[0]), "y": float(y + jitter[1]), "confidence": 0.9, "visible": True}
    keypoints["left_knee"] = keypoints["right_knee"] = {"x": None, "y": None, "confidence": 0.1, "visible": False}
    keypoints["left_ankle"] = keypoints["right_ankle"] = {"x": None, "y": None, "confidence": 0.1, "visible": False}

    xs = [p["x"] for p in keypoints.values() if p["visible"]]
    ys = [p["y"] for p in keypoints.values() if p["visible"]]
    bbox = [min(xs) - 0.15 * w, min(ys) - 0.25 * w, max(xs) + 0.15 * w, max(ys) + 0.1 * w]
    return keypoints, bbox


def build_background(width: int = 1280, height: int = 720) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        shade = 34 + int(24 * y / height)
        img[y, :] = (shade + 8, shade + 4, shade)
    cv2.rectangle(img, (0, 0), (width, 120), (52, 46, 40), -1)
    for x in range(60, width, 220):  # monitores de vigilancia en la pared
        cv2.rectangle(img, (x, 22), (x + 150, 100), (30, 30, 30), -1)
        cv2.rectangle(img, (x + 6, 28), (x + 144, 94), (70, 90, 60), -1)
    return img


def draw_guard(frame: np.ndarray, keypoints: Dict[str, Dict[str, Any]]) -> None:
    """Dibuja una figura humana simple a partir de los keypoints (para que el
    overlay del visualizador coincida con la imagen)."""
    def p(name: str) -> Tuple[int, int]:
        return int(keypoints[name]["x"]), int(keypoints[name]["y"])

    uniform, skin = (70, 55, 35), (150, 175, 205)
    ls, rs, lh, rh = p("left_shoulder"), p("right_shoulder"), p("left_hip"), p("right_hip")
    cv2.fillPoly(frame, [np.array([ls, rs, rh, lh], np.int32)], uniform)
    for a, b in (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")):
        cv2.line(frame, p(a), p(b), uniform, 22, cv2.LINE_AA)
    neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    head_center = ((p("left_ear")[0] + p("right_ear")[0]) // 2, (p("left_ear")[1] + p("nose")[1]) // 2)
    cv2.line(frame, neck, head_center, skin, 26, cv2.LINE_AA)
    radius = int(abs(p("right_ear")[0] - p("left_ear")[0]) * 0.62)
    cv2.circle(frame, head_center, radius, skin, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (head_center[0], head_center[1] - radius // 2), (radius, radius // 2), 0, 180, 360,
                (40, 40, 45), -1, cv2.LINE_AA)  # gorra
    for name in ("left_wrist", "right_wrist"):
        cv2.circle(frame, p(name), 12, skin, -1, cv2.LINE_AA)
    desk_top = int(keypoints["left_wrist"]["y"]) + 14
    cv2.rectangle(frame, (330, desk_top), (950, frame.shape[0]), (45, 70, 95), -1)
    cv2.rectangle(frame, (330, desk_top), (950, desk_top + 10), (60, 90, 120), -1)
    cv2.rectangle(frame, (720, desk_top - 40), (840, desk_top), (210, 210, 205), -1)  # bitacora
