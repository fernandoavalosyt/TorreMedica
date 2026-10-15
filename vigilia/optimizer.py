"""Optimizacion de video antes de YOLO: limita la tasa de analisis, reduce la
resolucion y mejora la imagen en poca luz (turnos de noche)."""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np


class FpsGate:
    """Deja pasar como maximo `fps` frames por segundo."""

    def __init__(self, fps: float) -> None:
        self.interval = 1.0 / fps
        self._next = 0.0

    def set_fps(self, fps: float) -> None:
        self.interval = 1.0 / fps

    def ready(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if now >= self._next:
            self._next = now + self.interval
            return True
        return False


class FrameOptimizer:
    def __init__(self, max_side: int = 640, dark_luma: float = 70.0) -> None:
        self.max_side = max_side
        self.dark_luma = dark_luma
        self.is_dark = False
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Devuelve (frame reducido, escala). escala = ancho_reducido / ancho_original."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else frame
        luma = float(small[::8, ::8].mean()) if small.ndim == 3 else float(small.mean())
        # Histeresis para no oscilar entre modos cerca del umbral.
        self.is_dark = luma < self.dark_luma or (self.is_dark and luma < self.dark_luma * 1.15)
        return (self.enhance(small) if self.is_dark else small), scale

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """CLAHE sobre la luminancia: recupera detalle de poca luz sin cambiar los colores."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
