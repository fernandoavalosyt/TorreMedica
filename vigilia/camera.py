"""Servicio de camara: conexion con credenciales (RTSP/RTSPS/HTTP-MJPEG),
reconexion automatica y siempre el frame mas reciente (sin acumular retraso)."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger("vigilia.camera")

# TCP evita la perdida de paquetes de UDP; stimeout (us) evita bloqueos si la camara deja de responder.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")

OPEN_TIMEOUT_MS = 8000
READ_TIMEOUT_MS = 8000
MAX_BACKOFF = 30.0


def _open_capture(cfg: CameraConfig) -> cv2.VideoCapture:
    source = cfg.source()
    if cfg.protocol == "webcam":
        cap = cv2.VideoCapture(source)
    elif cfg.protocol == "file":
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def test_connection(cfg: CameraConfig, timeout: float = 12.0) -> Dict[str, Any]:
    """Prueba una camara sin registrarla: alcanza el host, abre el stream y lee un frame."""
    if cfg.is_network:
        try:
            with socket.create_connection((cfg.host, cfg.port), timeout=min(4.0, timeout)):
                pass
        except OSError as exc:
            return {"ok": False, "message": f"No se alcanza {cfg.host}:{cfg.port} ({exc.__class__.__name__}). "
                                            "Revise IP, puerto, red y que la camara este encendida."}

    result: Dict[str, Any] = {}

    def probe() -> None:
        cap = None
        try:
            cap = _open_capture(cfg)
            if not cap.isOpened():
                result.update(ok=False, message="El stream no abre: revise usuario, contrasena y ruta del stream.")
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                result.update(ok=False, message="Conecta pero no entrega video (ruta o codec no compatible).")
                return
            h, w = frame.shape[:2]
            result.update(ok=True, message=f"Conexion correcta ({w}x{h}).", width=w, height=h)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de backend es un 'no conecta'
            result.update(ok=False, message=f"Error al abrir el stream: {exc.__class__.__name__}")
        finally:
            if cap is not None:
                cap.release()

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return {"ok": False, "message": "Tiempo de espera agotado al abrir el stream."}
    return result


class CameraStream(threading.Thread):
    """Hilo de captura. `latest()` entrega el ultimo frame; los intermedios se descartan."""

    def __init__(self, cfg: CameraConfig) -> None:
        super().__init__(name=f"camera-{cfg.id}", daemon=True)
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._frame_time = 0.0
        self.state = "connecting"          # connecting | connected | reconnecting | stopped
        self.last_error: Optional[str] = None
        self.connected_since: Optional[float] = None
        self.offline_since: Optional[float] = time.time()
        self.fps_in = 0.0

    def latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            return self._frame, self._seq, self._frame_time

    def stop(self) -> None:
        self._stop_event.set()

    def _publish(self, frame: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            if self._frame_time:
                self.fps_in = 0.9 * self.fps_in + 0.1 / max(1e-3, now - self._frame_time)
            self._frame, self._frame_time = frame, now
            self._seq += 1

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            cap = _open_capture(self.cfg)
            if not cap.isOpened():
                cap.release()
                self._mark_down(f"no se pudo abrir {self.cfg.safe_url()}")
                if self._stop_event.wait(backoff):
                    break
                backoff = min(MAX_BACKOFF, backoff * 2)
                continue

            backoff = 1.0
            self.state, self.last_error = "connected", None
            self.connected_since, self.offline_since = time.time(), None
            logger.info("[%s] conectada: %s", self.cfg.id, self.cfg.safe_url())
            file_fps = cap.get(cv2.CAP_PROP_FPS) if self.cfg.protocol == "file" else 0.0
            failures = 0
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    failures = 0
                    self._publish(frame)
                    if file_fps > 0:  # archivo de prueba: respetar su ritmo
                        self._stop_event.wait(1.0 / file_fps)
                    continue
                if self.cfg.protocol == "file":  # reiniciar el video en bucle
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures >= 5:
                    break
                self._stop_event.wait(0.2)
            cap.release()
            if not self._stop_event.is_set():
                self._mark_down("se perdio la senal del video")
                self._stop_event.wait(backoff)
        self.state = "stopped"

    def _mark_down(self, reason: str) -> None:
        if self.state != "reconnecting":
            logger.warning("[%s] %s; reintentando...", self.cfg.id, reason)
        self.state, self.last_error = "reconnecting", reason
        self.connected_since = None
        if self.offline_since is None:
            self.offline_since = time.time()
        with self._lock:
            self._frame = None

    def offline_seconds(self) -> float:
        return 0.0 if self.offline_since is None else time.time() - self.offline_since

    def status(self) -> Dict[str, Any]:
        return {"state": self.state, "last_error": self.last_error, "fps_in": round(self.fps_in, 1),
                "offline_seconds": round(self.offline_seconds(), 1)}
