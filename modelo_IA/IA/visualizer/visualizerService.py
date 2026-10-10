"""
visualizerService.py

Servicio 10: Visualizador "Vigilia AI - Guard Sleep Detection".

Dibuja sobre el frame de la camara el estado de cada guardia (calculado por
DrowsinessService) y lo muestra en una VENTANA LOCAL de OpenCV
(cv2.imshow + cv2.waitKey(1)), sin red de por medio: latencia minima.

  - Caja tactica + esqueleto por guardia (track_id persistente):
        verde     ALERTA (despierto)
        amarillo  SOMNOLIENTO  ("Merodeo de sueño: X.X seg")
        rojo      DORMIDO (parpadeante)
        naranja   INACTIVO (sin desplazamiento prolongado)
        purpura   GEMINI EVALUANDO... (verificacion Tier-2 en curso)
        verde     DESPIERTO - actividad (Gemini descarto el sueño)
  - Vector de cabeza: flecha desde la nariz hacia donde cae la cabeza.
  - HUD: ID del guardia, estado, tiempo con la cabeza abajo, LME y FPS.
  - Banner inferior parpadeante "ALERTA: GUARDIA DORMIDO".

Servidor web (Flask/MJPEG para React) CONGELADO: el codigo se conserva
intacto pero solo arranca con ENABLE_WEB_STREAM = True (variable de entorno
ENABLE_WEB_STREAM=true). Si no hay pantalla disponible (ej. Docker), el
servicio cae automaticamente al stream web (VISUALIZER_FALLBACK_TO_WEB).

Modo "live": se muestra SIEMPRE el ultimo frame de la camara con el ultimo
estado conocido de cada guardia. Un guardia sentado casi no se desplaza, asi
que el pequeño desfase de la inferencia no se nota y el video nunca se
atrasa.

Entradas (todas thread-safe y baratas; el dibujo ocurre en un hilo propio):
  on_frame(frame)            salida de S1 {camera_id, frame_id, timestamp, image}
  on_guard_frame(result)     salida de DrowsinessService {camera_id, guards: [...]}
  on_decision(decision)      salida de S7 (eventos generados)
  on_evidence(evidence)      salida de S8 (estado del clip)
  submit(meta)               frame + guardias ya calculados (simulacion o productor externo)

La ventana se atiende con run_window_loop(stop_event) desde el HILO PRINCIPAL
(requisito de HighGUI en macOS y lo mas estable en Windows/Linux).
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from flask import Flask, Response, request
from werkzeug.serving import make_server

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # sin Pillow: se usan las fuentes Hershey de OpenCV
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]


logger = logging.getLogger("visualizer_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "si")


# Servidor Flask/MJPEG hacia React: congelado por defecto. Su codigo sigue
# completo abajo (_build_app); basta ENABLE_WEB_STREAM=true para reactivarlo.
ENABLE_WEB_STREAM: bool = _env_bool("ENABLE_WEB_STREAM", False)

WINDOW_TITLE = "Vigilia AI - Guard Sleep Detection"


# =============================================================================
# Catalogos de presentacion
# =============================================================================

STATE_ALERT = "ALERT_ACTIVE"
STATE_DROWSY = "DROWSINESS_WARNING"
STATE_SLEEP = "SLEEPING_CRITICAL"
STATE_INACTIVE = "INACTIVE_GUARD_STATION"
STATE_VERIFYING = "VERIFYING_WITH_GEMINI"
STATE_WORKING = "WORKING_DISTRACTED"

STATE_BGR: Dict[str, Tuple[int, int, int]] = {
    STATE_ALERT: (60, 220, 60),        # verde
    STATE_DROWSY: (0, 215, 255),       # amarillo
    STATE_SLEEP: (0, 0, 255),          # rojo
    STATE_INACTIVE: (0, 140, 255),     # naranja
    STATE_VERIFYING: (255, 60, 200),   # purpura / azul neon
    STATE_WORKING: (60, 220, 60),      # verde
}
STATE_TEXT_BGR: Dict[str, Tuple[int, int, int]] = {
    STATE_ALERT: (15, 15, 15),
    STATE_DROWSY: (15, 15, 15),
    STATE_SLEEP: (255, 255, 255),
    STATE_INACTIVE: (15, 15, 15),
    STATE_VERIFYING: (255, 255, 255),
    STATE_WORKING: (15, 15, 15),
}
STATE_LABELS = {STATE_ALERT: "ALERTA", STATE_DROWSY: "SOMNOLIENTO", STATE_SLEEP: "DORMIDO",
                STATE_INACTIVE: "INACTIVO", STATE_VERIFYING: "VERIFICANDO CON GEMINI",
                STATE_WORKING: "DESPIERTO"}
STATE_SEVERITY = {STATE_ALERT: 0, STATE_WORKING: 0, STATE_DROWSY: 1, STATE_INACTIVE: 2, STATE_VERIFYING: 3,
                  STATE_SLEEP: 4}
ACTIVITY_LABELS = {
    "telefono": "usando teléfono", "escribiendo": "escribiendo", "leyendo": "leyendo", "radio": "usando radio",
    "mirando_monitor": "mirando monitor", "conversando": "conversando", "otra": "actividad", "ninguna": "en alerta",
}

EVENT_LABELS = {"SLEEPING_GUARD": "Guardia dormido", "GUARD_INACTIVE": "Guardia inactivo"}
PRIORITY_LABELS = {"CRITICAL": "crítica", "HIGH": "alta", "MEDIUM": "media", "LOW": "baja"}

COCO_KEYPOINTS: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_EDGES: Tuple[Tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
)
SKELETON_BGR = (255, 230, 120)
HUD_BG = (18, 14, 10)
LIVE_GREEN = (80, 220, 60)
OFFLINE_RED = (60, 60, 230)


# =============================================================================
# Configuracion
# =============================================================================

@dataclass(frozen=True)
class VisualizerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    jpeg_quality: int = 75
    max_width: int = 1280
    local_window: bool = True
    window_title: str = WINDOW_TITLE
    web_stream: bool = ENABLE_WEB_STREAM
    fallback_to_web: bool = True          # sin pantalla (Docker): servir el stream web
    overlay_max_age_seconds: float = 2.0  # estado de guardia mas viejo que esto no se dibuja
    stale_seconds: float = 3.0            # sin frames por mas tiempo => "SIN SEÑAL"
    banner_seconds: float = 12.0
    cors_origin: str = "*"
    title: str = "VIGILIA AI"
    alerts_url: Optional[str] = None
    font_path: Optional[str] = None

    @staticmethod
    def from_env() -> "VisualizerConfig":
        return VisualizerConfig(
            host=os.getenv("VISUALIZER_HOST", "127.0.0.1"),
            port=int(os.getenv("VISUALIZER_PORT", "5000")),
            jpeg_quality=int(os.getenv("VISUALIZER_JPEG_QUALITY", "75")),
            max_width=int(os.getenv("VISUALIZER_MAX_WIDTH", "1280")),
            local_window=_env_bool("VISUALIZER_LOCAL_WINDOW", True),
            web_stream=_env_bool("ENABLE_WEB_STREAM", ENABLE_WEB_STREAM),
            fallback_to_web=_env_bool("VISUALIZER_FALLBACK_TO_WEB", True),
            cors_origin=os.getenv("VISUALIZER_CORS_ORIGIN", "*"),
            alerts_url=os.getenv("VISUALIZER_ALERTS_URL") or None,
            font_path=os.getenv("VISUALIZER_FONT") or None,
        )


def gui_available() -> bool:
    """True si OpenCV tiene backend grafico y hay una pantalla a la que dibujar."""
    match = re.search(r"GUI:\s*(\S+)", cv2.getBuildInformation())
    if match and match.group(1).upper() == "NONE":
        return False  # opencv-python-headless
    if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        return False
    return True


# =============================================================================
# Utilidades de render
# =============================================================================

class FpsMeter:
    """FPS con promedio movil exponencial: el valor del HUD no parpadea."""

    def __init__(self, smoothing: float = 0.1) -> None:
        self._alpha = smoothing
        self._last: Optional[float] = None
        self._value = 0.0

    def tick(self, now: float) -> None:
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self._value = instant if self._value == 0.0 else self._value + self._alpha * (instant - self._value)
        self._last = now

    def value(self, now: float) -> float:
        if self._last is None or now - self._last > 2.0:
            return 0.0
        return self._value


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


class TextRenderer:
    """Texto antialiasado TrueType (acentos, ñ, °) via Pillow, en UNA pasada
    por frame. Sin Pillow o sin fuente cae a Hershey (sin acentos)."""

    _CANDIDATES = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    )

    def __init__(self, font_path: Optional[str] = None) -> None:
        self._path: Optional[str] = None
        self._fonts: Dict[int, Any] = {}
        if ImageFont is not None:
            for candidate in ([font_path] if font_path else []) + list(self._CANDIDATES):
                if candidate and os.path.isfile(candidate):
                    self._path = candidate
                    break
        self._ops: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = []
        if self._path is None:
            logger.warning("Sin fuente TrueType: el texto del visualizador usara Hershey (sin acentos)")

    @property
    def truetype(self) -> bool:
        return self._path is not None

    def _font(self, size: int) -> Any:
        font = self._fonts.get(size)
        if font is None:
            font = ImageFont.truetype(self._path, size)  # type: ignore[union-attr]
            self._fonts[size] = font
        return font

    def measure(self, text: str, size: int) -> Tuple[int, int]:
        if self.truetype:
            left, _, right, _ = self._font(size).getbbox(text)
            return int(right - left), int(size)
        scale = size / 30.0
        (w, h), _ = cv2.getTextSize(_strip_accents(text), cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 2)))
        return w, h

    def queue(self, text: str, top_left: Tuple[int, int], size: int, color: Tuple[int, int, int]) -> None:
        self._ops.append((text, top_left, size, color))

    def flush(self, img: np.ndarray) -> np.ndarray:
        ops, self._ops = self._ops, []
        if not ops:
            return img
        if self.truetype:
            # Solo se convierte a Pillow la region de cada texto (no el frame
            # completo). El array es BGR: Pillow lo trata como RGB y los
            # colores ya van en orden BGR, asi que los bytes quedan correctos.
            height, width = img.shape[:2]
            for text, (x, y), size, color in ops:
                font = self._font(size)
                left, top, right, bottom = font.getbbox(text)
                x1, y1 = max(0, x - 2), max(0, y - 2)
                x2, y2 = min(width, x + (right - left) + 4), min(height, y + (bottom - top) + 4)
                if x2 <= x1 or y2 <= y1:
                    continue
                region = Image.fromarray(np.ascontiguousarray(img[y1:y2, x1:x2]))  # type: ignore[union-attr]
                ImageDraw.Draw(region).text((x - x1 - left, y - y1 - top), text, font=font, fill=color)  # type: ignore[union-attr]
                img[y1:y2, x1:x2] = np.asarray(region)
            return img
        for text, (x, y), size, color in ops:
            scale = size / 30.0
            thickness = max(1, int(round(scale * 2)))
            plain = _strip_accents(text)
            (_, h), _ = cv2.getTextSize(plain, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.putText(img, plain, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return img


def _blend_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                color: Tuple[int, int, int], alpha: float) -> None:
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1.0 - alpha, 0, dst=roi)


def _scale_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * factor))) for c in color)  # type: ignore[return-value]


def _parse_keypoints(raw: Any) -> Dict[str, Tuple[int, int]]:
    """Formato S5 {nombre: {x, y, visible}}, {nombre: (x, y)} o lista COCO-17."""
    points: Dict[str, Tuple[int, int]] = {}
    if isinstance(raw, dict):
        for name, kp in raw.items():
            if isinstance(kp, dict):
                if kp.get("visible", True) and kp.get("x") is not None and kp.get("y") is not None:
                    points[name] = (int(round(kp["x"])), int(round(kp["y"])))
            elif isinstance(kp, (list, tuple)) and len(kp) >= 2 and kp[0] is not None:
                points[name] = (int(round(kp[0])), int(round(kp[1])))
    elif isinstance(raw, (list, tuple)):
        for name, kp in zip(COCO_KEYPOINTS, raw):
            if kp is not None and len(kp) >= 2 and (len(kp) < 3 or kp[2] is None or kp[2] >= 0.3):
                points[name] = (int(round(kp[0])), int(round(kp[1])))
    return points


def _guard_number(track_id: Any) -> str:
    suffix = str(track_id).rsplit("-", 1)[-1]
    return f"{int(suffix):02d}" if suffix.isdigit() else str(track_id)


def _blink(now: float, period: float = 1.0) -> bool:
    return (now % period) < period / 2.0


# =============================================================================
# Estado por camara
# =============================================================================

class _CameraView:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.frame: Optional[np.ndarray] = None
        self.frame_id = 0
        self.frame_ts = 0.0
        self.last_frame_wall = 0.0
        self.guards: List[Dict[str, Any]] = []
        self.guards_wall = 0.0
        self.alerts: Deque[Dict[str, Any]] = deque(maxlen=6)
        self.evidence: Dict[str, Dict[str, Any]] = {}
        self.dirty = False
        self.last_render_wall = 0.0
        self.fps = FpsMeter()
        self.summary: Dict[str, Any] = {}
        # Ultimo frame compuesto (protegido por VisualizerService._ready_lock)
        self.ready_bgr: Optional[np.ndarray] = None
        self.ready_jpeg: Optional[bytes] = None
        self.ready_version = 0


# =============================================================================
# Servicio
# =============================================================================

class VisualizerService:
    def __init__(self, config: Optional[VisualizerConfig] = None,
                 health_provider: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        self.config = config or VisualizerConfig.from_env()
        self._health_provider = health_provider
        self._views: Dict[str, _CameraView] = {}
        self._state_lock = threading.RLock()
        self._render_lock = threading.Lock()
        self._ready_lock = threading.Lock()
        self._ready_cond = threading.Condition(self._ready_lock)
        self._wake = threading.Event()
        self._text = TextRenderer(self.config.font_path)
        self._clients = 0
        self._stop = threading.Event()
        self._server = None
        self._render_thread: Optional[threading.Thread] = None
        self._render_ms = 0.0
        self.window_enabled = False
        self.web_enabled = False

    # -------------------------------------------------------------------
    # Entradas
    # -------------------------------------------------------------------

    def _view(self, camera_id: str) -> _CameraView:
        with self._state_lock:
            view = self._views.get(camera_id)
            if view is None:
                view = _CameraView(camera_id)
                self._views[camera_id] = view
            return view

    def on_frame(self, frame: Dict[str, Any]) -> None:
        view = self._view(str(frame["camera_id"]))
        with self._state_lock:
            view.frame = frame["image"]
            view.frame_id = int(frame["frame_id"])
            view.frame_ts = float(frame["timestamp"])
            view.last_frame_wall = time.time()
            view.fps.tick(view.last_frame_wall)  # FPS real del video que llega
            view.dirty = True
        self._wake.set()

    def on_guard_frame(self, result: Dict[str, Any]) -> None:
        view = self._view(str(result["camera_id"]))
        with self._state_lock:
            view.guards = list(result.get("guards") or [])
            view.guards_wall = time.time()
            view.dirty = True

    def on_decision(self, decision: Dict[str, Any]) -> None:
        if decision.get("decision") != "EVENT_GENERATED":
            return
        view = self._view(str(decision["camera_id"]))
        with self._state_lock:
            view.alerts.append({
                "event_id": str(decision.get("event_id")),
                "event_type": decision.get("event_type"),
                "priority": decision.get("priority"),
                "track_id": decision.get("track_id"),
                "expires": time.time() + self.config.banner_seconds,
            })

    def on_evidence(self, evidence: Dict[str, Any]) -> None:
        view = self._view(str(evidence["camera_id"]))
        with self._state_lock:
            view.evidence[str(evidence.get("event_id"))] = {"status": evidence.get("status"),
                                                            "format": evidence.get("format")}
            while len(view.evidence) > 20:
                view.evidence.pop(next(iter(view.evidence)))

    def submit(self, meta: Dict[str, Any]) -> None:
        """Frame + guardias ya calculados:
        {frame, frame_id, camera_id, timestamp, guards: [...], alerts: [str] (opcional)}"""
        if not isinstance(meta.get("frame"), np.ndarray):
            raise ValueError("submit() requiere 'frame' (np.ndarray BGR)")
        camera_id = str(meta.get("camera_id", "CAM-01"))
        self.on_frame({"camera_id": camera_id, "frame_id": meta.get("frame_id", 0),
                       "timestamp": meta.get("timestamp", time.time()), "image": meta["frame"]})
        self.on_guard_frame({"camera_id": camera_id, "guards": meta.get("guards") or []})
        for text in meta.get("alerts") or []:
            with self._state_lock:
                self._view(camera_id).alerts.append({"text": str(text), "expires": time.time() + 1.0})

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        cfg = self.config
        self.window_enabled = cfg.local_window and gui_available()
        if cfg.local_window and not self.window_enabled:
            logger.warning("No hay pantalla/GUI de OpenCV disponible (ej. Docker u opencv-headless): "
                           "ventana local desactivada")
        self.web_enabled = cfg.web_stream or (not self.window_enabled and cfg.fallback_to_web)
        if self.web_enabled:
            self._start_web()
        elif not self.window_enabled:
            logger.error("Visualizador sin salida: no hay ventana local y ENABLE_WEB_STREAM=false")
        self._render_thread = threading.Thread(target=self._render_loop, name="VisualizerRender", daemon=True)
        self._render_thread.start()
        logger.info("Visualizador listo | ventana local=%s | stream web=%s | fuente=%s",
                    "SI" if self.window_enabled else "NO",
                    f"SI (http://{cfg.host}:{cfg.port}/stream)" if self.web_enabled else "NO (congelado)",
                    "TrueType" if self._text.truetype else "Hershey")

    def _start_web(self) -> None:
        app = self._build_app()
        self._server = make_server(self.config.host, self.config.port, app, threaded=True)
        threading.Thread(target=self._server.serve_forever, name="VisualizerHTTP", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._ready_cond:
            self._ready_cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
        logger.info("Visualizador detenido")

    def run_window_loop(self, stop_event: threading.Event) -> bool:
        """Muestra la ventana local hasta que stop_event se active, se cierre la
        ventana o se pulse 'q'/ESC (que activan stop_event). Llamar desde el
        HILO PRINCIPAL. Devuelve False si no hay ventana disponible."""
        if not self.window_enabled:
            return False
        title = self.config.window_title
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1280, 720)
        except cv2.error as exc:
            logger.error("No se pudo crear la ventana de OpenCV (%s)", exc)
            self.window_enabled = False
            if self.config.fallback_to_web and not self.web_enabled:
                self.web_enabled = True
                self._start_web()
            return False

        shown = -1
        logger.info("Ventana '%s' abierta (q / ESC para salir)", title)
        try:
            while not stop_event.is_set() and not self._stop.is_set():
                view = self._primary_view()
                image = None
                if view is not None:
                    with self._ready_cond:
                        if view.ready_version == shown:
                            self._ready_cond.wait(timeout=0.05)
                        if view.ready_version != shown:
                            image, shown = view.ready_bgr, view.ready_version
                if image is not None:
                    cv2.imshow(title, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    stop_event.set()
                elif cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()  # el usuario cerro la ventana
                elif view is None:
                    time.sleep(0.05)
        finally:
            cv2.destroyAllWindows()
        return True

    def _primary_view(self) -> Optional[_CameraView]:
        with self._state_lock:
            return next(iter(self._views.values()), None)

    # -------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------

    def _render_loop(self) -> None:
        # Se renderiza en cuanto llega un frame nuevo (on_frame activa _wake):
        # la tasa la marca la camara. Sin esperas fijas, que en Windows se
        # redondean a multiplos de ~15.6 ms y recortan los FPS.
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            started = time.time()
            with self._state_lock:
                active = self.window_enabled or self._clients > 0
                cameras = list(self._views.keys())
            if not active:
                continue
            for camera_id in cameras:
                view = self._view(camera_id)
                with self._state_lock:
                    if not view.dirty and started - view.last_render_wall < 0.25:
                        continue
                    view.dirty = False
                try:
                    image = self.render(camera_id)
                    jpeg = self._encode(image) if self._clients > 0 else None
                except Exception:
                    logger.exception("Error renderizando la camara %s", camera_id)
                    continue
                with self._ready_cond:
                    view.ready_bgr = image          # se sobrescribe: los frames viejos se descartan
                    view.ready_jpeg = jpeg
                    view.ready_version += 1
                    self._ready_cond.notify_all()
                self._render_ms = (time.time() - started) * 1000.0

    def _encode(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        if not ok:
            raise RuntimeError("No se pudo codificar el frame del visualizador")
        return encoded.tobytes()

    def render(self, camera_id: str) -> np.ndarray:
        """Compone el frame actual de una camara (thread-safe)."""
        view = self._view(camera_id)
        now = time.time()
        with self._state_lock:
            frame = view.frame
            stale = view.last_frame_wall == 0.0 or now - view.last_frame_wall > self.config.stale_seconds
            guards = list(view.guards) if now - view.guards_wall <= self.config.overlay_max_age_seconds else []
            while view.alerts and view.alerts[0]["expires"] < now:
                view.alerts.popleft()
            alerts = [self._alert_text(a, view) for a in view.alerts]
            frame_id, frame_ts = view.frame_id, view.frame_ts
        with self._render_lock:
            if frame is None or stale:
                image = self._no_signal(view, now)
            else:
                image = self._draw(frame, view, guards, alerts, frame_ts, now)
            view.last_render_wall = now
        height, width = image.shape[:2]
        if width > self.config.max_width:
            image = cv2.resize(image, (self.config.max_width, int(height * self.config.max_width / width)),
                               interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def _alert_text(alert: Dict[str, Any], view: _CameraView) -> str:
        if "text" in alert:
            return alert["text"]
        text = (f"Evento: {EVENT_LABELS.get(str(alert['event_type']), alert['event_type'])} · "
                f"ID #{_guard_number(alert['track_id'])} · prioridad "
                f"{PRIORITY_LABELS.get(str(alert['priority']), alert['priority'])}")
        evidence = view.evidence.get(alert["event_id"])
        return text + (f" · evidencia {evidence['status']}" if evidence else "")

    # -------------------------------------------------------------------
    # Dibujo
    # -------------------------------------------------------------------

    def _draw(self, frame: np.ndarray, view: _CameraView, guards: List[Dict[str, Any]],
              alerts: List[str], frame_ts: float, now: float) -> np.ndarray:
        img = frame.copy()  # el frame original lo comparten otros servicios
        k = max(0.55, img.shape[1] / 1280.0)
        for guard in guards:
            points = _parse_keypoints(guard.get("keypoints"))
            if points:
                self._draw_skeleton(img, points, k)
        for guard in guards:
            self._draw_guard(img, guard, k, now)
        primary = max(guards, key=lambda g: (STATE_SEVERITY.get(g.get("state"), 0), g.get("state_seconds", 0)),
                      default=None)
        self._draw_hud(img, view, primary, len(guards), alerts, frame_ts, k, now)
        sleeping = [g for g in guards if g.get("state") == STATE_SLEEP]
        if sleeping:
            self._draw_sleep_banner(img, sleeping[0], k, now)
        view.summary = {
            "frame_id": view.frame_id,
            "guards": [{"track_id": g.get("track_id"), "state": g.get("state"),
                        "state_seconds": g.get("state_seconds")} for g in guards],
            "alerts": alerts,
            "fps": round(view.fps.value(now), 2),
        }
        return self._text.flush(img)

    @staticmethod
    def _draw_skeleton(img: np.ndarray, points: Dict[str, Tuple[int, int]], k: float) -> None:
        thickness = max(1, int(round(2 * k)))
        for a, b in COCO_EDGES:
            if a in points and b in points:
                cv2.line(img, points[a], points[b], SKELETON_BGR, thickness, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(img, point, max(2, int(3 * k)), (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_guard(self, img: np.ndarray, guard: Dict[str, Any], k: float, now: float) -> None:
        state = str(guard.get("state", STATE_ALERT))
        metrics = guard.get("metrics") or {}
        color = STATE_BGR.get(state, STATE_BGR[STATE_ALERT])
        height, width = img.shape[:2]
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in guard["bbox"])
        except (KeyError, TypeError, ValueError):
            return
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        # Caja tactica (esquinas reforzadas). Dormido: rojo parpadeante y mas grueso.
        base = max(2, int(round(2 * k)))
        thickness = base
        if state == STATE_SLEEP:
            on = _blink(now, 0.8)
            thickness = base * (3 if on else 2)
            color = color if on else _scale_color(color, 0.45)
            cv2.rectangle(img, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), _scale_color(STATE_BGR[STATE_SLEEP], 0.5),
                          max(1, base // 2), cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), _scale_color(color, 0.55), 1, cv2.LINE_AA)
        corner = int(max(10, min(0.22 * min(x2 - x1, y2 - y1), 42 * k)))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * corner, cy), color, thickness, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * corner), color, thickness, cv2.LINE_AA)

        if state == STATE_VERIFYING:
            # Animacion de espera: arco giratorio en la esquina de la caja.
            radius = max(10, int(14 * k))
            center = (x2 - radius - int(6 * k), y1 + radius + int(6 * k))
            start = int((now * 360) % 360)
            cv2.circle(img, center, radius, _scale_color(color, 0.35), max(2, int(3 * k)), cv2.LINE_AA)
            cv2.ellipse(img, center, (radius, radius), 0, start, start + 100, color, max(2, int(3 * k)), cv2.LINE_AA)

        # Vector de cabeza: hacia donde cae la cabeza (rojo si supera el umbral).
        head_vector = guard.get("head_vector")
        if head_vector:
            (nx, ny), (tx, ty) = head_vector
            arrow_color = (0, 0, 255) if metrics.get("head_drop") else (255, 255, 0)
            cv2.arrowedLine(img, (int(nx), int(ny)), (int(tx), int(ty)), arrow_color,
                            max(2, int(round(3 * k))), cv2.LINE_AA, tipLength=0.25)

        # Etiqueta
        number = _guard_number(guard.get("track_id"))
        verification = guard.get("verification") or {}
        lines = [f"ID: {number} | {STATE_LABELS.get(state, state)}"]
        if state == STATE_DROWSY:
            lines.append(f"Merodeo de sueño: {float(guard.get('state_seconds') or 0):.1f} seg")
        elif state == STATE_VERIFYING:
            dots = "." * (1 + int(now * 3) % 3)
            lines[0] = f"ID: {number} | GEMINI EVALUANDO{dots}"
            lines.append(f"Consulta en curso: {float(guard.get('state_seconds') or 0):.1f} s")
        elif state == STATE_SLEEP and verification.get("confirmed_by") == "gemini":
            lines[0] = f"ID: {number} | DORMIDO CONFIRMADO POR GEMINI"
            lines.append(f"Dormido: {float(metrics.get('t_sleep') or 0):.1f} s | "
                         f"confianza {float(verification.get('confidence') or 0):.0%}")
        elif state == STATE_SLEEP:
            lines.append(f"GUARDIA DORMIDO: {float(metrics.get('t_sleep') or 0):.1f} s"
                         + (" (veredicto local)" if verification.get("confirmed_by") == "local" else ""))
        elif state == STATE_WORKING:
            activity = ACTIVITY_LABELS.get(str(verification.get("activity")), str(verification.get("activity")))
            lines[0] = f"ID: {number} | DESPIERTO - {activity}"
            lines.append("Gemini descartó el sueño")
        elif state == STATE_INACTIVE:
            lines.append(f"Sin movimiento: {float(metrics.get('t_static') or 0):.0f} s")
        else:
            pitch = metrics.get("head_pitch_rel_deg")
            lines.append(f"Cabeza: {pitch:.0f}° | LME: {metrics.get('lme_level') or '—'}" if pitch is not None
                         else f"LME: {metrics.get('lme_level') or '—'}")
        if metrics.get("bob_count_10s"):
            lines.append(f"Cabeceos (10 s): {metrics['bob_count_10s']}")

        size = int(16 * k)
        pad = int(7 * k)
        line_h = int(size * 1.35)
        tag_w = max(self._text.measure(line, size)[0] for line in lines) + 2 * pad
        tag_h = line_h * len(lines) + pad
        tag_x = max(0, min(x1, width - tag_w))
        tag_y = y1 - tag_h - max(1, thickness // 2)
        if tag_y < 0:
            tag_y = y1 + thickness
        cv2.rectangle(img, (tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h), color, -1)
        text_color = STATE_TEXT_BGR.get(state, (15, 15, 15))
        for i, line in enumerate(lines):
            self._text.queue(line, (tag_x + pad, tag_y + pad // 2 + i * line_h + int(2 * k)), size, text_color)

    def _draw_hud(self, img: np.ndarray, view: _CameraView, primary: Optional[Dict[str, Any]], guard_count: int,
                  alerts: List[str], frame_ts: float, k: float, now: float) -> None:
        width = img.shape[1]
        size, small = int(18 * k), int(14 * k)
        pad = int(12 * k)
        bar_h = int(68 * k)
        _blend_rect(img, (0, 0), (width, bar_h), HUD_BG, 0.72)

        if primary is None:
            line1, line2 = "Sin guardia en cámara", "Esperando detección de persona…"
            state_color = (200, 200, 200)
        else:
            state = str(primary.get("state"))
            metrics = primary.get("metrics") or {}
            state_color = STATE_BGR.get(state, (200, 200, 200))
            line1 = f"ID Guardia: #{_guard_number(primary.get('track_id'))} | Estado: [{STATE_LABELS.get(state, state)}]"
            line2 = (f"Tiempo Cabeza Abajo: {float(metrics.get('t_head_down') or 0):.1f} s | "
                     f"LME (Movimiento): {metrics.get('lme_level') or '—'}")
            if guard_count > 1:
                line2 += f" | Guardias: {guard_count}"
        self._text.queue(line1, (pad, int(10 * k)), size, state_color)
        self._text.queue(line2, (pad, int(10 * k) + int(size * 1.45)), small, (220, 225, 230))

        live = f"LIVE  {view.camera_id}"
        fps = f"FPS: {view.fps.value(now):.1f}"
        latency = f"Retraso: {max(0.0, now - frame_ts) * 1000:.0f} ms" if frame_ts else ""
        right = max(self._text.measure(live, small)[0], self._text.measure(fps, size)[0],
                    self._text.measure(latency, small)[0]) + 2 * pad
        x0 = width - right
        dot_r = max(4, int(6 * k))
        cv2.circle(img, (x0 - dot_r, int(10 * k) + small // 2), dot_r,
                   _scale_color(LIVE_GREEN, 0.7 + 0.3 * math.sin(now * 4.0)), -1, cv2.LINE_AA)
        self._text.queue(live, (x0 + int(4 * k), int(8 * k)), small, (235, 235, 235))
        self._text.queue(fps, (x0 + int(4 * k), int(8 * k) + int(small * 1.35)), size, (255, 255, 0))
        if latency:
            self._text.queue(latency, (x0 + int(4 * k), int(8 * k) + int(small * 1.35) + int(size * 1.3)),
                             int(12 * k), (170, 180, 190))

        y = bar_h + int(6 * k)
        for alert in alerts[-3:]:
            tw, th = self._text.measure(alert, small)
            _blend_rect(img, (width - tw - 2 * pad, y), (width - pad // 2, y + th + pad // 2), (20, 20, 110), 0.8)
            self._text.queue(alert, (width - tw - int(1.5 * pad), y + pad // 4), small, (255, 255, 255))
            y += th + pad

    def _draw_sleep_banner(self, img: np.ndarray, guard: Dict[str, Any], k: float, now: float) -> None:
        height, width = img.shape[:2]
        bar_h = int(60 * k)
        top = height - bar_h
        on = _blink(now, 0.8)
        _blend_rect(img, (0, top), (width, height), (0, 0, 210) if on else (0, 0, 120), 0.78)
        cv2.line(img, (0, top), (width, top), (0, 0, 255), max(2, int(3 * k)))
        metrics = guard.get("metrics") or {}
        confirmed = (guard.get("verification") or {}).get("confirmed_by") == "gemini"
        text = (f"ALERTA: GUARDIA DORMIDO{'  ·  CONFIRMADO POR GEMINI' if confirmed else ''}  ·  "
                f"ID #{_guard_number(guard.get('track_id'))}  ·  {float(metrics.get('t_sleep') or 0):.1f} s")
        size = int(26 * k)
        tw, th = self._text.measure(text, size)
        icon = int(32 * k)
        gap = int(14 * k)
        x = max(int(12 * k), (width - (icon + gap + tw)) // 2)
        cy = top + bar_h // 2
        triangle = np.array([[x + icon // 2, cy - icon // 2], [x, cy + icon // 2], [x + icon, cy + icon // 2]], np.int32)
        cv2.fillPoly(img, [triangle], (0, 215, 255), cv2.LINE_AA)
        cv2.line(img, (x + icon // 2, cy - icon // 5), (x + icon // 2, cy + icon // 6), (20, 20, 20),
                 max(2, int(3 * k)), cv2.LINE_AA)
        cv2.circle(img, (x + icon // 2, cy + icon // 3), max(1, int(2 * k)), (20, 20, 20), -1, cv2.LINE_AA)
        self._text.queue(text, (x + icon + gap, cy - th // 2 - int(2 * k)), size, (255, 255, 255))

    def _no_signal(self, view: _CameraView, now: float) -> np.ndarray:
        img = np.full((720, 1280, 3), (23, 17, 11), dtype=np.uint8)
        img[::4, :] = (30, 24, 18)
        since = None if view.last_frame_wall == 0.0 else now - view.last_frame_wall
        title = "SIN SEÑAL"
        detail = "esperando el primer frame" if since is None else f"último frame hace {since:.0f} s"
        tw, _ = self._text.measure(title, 56)
        self._text.queue(title, ((1280 - tw) // 2, 300), 56, (150, 160, 180))
        sub = f"{view.camera_id}  ·  {detail}"
        sw, _ = self._text.measure(sub, 22)
        self._text.queue(sub, ((1280 - sw) // 2, 380), 22, (120, 130, 150))
        label = f"OFFLINE  {view.camera_id}  |  {self.config.title}"
        lw, lh = self._text.measure(label, 16)
        _blend_rect(img, (14, 14), (14 + lw + 40, 14 + lh + 20), HUD_BG, 0.65)
        cv2.circle(img, (30, 14 + (lh + 20) // 2), 6, OFFLINE_RED, -1, cv2.LINE_AA)
        self._text.queue(label, (44, 24), 16, (235, 235, 235))
        view.summary = {"frame_id": None, "guards": [], "alerts": [], "fps": 0.0}
        return self._text.flush(img)

    # =================================================================
    # Servidor web Flask/MJPEG (congelado: solo con ENABLE_WEB_STREAM)
    # =================================================================

    def _stream(self, camera_id: str) -> Iterator[bytes]:
        view = self._view(camera_id)
        with self._state_lock:
            self._clients += 1
        self._wake.set()
        last_version = -1
        try:
            while not self._stop.is_set():
                with self._ready_cond:
                    self._ready_cond.wait_for(
                        lambda: view.ready_version != last_version or self._stop.is_set(), timeout=1.0)
                    jpeg, image, version = view.ready_jpeg, view.ready_bgr, view.ready_version
                if version == last_version or image is None:
                    continue
                if jpeg is None:  # el render aun no habia codificado (primer cliente)
                    jpeg = self._encode(image)
                last_version = version
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
        except (GeneratorExit, BrokenPipeError, ConnectionError):
            pass  # el navegador/React cerro o refresco la conexion
        finally:
            with self._state_lock:
                self._clients -= 1

    def _state_summary(self) -> Dict[str, Any]:
        with self._state_lock:
            now = time.time()
            cameras = {camera_id: {**view.summary,
                                   "seconds_since_frame": round(now - view.last_frame_wall, 2)
                                   if view.last_frame_wall else None}
                       for camera_id, view in self._views.items()}
            return {"cameras": cameras, "stream_clients": self._clients, "render_ms": round(self._render_ms, 1),
                    "local_window": self.window_enabled}

    def _json(self, payload: Any, status: int = 200) -> Response:
        response = Response(json.dumps(payload, default=str), status=status, mimetype="application/json")
        response.headers["Access-Control-Allow-Origin"] = self.config.cors_origin
        response.headers["Cache-Control"] = "no-store"
        return response

    def _resolve_camera(self) -> Optional[str]:
        requested = request.args.get("camera")
        with self._state_lock:
            if requested:
                return requested if requested in self._views else None
            return next(iter(self._views), None)

    def _build_app(self) -> Flask:
        app = Flask("vigilia_visualizer")

        @app.after_request
        def _security_headers(response: Response) -> Response:
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            return response

        @app.get("/")
        def index() -> Response:
            with self._state_lock:
                cameras = list(self._views.keys())
            return Response(self._index_html(cameras), mimetype="text/html")

        @app.get("/stream")
        def stream() -> Response:
            camera_id = self._resolve_camera()
            if camera_id is None:
                return self._json({"error": "camera_not_found"}, 404)
            response = Response(self._stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Access-Control-Allow-Origin"] = self.config.cors_origin
            return response

        @app.get("/snapshot.jpg")
        def snapshot() -> Response:
            camera_id = self._resolve_camera()
            if camera_id is None:
                return self._json({"error": "camera_not_found"}, 404)
            response = Response(self._encode(self.render(camera_id)), mimetype="image/jpeg")
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/api/state")
        def api_state() -> Response:
            return self._json(self._state_summary())

        @app.get("/health")
        def health() -> Response:
            payload: Dict[str, Any] = {"status": "healthy", "version": "1.0.0"}
            if self._health_provider is not None:
                try:
                    payload = self._health_provider()
                except Exception as exc:
                    payload = {"status": "degraded", "version": "1.0.0", "error": str(exc)}
            payload["visualizer"] = {"stream_clients": self._clients, "render_ms": round(self._render_ms, 1),
                                     "local_window": self.window_enabled}
            return self._json(payload)

        return app

    def _index_html(self, cameras: Sequence[str]) -> str:
        options = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cameras)
        first = html.escape(cameras[0]) if cameras else ""
        alerts = (f'<a href="{html.escape(self.config.alerts_url)}" target="_blank" rel="noopener">Alertas</a>'
                  if self.config.alerts_url else "")
        return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(self.config.title)} · Guard Sleep Monitor</title>
<style>
:root{{--bg:#0b0f17;--panel:#111827;--line:#1f2a3a;--text:#e5e9f0;--muted:#8b96a8;--accent:#3cdc3c}}
*{{box-sizing:border-box;margin:0}}html,body{{height:100%}}
body{{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;display:grid;grid-template-rows:52px minmax(0,1fr);overflow:hidden}}
header{{display:flex;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--line)}}
header b{{letter-spacing:.6px}}header span{{color:var(--muted);font-size:13px}}
header nav{{margin-left:auto;display:flex;gap:14px;align-items:center;font-size:13px}}
a{{color:var(--accent);text-decoration:none}}select{{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 8px}}
main{{display:grid;place-items:center;padding:14px;min-height:0}}
img{{max-width:100%;max-height:100%;border-radius:10px;border:1px solid var(--line);background:#000}}
</style></head><body>
<header><b>{html.escape(self.config.title)}</b><span>Guard Sleep Monitor · stream en vivo</span>
<nav><select id="cam">{options}</select><a href="/api/state" target="_blank">Estado</a><a href="/health" target="_blank">Health</a>{alerts}</nav></header>
<main><img id="view" src="/stream?camera={first}" alt="Stream del visualizador"></main>
<script>document.getElementById('cam').addEventListener('change',function(e){{document.getElementById('view').src='/stream?camera='+encodeURIComponent(e.target.value);}});</script>
</body></html>"""
