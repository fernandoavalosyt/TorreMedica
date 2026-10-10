"""
drowsinessService.py

Servicio de deteccion de somnolencia del guardia (rol de S6 en "Vigilia - Guard
Sleep Monitor").

Consume los landmarks de S5 (YOLOv8-Pose o MediaPipe Pose, ambos normalizados
al esquema COCO-17) y las cajas/IDs de S4 (tracking), y por cada guardia
(track_id) calcula:

  1. Head Pitch Angle   angulo del vector oido-medio -> nariz respecto a la
                        "vertical" del cuerpo (perpendicular a la linea de
                        hombros). Tambien el roll (cabeza caida a un lado).
                        Head Drop si supera 35 grados (compensado por la
                        postura habitual del guardia, ver _pitch_relative).
  2. Shoulder Slump     compresion de la distancia vertical hombros-cadera
                        respecto a la postura despierta del guardia.
  3. LME                Landmark Motion Energy: desviacion de las posiciones
                        de cabeza y manos en una ventana de 60 cuadros
                        (~2.5 s), normalizada por el ancho de hombros.
                        LME ~ 0 => inmovilidad absoluta.
  4. Head Bobbing       caidas rapidas de la nariz con recuperacion brusca
                        (micro-suenos); se cuentan en los ultimos 10 s.

Maquina de estados por guardia (acumuladores de tiempo con tolerancia a
cuadros ruidosos e histeresis de salida):

  ALERT_ACTIVE            despierto: movimiento normal, cabeza erguida
  DROWSINESS_WARNING      (head drop O >= 2 cabeceos en 10 s) Y cuerpo inmovil > 3 s
  SLEEPING_CRITICAL       (head drop O slump) continuo + LME ~ 0 durante >= 15 s
  INACTIVE_GUARD_STATION  mismo track sin desplazamiento corporal > 60 s

Verificacion Tier-2 (opcional, GeminiJudgeService): cuando la condicion de
sueno lleva GEMINI_VERIFY_AFTER_SECONDS (10 s), el guardia pasa a
VERIFYING_WITH_GEMINI y se envia su recorte a Gemini Vision en segundo plano:
  is_sleeping=true   -> SLEEPING_CRITICAL (confirmado por Gemini)
  is_sleeping=false  -> WORKING_DISTRACTED (o ALERT_ACTIVE) y se reinician los
                        acumuladores en silencio durante un periodo de veto
  error / sin juez   -> fail-safe: decide la heuristica local (15 s)

Salidas:
  - Por frame: estado y metricas de cada guardia (listeners: visualizador).
  - Eventos compatibles con BehaviorResult para S7 (Decision Engine):
      guard_sleep     POSSIBLE (somnolencia) / SUSTAINED (dormido) / FINISHED
      guard_inactive  SUSTAINED / FINISHED

Toda la logica temporal usa el timestamp de los frames (tiempo de datos).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from functools import partial
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger("drowsiness_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

Point = Tuple[float, float]


# =============================================================================
# Estados y configuracion
# =============================================================================

class GuardState(str, Enum):
    ALERT_ACTIVE = "ALERT_ACTIVE"
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    SLEEPING_CRITICAL = "SLEEPING_CRITICAL"
    INACTIVE_GUARD_STATION = "INACTIVE_GUARD_STATION"
    VERIFYING_WITH_GEMINI = "VERIFYING_WITH_GEMINI"
    WORKING_DISTRACTED = "WORKING_DISTRACTED"


STATE_LABELS = {
    GuardState.ALERT_ACTIVE: "ALERTA",
    GuardState.DROWSINESS_WARNING: "SOMNOLIENTO",
    GuardState.SLEEPING_CRITICAL: "DORMIDO",
    GuardState.INACTIVE_GUARD_STATION: "INACTIVO",
    GuardState.VERIFYING_WITH_GEMINI: "VERIFICANDO",
    GuardState.WORKING_DISTRACTED: "DESPIERTO",
}
_SEVERITY = {
    GuardState.ALERT_ACTIVE: 0,
    GuardState.WORKING_DISTRACTED: 0,
    GuardState.DROWSINESS_WARNING: 1,
    GuardState.INACTIVE_GUARD_STATION: 2,
    GuardState.VERIFYING_WITH_GEMINI: 3,
    GuardState.SLEEPING_CRITICAL: 4,
}

HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
HAND_POINTS = ("left_wrist", "right_wrist")
BODY_POINTS = ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_elbow", "right_elbow")


@dataclass(frozen=True)
class DrowsinessConfig:
    head_drop_deg: float = 35.0              # pitch (hacia adelante) o roll (a un lado)
    head_drop_hysteresis_deg: float = 5.0    # sale de "cabeza abajo" solo bajo 35 - 5 grados
    pitch_baseline_cap_deg: float = 20.0     # la postura "normal" nunca puede absorber una cabeza caida
    baseline_min_samples: int = 30
    slump_ratio: float = 0.15                # compresion hombros-cadera vs postura despierta
    lme_window_frames: int = 60
    lme_window_seconds: float = 2.5
    lme_low_threshold: float = 0.025         # desviacion < 2.5 % del ancho de hombros => LME ~ 0
    body_motion_threshold: float = 0.04      # movimiento de hombros/manos para T_inmovil
    immobile_warning_seconds: float = 3.0
    sleep_seconds: float = 15.0              # configurable 10..30
    inactive_seconds: float = 60.0
    displacement_ratio: float = 0.25         # desplazamiento (en alturas de caja) que reinicia T_estatico
    bob_drop_ratio: float = 0.18             # caida de nariz (anchos de hombro) para contar un cabeceo
    bob_max_fall_seconds: float = 1.0
    bob_max_down_seconds: float = 2.5
    bob_recover_fraction: float = 0.6
    bob_window_seconds: float = 10.0
    bob_count_warning: int = 2
    gap_tolerance_seconds: float = 1.0       # cuadros ruidosos que no rompen un acumulador
    recovery_seconds: float = 2.0            # histeresis para bajar de estado
    heartbeat_seconds: float = 1.0
    track_ttl_seconds: float = 10.0
    verify_after_seconds: float = 10.0       # condicion de sueno que dispara la consulta a Gemini
    awake_veto_seconds: float = 15.0         # tras "despierto" de Gemini no se re-escala en este tiempo
    verification_timeout_seconds: float = 12.0  # sin respuesta en este tiempo -> veredicto local

    @staticmethod
    def from_env() -> "DrowsinessConfig":
        sleep_seconds = float(os.getenv("DROWSY_SLEEP_SECONDS", "15"))
        return DrowsinessConfig(
            head_drop_deg=float(os.getenv("DROWSY_HEAD_DROP_DEG", "35")),
            slump_ratio=float(os.getenv("DROWSY_SLUMP_RATIO", "0.15")),
            lme_low_threshold=float(os.getenv("DROWSY_LME_LOW", "0.025")),
            immobile_warning_seconds=float(os.getenv("DROWSY_IMMOBILE_SECONDS", "3")),
            sleep_seconds=min(30.0, max(10.0, sleep_seconds)),
            inactive_seconds=float(os.getenv("DROWSY_INACTIVE_SECONDS", "60")),
            verify_after_seconds=float(os.getenv("GEMINI_VERIFY_AFTER_SECONDS", "10")),
            awake_veto_seconds=float(os.getenv("GEMINI_AWAKE_VETO_SECONDS", "15")),
            verification_timeout_seconds=float(os.getenv("GEMINI_VERIFICATION_TIMEOUT", "12")),
        )


# =============================================================================
# Geometria de postura
# =============================================================================

def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    return None if norm < 1e-6 else v / norm


def parse_keypoints(raw: Any) -> Dict[str, np.ndarray]:
    """Formato S5 {nombre: {x, y, visible}} -> {nombre: array([x, y])} visibles."""
    points: Dict[str, np.ndarray] = {}
    if isinstance(raw, dict):
        for name, kp in raw.items():
            if isinstance(kp, dict) and kp.get("visible", True) and kp.get("x") is not None and kp.get("y") is not None:
                points[name] = np.array([float(kp["x"]), float(kp["y"])])
    return points


def _midpoint(points: Dict[str, np.ndarray], a: str, b: str) -> Optional[np.ndarray]:
    if a in points and b in points:
        return (points[a] + points[b]) / 2.0
    return points.get(a, points.get(b))


@dataclass
class PostureSample:
    timestamp: float
    scale: float                          # ancho de hombros (px), referencia de normalizacion
    up: np.ndarray                        # "vertical" del cuerpo (unitario, hacia arriba en imagen)
    pitch_deg: Optional[float]            # cabeza hacia adelante/abajo (+)
    roll_deg: Optional[float]             # cabeza ladeada (abs)
    nose_height: Optional[float]          # altura nariz sobre hombros, en anchos de hombro
    torso_ratio: Optional[float]          # (cadera.y - hombros.y) / ancho de hombros
    head_vector: Optional[Tuple[Point, Point]]
    head_points: Dict[str, np.ndarray]
    body_points: Dict[str, np.ndarray]
    observability: float


def compute_posture(points: Dict[str, np.ndarray], bbox: List[float], timestamp: float) -> PostureSample:
    x1, y1, x2, y2 = bbox
    box_height = max(1.0, y2 - y1)
    left_s, right_s = points.get("left_shoulder"), points.get("right_shoulder")
    shoulder_mid = _midpoint(points, "left_shoulder", "right_shoulder")
    hip_mid = _midpoint(points, "left_hip", "right_hip")

    shoulder_width = float(np.linalg.norm(right_s - left_s)) if left_s is not None and right_s is not None else 0.0
    scale = shoulder_width if shoulder_width > 0.12 * box_height else 0.25 * box_height

    # Vertical del cuerpo: perpendicular a la linea de hombros si es fiable;
    # si no (guardia de perfil), eje cadera->hombros; si no, vertical de imagen.
    up: Optional[np.ndarray] = None
    if left_s is not None and right_s is not None and shoulder_width > 0.3 * scale:
        s = right_s - left_s
        perp = np.array([s[1], -s[0]])
        up = _unit(perp if perp[1] < 0 else -perp)
    if up is None and hip_mid is not None and shoulder_mid is not None:
        up = _unit(shoulder_mid - hip_mid)
    if up is None:
        up = np.array([0.0, -1.0])

    nose = points.get("nose")
    ear_ref = _midpoint(points, "left_ear", "right_ear")
    both_ears = "left_ear" in points and "right_ear" in points

    pitch: Optional[float] = None
    head_vector: Optional[Tuple[Point, Point]] = None
    if nose is not None and ear_ref is not None:
        n = nose - ear_ref
        down = -float(np.dot(n, up))                       # + si la nariz cae por debajo de los oidos
        horizontal = float(np.linalg.norm(n - np.dot(n, up) * up))
        if both_ears:                                       # de frente la nariz casi no se separa horizontalmente
            horizontal = max(horizontal, float(np.linalg.norm(points["right_ear"] - points["left_ear"])) / 2.0)
        pitch = math.degrees(math.atan2(down, max(horizontal, 1e-6)))
        direction = _unit(n)
        if direction is not None:
            tip = nose + direction * 0.8 * scale
            head_vector = ((float(nose[0]), float(nose[1])), (float(tip[0]), float(tip[1])))

    roll: Optional[float] = None
    if both_ears and left_s is not None and right_s is not None:
        ear_line = points["right_ear"] - points["left_ear"]
        shoulder_line = right_s - left_s
        cross = ear_line[0] * shoulder_line[1] - ear_line[1] * shoulder_line[0]
        roll = abs(math.degrees(math.atan2(cross, float(np.dot(ear_line, shoulder_line)))))

    nose_height = float(np.dot(nose - shoulder_mid, up)) / scale if nose is not None and shoulder_mid is not None else None
    torso_ratio = None
    if hip_mid is not None and shoulder_mid is not None and shoulder_width > 0:
        torso_ratio = float(hip_mid[1] - shoulder_mid[1]) / shoulder_width

    required = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")
    return PostureSample(
        timestamp=timestamp,
        scale=scale,
        up=up,
        pitch_deg=pitch,
        roll_deg=roll,
        nose_height=nose_height,
        torso_ratio=torso_ratio,
        head_vector=head_vector,
        head_points={k: points[k] for k in HEAD_POINTS + HAND_POINTS if k in points},
        body_points={k: points[k] for k in BODY_POINTS if k in points},
        observability=sum(1 for k in required if k in points) / len(required),
    )


def motion_energy(samples: Deque[Tuple[float, float, Dict[str, np.ndarray]]]) -> Optional[float]:
    """Desviacion RMS de los puntos (normalizada por la escala) en la ventana.
    Solo usa puntos visibles en al menos la mitad de las muestras."""
    if len(samples) < 5:
        return None
    scale = float(np.median([s[1] for s in samples]))
    counts: Dict[str, int] = {}
    for _, _, pts in samples:
        for name in pts:
            counts[name] = counts.get(name, 0) + 1
    names = [n for n, c in counts.items() if c >= len(samples) / 2]
    if not names or scale <= 0:
        return None
    variances = []
    for name in names:
        coords = np.array([pts[name] for _, _, pts in samples if name in pts])
        variances.append(float(coords[:, 0].var() + coords[:, 1].var()))
    return math.sqrt(float(np.mean(variances))) / scale


# =============================================================================
# Acumuladores temporales
# =============================================================================

class _Hold:
    """Cuanto tiempo lleva una condicion siendo verdadera, tolerando huecos
    cortos (cuadros ruidosos o keypoints perdidos)."""

    def __init__(self, gap_tolerance: float) -> None:
        self._gap = gap_tolerance
        self.since: Optional[float] = None
        self.last_true: Optional[float] = None

    def update(self, condition: Optional[bool], now: float) -> float:
        if condition:
            if self.since is None:
                self.since = now
            self.last_true = now
        elif self.last_true is None or now - self.last_true > self._gap:
            self.since = None
        return self.duration(now)

    def duration(self, now: float) -> float:
        if self.since is None or self.last_true is None or now - self.last_true > self._gap:
            return 0.0
        return self.last_true - self.since


class _BobDetector:
    """Cabeceo: la nariz cae rapido (<= 1 s) mas de bob_drop_ratio anchos de
    hombro bajo su altura habitual y se recupera en <= 2.5 s. Una caida que
    no se recupera NO es un cabeceo (es cabeza caida sostenida)."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self._cfg = cfg
        self.baseline: Deque[float] = deque(maxlen=150)
        self.history: Deque[Tuple[float, float]] = deque(maxlen=60)
        self.down_since: Optional[float] = None
        self.peak_drop = 0.0
        self.events: Deque[float] = deque(maxlen=20)

    def update(self, height: Optional[float], now: float) -> None:
        if height is None:
            return
        self.history.append((now, height))
        if self.down_since is None:
            self.baseline.append(height)
        if len(self.baseline) < 10:
            return
        reference = float(np.median(self.baseline))
        drop = reference - height
        cfg = self._cfg
        if self.down_since is None:
            if drop > cfg.bob_drop_ratio:
                # Caida rapida: hace <= bob_max_fall_seconds la nariz estaba en su altura normal.
                recent = [h for t, h in self.history if now - t <= cfg.bob_max_fall_seconds]
                if recent and max(recent) > reference - cfg.bob_drop_ratio * 0.4:
                    self.down_since, self.peak_drop = now, drop
        else:
            self.peak_drop = max(self.peak_drop, drop)
            if now - self.down_since > cfg.bob_max_down_seconds:
                self.down_since = None  # caida sostenida: no es cabeceo
            elif drop < self.peak_drop * (1.0 - cfg.bob_recover_fraction):
                self.events.append(now)
                self.down_since = None

    def count(self, now: float) -> int:
        while self.events and now - self.events[0] > self._cfg.bob_window_seconds:
            self.events.popleft()
        return len(self.events)


@dataclass
class _Episode:
    behavior_type: str
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start: float = 0.0
    last_emit: float = 0.0
    state: Optional[str] = None


class _GuardTrack:
    def __init__(self, track_id: str, cfg: DrowsinessConfig) -> None:
        self.track_id = track_id
        self.camera_id = ""
        self.session_id = ""
        self.bbox: List[float] = [0, 0, 0, 0]
        self.class_name = "person"
        self.last_update = 0.0
        self.state = GuardState.ALERT_ACTIVE
        self.state_since: Optional[float] = None
        self.downgrade_since: Optional[float] = None
        self.lme_samples: Deque[Tuple[float, float, Dict[str, np.ndarray]]] = deque(maxlen=cfg.lme_window_frames)
        self.body_samples: Deque[Tuple[float, float, Dict[str, np.ndarray]]] = deque(maxlen=cfg.lme_window_frames)
        self.pitch_baseline: Deque[float] = deque(maxlen=300)
        self.torso_baseline: Deque[float] = deque(maxlen=300)
        self.torso_recent: Deque[Tuple[float, float]] = deque(maxlen=60)
        self.head_down = _Hold(cfg.gap_tolerance_seconds)
        self.lme_low = _Hold(cfg.gap_tolerance_seconds)
        self.immobile = _Hold(cfg.gap_tolerance_seconds)
        self.sleep_condition = _Hold(cfg.gap_tolerance_seconds)
        self.bobs = _BobDetector(cfg)
        self.static_anchor: Optional[Point] = None
        self.static_since: Optional[float] = None
        self.sudden_slump_at: Optional[float] = None
        self.episodes: Dict[str, _Episode] = {}
        self.last_status: Dict[str, Any] = {}
        self.head_dropped = False
        self.last_frame_id = 0
        # Verificacion Tier-2 (Gemini)
        self.pending_request: Optional[str] = None
        self.verification: Optional[Dict[str, Any]] = None
        self.sleep_confirmed = False
        self.local_fallback = False       # el juez fallo en este episodio: decide la heuristica local
        self.veto_until = 0.0
        self.distracted_activity: Optional[str] = None


# =============================================================================
# Servicio
# =============================================================================

class _BoundedQueue:
    def __init__(self, max_size: int) -> None:
        self._max = max_size
        self._items: Deque[Any] = deque()
        self._cond = threading.Condition()

    def put(self, item: Any) -> None:
        with self._cond:
            if len(self._items) >= self._max:
                self._items.popleft()
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        with self._cond:
            if not self._items:
                self._cond.wait(timeout=timeout)
            return self._items.popleft() if self._items else None

    def qsize(self) -> int:
        with self._cond:
            return len(self._items)


class DrowsinessService:
    """Analisis de somnolencia por guardia. Thread-safe: on_tracking y on_pose
    pueden llamarse desde hilos distintos."""

    def __init__(self, config: Optional[DrowsinessConfig] = None, judge: Any = None,
                 frame_provider: Optional[Callable[[str, int], Optional[np.ndarray]]] = None) -> None:
        """judge: GeminiJudgeService (opcional). frame_provider(camera_id, frame_id)
        devuelve el frame original para recortar al guardia."""
        self.config = config or DrowsinessConfig.from_env()
        self._judge = judge if judge is not None and frame_provider is not None else None
        self._frame_provider = frame_provider
        self._verify_requests: List[Tuple[_GuardTrack, str]] = []
        self._tracks: Dict[Tuple[str, str], _GuardTrack] = {}
        self._tracking_frames: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self._events = _BoundedQueue(1000)
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._metrics = {"frames_processed": 0, "events_emitted": 0, "transitions": 0, "total_ms": 0.0}

    # -------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """callback(resultado_por_frame) — ej. VisualizerService.on_guard_frame."""
        self._listeners.append(callback)

    def get_next_event(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Eventos compatibles con BehaviorResult (entrada de S7)."""
        return self._events.get(timeout=timeout)

    def on_tracking(self, result: Dict[str, Any]) -> None:
        """TrackingResult de S4: guarda cajas por frame y cierra tracks terminados."""
        camera_id = str(result.get("camera_id"))
        frame_id = int(result.get("frame_id", 0))
        tracks = list(result.get("tracks") or [])
        closing: List[Dict[str, Any]] = []
        with self._lock:
            frames = self._tracking_frames.setdefault(camera_id, {})
            frames[frame_id] = tracks
            for old in sorted(frames)[:-60]:
                del frames[old]
            now = float(result.get("timestamp") or 0.0)
            for track in tracks:
                if track.get("track_state") == "TERMINATED":
                    guard = self._tracks.pop((camera_id, str(track.get("track_id"))), None)
                    if guard is not None:
                        closing.extend(self._close_episodes(guard, now, "track_terminated"))
            for key in [k for k, g in self._tracks.items()
                        if k[0] == camera_id and now - g.last_update > self.config.track_ttl_seconds]:
                closing.extend(self._close_episodes(self._tracks.pop(key), now, "track_timeout"))
        for event in closing:
            self._emit(event)

    def on_pose(self, result: Dict[str, Any]) -> None:
        """SpecializedAIResult de S5: actualiza a cada guardia visible en ese frame."""
        started = time.perf_counter()
        camera_id = str(result.get("camera_id"))
        frame_id = int(result.get("frame_id", 0))
        now = float(result.get("timestamp") or time.time())
        session_id = str(result.get("session_id", ""))
        poses = {str(p.get("track_id")): p for p in result.get("poses") or []}
        events: List[Dict[str, Any]] = []
        guards: List[Dict[str, Any]] = []
        with self._lock:
            tracks = self._tracking_frames.get(camera_id, {}).get(frame_id)
            if tracks is None:  # sin tracking para ese frame: usar las cajas conocidas
                tracks = [{"track_id": g.track_id, "bbox": dict(zip(("x1", "y1", "x2", "y2"), g.bbox)),
                           "track_state": "ACTIVE", "class_name": g.class_name}
                          for (cam, _), g in self._tracks.items() if cam == camera_id and str(g.track_id) in poses]
            for track in tracks:
                if track.get("track_state") not in ("NEW", "ACTIVE", "RECOVERING") or track.get("class_name") != "person":
                    continue
                track_id = str(track.get("track_id"))
                bbox_raw = track.get("bbox") or {}
                try:
                    bbox = [float(bbox_raw[c]) for c in ("x1", "y1", "x2", "y2")]
                except (KeyError, TypeError, ValueError):
                    continue
                guard = self._tracks.get((camera_id, track_id))
                if guard is None:
                    guard = _GuardTrack(track_id, self.config)
                    guard.state_since = now
                    self._tracks[(camera_id, track_id)] = guard
                guard.camera_id, guard.session_id, guard.bbox = camera_id, session_id, bbox
                guard.last_frame_id = frame_id
                pose = poses.get(track_id)
                keypoints = pose.get("keypoints") if pose and pose.get("status") == "OK" else None
                events.extend(self._update_guard(guard, keypoints, now))
                guards.append(guard.last_status)
            self._metrics["frames_processed"] += 1
            self._metrics["total_ms"] += (time.perf_counter() - started) * 1000.0
            requests, self._verify_requests = self._verify_requests, []

        for event in events:
            self._emit(event)
        for guard, request_id in requests:
            self._dispatch_verification(guard, request_id)
        frame_result = {"camera_id": camera_id, "frame_id": frame_id, "timestamp": now, "guards": guards}
        for listener in self._listeners:
            try:
                listener(frame_result)
            except Exception:
                logger.exception("Listener de somnolencia fallo")

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            states: Dict[str, int] = {}
            for guard in self._tracks.values():
                states[guard.state.value] = states.get(guard.state.value, 0) + 1
            processed = self._metrics["frames_processed"]
            return {
                "guards_tracked": len(self._tracks),
                "states": states,
                "frames_processed": processed,
                "events_emitted": self._metrics["events_emitted"],
                "transitions": self._metrics["transitions"],
                "avg_processing_ms": round(self._metrics["total_ms"] / processed, 3) if processed else 0.0,
                "pending_events": self._events.qsize(),
                "config": {"head_drop_deg": self.config.head_drop_deg, "sleep_seconds": self.config.sleep_seconds,
                           "inactive_seconds": self.config.inactive_seconds},
            }

    # -------------------------------------------------------------------
    # Nucleo por guardia
    # -------------------------------------------------------------------

    def _update_guard(self, guard: _GuardTrack, keypoints: Any, now: float) -> List[Dict[str, Any]]:
        cfg = self.config
        guard.last_update = now
        points = parse_keypoints(keypoints) if keypoints else {}
        sample = compute_posture(points, guard.bbox, now) if points else None

        # --- 1. Cabeza: pitch relativo a la postura habitual + roll ---------
        pitch_rel = None
        head_drop: Optional[bool] = None
        if sample is not None and sample.pitch_deg is not None:
            baseline = (float(np.median(guard.pitch_baseline))
                        if len(guard.pitch_baseline) >= cfg.baseline_min_samples else 0.0)
            pitch_rel = sample.pitch_deg - min(baseline, cfg.pitch_baseline_cap_deg)
            # Histeresis: el ruido del estimador alrededor de 35 grados no debe
            # romper el acumulador de tiempo con la cabeza abajo.
            threshold = cfg.head_drop_deg - (cfg.head_drop_hysteresis_deg if guard.head_dropped else 0.0)
            head_drop = pitch_rel > threshold or (sample.roll_deg or 0.0) > threshold
            guard.head_dropped = head_drop

        # --- 2. Hombros encorvados ------------------------------------------
        slumped: Optional[bool] = None
        slump_value = None
        if sample is not None and sample.torso_ratio is not None:
            guard.torso_recent.append((now, sample.torso_ratio))
            if len(guard.torso_baseline) >= cfg.baseline_min_samples:
                reference = float(np.median(guard.torso_baseline))
                if reference > 0:
                    slump_value = 1.0 - sample.torso_ratio / reference
                    slumped = slump_value > cfg.slump_ratio
                    past = [r for t, r in guard.torso_recent if now - t <= 2.0]
                    if slumped and past and (max(past) - sample.torso_ratio) / reference > cfg.slump_ratio:
                        guard.sudden_slump_at = now  # colapso postural repentino

        # --- 3. LME (cabeza + manos) y movimiento corporal ----------------
        if sample is not None:
            guard.lme_samples.append((now, sample.scale, sample.head_points))
            guard.body_samples.append((now, sample.scale, sample.body_points))
        for samples in (guard.lme_samples, guard.body_samples):
            while samples and now - samples[0][0] > cfg.lme_window_seconds:
                samples.popleft()
        lme = motion_energy(guard.lme_samples)
        body_motion = motion_energy(guard.body_samples)
        lme_low = None if lme is None else lme < cfg.lme_low_threshold

        # --- 4. Cabeceo -----------------------------------------------------
        guard.bobs.update(sample.nose_height if sample else None, now)
        bob_count = guard.bobs.count(now)

        # --- Desplazamiento corporal (caja) --------------------------------
        cx, cy = (guard.bbox[0] + guard.bbox[2]) / 2.0, (guard.bbox[1] + guard.bbox[3]) / 2.0
        box_height = max(1.0, guard.bbox[3] - guard.bbox[1])
        if guard.static_anchor is None or math.hypot(cx - guard.static_anchor[0], cy - guard.static_anchor[1]) > \
                cfg.displacement_ratio * box_height:
            guard.static_anchor, guard.static_since = (cx, cy), now
        t_static = now - (guard.static_since or now)

        # --- Acumuladores ---------------------------------------------------
        t_head_down = guard.head_down.update(head_drop, now)
        t_lme_low = guard.lme_low.update(lme_low, now)
        body_still = None if body_motion is None else body_motion < cfg.body_motion_threshold
        # Sin pose: el cuerpo se considera inmovil si la caja no se desplaza.
        t_immobile = guard.immobile.update(body_still if body_still is not None else t_static > 0.5, now)
        sleeping_posture = bool(head_drop) or bool(slumped)
        t_sleep = guard.sleep_condition.update(sleeping_posture and bool(lme_low), now)

        # --- Estado objetivo y transicion con histeresis -------------------
        if t_sleep >= cfg.sleep_seconds:
            local_target = GuardState.SLEEPING_CRITICAL
        elif t_static >= cfg.inactive_seconds:
            local_target = GuardState.INACTIVE_GUARD_STATION
        elif (bool(head_drop) or bob_count >= cfg.bob_count_warning) and t_immobile > cfg.immobile_warning_seconds:
            local_target = GuardState.DROWSINESS_WARNING
        else:
            local_target = GuardState.ALERT_ACTIVE
        target = self._judged_target(guard, local_target, t_sleep, now)

        previous = guard.state
        # Subir de gravedad, igualar, o salir de la verificacion es inmediato;
        # bajar de gravedad exige recovery_seconds (histeresis).
        if _SEVERITY[target] >= _SEVERITY[guard.state] or guard.state == GuardState.VERIFYING_WITH_GEMINI:
            if target != guard.state:
                guard.state, guard.state_since = target, now
            guard.downgrade_since = None
        elif target != guard.state:
            guard.downgrade_since = guard.downgrade_since or now
            if now - guard.downgrade_since >= cfg.recovery_seconds:
                guard.state, guard.state_since, guard.downgrade_since = target, now, None
        else:
            guard.downgrade_since = None

        # La postura "normal" solo se aprende mientras el guardia esta despierto.
        if guard.state == GuardState.ALERT_ACTIVE and sample is not None and not head_drop:
            if sample.pitch_deg is not None:
                guard.pitch_baseline.append(sample.pitch_deg)
            if sample.torso_ratio is not None and not slumped:
                guard.torso_baseline.append(sample.torso_ratio)

        metrics = {
            "head_pitch_deg": None if sample is None or sample.pitch_deg is None else round(sample.pitch_deg, 1),
            "head_pitch_rel_deg": None if pitch_rel is None else round(pitch_rel, 1),
            "head_roll_deg": None if sample is None or sample.roll_deg is None else round(sample.roll_deg, 1),
            "head_drop": head_drop,
            "shoulder_slump": None if slump_value is None else round(slump_value, 3),
            "slumped": slumped,
            "sudden_slump": guard.sudden_slump_at is not None and now - guard.sudden_slump_at < 5.0,
            "lme": None if lme is None else round(lme, 4),
            "lme_level": None if lme_low is None else ("Low" if lme_low else "Normal"),
            "body_motion": None if body_motion is None else round(body_motion, 4),
            "bob_count_10s": bob_count,
            "t_head_down": round(t_head_down, 2),
            "t_lme_low": round(t_lme_low, 2),
            "t_immobile": round(t_immobile, 2),
            "t_sleep": round(t_sleep, 2),
            "t_static": round(t_static, 2),
            "observability": 0.0 if sample is None else round(sample.observability, 2),
            "verification": self._verification_summary(guard, now),
        }
        # Vector de cabeza: misma direccion oido->nariz, longitud proporcional a
        # la inclinacion (corto con la cabeza erguida, largo al caer).
        head_vector = None
        if sample is not None and sample.head_vector is not None:
            (nx, ny), (tx, ty) = sample.head_vector
            direction = _unit(np.array([tx - nx, ty - ny]))
            tilt = max(pitch_rel or 0.0, sample.roll_deg or 0.0)
            if direction is not None:
                length = sample.scale * float(np.clip(tilt / 50.0, 0.12, 1.0))
                head_vector = ((nx, ny), (nx + float(direction[0]) * length, ny + float(direction[1]) * length))

        guard.last_status = {
            "track_id": guard.track_id,
            "camera_id": guard.camera_id,
            "bbox": list(guard.bbox),
            "state": guard.state.value,
            "state_label": STATE_LABELS[guard.state],
            "state_seconds": round(now - (guard.state_since or now), 2),
            "metrics": metrics,
            "head_vector": head_vector,
            "keypoints": keypoints,
            "verification": metrics["verification"],
        }

        events: List[Dict[str, Any]] = []
        if guard.state != previous:
            self._metrics["transitions"] += 1
            logger.info("Guardia %s: %s -> %s | pitch=%s LME=%s cabeceos=%d t_sleep=%.1fs",
                        guard.track_id, previous.value, guard.state.value, metrics["head_pitch_rel_deg"],
                        metrics["lme"], bob_count, t_sleep)
        events.extend(self._episode_events(guard, now, metrics))
        return events

    # -------------------------------------------------------------------
    # Verificacion Tier-2 con Gemini
    # -------------------------------------------------------------------

    def _judged_target(self, guard: _GuardTrack, local_target: GuardState, t_sleep: float,
                       now: float) -> GuardState:
        """Aplica el veredicto de Gemini sobre el estado propuesto por la
        heuristica local. Sin juez (o si fallo) manda la heuristica local."""
        cfg = self.config
        if self._judge is None:
            return local_target

        if t_sleep <= 0.0:  # la condicion de sueno termino: nuevo episodio
            guard.local_fallback = False
            if guard.sleep_confirmed:
                guard.sleep_confirmed = False
            if guard.pending_request is not None:  # desperto mientras se verificaba
                guard.pending_request = None
                guard.verification = {**(guard.verification or {}), "status": "cancelled", "finished": now}

        if guard.sleep_confirmed:
            return GuardState.SLEEPING_CRITICAL

        if now < guard.veto_until:
            if guard.distracted_activity and guard.distracted_activity != "ninguna":
                return GuardState.WORKING_DISTRACTED
            return GuardState.ALERT_ACTIVE

        if guard.pending_request is not None:
            started = (guard.verification or {}).get("started", now)
            if now - started <= cfg.verification_timeout_seconds:
                return GuardState.VERIFYING_WITH_GEMINI
            guard.pending_request = None  # sin respuesta a tiempo: fail-safe local
            guard.local_fallback = True
            guard.verification = {**(guard.verification or {}), "status": "error", "error": "no_response",
                                  "finished": now}
            logger.warning("Guardia %s: Gemini no respondio en %.0f s, decide la heuristica local",
                           guard.track_id, cfg.verification_timeout_seconds)
            return local_target

        if (t_sleep >= cfg.verify_after_seconds and not guard.local_fallback
                and self._judge.can_verify(f"{guard.camera_id}:{guard.track_id}")):
            request_id = uuid.uuid4().hex
            guard.pending_request = request_id
            guard.verification = {"status": "pending", "started": now, "request_id": request_id}
            self._verify_requests.append((guard, request_id))
            return GuardState.VERIFYING_WITH_GEMINI

        return local_target

    def _dispatch_verification(self, guard: _GuardTrack, request_id: str) -> None:
        """Fuera del lock: recorta al guardia y lo envia al juez (no bloquea)."""
        from IA.gemini_judge.geminiJudgeService import crop_with_context

        frame = self._frame_provider(guard.camera_id, guard.last_frame_id) if self._frame_provider else None
        crop = crop_with_context(frame, guard.bbox, self._judge.config.padding_ratio) if frame is not None else None
        metrics = guard.last_status.get("metrics", {})
        context = {
            "cabeza_inclinada_grados": metrics.get("head_pitch_rel_deg"),
            "segundos_inmovil": metrics.get("t_sleep"),
            "movimiento_LME": metrics.get("lme_level"),
            "hombros_encorvados": metrics.get("slumped"),
        }
        accepted = crop is not None and self._judge.verify_drowsiness_async(
            crop, f"{guard.camera_id}:{guard.track_id}",
            partial(self._on_verdict, guard.camera_id, guard.track_id, request_id), context)
        if not accepted:
            self._on_verdict(guard.camera_id, guard.track_id, request_id, None, None,
                             "not_sent" if crop is not None else "no_frame")

    def _on_verdict(self, camera_id: str, track_id: str, request_id: str, _judge_key: Optional[str],
                    verdict: Optional[Dict[str, Any]], error: Optional[str]) -> None:
        """Callback del juez (hilo del pool). El cambio de estado se aplica en
        el siguiente frame del guardia (_judged_target)."""
        with self._lock:
            guard = self._tracks.get((camera_id, track_id))
            if guard is None or guard.pending_request != request_id:
                return  # veredicto obsoleto (el guardia desperto o desaparecio)
            guard.pending_request = None
            now = guard.last_update
            base = {**(guard.verification or {}), "finished": now}
            if verdict is None:
                guard.local_fallback = True
                guard.verification = {**base, "status": "error", "error": error or "unknown"}
            elif verdict.get("is_sleeping"):
                guard.sleep_confirmed = True
                guard.verification = {**base, "status": "confirmed", "verdict": verdict}
            else:
                guard.verification = {**base, "status": "rejected", "verdict": verdict}
                guard.distracted_activity = verdict.get("activity_detected")
                guard.veto_until = now + self.config.awake_veto_seconds
                # Gemini dice despierto: reinicio silencioso de los acumuladores.
                gap = self.config.gap_tolerance_seconds
                guard.sleep_condition, guard.head_down = _Hold(gap), _Hold(gap)
                guard.lme_low, guard.immobile = _Hold(gap), _Hold(gap)

    @staticmethod
    def _verification_summary(guard: _GuardTrack, now: float) -> Optional[Dict[str, Any]]:
        info = guard.verification
        if info is None:
            return None
        verdict = info.get("verdict") or {}
        status = info.get("status")
        return {
            "status": status,
            "elapsed": round((info.get("finished") or now) - info.get("started", now), 2),
            # Del ultimo veredicto (persiste mientras dura la histeresis de salida).
            "confirmed_by": {"confirmed": "gemini", "error": "local"}.get(str(status)),
            "is_sleeping": verdict.get("is_sleeping"),
            "confidence": verdict.get("confidence"),
            "activity": verdict.get("activity_detected"),
            "eyes": verdict.get("eyes_state"),
            "reasoning": verdict.get("reasoning"),
            "error": info.get("error"),
        }

    # -------------------------------------------------------------------
    # Eventos para S7 (formato BehaviorResult)
    # -------------------------------------------------------------------

    def _episode_events(self, guard: _GuardTrack, now: float, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        desired: Dict[str, str] = {}
        if guard.state in (GuardState.DROWSINESS_WARNING, GuardState.VERIFYING_WITH_GEMINI):
            desired["guard_sleep"] = "POSSIBLE"
        elif guard.state == GuardState.SLEEPING_CRITICAL:
            desired["guard_sleep"] = "SUSTAINED"
        elif guard.state == GuardState.INACTIVE_GUARD_STATION:
            desired["guard_inactive"] = "SUSTAINED"

        events: List[Dict[str, Any]] = []
        for behavior_type in [b for b in guard.episodes if b not in desired]:
            reason = "gemini_awake" if guard.state == GuardState.WORKING_DISTRACTED else "recovered"
            events.append(self._behavior_event(guard, guard.episodes.pop(behavior_type), "FINISHED", now, metrics,
                                               reason))
        for behavior_type, state in desired.items():
            episode = guard.episodes.get(behavior_type)
            if episode is None:
                episode = _Episode(behavior_type=behavior_type, start=now)
                guard.episodes[behavior_type] = episode
            if episode.state != state or now - episode.last_emit >= self.config.heartbeat_seconds:
                episode.state = state
                episode.last_emit = now
                events.append(self._behavior_event(guard, episode, state, now, metrics, None))
        return events

    def _close_episodes(self, guard: _GuardTrack, now: float, reason: str) -> List[Dict[str, Any]]:
        events = [self._behavior_event(guard, ep, "FINISHED", now, guard.last_status.get("metrics", {}), reason)
                  for ep in guard.episodes.values()]
        guard.episodes.clear()
        return events

    def _behavior_event(self, guard: _GuardTrack, episode: _Episode, state: str, now: float,
                        metrics: Dict[str, Any], finish_reason: Optional[str]) -> Dict[str, Any]:
        duration = max(0.0, now - episode.start)
        if episode.behavior_type == "guard_sleep":
            label = {"POSSIBLE": "somnolencia", "SUSTAINED": "sueño sostenido", "FINISHED": "sueño finalizado"}[state]
            strength = min(1.0, (metrics.get("t_sleep") or 0.0) / self.config.sleep_seconds) if state != "POSSIBLE" else 0.5
        else:
            label = {"SUSTAINED": "inactividad prolongada", "FINISHED": "inactividad finalizada"}.get(state, "inactividad")
            strength = min(1.0, (metrics.get("t_static") or 0.0) / self.config.inactive_seconds)
        evidence = {**metrics, "guard_state": guard.state.value}
        if finish_reason:
            evidence["finish_reason"] = finish_reason
        return {
            "result_id": str(uuid.uuid4()),
            "episode_id": episode.episode_id,
            "camera_id": guard.camera_id,
            "session_id": guard.session_id or "unknown",
            "track_id": guard.track_id,
            "subject_id": f"{guard.camera_id}:{guard.track_id}",
            "timestamp": now,
            "behavior_type": episode.behavior_type,
            "state": state,
            "start_time": episode.start,
            "duration": round(duration, 3),
            "last_update": now,
            "zone_id": None,
            "evidence_strength": round(strength, 3),
            "observability": metrics.get("observability", 0.0),
            "evidence": evidence,
            "description": f"Evidencia de {label} del guardia durante {duration:.1f} s",
        }

    def _emit(self, event: Dict[str, Any]) -> None:
        self._metrics["events_emitted"] += 1
        self._events.put(event)
