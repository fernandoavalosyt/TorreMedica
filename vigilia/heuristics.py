"""Heuristica local de somnolencia a partir de los puntos del cuerpo.

Calcula una puntuacion 0-1 por frame (postura + ojos + bostezos + inmovilidad),
la suaviza en el tiempo y la convierte en un estado local:
    despierta -> somnolienta -> dormida
Es barata (sin red neuronal propia) y su funcion principal es decidir CUANDO
vale la pena gastar una consulta a Gemini. Los umbrales son punto de partida:
calibrar con video real de la zona de guardia.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from .vision import (KP_CONF, L_EAR, L_EYE, L_HIP, L_SH, NOSE, R_EAR, R_EYE, R_HIP, R_SH, FaceMetrics, Person)

MOTION_THRESHOLD = 0.035      # desplazamiento medio (en anchos de hombro) por analisis; por debajo = quieto
YAWN_JAW_OPEN = 0.55
YAWN_MIN_SECONDS = 1.2
SUSPECT_SCORE = 0.40
SLEEP_SCORE = 0.60
EYES_WINDOW = 10.0            # s para el porcentaje de ojos cerrados (PERCLOS)
YAWN_WINDOW = 120.0


def _ramp(value: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def _pt(kpts: np.ndarray, idx: int) -> Optional[np.ndarray]:
    return kpts[idx, :2] if kpts[idx, 2] >= KP_CONF else None


def _mean_pts(*points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    valid = [p for p in points if p is not None]
    return np.mean(valid, axis=0) if valid else None


@dataclass
class PoseMetrics:
    head_drop: Optional[float]       # (cabeza - hombros)/ancho de hombros; ~-0.5 erguido, >0 cabeza sobre el pecho
    head_lateral: Optional[float]
    tilt_deg: Optional[float]
    torso_deg: Optional[float]       # inclinacion del tronco respecto a la vertical
    aspect: float                    # ancho/alto del bbox
    face_visible: bool
    head_hidden: bool                # hombros visibles pero ningun punto de la cabeza
    scale: float
    posture_score: float
    posture: str


def analyze_pose(kpts: np.ndarray, bbox: np.ndarray) -> PoseMetrics:
    bw, bh = max(1.0, float(bbox[2] - bbox[0])), max(1.0, float(bbox[3] - bbox[1]))
    ls, rs = _pt(kpts, L_SH), _pt(kpts, R_SH)
    both_sh = ls is not None and rs is not None
    sh_mid = (ls + rs) / 2 if both_sh else None
    scale = max(float(np.linalg.norm(ls - rs)) if both_sh else 0.0, 0.22 * bh)

    nose, le, re = _pt(kpts, NOSE), _pt(kpts, L_EYE), _pt(kpts, R_EYE)
    lear, rear = _pt(kpts, L_EAR), _pt(kpts, R_EAR)
    head = nose if nose is not None else _mean_pts(le, re) if (le is not None or re is not None) \
        else _mean_pts(lear, rear)
    face_visible = nose is not None and (le is not None or re is not None)
    head_hidden = sh_mid is not None and head is None

    head_drop = head_lateral = None
    if head is not None and sh_mid is not None:
        head_drop = float((head[1] - sh_mid[1]) / scale)
        head_lateral = float(abs(head[0] - sh_mid[0]) / scale)

    tilt = None
    pair = (le, re) if le is not None and re is not None else (lear, rear) if lear is not None and rear is not None \
        else None
    if pair is not None:
        angle = abs(math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0])))
        tilt = min(angle, 180.0 - angle)

    torso = None
    lh, rh = _pt(kpts, L_HIP), _pt(kpts, R_HIP)
    if sh_mid is not None and lh is not None and rh is not None:
        vec = (lh + rh) / 2 - sh_mid
        torso = math.degrees(math.atan2(abs(float(vec[0])), float(vec[1])))

    aspect = bw / bh
    score = (0.50 * (_ramp(head_drop, -0.35, 0.05) if head_drop is not None else 0.0)
             + 0.15 * (_ramp(head_lateral, 0.35, 1.0) if head_lateral is not None else 0.0)
             + 0.15 * (_ramp(tilt, 15, 45) if tilt is not None else 0.0)
             + 0.25 * (_ramp(torso, 25, 60) if torso is not None else 0.0)
             + 0.45 * _ramp(aspect, 1.0, 1.6)
             + 0.25 * (1.0 if head_hidden else 0.0))
    score = min(1.0, score)

    if aspect >= 1.3 or (torso is not None and torso >= 60):
        posture = "recostado"
    elif head_drop is not None and head_drop > -0.15:
        posture = "cabeza_baja"
    elif head_hidden:
        posture = "sin_cabeza"
    elif (tilt is not None and tilt >= 25) or (head_lateral is not None and head_lateral >= 0.6):
        posture = "cabeza_ladeada"
    elif torso is not None and torso >= 30:
        posture = "inclinado"
    elif not face_visible:
        posture = "sin_rostro"
    else:
        posture = "erguido"
    return PoseMetrics(head_drop, head_lateral, tilt, torso, aspect, face_visible, head_hidden, scale, score, posture)


class Sustain:
    """Mide cuanto tiempo se cumple una condicion, tolerando huecos cortos (`grace`)."""

    def __init__(self, grace: float = 2.0) -> None:
        self.grace = grace
        self.since: Optional[float] = None
        self._last_true = 0.0

    def update(self, condition: bool, now: float) -> float:
        if condition:
            if self.since is None:
                self.since = now
            self._last_true = now
        elif self.since is not None and now - self._last_true > self.grace:
            self.since = None
        return 0.0 if self.since is None else now - self.since

    def reset(self) -> None:
        self.since = None


@dataclass
class TrackState:
    track_id: int
    camera_id: str
    first_seen: float
    last_seen: float
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    kpts: np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
    metrics: Optional[PoseMetrics] = None
    updated: bool = False
    # puntuacion local
    score: float = 0.0
    score_ema: float = 0.0
    local_state: str = "despierta"
    # movimiento
    prev_kpts: Optional[np.ndarray] = None
    motion_ema: float = 1.0
    still_since: Optional[float] = None
    # cara
    last_face: Optional[FaceMetrics] = None
    eyes: Deque[Tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=120))
    face_seen: Deque[Tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=200))
    yawn_times: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    yawn_flag: bool = False
    # temporizadores
    yawn_sustain: Sustain = field(default_factory=lambda: Sustain(1.0))
    suspect_timer: Sustain = field(default_factory=Sustain)
    sleep_timer: Sustain = field(default_factory=Sustain)
    # Gemini
    verdict: Optional[Dict[str, Any]] = None
    verdict_time: float = 0.0
    veto_until: float = 0.0
    review_pending: bool = False
    # recortes JPEG recientes para Gemini: (timestamp, bytes)
    crops: Deque[Tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=12))
    last_crop_time: float = 0.0
    # decision final / alertas
    level: str = "ok"                # ok | somnolencia | dormido
    level_source: str = ""
    level_reason: str = ""
    ok_since: Optional[float] = None
    alert_level_sent: str = "ok"
    alert_sent_at: float = 0.0

    # ---- indicadores derivados -------------------------------------------
    def still_seconds(self, now: float) -> float:
        return 0.0 if self.still_since is None else now - self.still_since

    def eyes_closed_ratio(self, now: float) -> Optional[float]:
        samples = [closed for t, closed in self.eyes if now - t <= EYES_WINDOW]
        return float(np.mean(samples)) if len(samples) >= 3 else None

    def face_visible_ratio(self, now: float) -> float:
        samples = [v for t, v in self.face_seen if now - t <= 20.0]
        return float(np.mean(samples)) if samples else 0.0

    def yawns_recent(self, now: float) -> int:
        return sum(1 for t in self.yawn_times if now - t <= YAWN_WINDOW)

    def summary(self, now: float) -> Dict[str, Any]:
        """Indicadores compactos (se envian a Gemini como contexto y a la API)."""
        closed = self.eyes_closed_ratio(now)
        return {
            "postura": self.metrics.posture if self.metrics else "sin_datos",
            "puntuacion_local": round(self.score_ema, 2),
            "ojos_cerrados_pct": None if closed is None else round(closed * 100),
            "bostezos_2min": self.yawns_recent(now),
            "inmovil_seg": round(self.still_seconds(now)),
            "rostro_visible_pct": round(self.face_visible_ratio(now) * 100),
        }


def update_track(ts: TrackState, person: Person, face: Optional[FaceMetrics], now: float,
                 suspect_seconds: float, sleep_seconds: float) -> None:
    """Actualiza el estado de un track con un nuevo analisis de pose (y cara si hubo)."""
    dt = min(2.0, max(0.05, now - ts.last_seen)) if ts.updated else 0.2
    ts.last_seen, ts.updated = now, True
    ts.bbox, ts.kpts = person.bbox, person.kpts
    metrics = analyze_pose(person.kpts, person.bbox)
    ts.metrics = metrics

    # Movimiento de la parte superior del cuerpo (nariz..munecas), en anchos de hombro.
    if ts.prev_kpts is not None:
        both = (person.kpts[:11, 2] >= KP_CONF) & (ts.prev_kpts[:11, 2] >= KP_CONF)
        if both.sum() >= 3:
            disp = np.linalg.norm(person.kpts[:11, :2][both] - ts.prev_kpts[:11, :2][both], axis=1).mean()
            ts.motion_ema = 0.6 * ts.motion_ema + 0.4 * float(disp / metrics.scale)
    ts.prev_kpts = person.kpts.copy()
    if ts.motion_ema > MOTION_THRESHOLD:
        ts.still_since = None
    elif ts.still_since is None:
        ts.still_since = now

    ts.face_seen.append((now, metrics.face_visible))
    if face is not None:
        ts.last_face = face
        ts.eyes.append((now, face.eyes_closed))
        yawning_for = ts.yawn_sustain.update(face.jaw_open >= YAWN_JAW_OPEN, now)
        if yawning_for >= YAWN_MIN_SECONDS and not ts.yawn_flag:
            ts.yawn_flag = True
            ts.yawn_times.append(now)
        elif yawning_for == 0.0:
            ts.yawn_flag = False

    closed = ts.eyes_closed_ratio(now)
    yawns = ts.yawns_recent(now)
    total = (metrics.posture_score
             + (0.45 * closed if closed is not None else 0.0)
             + (0.20 if yawns >= 3 else 0.10 if yawns >= 1 else 0.0)
             + 0.20 * _ramp(ts.still_seconds(now), 15, 60))
    ts.score = min(1.0, total)
    alpha = 1.0 - math.exp(-dt / 2.0)
    ts.score_ema += alpha * (ts.score - ts.score_ema)

    suspect_for = ts.suspect_timer.update(ts.score_ema >= SUSPECT_SCORE, now)
    sleep_for = ts.sleep_timer.update(ts.score_ema >= SLEEP_SCORE, now)
    if sleep_for >= sleep_seconds:
        ts.local_state = "dormida"
    elif suspect_for >= suspect_seconds:
        ts.local_state = "somnolienta"
    else:
        ts.local_state = "despierta"
