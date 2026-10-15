"""Pipeline de una camara: captura -> optimizador -> YOLO-pose+tracking ->
heuristica -> (Gemini si hay sospecha) -> decision -> alerta -> video anotado."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .alerts import AlertManager
from .camera import CameraStream
from .config import CameraConfig, Settings
from .decision import decide
from .gemini import GeminiJudge
from .heuristics import TrackState, update_track
from .optimizer import FpsGate, FrameOptimizer
from .render import draw_tracks, placeholder
from .vision import FaceAnalyzer, Person, PoseTracker

logger = logging.getLogger("vigilia.pipeline")

TRACK_TTL = 10.0              # s sin ver a una persona para olvidar su ID
CROP_INTERVAL = 1.5           # s entre recortes guardados para Gemini
FACE_INTERVAL = 0.5           # s entre analisis faciales por persona
MIN_PERSON_HEIGHT = 0.10      # fraccion del alto del frame: ignora personas muy lejanas
REVIEW_REFRESH = 60.0         # s antes de volver a consultar a una persona ya dictaminada


def _in_roi(roi: Optional[List[List[float]]], point_xy: tuple, w: int, h: int) -> bool:
    if not roi:
        return True
    polygon = np.array([[x * w, y * h] for x, y in roi], np.float32)
    return cv2.pointPolygonTest(polygon, (float(point_xy[0]), float(point_xy[1])), False) >= 0


class CameraPipeline(threading.Thread):
    def __init__(self, cfg: CameraConfig, settings: Settings, model_path: str, judge: GeminiJudge,
                 alerts: AlertManager) -> None:
        super().__init__(name=f"pipeline-{cfg.id}", daemon=True)
        self.cfg, self.s, self.judge, self.alerts = cfg, settings, judge, alerts
        self.stream = CameraStream(cfg)
        self.optimizer = FrameOptimizer(settings.max_side)
        self.tracker = PoseTracker(model_path, settings.device, settings.yolo_conf)
        self.face = FaceAnalyzer(settings.models_dir) if settings.face_analysis != "off" else None
        if settings.face_analysis == "on" and self.face is not None and not self.face.available:
            raise RuntimeError(f"FACE_ANALYSIS=on pero no esta disponible: {self.face.reason}")
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._tracks: Dict[int, TrackState] = {}
        self._jpeg: Optional[bytes] = None
        self._last_seq = -1
        self._fps_out = 0.0
        self._last_infer = 0.0
        self._last_face: Dict[int, float] = {}
        self._had_people = False
        self._offline_flagged = False

    # ---- API para la web ---------------------------------------------------
    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            people = [{
                "id": ts.track_id, "level": ts.level, "source": ts.level_source, "local_state": ts.local_state,
                "reason": ts.level_reason, "seconds_visible": round(now - ts.first_seen),
                "verdict": ts.verdict, **ts.summary(now),
            } for ts in sorted(self._tracks.values(), key=lambda t: t.track_id)]
        return {"camera": self.cfg.public(), "stream": self.stream.status(), "fps": round(self._fps_out, 1),
                "dark_mode": self.optimizer.is_dark, "face_analysis": bool(self.face and self.face.available),
                "people": people}

    def stop(self) -> None:
        self._stop_event.set()
        self.stream.stop()

    # ---- bucle -----------------------------------------------------------
    def run(self) -> None:
        self.stream.start()
        gate = FpsGate(self.s.inference_fps)
        while not self._stop_event.is_set():
            try:
                self._tick(gate)
            except Exception:  # noqa: BLE001 - un frame malo no debe tumbar la camara
                logger.exception("[%s] error procesando frame", self.cfg.id)
                self._stop_event.wait(0.5)
        self.stream.join(timeout=3)

    def _tick(self, gate: FpsGate) -> None:
        frame, seq, frame_time = self.stream.latest()
        now = time.time()
        self._check_offline()
        if frame is None:
            with self._lock:
                self._tracks.clear()
            self._publish(placeholder(f"{self.cfg.name}: SIN SENAL"))
            self._stop_event.wait(0.5)
            return
        # Con la zona vacia se analiza mas despacio: ahorra CPU.
        gate.set_fps(self.s.inference_fps if self._had_people else self.s.idle_fps)
        if seq == self._last_seq or not gate.ready(now):
            self._stop_event.wait(0.02)
            return
        self._last_seq = seq

        small, scale = self.optimizer.process(frame)
        h, w = small.shape[:2]
        people = [p for p in self.tracker.infer(small, self.s.max_side) if self._accept(p, w, h)]
        self._had_people = bool(people)
        started = time.time()
        self._fps_out = 0.9 * self._fps_out + 0.1 / max(1e-3, started - self._last_infer) if self._last_infer \
            else self._fps_out
        self._last_infer = started

        with self._lock:
            seen = set()
            for person in people:
                ts = self._tracks.get(person.track_id)
                if ts is None:
                    ts = self._tracks[person.track_id] = TrackState(person.track_id, self.cfg.id, now, now)
                seen.add(person.track_id)
                face = self._face_metrics(ts, small, person, now)
                update_track(ts, person, face, now, self.s.suspect_seconds, self.s.sleep_seconds)
                self._store_crop(ts, frame, scale, now)
            for track_id in [t for t, ts in self._tracks.items() if now - ts.last_seen > TRACK_TTL]:
                del self._tracks[track_id]
                self._last_face.pop(track_id, None)
            tracks = list(self._tracks.values())

            for ts in tracks:
                if ts.track_id not in seen:
                    continue
                self._maybe_review(ts, now)
                # La evidencia se dibuja solo si realmente se emite una alerta.
                self.alerts.evaluate(ts, decide(ts, now, self.s), now, self.cfg.name,
                                     lambda: draw_tracks(small, tracks, self.cfg.name, self._fps_out, self.cfg.roi))

        self._publish(draw_tracks(small, tracks, self.cfg.name, self._fps_out, self.cfg.roi, self.optimizer.is_dark))

    # ---- piezas -------------------------------------------------------------
    def _accept(self, person: Person, w: int, h: int) -> bool:
        if (person.bbox[3] - person.bbox[1]) < MIN_PERSON_HEIGHT * h:
            return False
        # La zona de guardia se evalua con el punto medio del torso para tolerar escritorios que ocultan las piernas.
        center = ((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2)
        return _in_roi(self.cfg.roi, center, w, h)

    def _face_metrics(self, ts: TrackState, small: np.ndarray, person: Person, now: float):
        if self.face is None or not self.face.available or now - self._last_face.get(ts.track_id, 0) < FACE_INTERVAL:
            return None
        self._last_face[ts.track_id] = now
        return self.face.analyze(small, person.kpts)

    def _store_crop(self, ts: TrackState, frame: np.ndarray, scale: float, now: float) -> None:
        """Guarda recortes pequenos (cabeza y torso, resolucion original) para Gemini."""
        if self.judge.disabled_reason is not None:  # sin Gemini no hace falta guardar recortes
            return
        if now - ts.last_crop_time < CROP_INTERVAL:
            return
        ts.last_crop_time = now
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = (ts.bbox / scale)
        pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.10
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right, bottom = min(fw, int(x2 + pad_x)), min(fh, int(y2 + pad_y))
        if right - left < 24 or bottom - top < 24:
            return
        crop = frame[top:bottom, left:right]
        if self.optimizer.is_dark:
            crop = self.optimizer.enhance(crop)
        side = self.s.gemini_crop_side
        longest = max(crop.shape[:2])
        if longest > side:
            k = side / longest
            crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            ts.crops.append((now, jpeg.tobytes()))

    def _maybe_review(self, ts: TrackState, now: float) -> None:
        if ts.local_state == "despierta" or ts.review_pending or len(ts.crops) < 2:
            return
        if ts.verdict is not None and now - ts.verdict_time < REVIEW_REFRESH:
            return
        key = f"{self.cfg.id}-{ts.track_id}"
        recent = [(t, j) for t, j in ts.crops if now - t <= 12.0] or list(ts.crops)
        n = self.s.gemini_frames
        step = max(1, len(recent) / n)
        chosen = [recent[min(len(recent) - 1, int(i * step))] for i in range(n)] if len(recent) > n else recent
        frames = [(now - t, j) for t, j in chosen]
        context = {**ts.summary(now), "estado_local": ts.local_state, "visible_seg": round(now - ts.first_seen)}
        ts.review_pending = True
        if not self.judge.review(key, frames, context, lambda k, v, e, track=ts: self._on_verdict(track, v, e)):
            ts.review_pending = False

    def _on_verdict(self, ts: TrackState, verdict: Optional[Dict[str, Any]], error: Optional[str]) -> None:
        now = time.time()
        with self._lock:
            ts.review_pending = False
            if verdict is None:
                return
            ts.verdict, ts.verdict_time = verdict, now
            if verdict["state"] == "despierta" and verdict["confidence"] >= self.s.min_conf_drowsy:
                ts.veto_until = now + self.s.veto_seconds
            else:
                ts.veto_until = 0.0

    def _check_offline(self) -> None:
        offline = self.stream.offline_seconds()
        if offline >= self.s.camera_offline_seconds:
            self.alerts.system_alert(self.cfg.id, self.cfg.name, "offline",
                                     f"Camara sin senal desde hace {offline:.0f} s: no hay vigilancia en esta zona")
            self._offline_flagged = True
        elif self._offline_flagged and offline == 0.0:
            self.alerts.system_alert(self.cfg.id, self.cfg.name, "offline", None)
            self._offline_flagged = False

    def _publish(self, image: np.ndarray) -> None:
        ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._lock:
                self._jpeg = jpeg.tobytes()
