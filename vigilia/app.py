"""Supervisor: crea y administra los pipelines de camara en caliente."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from .alerts import AlertManager
from .config import CameraConfig, CameraStore, Settings
from .gemini import GeminiJudge
from .pipeline import CameraPipeline
from .vision import ensure_model

logger = logging.getLogger("vigilia.app")


class Vigilia:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = CameraStore(settings.data_dir / "cameras.json", settings.max_cameras)
        self.alerts = AlertManager(settings)
        self.judge = GeminiJudge(settings)
        self.model_path = ensure_model(settings.models_dir, settings.yolo_model)
        self._pipelines: Dict[str, CameraPipeline] = {}
        self._lock = threading.Lock()
        self.started = time.time()

    def start(self) -> None:
        for cam in self.store.list():
            if cam.enabled:
                self._start_pipeline(cam)
        if not self._pipelines:
            logger.warning("Sin camaras configuradas: agreguelas desde el panel web.")

    def stop(self) -> None:
        with self._lock:
            pipelines = list(self._pipelines.values())
            self._pipelines.clear()
        for p in pipelines:
            p.stop()
        for p in pipelines:
            p.join(timeout=5)
        self.judge.shutdown()

    def _start_pipeline(self, cam: CameraConfig) -> None:
        pipeline = CameraPipeline(cam, self.s, self.model_path, self.judge, self.alerts)
        with self._lock:
            self._pipelines[cam.id] = pipeline
        pipeline.start()

    # ---- administracion de camaras (API) ----------------------------------
    def save_camera(self, cam: CameraConfig) -> CameraConfig:
        cam = self.store.upsert(cam)
        self._stop_pipeline(cam.id)
        if cam.enabled:
            self._start_pipeline(cam)
        return cam

    def delete_camera(self, cam_id: str) -> bool:
        removed = self.store.remove(cam_id)
        self._stop_pipeline(cam_id)
        return removed

    def _stop_pipeline(self, cam_id: str) -> None:
        with self._lock:
            pipeline = self._pipelines.pop(cam_id, None)
        if pipeline:
            pipeline.stop()
            pipeline.join(timeout=5)

    def pipeline(self, cam_id: str) -> Optional[CameraPipeline]:
        with self._lock:
            return self._pipelines.get(cam_id)

    # ---- estado -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            pipelines = list(self._pipelines.values())
        cameras: List[Dict[str, Any]] = [p.status() for p in pipelines]
        running = {c["camera"]["id"] for c in cameras}
        stopped = [{"camera": c.public(), "stream": {"state": "disabled"}, "people": [], "fps": 0}
                   for c in self.store.list() if c.id not in running]
        return {"uptime_seconds": round(time.time() - self.started), "cameras": cameras + stopped,
                "gemini": self.judge.health(), "active_alerts": self.alerts.active_count(),
                "max_cameras": self.s.max_cameras}
