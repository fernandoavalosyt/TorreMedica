"""Alertas: deduplicacion por persona, evidencia (foto anotada), registro JSONL,
webhook opcional y estado de reconocimiento (ack) para el panel."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from .config import Settings
from .decision import LEVEL_RANK
from .heuristics import TrackState

logger = logging.getLogger("vigilia.alerts")

RESOLVE_AFTER_OK_SECONDS = 8.0


class AlertManager:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.dir = settings.data_dir / "alerts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.dir / "alerts.jsonl"
        self._lock = threading.Lock()
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._system_active: set[str] = set()
        self._load_recent()
        self._purge_old()

    # ---- consulta ----------------------------------------------------------
    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-limit:][::-1]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._alerts if not a["ack"] and a["level"] != "resuelto")

    def ack(self, alert_id: str) -> bool:
        with self._lock:
            for alert in self._alerts:
                if alert["id"] == alert_id:
                    alert["ack"] = True
                    self._append_log({"event": "ack", "id": alert_id, "time": time.time()})
                    return True
        return False

    def snapshot_path(self, name: str) -> Optional[Path]:
        path = (self.dir / name).resolve()
        return path if path.parent == self.dir.resolve() and path.is_file() else None

    # ---- alertas de personas ---------------------------------------------
    def evaluate(self, ts: TrackState, decision: tuple, now: float, camera_name: str,
                 snapshot: Callable[[], Optional[np.ndarray]]) -> None:
        level, source, reason = decision
        ts.level, ts.level_source, ts.level_reason = level, source, reason
        if level == "ok":
            if ts.ok_since is None:
                ts.ok_since = now
            if ts.alert_level_sent != "ok" and now - ts.ok_since >= RESOLVE_AFTER_OK_SECONDS:
                self._emit(ts, "resuelto", source, "La persona volvio a estado de alerta normal", now,
                           camera_name, None)
                ts.alert_level_sent = "ok"
            return
        ts.ok_since = None
        escalated = LEVEL_RANK[level] > LEVEL_RANK[ts.alert_level_sent]
        if escalated or now - ts.alert_sent_at >= self.s.realert_seconds:
            self._emit(ts, level, source, reason, now, camera_name, snapshot())
            ts.alert_level_sent, ts.alert_sent_at = level, now

    def _emit(self, ts: TrackState, level: str, source: str, reason: str, now: float, camera_name: str,
              image: Optional[np.ndarray]) -> None:
        alert_id = uuid.uuid4().hex[:8]
        photo = self._save_snapshot(alert_id, ts.camera_id, image)
        record = {
            "id": alert_id, "time": now, "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "camera_id": ts.camera_id, "camera_name": camera_name, "track_id": ts.track_id, "level": level,
            "source": source, "reason": reason, "indicators": ts.summary(now), "verdict": ts.verdict,
            "snapshot": photo, "ack": level == "resuelto",
        }
        self._store(record)
        logger.warning("ALERTA %s | %s | persona ID %s | %s (%s)", level.upper(), camera_name, ts.track_id,
                       reason, source or "-")

    # ---- alertas de sistema (camara sin senal) ---------------------------
    def system_alert(self, camera_id: str, camera_name: str, key: str, message: Optional[str]) -> None:
        """message=None resuelve la alerta de sistema `key`; si no, la emite una sola vez."""
        full_key = f"{camera_id}:{key}"
        if message is None:
            self._system_active.discard(full_key)
            return
        if full_key in self._system_active:
            return
        self._system_active.add(full_key)
        now = time.time()
        self._store({"id": uuid.uuid4().hex[:8], "time": now,
                     "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "camera_id": camera_id,
                     "camera_name": camera_name, "track_id": None, "level": "sistema", "source": "sistema",
                     "reason": message, "indicators": {}, "verdict": None, "snapshot": None, "ack": False})
        logger.error("ALERTA SISTEMA | %s | %s", camera_name, message)

    # ---- almacenamiento ---------------------------------------------------
    def _store(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._alerts.append(record)
            self._append_log(record)
        if self.s.alert_webhook_url:
            threading.Thread(target=self._post_webhook, args=(record,), daemon=True).start()

    def _append_log(self, record: Dict[str, Any]) -> None:
        try:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("No se pudo escribir %s", self.log_file)

    def _save_snapshot(self, alert_id: str, camera_id: str, image: Optional[np.ndarray]) -> Optional[str]:
        if image is None:
            return None
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{camera_id}_{alert_id}.jpg"
        try:
            cv2.imwrite(str(self.dir / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return name
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo guardar la evidencia")
            return None

    def _post_webhook(self, record: Dict[str, Any]) -> None:
        try:
            import requests

            requests.post(self.s.alert_webhook_url, json=record, timeout=5).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook fallo: %s", exc.__class__.__name__)

    def _load_recent(self) -> None:
        if not self.log_file.is_file():
            return
        try:
            for line in self.log_file.read_text(encoding="utf-8").splitlines()[-400:]:
                item = json.loads(line)
                if item.get("event") == "ack":
                    for alert in self._alerts:
                        if alert["id"] == item["id"]:
                            alert["ack"] = True
                elif "level" in item:
                    self._alerts.append(item)
        except (OSError, ValueError):
            logger.warning("No se pudo leer el historial de alertas previo")

    def _purge_old(self) -> None:
        limit = time.time() - self.s.alert_retention_days * 86400
        for photo in self.dir.glob("*.jpg"):
            try:
                if photo.stat().st_mtime < limit:
                    photo.unlink()
            except OSError:
                pass
