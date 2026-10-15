"""Configuracion de Vigilia: variables de entorno (validadas) y camaras.

Todo parametro numerico se valida contra un rango; los errores se acumulan y
se reportan juntos (ConfigError) para corregir el .env de una sola vez.
Las camaras viven en data/cameras.json y se administran desde el panel web.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlsplit

logger = logging.getLogger("vigilia.config")

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    pass


# =============================================================================
# Ajustes generales
# =============================================================================

class _Reader:
    def __init__(self) -> None:
        self.errors: List[str] = []

    def _raw(self, name: str) -> Optional[str]:
        raw = os.getenv(name)
        return None if raw is None or not raw.strip() else raw.strip()

    def num(self, name: str, default: float, lo: float, hi: float, cast=float):
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            value = cast(raw)
        except ValueError:
            self.errors.append(f"{name}='{raw}' no es un numero valido")
            return default
        if not lo <= value <= hi:
            self.errors.append(f"{name}={value} fuera de rango [{lo}, {hi}]")
            return default
        return value

    def text(self, name: str, default: str = "") -> str:
        raw = self._raw(name)
        return default if raw is None else raw

    def flag(self, name: str, default: bool) -> bool:
        raw = self._raw(name)
        return default if raw is None else raw.lower() in ("1", "true", "yes", "si", "on")

    def choice(self, name: str, default: str, options: tuple) -> str:
        raw = (self._raw(name) or default).lower()
        if raw not in options:
            self.errors.append(f"{name}='{raw}' invalido (opciones: {', '.join(options)})")
            return default
        return raw


@dataclass(frozen=True)
class Settings:
    # Rutas
    data_dir: Path
    models_dir: Path
    # Vision local
    yolo_model: str
    device: str
    yolo_conf: float
    inference_fps: float
    idle_fps: float
    max_side: int
    face_analysis: str
    # Gemini
    gemini_api_key: Optional[str]
    gemini_enabled: bool
    gemini_model: str
    gemini_timeout: float
    gemini_cooldown: float
    gemini_frames: int
    gemini_crop_side: int
    gemini_max_calls_per_hour: int
    gemini_thinking_level: Optional[str]
    # Reglas de decision
    suspect_seconds: float
    sleep_seconds: float
    local_only_drowsy_seconds: float
    local_only_sleep_seconds: float
    veto_seconds: float
    verdict_ttl: float
    min_conf_asleep: float
    min_conf_drowsy: float
    # Alertas
    realert_seconds: float
    alert_webhook_url: Optional[str]
    alert_retention_days: int
    camera_offline_seconds: float
    # Web
    web_host: str
    web_port: int
    web_user: Optional[str]
    web_password: Optional[str]
    max_cameras: int
    log_level: str

    @staticmethod
    def from_env() -> "Settings":
        r = _Reader()
        key = r.text("GEMINI_API_KEY") or None
        if key and len(key) < 20:
            r.errors.append("GEMINI_API_KEY parece incompleta (menos de 20 caracteres)")
        webhook = r.text("ALERT_WEBHOOK_URL") or None
        if webhook and not re.match(r"^https?://[^\s]+$", webhook):
            r.errors.append("ALERT_WEBHOOK_URL debe ser una URL http(s)")
        web_user, web_password = r.text("DASHBOARD_USER") or None, r.text("DASHBOARD_PASSWORD") or None
        if bool(web_user) != bool(web_password):
            r.errors.append("DASHBOARD_USER y DASHBOARD_PASSWORD deben definirse juntos")
        web_host = r.text("WEB_HOST", "127.0.0.1")
        if web_host not in ("127.0.0.1", "localhost") and not web_password:
            r.errors.append("WEB_HOST expuesto a la red requiere DASHBOARD_USER y DASHBOARD_PASSWORD")

        settings = Settings(
            data_dir=Path(r.text("VIGILIA_DATA_DIR", str(ROOT / "data"))),
            models_dir=Path(r.text("VIGILIA_MODELS_DIR", str(ROOT / "models"))),
            yolo_model=r.text("YOLO_MODEL", "yolov8n-pose.pt"),
            device=r.text("DEVICE", "cpu"),
            yolo_conf=r.num("YOLO_CONF", 0.40, 0.05, 0.95),
            inference_fps=r.num("INFERENCE_FPS", 5.0, 0.5, 30.0),
            idle_fps=r.num("IDLE_FPS", 1.5, 0.2, 10.0),
            max_side=r.num("MAX_FRAME_SIDE", 640, 320, 1280, int),
            face_analysis=r.choice("FACE_ANALYSIS", "auto", ("auto", "on", "off")),
            gemini_api_key=key,
            gemini_enabled=r.flag("GEMINI_ENABLED", True),
            gemini_model=r.text("GEMINI_MODEL", "gemini-3.8-flash"),
            gemini_timeout=r.num("GEMINI_TIMEOUT_SECONDS", 20.0, 3.0, 120.0),
            gemini_cooldown=r.num("GEMINI_COOLDOWN_SECONDS", 20.0, 5.0, 600.0),
            gemini_frames=r.num("GEMINI_FRAMES", 4, 1, 8, int),
            gemini_crop_side=r.num("GEMINI_CROP_SIDE", 384, 128, 768, int),
            gemini_max_calls_per_hour=r.num("GEMINI_MAX_CALLS_PER_HOUR", 120, 1, 5000, int),
            gemini_thinking_level=r.text("GEMINI_THINKING_LEVEL") or None,
            suspect_seconds=r.num("SUSPECT_SECONDS", 5.0, 1.0, 120.0),
            sleep_seconds=r.num("SLEEP_SECONDS", 12.0, 2.0, 600.0),
            local_only_drowsy_seconds=r.num("LOCAL_ONLY_DROWSY_SECONDS", 20.0, 3.0, 600.0),
            local_only_sleep_seconds=r.num("LOCAL_ONLY_SLEEP_SECONDS", 30.0, 5.0, 900.0),
            veto_seconds=r.num("GEMINI_AWAKE_VETO_SECONDS", 60.0, 5.0, 900.0),
            verdict_ttl=r.num("GEMINI_VERDICT_TTL_SECONDS", 90.0, 10.0, 900.0),
            min_conf_asleep=r.num("MIN_CONF_ASLEEP", 0.60, 0.3, 0.99),
            min_conf_drowsy=r.num("MIN_CONF_DROWSY", 0.50, 0.3, 0.99),
            realert_seconds=r.num("REALERT_SECONDS", 120.0, 10.0, 3600.0),
            alert_webhook_url=webhook,
            alert_retention_days=r.num("ALERT_RETENTION_DAYS", 30, 1, 3650, int),
            camera_offline_seconds=r.num("CAMERA_OFFLINE_SECONDS", 60.0, 10.0, 3600.0),
            web_host=web_host,
            web_port=r.num("WEB_PORT", 5000, 1, 65535, int),
            web_user=web_user,
            web_password=web_password,
            max_cameras=r.num("MAX_CAMERAS", 2, 1, 4, int),
            log_level=r.choice("LOG_LEVEL", "info", ("debug", "info", "warning", "error")),
        )
        if settings.sleep_seconds <= settings.suspect_seconds:
            r.errors.append("SLEEP_SECONDS debe ser mayor que SUSPECT_SECONDS")
        if r.errors:
            raise ConfigError("Configuracion invalida:\n  - " + "\n  - ".join(r.errors))
        return settings

    @property
    def gemini_active(self) -> bool:
        return self.gemini_enabled and bool(self.gemini_api_key)


# =============================================================================
# Camaras
# =============================================================================

NETWORK_PROTOCOLS = ("rtsp", "rtsps", "http", "https")
LOCAL_PROTOCOLS = ("webcam", "file")
DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}

# Ruta del sub-stream (menor resolucion = menos CPU) segun la marca.
BRAND_PATHS = {
    "generico": "/",
    "hikvision": "/Streaming/Channels/102",
    "dahua": "/cam/realmonitor?channel=1&subtype=1",
    "tapo": "/stream2",
    "reolink": "/h264Preview_01_sub",
    "uniview": "/media/video2",
    "axis": "/axis-media/media.amp",
}

_CAM_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_HOSTNAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    protocol: str
    host: str = ""
    port: int = 0
    user: str = ""
    password: str = ""
    path: str = "/"
    brand: str = "generico"
    roi: Optional[List[List[float]]] = None   # poligono normalizado (0-1) de la zona de guardia
    enabled: bool = True

    @property
    def is_network(self) -> bool:
        return self.protocol in NETWORK_PROTOCOLS

    def source(self, *, mask: bool = False) -> Any:
        """Lo que se entrega a cv2.VideoCapture (URL, indice de webcam o archivo)."""
        if self.protocol == "webcam":
            return int(self.host or 0)
        if self.protocol == "file":
            return self.path
        auth = ""
        if self.user:
            auth = quote(self.user, safe="") + (":" + ("***" if mask else quote(self.password, safe="")) if self.password else "") + "@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}{self.path}"

    def safe_url(self) -> str:
        return str(self.source(mask=True))

    def public(self) -> Dict[str, Any]:
        """Representacion sin contrasena para la API."""
        return {"id": self.id, "name": self.name, "protocol": self.protocol, "host": self.host, "port": self.port,
                "user": self.user, "has_password": bool(self.password), "path": self.path, "brand": self.brand,
                "roi": self.roi, "enabled": self.enabled, "url": self.safe_url()}

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "protocol": self.protocol, "host": self.host, "port": self.port,
                "user": self.user, "password": self.password, "path": self.path, "brand": self.brand,
                "roi": self.roi, "enabled": self.enabled}


def _clean_text(value: Any, field_name: str, max_len: int, required: bool = False) -> str:
    text = "" if value is None else str(value)
    if _CONTROL.search(text):
        raise ValueError(f"{field_name}: contiene caracteres no permitidos")
    text = text.strip()
    if len(text) > max_len:
        raise ValueError(f"{field_name}: maximo {max_len} caracteres")
    if required and not text:
        raise ValueError(f"{field_name}: es obligatorio")
    return text


def _validate_host(host: str) -> str:
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if not _HOSTNAME.match(host):
        raise ValueError("host: debe ser una IP o un nombre de host valido (sin esquema, ruta ni puerto)")
    return host


def _validate_roi(roi: Any) -> Optional[List[List[float]]]:
    if not roi:
        return None
    if not isinstance(roi, list) or not 3 <= len(roi) <= 16:
        raise ValueError("roi: lista de 3 a 16 puntos [x, y]")
    points = []
    for point in roi:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("roi: cada punto debe ser [x, y]") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("roi: coordenadas normalizadas entre 0 y 1")
        points.append([x, y])
    return points


def validate_camera(data: Dict[str, Any], *, allow_local: bool = False, existing_password: str = "") -> CameraConfig:
    """Valida y normaliza una camara (payload de la API o del .env).

    Acepta tambien `url` completa (rtsp://usuario:clave@host:554/ruta), que se
    descompone en campos. Si no llega contrasena se conserva `existing_password`
    (edicion desde el panel sin reenviar la clave)."""
    if not isinstance(data, dict):
        raise ValueError("payload invalido")
    data = dict(data)
    url = _clean_text(data.get("url"), "url", 512)
    if url:
        parts = urlsplit(url)
        data.setdefault("protocol", parts.scheme)
        try:
            port = parts.port
        except ValueError:
            raise ValueError("url: puerto invalido") from None
        data["host"] = parts.hostname or ""
        data["port"] = port or data.get("port")
        # Usuario/clave dentro de la URL mandan; si no vienen, se conservan los campos separados.
        if parts.username:
            data["user"] = unquote(parts.username)
        if parts.password:
            data["password"] = unquote(parts.password)
        if parts.path or parts.query:
            data["path"] = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")

    protocol = _clean_text(data.get("protocol") or "rtsp", "protocol", 10).lower()
    if protocol not in NETWORK_PROTOCOLS + (LOCAL_PROTOCOLS if allow_local else ()):
        raise ValueError(f"protocol: use uno de {', '.join(NETWORK_PROTOCOLS)}")

    cam_id = _clean_text(data.get("id"), "id", 32)
    if cam_id and not _CAM_ID.match(cam_id):
        raise ValueError("id: solo letras, numeros, '-' y '_' (maximo 32)")
    name = _clean_text(data.get("name"), "name", 60) or cam_id or "Camara"
    brand = _clean_text(data.get("brand") or "generico", "brand", 20).lower()
    if brand not in BRAND_PATHS:
        raise ValueError(f"brand: use uno de {', '.join(BRAND_PATHS)}")
    roi = _validate_roi(data.get("roi"))
    enabled = bool(data.get("enabled", True))

    host, port, user, password, path = "", 0, "", "", "/"
    if protocol in NETWORK_PROTOCOLS:
        host = _validate_host(_clean_text(data.get("host"), "host", 253, required=True))
        try:
            port = int(data.get("port") or DEFAULT_PORTS[protocol])
        except (TypeError, ValueError):
            raise ValueError("port: debe ser un numero") from None
        if not 1 <= port <= 65535:
            raise ValueError("port: fuera de rango (1-65535)")
        user = _clean_text(data.get("user"), "user", 64)
        password = _clean_text(data.get("password"), "password", 128) or (existing_password if user else "")
        if password and not user:
            raise ValueError("password: se indico contrasena sin usuario")
        path = _clean_text(data.get("path"), "path", 256) or BRAND_PATHS[brand]
        if "://" in path or re.search(r"\s", path):
            raise ValueError("path: debe ser solo la ruta del stream (ej. /Streaming/Channels/102)")
        if not path.startswith("/"):
            path = "/" + path
    elif protocol == "webcam":
        host = _clean_text(data.get("host") or "0", "host", 3)
        if not host.isdigit():
            raise ValueError("host: para webcam use el indice (0, 1, ...)")
    else:  # file
        path = _clean_text(data.get("path"), "path", 512, required=True)
        if not Path(path).is_file():
            raise ValueError(f"path: el archivo '{path}' no existe")

    return CameraConfig(id=cam_id or "", name=name, protocol=protocol, host=host, port=port, user=user,
                        password=password, path=path, brand=brand, roi=roi, enabled=enabled)


class CameraStore:
    """Persistencia de camaras en data/cameras.json (escritura atomica).

    Las contrasenas se guardan en texto plano en ese archivo: protegerlo con
    permisos del sistema operativo (se intenta chmod 600)."""

    def __init__(self, path: Path, max_cameras: int) -> None:
        self.path = path
        self.max_cameras = max_cameras
        self._lock = threading.Lock()
        self._cameras: Dict[str, CameraConfig] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for item in raw.get("cameras", []):
                    cam = validate_camera(item, allow_local=True)
                    self._cameras[cam.id] = cam
            except (OSError, ValueError, AttributeError) as exc:
                raise ConfigError(f"No se pudo leer {self.path}: {exc}") from exc
        elif self._seed_from_env():
            self._save()

    def _seed_from_env(self) -> bool:
        seeded = False
        for index in range(1, self.max_cameras + 1):
            prefix = f"CAMERA_{index}_"
            fields = {k[len(prefix):].lower(): v for k, v in os.environ.items() if k.startswith(prefix) and v.strip()}
            if not fields:
                continue
            fields.setdefault("id", f"CAM-{index:02d}")
            try:
                cam = validate_camera(fields, allow_local=True)
            except ValueError as exc:
                raise ConfigError(f"{prefix}*: {exc}") from exc
            self._cameras[cam.id] = cam
            seeded = True
        return seeded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cameras": [c.to_json() for c in self._cameras.values()]}, indent=2),
                       encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def list(self) -> List[CameraConfig]:
        with self._lock:
            return list(self._cameras.values())

    def get(self, cam_id: str) -> Optional[CameraConfig]:
        with self._lock:
            return self._cameras.get(cam_id)

    def upsert(self, cam: CameraConfig) -> CameraConfig:
        with self._lock:
            if not cam.id:
                n = 1
                while f"CAM-{n:02d}" in self._cameras:
                    n += 1
                cam = CameraConfig(**{**cam.__dict__, "id": f"CAM-{n:02d}"})
            if cam.id not in self._cameras and len(self._cameras) >= self.max_cameras:
                raise ValueError(f"maximo {self.max_cameras} camaras (MAX_CAMERAS)")
            self._cameras[cam.id] = cam
            self._save()
            return cam

    def remove(self, cam_id: str) -> bool:
        with self._lock:
            if self._cameras.pop(cam_id, None) is None:
                return False
            self._save()
            return True
