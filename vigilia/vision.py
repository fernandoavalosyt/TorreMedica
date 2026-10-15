"""Vision local: YOLO-pose (deteccion de personas + tracking con ID + 17 puntos
del cuerpo en un solo modelo) y analisis facial opcional (ojos/bostezo)."""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("vigilia.vision")

# Indices COCO de los 17 puntos
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = range(13)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KP_CONF = 0.40


@dataclass
class Person:
    track_id: int
    bbox: np.ndarray        # x1, y1, x2, y2 (coordenadas del frame reducido)
    conf: float
    kpts: np.ndarray        # (17, 3): x, y, confianza


def ensure_model(models_dir: Path, name: str) -> str:
    """Devuelve la ruta del modelo; si falta, lo descarga (asset oficial de Ultralytics)."""
    target = models_dir / name
    if target.is_file():
        return str(target)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando modelo %s ...", name)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = attempt_download_asset(str(target))
    if downloaded and Path(downloaded).is_file() and Path(downloaded).resolve() != target.resolve():
        shutil.copyfile(downloaded, target)
    if not target.is_file():
        raise RuntimeError(f"No se pudo obtener el modelo '{name}'. Copielo manualmente en {models_dir}.")
    return str(target)


class PoseTracker:
    """YOLO-pose + ByteTrack. Una instancia por camara (el tracker conserva estado)."""

    def __init__(self, model_path: str, device: str, conf: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def infer(self, frame: np.ndarray, imgsz: int) -> List[Person]:
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.conf,
                                   imgsz=imgsz, device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        return [Person(int(i), boxes[n], float(confs[n]), kpts[n]) for n, i in enumerate(ids)]


@dataclass
class FaceMetrics:
    eyes_closed: bool
    blink: float        # 0-1, promedio de eyeBlinkLeft/Right
    jaw_open: float     # 0-1


_FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
                   "face_landmarker.task")
_download_lock = threading.Lock()


class FaceAnalyzer:
    """Ojos cerrados y boca abierta (bostezo) con MediaPipe FaceLandmarker sobre el
    recorte de la cabeza. Opcional: si falta mediapipe o el modelo, se desactiva y
    Gemini evalua el rostro."""

    EYES_CLOSED_THRESHOLD = 0.55

    def __init__(self, models_dir: Path) -> None:
        self.available = False
        self.reason: Optional[str] = None
        self._landmarker = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_file = self._ensure_model(models_dir)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
                running_mode=mp_vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self.available = True
        except Exception as exc:  # noqa: BLE001 - la cara es opcional
            self.reason = f"{exc.__class__.__name__}: {exc}"[:160]
            logger.warning("Analisis facial desactivado (%s). Gemini evaluara ojos y bostezos.", self.reason)

    @staticmethod
    def _ensure_model(models_dir: Path) -> Path:
        target = models_dir / "face_landmarker.task"
        with _download_lock:
            if not target.is_file():
                import requests

                models_dir.mkdir(parents=True, exist_ok=True)
                response = requests.get(_FACE_MODEL_URL, timeout=30)
                response.raise_for_status()
                target.write_bytes(response.content)
        return target

    @staticmethod
    def head_crop(image: np.ndarray, kpts: np.ndarray) -> Optional[np.ndarray]:
        """Recorte cuadrado de la cabeza a partir de nariz/ojos/orejas visibles."""
        pts = [kpts[i] for i in range(5) if kpts[i][2] >= KP_CONF]
        if len(pts) < 3 or kpts[NOSE][2] < KP_CONF:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        has_ears = kpts[L_EAR][2] >= KP_CONF and kpts[R_EAR][2] >= KP_CONF
        side = extent * (1.9 if has_ears else 3.4)
        if side < 40:
            return None
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        h, w = image.shape[:2]
        x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        x2, y2 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        return image[y1:y2, x1:x2]

    def analyze(self, image: np.ndarray, kpts: np.ndarray) -> Optional[FaceMetrics]:
        if not self.available:
            return None
        crop = self.head_crop(image, kpts)
        if crop is None:
            return None
        if min(crop.shape[:2]) < 160:
            scale = 160 / min(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        except Exception:  # noqa: BLE001
            logger.debug("FaceLandmarker fallo", exc_info=True)
            return None
        if not result.face_blendshapes:
            return None
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
        return FaceMetrics(eyes_closed=blink >= self.EYES_CLOSED_THRESHOLD, blink=blink,
                           jaw_open=scores.get("jawOpen", 0.0))
