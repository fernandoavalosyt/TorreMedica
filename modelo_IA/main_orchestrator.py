"""
main_orchestrator.py

Orquestador de "Vigilia - Guard Sleep Monitor". Instancia los servicios del
pipeline, los conecta con hilos "pump" (salida de un servicio -> entrada de
los siguientes) y muestra el resultado en la ventana local del visualizador.

Flujo:
    S1 Video ──┬─> S2 Preprocessing (10 fps) ─> S3 YOLO ─> S4 Tracking ──┬─> S5 Pose ─┐
               ├─> S5 (cache de frames originales)                       └──────┐     │
               ├─> S8 (ring buffer de video, ~12 fps)                           v     v
               └─> S10 Visualizador (25 fps, ventana local)        S6 DrowsinessService
                                                                         │ (estado por guardia -> S10)
                                                    Juez Gemini (Tier-2) <─┤ candidato a dormido: recorte
                                                                         │ del frame -> veredicto JSON
                          S9 Alertas <─ S8 Evidencia <─ S7 Decision <────┘ (eventos guard_sleep / guard_inactive)

Tasas: la camara se captura y se muestra a VIDEO_TARGET_FPS (25) para que la
ventana sea fluida; la inferencia corre a INFERENCE_FPS (10). Las colas del
camino de inferencia son de pocos frames: si la CPU no alcanza, se descartan
frames viejos en vez de acumular retraso.

Uso local (desde cualquier carpeta):
    python modelo_IA/main_orchestrator.py              camara IP + modelos (ventana local)
    python modelo_IA/main_orchestrator.py --simulate   guardia sintetico, sin camara ni modelos
    q / ESC en la ventana (o Ctrl+C) para salir
Docker (sin pantalla): el visualizador cae al stream web http://localhost:5000/stream

Configuracion 100% por variables de entorno (ver modelo_IA/.env.example).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from IA.alert_service.alertService import AlertConfig, AlertService
from IA.decision_engine.decisionService.decisionService import DecisionConfig, DecisionService, load_rules_file
from IA.drowsiness.drowsinessService import DrowsinessConfig, DrowsinessService
from IA.drowsiness.guardSimulation import CYCLE_SECONDS, build_background, draw_guard, guard_pose
from IA.event_evidence.evidenceService.evidenceService import EvidenceConfig, EvidenceService
from IA.gemini_judge.geminiJudgeService import GeminiJudgeConfig, GeminiJudgeService
from IA.pose_specialized.poseService.poseService import PoseConfig, PoseService
from IA.preprocessing.preprocessingService.preprocessingService import CameraProcessingConfig, PreprocessingService
from IA.tracking.trackingService.trackingService import TrackConfig, TrackingService
from IA.video_ingestion.videoService import CameraWorker, build_camera_worker_from_env
from IA.visualizer.visualizerService import VisualizerConfig, VisualizerService
from IA.yolo_detection.yoloDetectionService.yoloDetectionService import YoloDetectionService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_VERSION = "2.0.0"
GUARD_RULES_FILE = "config/rules_guard.json"

logger = logging.getLogger("orchestrator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "si")


def _apply_defaults() -> None:
    """Valores por defecto del modo guardia (las variables ya definidas mandan)."""
    os.environ.setdefault("VIDEO_TARGET_FPS", "25")         # captura y ventana fluidas
    os.environ.setdefault("INFERENCE_FPS", "10")            # YOLO + pose
    os.environ.setdefault("VIDEO_BUFFER_MAX_SIZE", "4")
    os.environ.setdefault("MODEL_INPUT_QUEUE_SIZE", "4")
    os.environ.setdefault("POSE_INPUT_QUEUE_SIZE", "4")
    os.environ.setdefault("POSE_FRAME_CACHE_SIZE", "150")   # ~6 s a 25 fps
    os.environ.setdefault("DECISION_RULES_FILE", GUARD_RULES_FILE)
    os.environ.setdefault("DECISION_CONTEXT_FILE", "config/camera_context.json")


# =============================================================================
# Pesos de los modelos
# =============================================================================

def ensure_model_file(env_var: str, default_path: str) -> str:
    """Garantiza que exista el archivo de pesos. Si falta y
    AUTO_DOWNLOAD_MODELS=true (default), descarga el asset oficial de
    Ultralytics con ese nombre (ej. yolov8n.pt, yolov8n-pose.pt)."""
    path = os.getenv(env_var) or default_path
    os.environ[env_var] = path
    if os.path.isfile(path):
        return path
    if not _env_bool("AUTO_DOWNLOAD_MODELS", True):
        raise SystemExit(f"Falta el modelo '{path}' ({env_var}) y AUTO_DOWNLOAD_MODELS=false")

    logger.info("Pesos no encontrados, descargando %s ...", path)
    from ultralytics.utils.downloads import attempt_download_asset

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    downloaded = attempt_download_asset(path)
    # Ultralytics puede devolver una copia ya existente en su carpeta global.
    if downloaded and os.path.abspath(downloaded) != os.path.abspath(path) and os.path.isfile(downloaded):
        shutil.copyfile(downloaded, path)
    if not os.path.isfile(path):
        raise SystemExit(f"No se pudo obtener el modelo '{path}'. Descargalo manualmente a esa ruta.")
    return path


# =============================================================================
# Pumps: conectores entre servicios
# =============================================================================

Sink = Tuple[str, Callable[[Dict[str, Any]], Any]]


class Pump(threading.Thread):
    """Lee de la salida de un servicio y entrega cada item a uno o varios
    destinos. El fallo de un destino no afecta a los demas."""

    def __init__(self, name: str, source: Callable[[float], Optional[Dict[str, Any]]], sinks: List[Sink],
                 stop_event: threading.Event) -> None:
        super().__init__(name=f"pump-{name}", daemon=True)
        self.label = name
        self._source = source
        self._sinks = sinks
        self._stop_event = stop_event
        self.moved = 0
        self.errors: Dict[str, int] = {sink_name: 0 for sink_name, _ in sinks}
        self.last_item_wall: Optional[float] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._source(0.5)
            except Exception:
                logger.exception("La fuente del pump %s fallo", self.label)
                self._stop_event.wait(0.5)
                continue
            if item is None:
                continue
            self.moved += 1
            self.last_item_wall = time.time()
            for sink_name, sink in self._sinks:
                try:
                    sink(item)
                except Exception:
                    self.errors[sink_name] += 1
                    count = self.errors[sink_name]
                    if count <= 3 or count % 100 == 0:
                        logger.exception("Destino %s del pump %s fallo (%d errores)", sink_name, self.label, count)

    def stats(self) -> Dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "moved": self.moved,
            "errors": {k: v for k, v in self.errors.items() if v},
            "seconds_since_last": None if self.last_item_wall is None else round(time.time() - self.last_item_wall, 2),
        }


def _decimate(sink: Callable[[Dict[str, Any]], Any], max_fps: float) -> Callable[[Dict[str, Any]], None]:
    """Entrega como maximo max_fps items por segundo (segun el timestamp del frame)."""
    state = {"last": 0.0}
    interval = 1.0 / max_fps

    def wrapped(item: Dict[str, Any]) -> None:
        ts = float(item.get("timestamp") or time.time())
        if ts - state["last"] >= interval:
            state["last"] = ts
            sink(item)

    return wrapped


class RecentFrames:
    """Ultimos frames originales por camara, para que el juez Gemini recorte
    exactamente el frame que disparo la verificacion (o el mas reciente)."""

    def __init__(self, max_frames: int = 150) -> None:
        self._max = max_frames
        self._frames: Dict[str, "OrderedDict[int, np.ndarray]"] = {}
        self._lock = threading.Lock()

    def put(self, frame: Dict[str, Any]) -> None:
        with self._lock:
            per_camera = self._frames.setdefault(str(frame["camera_id"]), OrderedDict())
            per_camera[int(frame["frame_id"])] = frame["image"]
            while len(per_camera) > self._max:
                per_camera.popitem(last=False)

    def get(self, camera_id: str, frame_id: int) -> Optional[np.ndarray]:
        with self._lock:
            per_camera = self._frames.get(camera_id)
            if not per_camera:
                return None
            image = per_camera.get(frame_id)
            return image if image is not None else next(reversed(per_camera.values()))


def _build_visualizer(health_provider: Callable[[], Dict[str, Any]]) -> VisualizerService:
    config = VisualizerConfig.from_env()
    if config.alerts_url is None:
        config = replace(config, alerts_url=f"http://localhost:{os.getenv('ALERT_PORT', '8080')}/")
    return VisualizerService(config, health_provider=health_provider)


# =============================================================================
# Pipeline real (camara + modelos)
# =============================================================================

class Pipeline:
    def __init__(self) -> None:
        _apply_defaults()
        self.camera_id = os.getenv("VIDEO_CAMERA_ID", "CAM-01")
        self._stop_event = threading.Event()
        self._pumps: List[Pump] = []
        self._start_wall = time.time()

        ensure_model_file("MODEL_PATH", "models/yolov8n.pt")
        ensure_model_file("POSE_MODEL_PATH", "models/yolov8n-pose.pt")

        logger.info("Inicializando servicios | camara=%s", self.camera_id)
        self.s1: CameraWorker = build_camera_worker_from_env()
        self.s2 = PreprocessingService()
        self.s3 = YoloDetectionService()
        self.s4 = TrackingService(config=TrackConfig())
        self.s5 = PoseService(config=PoseConfig.from_env())
        self.frames = RecentFrames()
        self.judge = GeminiJudgeService(GeminiJudgeConfig.from_env())
        self.s6 = DrowsinessService(DrowsinessConfig.from_env(), judge=self.judge, frame_provider=self.frames.get)
        self.s7 = DecisionService(config=DecisionConfig.from_env())
        self.s8 = EvidenceService(config=EvidenceConfig.from_env())
        self.s9 = AlertService(AlertConfig(
            host=os.getenv("ALERT_HOST", "127.0.0.1"),
            port=int(os.getenv("ALERT_PORT", "8080")),
            evidence_root=os.getenv("EVIDENCE_STORAGE_ROOT", "evidence_store"),
            open_browser=_env_bool("ALERT_OPEN_BROWSER", True),
        ))
        self.visualizer = _build_visualizer(self.health)
        self.s6.add_listener(self.visualizer.on_guard_frame)

    def _wire(self) -> None:
        cam = self.camera_id
        s1, s2, s3, s4, s5, s6, s7, s8, s9, viz = (self.s1, self.s2, self.s3, self.s4, self.s5,
                                                  self.s6, self.s7, self.s8, self.s9, self.visualizer)
        stop = self._stop_event
        evidence_fps = float(os.getenv("EVIDENCE_BUFFER_FPS", "12"))
        self._pumps = [
            Pump("video", lambda t: s1.get_next_frame(timeout=t), [
                ("visualizer", viz.on_frame),
                ("judge_frames", self.frames.put),
                ("preprocessing", lambda f: s2.submit_frame(cam, f)),
                ("pose_frame_cache", lambda f: s5.cache_frame(cam, int(f["frame_id"]), f["image"])),
                ("evidence_buffer", _decimate(
                    lambda f: s8.buffer.append_frame(cam, float(f["timestamp"]), f["image"]), evidence_fps)),
            ], stop),
            Pump("preprocessing", lambda t: s2.get_next_processed(cam, timeout=t), [
                ("detection", s3.submit_frame),
            ], stop),
            Pump("detection", lambda t: s3.get_next_result(cam, timeout=t), [
                ("tracking", s4.submit_detection_result),
            ], stop),
            Pump("tracking", lambda t: s4.get_next_result(cam, timeout=t), [
                ("drowsiness", s6.on_tracking),
                ("pose", s5.submit_tracking_result),
            ], stop),
            Pump("pose", lambda t: s5.get_next_result(cam, timeout=t), [
                ("drowsiness", s6.on_pose),
            ], stop),
            Pump("drowsiness", lambda t: s6.get_next_event(timeout=t), [
                ("decision", s7.submit),
            ], stop),
            Pump("decision", lambda t: s7.get_next_decision(timeout=t), [
                ("evidence", s8.submit_decision),
                ("visualizer", viz.on_decision),
            ], stop),
            Pump("evidence", lambda t: s8.get_next_result(timeout=t), [
                ("alerts", s9.receive_evidence),
                ("visualizer", viz.on_evidence),
            ], stop),
        ]

    def start(self) -> None:
        # De atras hacia adelante: cada consumidor listo antes que su productor.
        self.s9.start()
        self.s8.start()
        self.s7.start()
        self.s5.start()
        self.s3.start()
        buffer_size = int(os.getenv("PREPROCESS_BUFFER_SIZE", "4"))
        target = int(os.getenv("PREPROCESS_TARGET_SIZE", "640"))
        self.s2.register_camera(CameraProcessingConfig(
            camera_id=self.camera_id, target_width=target, target_height=target,
            target_fps=float(os.getenv("INFERENCE_FPS", "10")),
        ), input_buffer_size=buffer_size, output_buffer_size=buffer_size)
        self.visualizer.start()
        self._wire()
        for pump in self._pumps:
            pump.start()
        self.s1.start()
        logger.info("Pipeline en marcha | camara=%s | alertas=http://localhost:%s/", self.camera_id, self.s9.config.port)

    def stop(self) -> None:
        logger.info("Deteniendo pipeline...")
        self.s1.stop()
        self._stop_event.set()
        for pump in self._pumps:
            pump.join(timeout=2.0)
        for name, stopper in (("preprocessing", self.s2.stop_all), ("detection", self.s3.stop),
                              ("tracking", self.s4.stop_all), ("pose", self.s5.stop), ("decision", self.s7.stop),
                              ("evidence", lambda: self.s8.stop(drain_timeout=8.0)), ("alerts", self.s9.stop),
                              ("gemini_judge", self.judge.shutdown), ("visualizer", self.visualizer.stop)):
            try:
                stopper()
            except Exception:
                logger.exception("Error deteniendo %s", name)
        logger.info("Pipeline detenido")

    def health(self) -> Dict[str, Any]:
        services: Dict[str, Any] = {}
        for name, provider in (("video_ingestion", self.s1.health_check), ("preprocessing", self.s2.health_check),
                               ("detection", self.s3.health_check), ("tracking", self.s4.health_check),
                               ("pose", self.s5.health_check), ("drowsiness", self.s6.health_check),
                               ("decision", self.s7.health_check), ("evidence", self.s8.health_check),
                               ("alerts", self.s9.health_check), ("gemini_judge", self.judge.health_check)):
            try:
                services[name] = provider()
            except Exception as exc:
                services[name] = {"error": str(exc)}
        camera_state = services.get("video_ingestion", {}).get("connection_state")
        pumps = {pump.label: pump.stats() for pump in self._pumps}
        pumps_ok = bool(pumps) and all(p["alive"] for p in pumps.values())
        return {
            "status": "healthy" if camera_state == "CONNECTED" and pumps_ok else "degraded",
            "version": PIPELINE_VERSION,
            "service": "modelo-ia",
            "mode": "guard_sleep_monitor",
            "camera_id": self.camera_id,
            "camera_state": camera_state,
            "uptime_seconds": round(time.time() - self._start_wall, 1),
            "pumps": pumps,
            "services": services,
        }


# =============================================================================
# Simulacion: guardia sintetico -> S6 real -> S7 real -> visualizador
# =============================================================================

class GuardSimulation:
    """Ejecuta la matematica real de somnolencia (S6) y las reglas reales de
    S7 sobre un guardia sintetico (IA/drowsiness/guardSimulation.py). Sin
    camara, sin modelos y sin GPU."""

    def __init__(self, stop_event: threading.Event) -> None:
        _apply_defaults()
        self.camera_id = os.getenv("VIDEO_CAMERA_ID", "CAM-01")
        self._stop_event = stop_event
        self._fps = float(os.getenv("SIMULATION_FPS", "25"))
        self._start = time.time()
        self.frames = RecentFrames()
        self.judge = GeminiJudgeService(GeminiJudgeConfig.from_env())
        self.drowsiness = DrowsinessService(DrowsinessConfig.from_env(), judge=self.judge,
                                            frame_provider=self.frames.get)
        self.decision = DecisionService(config=DecisionConfig(workers=1), rules=load_rules_file(GUARD_RULES_FILE))
        self.visualizer = _build_visualizer(self.health)
        self.drowsiness.add_listener(self.visualizer.on_guard_frame)
        self._pumps = [
            Pump("drowsiness", lambda t: self.drowsiness.get_next_event(timeout=t),
                 [("decision", self.decision.submit)], stop_event),
            Pump("decision", lambda t: self.decision.get_next_decision(timeout=t),
                 [("visualizer", self.visualizer.on_decision)], stop_event),
        ]
        self._thread = threading.Thread(target=self._run, name="GuardSimulation", daemon=True)

    def start(self) -> None:
        self.decision.start()
        self.visualizer.start()
        for pump in self._pumps:
            pump.start()
        self._thread.start()
        logger.info("Simulacion de guardia en marcha (ciclo de %.0f s: despierto -> cabeceos -> dormido -> despierta)",
                    CYCLE_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        self.judge.shutdown()
        self.decision.stop()
        self.visualizer.stop()

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "version": PIPELINE_VERSION, "service": "modelo-ia", "mode": "simulation",
                "camera_id": self.camera_id, "drowsiness": self.drowsiness.health_check(),
                "gemini_judge": self.judge.health_check()}

    def _run(self) -> None:
        background = build_background()
        rng = np.random.default_rng()
        frame_id = 0
        interval = 1.0 / max(1.0, self._fps)
        track_id = f"{self.camera_id}-1"
        next_frame = time.time()
        while not self._stop_event.is_set():
            # Agenda fija: el redondeo del temporizador de Windows (~15.6 ms)
            # no se acumula y la tasa media se mantiene en SIMULATION_FPS.
            next_frame += interval
            started = time.time()
            keypoints, bbox = guard_pose(started - self._start, rng=rng)
            frame = background.copy()
            draw_guard(frame, keypoints)
            frame_id += 1
            frame_msg = {"camera_id": self.camera_id, "frame_id": frame_id, "timestamp": started, "image": frame}
            self.frames.put(frame_msg)
            self.visualizer.on_frame(frame_msg)
            track = {"track_id": track_id, "class_name": "person", "track_state": "ACTIVE",
                     "bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox))}
            self.drowsiness.on_tracking({"camera_id": self.camera_id, "frame_id": frame_id,
                                         "timestamp": started, "tracks": [track]})
            self.drowsiness.on_pose({"camera_id": self.camera_id, "session_id": "simulation", "frame_id": frame_id,
                                     "timestamp": started,
                                     "poses": [{"track_id": track_id, "status": "OK", "keypoints": keypoints}]})
            delay = next_frame - time.time()
            if delay > 0:
                self._stop_event.wait(delay)
            elif delay < -1.0:
                next_frame = time.time()  # muy atrasado (equipo cargado): no intentar recuperar


# =============================================================================
# Entrada
# =============================================================================

def main() -> None:
    os.chdir(BASE_DIR)  # rutas relativas (models/, config/, evidence_store/) siempre desde modelo_IA/
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)  # las variables del entorno/compose mandan
        load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"), override=False)  # GEMINI_API_KEY
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Vigilia - Guard Sleep Monitor")
    parser.add_argument("--simulate", action="store_true",
                        help="guardia sintetico sin camara ni modelos (tambien ORCHESTRATOR_MODE=simulate)")
    parser.add_argument("--web", action="store_true", help="activa el stream web Flask (ENABLE_WEB_STREAM=true)")
    parser.add_argument("--no-window", action="store_true", help="sin ventana local de OpenCV")
    parser.add_argument("--port", type=int, help="puerto del stream web (por defecto VISUALIZER_PORT o 5000)")
    args = parser.parse_args()
    if args.web:
        os.environ["ENABLE_WEB_STREAM"] = "true"
    if args.no_window:
        os.environ["VISUALIZER_LOCAL_WINDOW"] = "false"
    if args.port:
        os.environ["VISUALIZER_PORT"] = str(args.port)

    stop_requested = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("Senal %s recibida, cerrando...", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    simulate = args.simulate or os.getenv("ORCHESTRATOR_MODE", "").lower() == "simulate"
    runner = GuardSimulation(stop_requested) if simulate else Pipeline()
    runner.start()
    try:
        # La ventana de OpenCV vive en el hilo principal. Sin pantalla (Docker)
        # se espera aqui y el visualizador sirve el stream web.
        if not runner.visualizer.run_window_loop(stop_requested):
            while not stop_requested.wait(0.5):
                pass
    finally:
        runner.stop()


if __name__ == "__main__":
    sys.exit(main())
