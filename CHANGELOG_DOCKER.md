# Vigilia — Registro de Cambios Docker

> Cada sección documenta qué se hizo, qué archivo se creó/modificó y por qué.

---

## 2026-09-19 — Configuración inicial de infraestructura Docker

### Archivos Root creados

| Archivo | Qué hace |
|---------|----------|
| `.env` | Variables de entorno (credenciales dev, puertos, DSNs) |
| `.env.example` | Template de referencia sin valores sensibles |
| `.dockerignore` | Excluye archivos innecesarios del build Docker |
| `.gitignore` | Ignora archivos generados y sensibles en Git |
| `docker-compose.yml` | Orquesta los 8 servicios (IA, backend, frontend, postgres, mongodb, redis, rabbitmq, minio) |
| `docker-compose.override.yml` | Overrides para desarrollo local (hot-reload, volumes) |

### modelo_IA/ — Servicio de Inteligencia Artificial

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Imagen Python 3.11 (opción CUDA comentada para futuro con GPU NVIDIA) |
| `requirements.txt` | Dependencias: YOLO, OpenCV, FastAPI, RabbitMQ, Redis, MinIO, Gemini |
| `app/main.py` | Entry point FastAPI con health check |
| `app/config.py` | Configuración con Pydantic (conexiones a servicios) |
| `app/__init__.py` | Marca el directorio como paquete Python |
| `app/core/__init__.py` | Marca core/ como paquete Python |

### backend/ — Servicio Backend API

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Imagen Python 3.11 con uvicorn hot-reload |
| `requirements.txt` | Dependencias: FastAPI, SQLAlchemy, Alembic, MongoDB, Redis, JWT |
| `alembic.ini` | Configuración de migraciones de base de datos |
| `app/main.py` | Entry point FastAPI con CORS y health check |
| `app/config.py` | Configuración de conexiones a toda la infraestructura |
| `app/__init__.py` | Marca el directorio como paquete Python |

### frontend/ — Servicio Frontend

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Node 20 con multi-stage build (dev + producción con Nginx) |
| `nginx.conf` | Configuración Nginx para SPA en producción |
| Proyecto Vite | React + TypeScript inicializado con `npm create vite@latest` |

---

### Notas importantes

- **GPU NVIDIA**: No disponible actualmente. La línea CUDA está **comentada** en `modelo_IA/Dockerfile`. Cuando tengas GPU, descomenta la línea `FROM nvidia/cuda:...` y comenta `FROM python:3.11-slim`.
- **Versiones usadas**: Python 3.11, Node 20, PostgreSQL 16, MongoDB 7, Redis 7, RabbitMQ 3.
- **Para levantar todo**: `docker compose up -d` desde la raíz del proyecto.

---

## 2026-09-24 — Pipeline de IA (S1..S9) + Visualizador Byrack (S10)

### modelo_IA/

| Archivo | Qué hace |
|---------|----------|
| `main_orchestrator.py` | Nuevo entry point: instancia S1..S9, los conecta con hilos "pump" y levanta el visualizador |
| `IA/visualizer/visualizerService.py` | S10: dibuja tracks, esqueleto, comportamientos, alertas y evidencia sobre el frame original; MJPEG en `:8001/stream` |
| `IA/*` | Carpetas renombradas sin guiones (`video_ingestion`, `yolo_detection`, …) + `__init__.py` para poder importarlas |
| `IA/video_ingestion/videoService.py` | Cámara IP por RTSP activa (`RTSP_URL`, `RTSP_USER`, `RTSP_PASSWORD`); `frame_id` ya no se reinicia al reconectar |
| `Dockerfile` / `requirements.txt` | Recreados: Python 3.11 (bookworm), torch CPU, OpenCV headless, Flask; `CMD python main_orchestrator.py` |
| `.dockerignore` | Excluye `venv/`, `evidence_store/` y pesos del contexto de build |
| `config/zones.json`, `config/camera_context.json` | Zonas de ejemplo (1920x1080) con los mismos `zone_id`; salida con merodeo a 3 s |
| `.env.example` | Variables del pipeline |

### Raíz

| Archivo | Cambio |
|---------|--------|
| `docker-compose.yml` | `modelo-ia`: puerto 8080 (alertas), `VISUALIZER_HOST/ALERT_HOST=0.0.0.0`, `ALERT_OPEN_BROWSER=false`, `stop_grace_period: 30s` |
| `docker-compose.override.yml` | `modelo-ia` arranca con `python main_orchestrator.py` (antes `uvicorn app.main:app`, que ya no existe) |
| `.env.example` | `ALERT_PORT` |
| `.gitignore` | `modelo_IA/evidence_store/` |

### Notas

- El puerto 8001 se mantiene: backend (`IA_SERVICE_URL`) y frontend (`/health`) ya apuntan ahí. Frontend: `<img src="http://localhost:8001/stream" />`.
- Los pesos `yolov8n.pt` y `yolov8n-pose.pt` se descargan solos al primer arranque dentro del volumen `ia-models` (`AUTO_DOWNLOAD_MODELS=true`).

---

## 2026-10-04 — Visualizador Byrack AI (Command Center) en el puerto 5000

| Archivo | Cambio |
|---------|--------|
| `modelo_IA/IA/visualizer/visualizerService.py` | Overlay estilo centro de control: cajas tácticas por estado (NORMAL verde, LOITERING amarillo, POSSIBLE_THEFT rojo), etiquetas `ID \| clase %`, `Merodeo: Xs`, trayectorias de 30 puntos, HUD `[LIVE]` + FPS suavizado, banner de alerta crítica, JPEG 75. Nueva entrada `submit(meta)` con el contrato FrameMeta |
| `modelo_IA/main_orchestrator.py` | Colas cortas en el camino de video (4 frames) para no acumular retraso; modo `--simulate` (escena sintética sin cámara ni modelos) |
| `docker-compose.yml`, `.env`, `.env.example`, `modelo_IA/.env*` | Visualizador en el puerto **5000** (antes 8001) |
| `modelo_IA/Dockerfile` | `EXPOSE 5000`, healthcheck en 5000, `fonts-dejavu-core` |
| `modelo_IA/requirements.txt` | `Pillow` |
| `backend/app/config.py` | `IA_SERVICE_URL=http://modelo-ia:5000` |
| `frontend/src/App.jsx` | Health del IA en `VITE_IA_URL` (default `http://localhost:5000`) |

Stream en React: `<img src="http://localhost:5000/stream" alt="Byrack" />`

---

## 2026-10-05 — Re-enfoque: Vigilia Guard Sleep Monitor

| Archivo | Cambio |
|---------|--------|
| `modelo_IA/IA/drowsiness/drowsinessService.py` | **Nuevo S6 del guardia.** Head pitch/roll (35° con histéresis y postura base del guardia), shoulder slump, LME (60 cuadros / 2.5 s), cabeceo (≥2 en 10 s). Estados ALERT_ACTIVE → DROWSINESS_WARNING → SLEEPING_CRITICAL → INACTIVE_GUARD_STATION. Emite eventos `guard_sleep` / `guard_inactive` para S7 |
| `modelo_IA/IA/drowsiness/guardSimulation.py` | Guardia sintético (despierto → cabeceos → dormido → despierta) para probar sin cámara ni modelos |
| `modelo_IA/IA/visualizer/visualizerService.py` | Ventana local `cv2.imshow` + `waitKey(1)` (~25 fps). Flask/MJPEG congelado con `ENABLE_WEB_STREAM=False`. Overlay de guardia: verde/amarillo/rojo parpadeante, vector de cabeza, HUD de estado, LME y FPS |
| `modelo_IA/main_orchestrator.py` | S6 = DrowsinessService (el S6 retail ya no se instancia). Captura 25 fps / inferencia 10 fps. `--simulate`, `--web`, `--no-window` |
| `modelo_IA/config/rules_guard.json` | Reglas S7: `GUARD_SLEEPING` (alta, crítica en turno nocturno), `GUARD_INACTIVE`; `GUARD_DROWSY` desactivada |
| `modelo_IA/config/camera_context.json` | Cámara `security` con turnos `day_shift` / `night_shift` |
| `modelo_IA/IA/pose_specialized/poseService/poseService.py` | También procesa tracks `RECOVERING` |
| `modelo_IA/IA/alert_service/alertService.py` | Etiquetas de eventos del guardia |
| `docker-compose.yml` | En el contenedor no hay pantalla: `VISUALIZER_LOCAL_WINDOW=false`, `ENABLE_WEB_STREAM=true` |
| `modelo_IA/.env*` | `VIDEO_TARGET_FPS=25`, `INFERENCE_FPS=10`, `DROWSY_*`, `ENABLE_WEB_STREAM=false` |

---

## 2026-10-05 — Juez Final Tier-2 con Gemini Vision

| Archivo | Cambio |
|---------|--------|
| `modelo_IA/IA/gemini_judge/geminiJudgeService.py` | **Nuevo.** Verificación con `google-genai` (salida JSON con esquema Pydantic), `ThreadPoolExecutor`, cooldown de 10 s por guardia, recorte con 20 % de contexto en JPEG en memoria. Fail-safe: key inválida → se desactiva; 429 → pausa; timeout/5xx → veredicto local |
| `modelo_IA/IA/drowsiness/drowsinessService.py` | Estados `VERIFYING_WITH_GEMINI` y `WORKING_DISTRACTED`. A los 10 s de condición de sueño consulta a Gemini: `true` → SLEEPING_CRITICAL; `false` → despierto + reinicio de acumuladores + veto 15 s; sin respuesta en 12 s → heurística local |
| `modelo_IA/IA/visualizer/visualizerService.py` | Púrpura `GEMINI EVALUANDO...` con animación, rojo `DORMIDO CONFIRMADO POR GEMINI`, verde `DESPIERTO - actividad` |
| `modelo_IA/main_orchestrator.py` | Crea el juez, buffer de frames recientes para el recorte, carga `GEMINI_API_KEY` del `.env` raíz |
| `modelo_IA/requirements.txt`, `modelo_IA/.env.example` | `google-genai`, `pydantic`, variables `GEMINI_*` |

Modelo por defecto `gemini-3.8-flash` (`gemini-1.5-flash` fue retirado de la API). Requiere `GEMINI_API_KEY` válida de Google AI Studio en el `.env` raíz; sin ella el sistema funciona solo con la heurística local.
