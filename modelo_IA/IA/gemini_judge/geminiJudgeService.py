"""
geminiJudgeService.py

"Juez Final" (Tier-2 Verification) de Vigilia con Gemini Vision.

Cuando la heuristica local (pose + LME) marca a un guardia como candidato a
dormido, DrowsinessService recorta su region (bbox + 20 % de contexto) y la
envia aqui. Gemini responde un veredicto JSON estructurado (esquema Pydantic)
que decide si el guardia realmente duerme o esta trabajando/distraido.

Garantias:
  - NUNCA bloquea al llamador: la peticion HTTP corre en un ThreadPoolExecutor
    y el resultado llega por callback (en un hilo del pool).
  - Anti-spam: una peticion en vuelo y un cooldown (10 s) por track_id.
  - Fail-safe: ante falta de API key, key invalida, rate limit, timeout o
    respuesta invalida, el callback recibe error=... y el sistema conserva el
    veredicto local. Una key invalida desactiva el juez; un 429 lo pausa.
  - La imagen se codifica a JPEG en memoria (cv2.imencode), sin archivos
    temporales. cv2.imencode espera BGR y produce un JPEG con colores
    correctos, asi que no se convierte a RGB antes de codificar.

Configuracion (variables de entorno):
  GEMINI_API_KEY             API key de Google AI Studio (obligatoria para activar el juez)
  GEMINI_JUDGE_ENABLED       true/false (default true si hay API key)
  GEMINI_JUDGE_MODEL         default gemini-3.8-flash (gemini-1.5-flash ya no existe en la API)
  GEMINI_TIMEOUT_SECONDS     default 8
  GEMINI_COOLDOWN_SECONDS    default 10 (por track_id)
  GEMINI_RATE_LIMIT_PAUSE    default 60 s de pausa tras un HTTP 429
  GEMINI_THINKING_LEVEL      opcional (ej. low) para modelos con razonamiento
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("gemini_judge")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


SYSTEM_PROMPT = """Eres un auditor experto en seguridad física especializado en evaluación semántica de estado de alerta en personal de vigilancia.
Analizarás una imagen recortada de un guardia de seguridad capturado por una cámara de vigilancia.

Tu objetivo es responder estricta y únicamente en formato JSON evaluando los siguientes 3 aspectos:

1. CONTEXTO Y ACTIVIDAD:
   - ¿Tiene un teléfono celular encendido en la mano, un cuaderno con pluma, un radio de comunicación o está mirando fijamente un monitor?
   - Si está realizando alguna de estas actividades, NO está dormido (está trabajando/distraído, pero despierto).

2. EVALUACIÓN MICRO-VISUAL:
   - ¿Tiene los párpados completamente cerrados?
   - ¿Tiene la mandíbula relajada / boca abierta por completo?
   - ¿Muestra una expresión facial de pérdida de conciencia?

3. SOPORTE DE POSTURA Y GRAVEDAD:
   - ¿Conserva tensión muscular para sostener su cuerpo/cabeza?
   - ¿Su cuerpo está completamente desplomado, apoyado pesadamente contra la pared o la mesa por pérdida de tono muscular?

VEREDICTO FINAL (`is_sleeping`):
- `true`: Solo si la postura indica colapso/relajación muscular Y NO hay actividad/dispositivo en uso Y la expresión o párpados denotan sueño profundo.
- `false`: Si está leyendo, usando el celular, escribiendo, o simplemente mirando hacia abajo conservando tensión.

Si un rasgo no es visible en la imagen (ej. los ojos de una persona de espaldas), indícalo como no visible en lugar de suponerlo."""


class GeminiVerdict(BaseModel):
    """Esquema de salida estructurada que Gemini debe respetar."""

    is_sleeping: bool = Field(description="Veredicto final: true solo si duerme con alta certeza.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza del veredicto entre 0 y 1.")
    activity_detected: Literal[
        "ninguna", "telefono", "escribiendo", "leyendo", "radio", "mirando_monitor", "conversando", "otra"
    ] = Field(description="Actividad visible que indica que esta despierto, o 'ninguna'.")
    eyes_state: Literal["cerrados", "abiertos", "no_visibles"] = Field(description="Estado de los parpados.")
    jaw_relaxed: bool = Field(description="Mandibula relajada o boca completamente abierta.")
    muscle_tone_lost: bool = Field(description="Cuerpo o cabeza desplomados por perdida de tono muscular.")
    reasoning: str = Field(description="Justificacion breve (maximo 2 frases) en español.")


VerdictCallback = Callable[[str, Optional[Dict[str, Any]], Optional[str]], None]
"""callback(track_id, verdict | None, error | None). Se invoca exactamente una vez por peticion aceptada."""


@dataclass(frozen=True)
class GeminiJudgeConfig:
    api_key: Optional[str]
    enabled: bool = True
    model: str = "gemini-3.8-flash"
    timeout_seconds: float = 8.0
    cooldown_seconds: float = 10.0
    rate_limit_pause_seconds: float = 60.0
    max_workers: int = 2
    max_image_side: int = 768
    jpeg_quality: int = 85
    padding_ratio: float = 0.20
    thinking_level: Optional[str] = None

    @staticmethod
    def from_env() -> "GeminiJudgeConfig":
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
        return GeminiJudgeConfig(
            api_key=api_key,
            enabled=os.getenv("GEMINI_JUDGE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "si"),
            model=os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.8-flash"),
            timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "8")),
            cooldown_seconds=float(os.getenv("GEMINI_COOLDOWN_SECONDS", "10")),
            rate_limit_pause_seconds=float(os.getenv("GEMINI_RATE_LIMIT_PAUSE", "60")),
            thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or None,
        )


def crop_with_context(frame: np.ndarray, bbox: Any, padding_ratio: float = 0.20) -> Optional[np.ndarray]:
    """Recorta la bbox [x1, y1, x2, y2] ampliada padding_ratio por lado
    (escritorio, manos, telefono, monitor) y devuelve una COPIA del recorte."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if right - left < 16 or bottom - top < 16:
        return None
    return frame[top:bottom, left:right].copy()


class GeminiJudgeService:
    def __init__(self, config: Optional[GeminiJudgeConfig] = None) -> None:
        self.config = config or GeminiJudgeConfig.from_env()
        self._client = None
        self._disabled_reason: Optional[str] = None
        self._paused_until = 0.0
        self._pending: Dict[str, float] = {}
        self._last_request: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._metrics = {"requests": 0, "sleeping": 0, "awake": 0, "errors": 0, "total_latency_ms": 0.0,
                         "skipped_cooldown": 0}
        self._last_error: Optional[str] = None

        if not self.config.enabled:
            self._disabled_reason = "GEMINI_JUDGE_ENABLED=false"
        elif not self.config.api_key:
            self._disabled_reason = "falta GEMINI_API_KEY"
        else:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self.config.api_key,
                    http_options=types.HttpOptions(timeout=int(self.config.timeout_seconds * 1000)),
                )
                self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers,
                                                    thread_name_prefix="gemini-judge")
            except ImportError:
                self._disabled_reason = "paquete google-genai no instalado"
            except Exception as exc:  # nunca debe impedir que arranque el pipeline
                self._disabled_reason = f"no se pudo crear el cliente: {type(exc).__name__}"

        if self._disabled_reason:
            logger.warning("Juez Gemini DESACTIVADO (%s): se usara solo el veredicto local", self._disabled_reason)
        else:
            logger.info("Juez Gemini activo | modelo=%s timeout=%.0fs cooldown=%.0fs",
                        self.config.model, self.config.timeout_seconds, self.config.cooldown_seconds)

    # -------------------------------------------------------------------
    # API publica
    # -------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True si el juez puede aceptar peticiones ahora mismo."""
        return self._disabled_reason is None and time.time() >= self._paused_until

    def can_verify(self, track_id: str) -> bool:
        """True si una peticion para este track seria aceptada (sin enviarla)."""
        if not self.available:
            return False
        with self._lock:
            return track_id not in self._pending and \
                time.time() - self._last_request.get(track_id, 0.0) >= self.config.cooldown_seconds

    def verify_drowsiness_async(self, image_crop: np.ndarray, track_id: str, callback: VerdictCallback,
                                context: Optional[Dict[str, Any]] = None) -> bool:
        """Encola la verificacion y retorna de inmediato. True si se acepto;
        False si el juez no esta disponible, el track ya tiene una peticion en
        vuelo o esta en cooldown (en ese caso NO se invoca el callback)."""
        if not self.available or self._executor is None:
            return False
        if image_crop is None or image_crop.size == 0:
            return False
        now = time.time()
        with self._lock:
            if track_id in self._pending or now - self._last_request.get(track_id, 0.0) < self.config.cooldown_seconds:
                self._metrics["skipped_cooldown"] += 1
                return False
            self._pending[track_id] = now
            self._last_request[track_id] = now
            self._metrics["requests"] += 1
        try:
            self._executor.submit(self._verify, image_crop, track_id, callback, context or {})
        except RuntimeError:  # executor cerrado
            with self._lock:
                self._pending.pop(track_id, None)
            return False
        return True

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            done = self._metrics["sleeping"] + self._metrics["awake"]
            return {
                "active": self._disabled_reason is None,
                "disabled_reason": self._disabled_reason,
                "paused_seconds": max(0.0, round(self._paused_until - time.time(), 1)),
                "model": self.config.model,
                "pending": len(self._pending),
                **{k: v for k, v in self._metrics.items() if k != "total_latency_ms"},
                "avg_latency_ms": round(self._metrics["total_latency_ms"] / done, 1) if done else 0.0,
                "last_error": self._last_error,
            }

    # -------------------------------------------------------------------
    # Trabajo en segundo plano
    # -------------------------------------------------------------------

    def _encode(self, image: np.ndarray) -> bytes:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > self.config.max_image_side:
            scale = self.config.max_image_side / longest
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        if not ok:
            raise ValueError("no se pudo codificar el recorte a JPEG")
        return encoded.tobytes()

    def _verify(self, image_crop: np.ndarray, track_id: str, callback: VerdictCallback, context: Dict[str, Any]) -> None:
        from google.genai import errors, types

        request_id = uuid.uuid4().hex[:8]
        started = time.time()
        verdict: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        try:
            jpeg = self._encode(image_crop)
            details = ", ".join(f"{k}={v}" for k, v in context.items() if v is not None)
            prompt = ("Evalúa si este guardia de seguridad está dormido. "
                      + (f"Señales del sensor local (solo orientativas, decide por la imagen): {details}." if details else ""))
            config_kwargs: Dict[str, Any] = {
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": GeminiVerdict,
                "temperature": 0.0,
                # Sin herramientas: evita el aviso del SDK sobre function calling automatico.
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if self.config.thinking_level:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.config.thinking_level)
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self.config.model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            parsed = response.parsed
            if isinstance(parsed, GeminiVerdict):
                verdict = parsed.model_dump()
            else:
                verdict = GeminiVerdict.model_validate_json(response.text or "").model_dump()
        except errors.ClientError as exc:
            error = self._handle_client_error(exc)
        except errors.ServerError as exc:
            error = f"server_error_{exc.code}"
        except Exception as exc:
            name = type(exc).__name__
            error = "timeout" if "Timeout" in name else f"{name}: {str(exc)[:120]}"

        latency_ms = (time.time() - started) * 1000.0
        with self._lock:
            self._pending.pop(track_id, None)
            if verdict is not None:
                self._metrics["sleeping" if verdict["is_sleeping"] else "awake"] += 1
                self._metrics["total_latency_ms"] += latency_ms
            else:
                self._metrics["errors"] += 1
                self._last_error = error
        if verdict is not None:
            logger.info("Gemini [%s] track=%s -> is_sleeping=%s conf=%.2f actividad=%s ojos=%s (%.0f ms)",
                        request_id, track_id, verdict["is_sleeping"], verdict["confidence"],
                        verdict["activity_detected"], verdict["eyes_state"], latency_ms)
        else:
            logger.warning("Gemini [%s] track=%s fallo (%s) en %.0f ms: se conserva el veredicto local",
                           request_id, track_id, error, latency_ms)
        try:
            callback(track_id, verdict, error)
        except Exception:
            logger.exception("El callback del juez Gemini fallo")

    def _handle_client_error(self, exc: Any) -> str:
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or exc)
        if code == 429:
            self._paused_until = time.time() + self.config.rate_limit_pause_seconds
            logger.warning("Gemini: limite de peticiones (429). Juez en pausa %.0f s", self.config.rate_limit_pause_seconds)
            return "rate_limited"
        if code in (401, 403) or (code == 400 and "API key" in message):
            self._disabled_reason = f"API key invalida o sin permisos (HTTP {code})"
            logger.error("Gemini: %s. Juez desactivado; se usara solo el veredicto local", self._disabled_reason)
            return "invalid_api_key"
        if code == 404:
            self._disabled_reason = f"modelo '{self.config.model}' no disponible (HTTP 404)"
            logger.error("Gemini: %s. Revisa GEMINI_JUDGE_MODEL", self._disabled_reason)
            return "model_not_found"
        return f"client_error_{code}"
