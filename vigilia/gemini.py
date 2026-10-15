"""Juez Gemini Vision: revisa una secuencia corta de recortes de UNA persona y
dictamina si esta despierta, somnolienta o dormida.

Ahorro de tokens: nunca se envia el video completo. Solo se consulta cuando la
heuristica local sospecha, con pocos recortes pequenos (cabeza y torso, no la
escena) y un resumen numerico del sensor local. Hay cooldown por persona,
tope de llamadas por hora y pausa automatica ante limites de cuota.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger("vigilia.gemini")

SYSTEM_PROMPT = """Eres un supervisor de seguridad hospitalaria. Recibes una secuencia cronologica de recortes de UNA persona (guardia en su puesto) tomados por una camara, y un resumen numerico de sensores locales.
Decide si esta DESPIERTA, SOMNOLIENTA (quedandose dormida) o DORMIDA.

Senales de sueno: ojos cerrados de forma sostenida, parpadeo lento, cabeceos o cabeza caida/apoyada, bostezos repetidos, cuerpo desplomado o recostado, inmovilidad prolongada, relajacion muscular (mandibula suelta).
Senales de vigilia: usa celular, escribe, lee, habla, mira un monitor, se mueve con intencion, conserva tension en cuello y tronco aunque mire hacia abajo.
Si el rostro no es visible, decide por postura, movimiento y tiempo. Un rasgo no visible se reporta como no visible, nunca se supone.
Usa 'dormida' solo con evidencia clara y sostenida en varios recortes; con evidencia parcial usa 'somnolienta'; si las imagenes no bastan, 'indeterminado'. El resumen numerico es orientativo: manda lo que ves."""


class Verdict(BaseModel):
    state: Literal["despierta", "somnolienta", "dormida", "indeterminado"]
    confidence: float = Field(ge=0.0, le=1.0)
    eyes: Literal["abiertos", "cerrados", "parpadeo_lento", "no_visibles"]
    yawning: bool
    head: Literal["erguida", "cabeceando", "caida_adelante", "caida_lateral", "apoyada", "no_visible"]
    posture: Literal["activa", "relajada", "desplomada", "recostada"]
    activity: Literal["ninguna", "celular", "escribiendo", "leyendo", "hablando", "monitor", "otra"]
    reasoning: str = Field(description="Maximo 2 frases en espanol.")


VerdictCallback = Callable[[str, Optional[Dict[str, Any]], Optional[str]], None]


class GeminiJudge:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._client = None
        self._types = None
        self._errors_mod = None
        self.disabled_reason: Optional[str] = None
        self._paused_until = 0.0
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._last_request: Dict[str, float] = {}
        self._call_times: Deque[float] = deque()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.stats = {"calls": 0, "dormida": 0, "somnolienta": 0, "despierta": 0, "indeterminado": 0,
                      "errors": 0, "tokens": 0, "skipped_budget": 0}
        self.last_error: Optional[str] = None

        if not settings.gemini_enabled:
            self.disabled_reason = "GEMINI_ENABLED=false"
        elif not settings.gemini_api_key:
            self.disabled_reason = "falta GEMINI_API_KEY"
        else:
            try:
                from google import genai
                from google.genai import errors, types

                self._client = genai.Client(api_key=settings.gemini_api_key,
                                            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout * 1000)))
                self._types, self._errors_mod = types, errors
                self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")
            except ImportError:
                self.disabled_reason = "paquete google-genai no instalado"
            except Exception as exc:  # noqa: BLE001
                self.disabled_reason = f"no se pudo crear el cliente: {exc.__class__.__name__}"
        if self.disabled_reason:
            logger.warning("Gemini DESACTIVADO (%s): se usara solo la heuristica local.", self.disabled_reason)
        else:
            logger.info("Gemini activo | modelo=%s | maximo %d llamadas/hora", settings.gemini_model,
                        settings.gemini_max_calls_per_hour)

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.disabled_reason is None and time.time() >= self._paused_until

    def can_review(self, key: str, now: float) -> bool:
        if not self.available:
            return False
        with self._lock:
            if key in self._pending or now - self._last_request.get(key, 0.0) < self.s.gemini_cooldown:
                return False
            while self._call_times and now - self._call_times[0] > 3600:
                self._call_times.popleft()
            return len(self._call_times) < self.s.gemini_max_calls_per_hour

    def review(self, key: str, frames: List[Tuple[float, bytes]], context: Dict[str, Any],
               callback: VerdictCallback) -> bool:
        """Encola la revision (no bloquea). `frames` = [(segundos_atras, jpeg)]. Devuelve False si se rechaza."""
        now = time.time()
        if not frames or self._executor is None or not self.can_review(key, now):
            if self.available and frames:
                self.stats["skipped_budget"] += 1
            return False
        with self._lock:
            self._pending.add(key)
            self._last_request[key] = now
            self._call_times.append(now)
        try:
            self._executor.submit(self._run, key, frames, context, callback)
        except RuntimeError:
            with self._lock:
                self._pending.discard(key)
            return False
        return True

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def health(self) -> Dict[str, Any]:
        return {"active": self.disabled_reason is None, "disabled_reason": self.disabled_reason,
                "model": self.s.gemini_model, "paused_seconds": max(0, round(self._paused_until - time.time())),
                "pending": len(self._pending), "last_error": self.last_error, **self.stats}

    # ------------------------------------------------------------------
    def _run(self, key: str, frames: List[Tuple[float, bytes]], context: Dict[str, Any],
             callback: VerdictCallback) -> None:
        types = self._types
        started = time.time()
        verdict: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        tokens = 0
        try:
            parts: List[Any] = []
            for age, jpeg in frames:
                parts.append(f"Recorte tomado hace {age:.0f} s:")
                parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
            summary = ", ".join(f"{k}={v}" for k, v in context.items() if v is not None)
            parts.append(f"Resumen de sensores locales (orientativo): {summary}. Evalua el estado de la persona.")
            config: Dict[str, Any] = {
                "system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json",
                "response_schema": Verdict, "temperature": 0.0, "max_output_tokens": 400,
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if self.s.gemini_thinking_level:
                config["thinking_config"] = types.ThinkingConfig(thinking_level=self.s.gemini_thinking_level)
            response = self._client.models.generate_content(
                model=self.s.gemini_model, contents=parts, config=types.GenerateContentConfig(**config))
            parsed = response.parsed
            verdict = (parsed if isinstance(parsed, Verdict)
                       else Verdict.model_validate_json(response.text or "")).model_dump()
            usage = getattr(response, "usage_metadata", None)
            tokens = int(getattr(usage, "total_token_count", 0) or 0)
        except self._errors_mod.ClientError as exc:
            error = self._client_error(exc)
        except self._errors_mod.ServerError as exc:
            error = f"server_error_{exc.code}"
        except Exception as exc:  # noqa: BLE001
            error = "timeout" if "Timeout" in exc.__class__.__name__ else f"{exc.__class__.__name__}: {str(exc)[:100]}"

        latency = (time.time() - started) * 1000
        with self._lock:
            self._pending.discard(key)
            if verdict is not None:
                self.stats["calls"] += 1
                self.stats[verdict["state"]] += 1
                self.stats["tokens"] += tokens
            else:
                self.stats["errors"] += 1
                self.last_error = error
        if verdict is not None:
            logger.info("Gemini %s -> %s (conf %.2f, ojos=%s, %d tokens, %.0f ms)", key, verdict["state"],
                        verdict["confidence"], verdict["eyes"], tokens, latency)
        else:
            logger.warning("Gemini %s fallo (%s, %.0f ms); se conserva la heuristica local", key, error, latency)
        try:
            callback(key, verdict, error)
        except Exception:  # noqa: BLE001
            logger.exception("Callback de Gemini fallo")

    def _client_error(self, exc: Any) -> str:
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or exc)
        if code == 429:
            self._paused_until = time.time() + 60
            logger.warning("Gemini: limite de cuota (429). Pausa de 60 s.")
            return "rate_limited"
        if code in (401, 403) or (code == 400 and "API key" in message):
            self.disabled_reason = f"API key invalida o sin permisos (HTTP {code})"
            logger.error("Gemini: %s. Desactivado hasta reiniciar.", self.disabled_reason)
            return "invalid_api_key"
        if code == 404:
            self.disabled_reason = f"modelo '{self.s.gemini_model}' no disponible (HTTP 404). Revise GEMINI_MODEL"
            logger.error("Gemini: %s", self.disabled_reason)
            return "model_not_found"
        return f"client_error_{code}"
