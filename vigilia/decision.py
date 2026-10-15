"""Decision final: combina el estado local y el dictamen de Gemini.

Niveles: ok < somnolencia (se esta quedando dormida) < dormido.

Reglas, en orden:
 1. Veto: si Gemini dijo "despierta" con confianza suficiente, no hay alerta durante GEMINI_AWAKE_VETO_SECONDS.
 2. Dictamen vigente de Gemini (solo mientras la heuristica local siga sospechando):
      dormida   (conf >= MIN_CONF_ASLEEP) -> dormido
      dormida/somnolienta (conf >= MIN_CONF_DROWSY) -> somnolencia
 3. Revision de Gemini en curso: se mantiene el nivel actual (no se escala a ciegas).
 4. Respaldo local (Gemini apagado, sin cuota, indeterminado o con error): el estado local
    debe sostenerse mas tiempo antes de alertar.
"""

from __future__ import annotations

from typing import Tuple

from .config import Settings
from .heuristics import TrackState

LEVEL_RANK = {"ok": 0, "somnolencia": 1, "dormido": 2}
STILL_SUSPICIOUS = 0.25   # un dictamen de Gemini deja de valer si la puntuacion local ya bajo de esto


def decide(ts: TrackState, now: float, s: Settings) -> Tuple[str, str, str]:
    """Devuelve (nivel, fuente, motivo)."""
    if now < ts.veto_until:
        return "ok", "gemini", "Gemini: persona despierta"

    verdict = ts.verdict
    if verdict is not None and now - ts.verdict_time <= s.verdict_ttl and ts.score_ema >= STILL_SUSPICIOUS:
        state, conf = verdict["state"], verdict["confidence"]
        if state == "dormida" and conf >= s.min_conf_asleep:
            return "dormido", "gemini", verdict["reasoning"]
        if state in ("dormida", "somnolienta") and conf >= s.min_conf_drowsy:
            return "somnolencia", "gemini", verdict["reasoning"]

    if ts.review_pending:
        return ts.level, ts.level_source, ts.level_reason

    sleep_for = 0.0 if ts.sleep_timer.since is None else now - ts.sleep_timer.since
    drowsy_for = 0.0 if ts.suspect_timer.since is None else now - ts.suspect_timer.since
    if ts.local_state == "dormida" and sleep_for >= s.local_only_sleep_seconds:
        return "dormido", "local", f"Postura/ojos/inmovilidad compatibles con sueno durante {sleep_for:.0f} s"
    if ts.local_state != "despierta" and drowsy_for >= s.local_only_drowsy_seconds:
        return "somnolencia", "local", f"Signos de somnolencia sostenidos durante {drowsy_for:.0f} s"
    return "ok", "", ""
