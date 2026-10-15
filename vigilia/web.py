"""Panel web (Flask): video en vivo, personas con ID, alertas y alta de camaras con credenciales."""

from __future__ import annotations

import hmac
import logging
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import __version__
from .app import Vigilia
from .camera import test_connection
from .config import validate_camera

logger = logging.getLogger("vigilia.web")
STATIC = Path(__file__).parent / "static"


def create_web(app_core: Vigilia) -> Flask:
    web = Flask(__name__, static_folder=None)
    web.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    s = app_core.s

    @web.before_request
    def guard():
        if s.web_user and s.web_password:
            auth = request.authorization
            if not (auth and hmac.compare_digest(auth.username or "", s.web_user)
                    and hmac.compare_digest(auth.password or "", s.web_password)):
                return Response("Autenticacion requerida", 401, {"WWW-Authenticate": 'Basic realm="Vigilia"'})
        # Los cambios solo se aceptan como JSON (evita formularios cruzados desde otros sitios).
        if request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/"):
            if request.method != "DELETE" and not request.is_json:
                return jsonify(error="Content-Type debe ser application/json"), 415
            if request.headers.get("X-Requested-With") != "vigilia":
                return jsonify(error="Falta la cabecera X-Requested-With"), 400

    @web.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @web.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @web.get("/healthz")
    def healthz():
        return jsonify(status="ok", version=__version__)

    @web.get("/api/status")
    def status():
        return jsonify(app_core.status())

    # ---- video ------------------------------------------------------------
    @web.get("/stream/<cam_id>")
    def stream(cam_id: str):
        pipeline = app_core.pipeline(cam_id)
        if pipeline is None:
            abort(404)

        def generate():
            last = None
            while True:
                jpeg = pipeline.jpeg()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() \
                        + b"\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.1)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---- alertas ------------------------------------------------------------
    @web.get("/api/alerts")
    def alerts():
        limit = request.args.get("limit", 50, type=int)
        return jsonify(app_core.alerts.recent(max(1, min(limit, 200))))

    @web.post("/api/alerts/<alert_id>/ack")
    def ack(alert_id: str):
        return (jsonify(ok=True) if app_core.alerts.ack(alert_id) else (jsonify(error="no existe"), 404))

    @web.get("/evidence/<name>")
    def evidence(name: str):
        path = app_core.alerts.snapshot_path(name)
        if path is None:
            abort(404)
        return send_from_directory(path.parent, path.name)

    # ---- camaras (credenciales) ---------------------------------------------
    @web.get("/api/cameras")
    def cameras():
        return jsonify([c.public() for c in app_core.store.list()])

    def _parse(payload):
        if not isinstance(payload, dict):
            raise ValueError("payload invalido")
        previous = app_core.store.get(str(payload.get("id") or ""))
        return validate_camera(payload, existing_password=previous.password if previous else "")

    @web.post("/api/cameras/test")
    def camera_test():
        try:
            cfg = _parse(request.get_json())
        except ValueError as exc:
            return jsonify(ok=False, message=str(exc)), 400
        return jsonify(test_connection(cfg))

    @web.post("/api/cameras")
    def camera_save():
        try:
            cfg = _parse(request.get_json())
            saved = app_core.save_camera(cfg)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(saved.public()), 201

    @web.delete("/api/cameras/<cam_id>")
    def camera_delete(cam_id: str):
        return (jsonify(ok=True) if app_core.delete_camera(cam_id) else (jsonify(error="no existe"), 404))

    @web.errorhandler(400)
    @web.errorhandler(413)
    def bad_request(exc):
        return jsonify(error="solicitud invalida"), exc.code

    return web
