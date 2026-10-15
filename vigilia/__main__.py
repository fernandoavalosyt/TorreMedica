"""Arranque:  python -m vigilia"""

from __future__ import annotations

import logging
import signal
import sys
import threading

from dotenv import load_dotenv

from .config import ROOT, ConfigError, Settings


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    logging.basicConfig(level=settings.log_level.upper(), datefmt="%H:%M:%S",
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from .app import Vigilia
    from .web import create_web

    try:
        core = Vigilia(settings)
    except (ConfigError, RuntimeError) as exc:
        logging.getLogger("vigilia").error("%s", exc)
        return 2
    core.start()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    web = create_web(core)
    threading.Thread(target=lambda: web.run(host=settings.web_host, port=settings.web_port, threaded=True,
                                            use_reloader=False), daemon=True, name="web").start()
    logging.getLogger("vigilia").info("Panel: http://%s:%s", settings.web_host, settings.web_port)
    try:
        while not stop.wait(0.5):
            pass
    finally:
        logging.getLogger("vigilia").info("Deteniendo...")
        core.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
