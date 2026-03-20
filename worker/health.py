from __future__ import annotations

"""Simple HTTP health endpoint for the GPU worker (optional, for monitoring)."""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress default stdout logging


def start_health_server(port: int = 8080) -> bool:
    try:
        server = HTTPServer(("0.0.0.0", port), _Handler)
    except OSError as exc:
        # Health endpoint is optional; worker must continue even if port is busy.
        log.warning("Health endpoint disabled on port %s: %s", port, exc)
        return False

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return True
