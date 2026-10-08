"""Small local HTTP API; designed for a demo and easy inspection."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .evaluation import evaluate
from .models import Dataset, parse_dataset
from .report import render_report
from .storage import RunStore


MAX_BODY_BYTES = 2 * 1024 * 1024
LOG = logging.getLogger(__name__)


def make_handler(store: RunStore, demo: Dataset, demo_run_id: str, demo_comparison: dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RAGFailureLab/0.1"

        def log_message(self, format: str, *args: object) -> None:
            LOG.info(json.dumps({"remote": self.client_address[0], "message": format % args}))

        def _write(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._write(status, json.dumps(data, separators=(",", ":")).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            path = urlsplit(self.path).path
            if path in {"/", "/report"}:
                run = store.get(demo_run_id)
                assert run is not None
                self._write(200, render_report(run, demo_comparison).encode(), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._json(200, {"status": "ok"})
            elif path == "/api/runs":
                self._json(200, {"runs": store.list()})
            elif path.startswith("/api/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) not in {3, 4} or parts[:2] != ["api", "runs"]:
                    self._json(404, {"error": "not found"})
                    return
                run = store.get(parts[2])
                if run is None:
                    self._json(404, {"error": "run not found"})
                elif len(parts) == 4 and parts[3] == "report":
                    self._write(200, render_report(run).encode(), "text/html; charset=utf-8")
                elif len(parts) == 3:
                    self._json(200, run)
                else:
                    self._json(404, {"error": "not found"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            if urlsplit(self.path).path != "/api/runs":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    raise ValueError("request body must be between 1 byte and 2 MiB")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                dataset = parse_dataset(request.get("dataset"))
                mode = request.get("mode", "bm25")
                top_k = request.get("top_k", 3)
                if not isinstance(mode, str):
                    raise ValueError("mode must be keyword, bm25, or recorded")
                if not isinstance(top_k, int) or isinstance(top_k, bool):
                    raise ValueError("top_k must be an integer")
                run = evaluate(dataset, mode=mode, top_k=top_k)
                store.save(run)
                self._json(201, {"id": run["id"], "summary": run["summary"], "url": f"/api/runs/{run['id']}"})
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def serve(host: str, port: int, store: RunStore, demo: Dataset, demo_run_id: str, demo_comparison: dict[str, Any]) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(store, demo, demo_run_id, demo_comparison))
    LOG.info("serving on http://%s:%d", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
