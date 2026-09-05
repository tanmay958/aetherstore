#!/usr/bin/env python3
"""Serve the dashboard locally, standing in for Cloudflare Pages.

In production the page and the API share an origin: Cloudflare serves the
static files and a Pages Function forwards /api/* to Cloud Run. That is what
makes the API key work, because the key lives in the Function where a visitor
cannot read it, and it is also why the page needs no CORS at all.

Opening index.html against the deployed service directly does not reproduce
that. The browser sees two origins, asks Cloud Run for CORS permission, is
refused, and every request fails. Adding CORS to the service to make local
development easier would be exactly the wrong fix: it would weaken the
same-origin property that the deployment depends on, to solve a problem that
only exists on a laptop.

So this does locally what the Function does remotely.

    python3 web/dev-server.py
    open http://localhost:8500

    AETHER_ORIGIN=...  which service to forward to
    AETHER_API_KEY=... sent as x-aether-key, when the service requires one
"""

from __future__ import annotations

import http.server
import os
import urllib.error
import urllib.request
from pathlib import Path

ORIGIN = os.environ.get(
    "AETHER_ORIGIN", "https://aether-441173057461.us-central1.run.app"
)
API_KEY = os.environ.get("AETHER_API_KEY", "")
PORT = int(os.environ.get("PORT", 8500))
WEB = Path(__file__).parent

# The same allowlist the Function enforces, for the same reason: /api/* must
# not become a tunnel to whatever paths the service grows later.
ALLOWED = {"search", "predict", "index", "model"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self):  # noqa: N802 - the base class names it
        if self.path.startswith("/api/"):
            return self.proxy("GET")
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self.proxy("POST")
        self.send_error(405)

    def proxy(self, method: str) -> None:
        path = self.path[len("/api/"):]
        route = path.split("?")[0].strip("/")
        if route not in ALLOWED:
            return self.send_error(404, f"no such endpoint: {route}")

        body = None
        if method == "POST":
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length)

        request = urllib.request.Request(
            f"{ORIGIN}/api/{path}",
            data=body,
            method=method,
            headers={
                "accept": "application/json",
                **({"content-type": "application/json"} if body else {}),
                **({"x-aether-key": API_KEY} if API_KEY else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as upstream:
                payload, status = upstream.read(), upstream.status
        except urllib.error.HTTPError as error:
            payload, status = error.read(), error.code
        except Exception as error:  # noqa: BLE001
            payload = f'{{"detail":"upstream unreachable: {error}"}}'.encode()
            status = 502

        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")


if __name__ == "__main__":
    print(f"dashboard  http://localhost:{PORT}")
    print(f"  /api/*  ->  {ORIGIN}/api/*" + ("  (with key)" if API_KEY else ""))
    http.server.ThreadingHTTPServer(("", PORT), Handler).serve_forever()
