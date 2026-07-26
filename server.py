"""
server.py — HealthBot application entrypoint.

Serves:
  - The static HTML/CSS/JS chat frontend (static/index.html, /static/css,
    /static/js) at "/".
  - The JSON API defined in api/routes.py, mounted at "/api/...".

Run with:
    python3 server.py
or, for production-style serving:
    gunicorn -w 1 -b 0.0.0.0:5000 "server:app"

Note on worker count: keep `-w 1` unless you swap the in-memory session
store (chatbot/session_store.py) for a shared backend like Redis — see
that module's docstring for why.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

from api.routes import api
from config import APP_TITLE, FLASK_DEBUG, FLASK_HOST, FLASK_PORT

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.errorhandler(404)
    def not_found(_err):
        # Keep API 404s as JSON; let everything else fall through to the SPA.
        from flask import jsonify, request

        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(STATIC_DIR, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    print(f"\n{APP_TITLE}")
    print(f"Starting on http://{FLASK_HOST}:{FLASK_PORT}  (Ctrl+C to stop)\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
