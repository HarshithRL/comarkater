"""Flask UI factory — internal dashboard, mounted on same Databricks App."""

import os

from flask import Flask


def create_ui_app() -> Flask:
    """Create and configure the Flask UI application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "comarketer-dev-only")

    from ui.routes import ui_bp
    app.register_blueprint(ui_bp)
    return app
