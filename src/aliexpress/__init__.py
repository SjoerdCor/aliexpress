"""Flask application factory.

Importing from this package yields ``create_app``, which wires together the
Flask application, extensions, blueprints, error handlers, and CLI commands.
The root ``app.py`` is a thin launcher that calls this factory.
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, render_template

from aliexpress.appconfig import DevelopmentConfig, ProductionConfig
from aliexpress.cli import admins as admins_cli
from aliexpress.cli import schools as schools_cli
from aliexpress.extensions import db, limiter, login_manager
from aliexpress.http_errors import register_error_handlers
from aliexpress.logging_config import add_file_handler, configure_logging
from aliexpress.routes.admin import admin_bp
from aliexpress.routes.auth import auth_bp, load_user
from aliexpress.routes.processes import processes_bp
from aliexpress.routes.results import results_bp
from aliexpress.routes.wizard import wizard_bp

configure_logging()
_logger = logging.getLogger(__name__)


def ensure_secret_key(flask_app):
    """Refuse to start without a signing key.

    An empty SECRET_KEY makes session cookies unsignable (and login state forgeable),
    so fail fast at startup rather than at the first request. Development supplies a
    fallback key in DevelopmentConfig, so this only bites a misconfigured production deploy.
    """
    if not flask_app.config.get("SECRET_KEY"):
        raise RuntimeError(
            "SECRET_KEY is not set; refusing to start. Set it in the environment (.env)."
        )


def create_app(test_config=None):
    """Create and configure a Flask application instance.

    ``test_config`` is a dict of settings that override the defaults; it is
    used by the test suite to inject TESTING=True, an in-memory SQLite URI,
    and a temporary STORAGE_DIR without touching the real instance folder.
    When ``test_config`` is provided the file log handler is also skipped so
    repeated fixture calls do not accumulate duplicate handlers.
    """
    load_dotenv()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    instance_path = os.path.join(project_root, "instance")

    app = Flask(
        __name__,
        root_path=project_root,
        instance_path=instance_path,
    )

    env = os.getenv("FLASK_ENV", "production")
    config_class = DevelopmentConfig if env == "development" else ProductionConfig
    app.config.from_object(config_class)

    if test_config is not None:
        app.config.update(test_config)

    ensure_secret_key(app)

    os.makedirs(app.instance_path, exist_ok=True)
    if "STORAGE_DIR" not in app.config:
        app.config["STORAGE_DIR"] = os.path.join(app.instance_path, "storage")
    os.makedirs(app.config["STORAGE_DIR"], exist_ok=True)
    _logger.debug("Created dir if not exists: %s", app.config["STORAGE_DIR"])

    # File handler only in production: add_file_handler is not idempotent and
    # would accumulate duplicate handlers if called on every test fixture invocation.
    if test_config is None:
        log_dir = os.path.join(app.instance_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        add_file_handler(os.path.join(log_dir, "aliexpress.log"))

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Je moet ingelogd zijn om deze pagina te bekijken."
    login_manager.login_message_category = "error"
    limiter.init_app(app)

    login_manager.user_loader(load_user)

    app.cli.add_command(schools_cli, "schools")
    app.cli.add_command(admins_cli, "admins")

    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(processes_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(wizard_bp)

    register_error_handlers(app)

    @app.route("/")
    def home():
        """Display home page"""
        return render_template("home.html")

    with app.app_context():
        db.create_all()

    return app
