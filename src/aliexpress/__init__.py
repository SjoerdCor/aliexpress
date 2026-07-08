"""Flask application factory.

Importing from this package yields ``create_app``, which wires together the
Flask application, extensions, blueprints, error handlers, and CLI commands.
The root ``app.py`` is a thin launcher that calls this factory.
"""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, g, render_template, request, session

from aliexpress.logging_config import (
    add_file_handler,
    configure_logging,
    pop_log_context,
    push_log_context,
)
from aliexpress.web.admin_seed import ensure_admin_password, seed_admin_from_env
from aliexpress.web.appconfig import DevelopmentConfig, ProductionConfig
from aliexpress.web.cli import schools as schools_cli
from aliexpress.web.extensions import db, limiter, login_manager
from aliexpress.web.http_errors import register_error_handlers
from aliexpress.web.routes.admin import admin_bp
from aliexpress.web.routes.auth import auth_bp, load_user
from aliexpress.web.routes.processes import processes_bp
from aliexpress.web.routes.results import results_bp
from aliexpress.web.routes.roster import roster_bp
from aliexpress.web.routes.wizard import wizard_bp

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


def _configure_secrets(app, env, test_config):
    """Resolve SECRET_KEY/ADMIN_PASSWORD from the environment and fail fast if unusable.

    Read here, after ``load_dotenv()``, so ``.env`` is already in ``os.environ``; they
    cannot live in the Config class body because class attributes are evaluated at
    import time, before ``load_dotenv()`` runs (and before uv injects ``.env`` on
    non-uv launchers). ``test_config`` is applied before the guards so tests can
    override either value.
    """
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or (
        "dev-fallback-secret" if env == "development" else None
    )
    app.config["ADMIN_PASSWORD"] = os.getenv("ADMIN_PASSWORD")

    if test_config is not None:
        app.config.update(test_config)

    ensure_secret_key(app)
    ensure_admin_password(app)


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

    _configure_secrets(app, env, test_config)

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

    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(processes_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(roster_bp)
    app.register_blueprint(wizard_bp)

    register_error_handlers(app)

    @app.before_request
    def _push_log_ctx():
        """Bind school/process/endpoint to the log context for this request."""
        # pylint: disable=import-outside-toplevel
        from flask_login import current_user

        from aliexpress.web.routes.auth import effective_school_id

        school = effective_school_id() if current_user.is_authenticated else None
        token = push_log_context(
            school=school,
            process=session.get("process_id"),
            phase=request.endpoint,
        )
        g.log_ctx_token = token

    @app.teardown_request
    def _pop_log_ctx(exc):  # pylint: disable=unused-argument
        """Restore the log context after each request (Werkzeug reuses threads).

        Guard against double-reset: Flask's test client preserves the last request
        context across requests and pops it at __exit__, which calls teardown_request
        a second time with the same (already-used) token.
        """
        token = getattr(g, "log_ctx_token", None)
        if token is not None:
            g.log_ctx_token = None
            pop_log_context(token)

    @app.route("/")
    def home():
        """Display home page"""
        return render_template("home.html")

    with app.app_context():
        db.create_all()
        seed_admin_from_env(app)

    return app
