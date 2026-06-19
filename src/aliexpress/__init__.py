"""Flask application factory.

Importing from this package yields ``create_app``, which wires together the
Flask application, extensions, and (later) blueprints.  The root ``app.py``
will eventually shrink to a one-liner that calls this factory.
"""

import os

from flask import Flask

from aliexpress.appconfig import DevelopmentConfig, ProductionConfig
from aliexpress.extensions import db, limiter, login_manager


def create_app(test_config=None):
    """Create and configure a Flask application instance.

    ``test_config`` is a dict of settings that override the defaults; it is
    used by the test suite to inject TESTING=True, an in-memory SQLite URI,
    and a temporary STORAGE_DIR without touching the real instance folder.
    """
    # Resolve the project root so Flask finds templates/ and static/ at the
    # right location regardless of where this package is installed.
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

    os.makedirs(app.instance_path, exist_ok=True)

    # Derived path: tests override via test_config; production gets instance/storage.
    if "STORAGE_DIR" not in app.config:
        app.config["STORAGE_DIR"] = os.path.join(app.instance_path, "storage")

    db.init_app(app)
    login_manager.init_app(app)
    limiter.init_app(app)

    return app
