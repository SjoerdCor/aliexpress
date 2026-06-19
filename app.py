"""The flask server that governs the app"""

import logging
import os
import webbrowser

from dotenv import load_dotenv
from flask import Flask, render_template

from aliexpress.admin import admin_bp
from aliexpress.cli import admins as admins_cli
from aliexpress.cli import schools as schools_cli
from aliexpress.extensions import db, limiter, login_manager
from aliexpress.http_errors import register_error_handlers
from aliexpress.logging_config import add_file_handler, configure_logging
from aliexpress.routes.auth import auth_bp, load_user
from aliexpress.routes.processes import processes_bp
from aliexpress.routes.results import results_bp
from aliexpress.routes.wizard import wizard_bp

configure_logging()
logger = logging.getLogger("aliexpress.app")  # file handler added below


load_dotenv()

env = os.getenv("FLASK_ENV", "production")
if env == "development":
    from aliexpress.appconfig import DevelopmentConfig as ConfigClass
else:
    from aliexpress.appconfig import ProductionConfig as ConfigClass


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


app = Flask(__name__)
app.config.from_object(ConfigClass)
ensure_secret_key(app)
LOG_DIR = os.path.join(app.instance_path, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
add_file_handler(os.path.join(LOG_DIR, "aliexpress.log"))
app.config["STORAGE_DIR"] = os.path.join(app.instance_path, "storage")
os.makedirs(app.config["STORAGE_DIR"], exist_ok=True)
logger.debug("Created dir if not exists: %s", app.config["STORAGE_DIR"])

# A relative SQLite URI is resolved against the instance folder, which must exist first.
os.makedirs(app.instance_path, exist_ok=True)
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message = "Je moet ingelogd zijn om deze pagina te bekijken."
login_manager.login_message_category = "error"

# Rate-limit the login route against brute force. In-memory storage suffices for a single
# process; a multi-process deployment (the future EU server) needs a shared backend (Redis).
limiter.init_app(app)

with app.app_context():
    db.create_all()


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


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=env == "development")
