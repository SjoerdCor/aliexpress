"""The flask server that governs the app"""

import json
import logging
import os
import webbrowser

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import login_required

from aliexpress.admin import admin_bp
from aliexpress.cli import admins as admins_cli
from aliexpress.cli import schools as schools_cli
from aliexpress.extensions import db, limiter, login_manager
from aliexpress.http_errors import register_error_handlers
from aliexpress.logging_config import add_file_handler, configure_logging
from aliexpress.models import Process
from aliexpress.routes.auth import auth_bp, effective_school_id, load_user
from aliexpress.routes.processes import processes_bp, require_process
from aliexpress.routes.wizard import wizard_bp
from aliexpress.storage import get_file_path

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
app.register_blueprint(wizard_bp)


register_error_handlers(app)


@app.route("/")
def home():
    """Display home page"""
    return render_template("home.html")


@app.route("/download_preferences")
@login_required
@require_process
def download_preferences():
    """Download the filled-in preferences file as the teacher uploaded it."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "preferences.xlsx")
    if not os.path.exists(path):
        logger.warning(
            "Download of filled-in preferences requested but none stored for process %s",
            process_id,
        )
        abort(404)
    logger.info("Serving stored preferences upload for process %s", process_id)
    return send_file(
        path,
        as_attachment=True,
        download_name="voorkeuren (ingevuld).xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/status")
@login_required
@require_process
def status():
    """Return the current process's run status and log lines as JSON."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_name = session["process_id"]
    proc = Process.by_name(school_id, process_name)
    if proc is None or proc.run is None:
        return jsonify({"status_studentdistribution": "unknown", "logs": []})
    run = proc.run
    payload = {
        "status_studentdistribution": run.status,
        "logs": [line.text for line in run.log_lines],
    }
    if run.status == "error" and run.message:
        payload["message"] = run.message
    return jsonify(payload)


@app.route("/processing")
@login_required
@require_process
def processing():
    """Display processing page"""
    return render_template("processing.html")


@app.route("/handle-error", methods=["POST"])
@login_required
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    flash(data["message"], "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@app.route("/sociogram")
@login_required
@require_process
def show_sociogram():
    """Display the sociogram for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "sociogram.html")
    if not os.path.exists(path):
        flash("Sociogram niet beschikbaar.", "error")
        return redirect(url_for("processes.index"))
    with open(path, encoding="utf-8") as fh:
        plotly_div = fh.read()
    return render_template("sociogram.html", plotly_div=plotly_div)


@app.route("/result")
@login_required
@require_process
def result_page():
    """Display result for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "result_tables.json")
    if not os.path.exists(path):
        flash("Resultaat niet beschikbaar.", "error")
        return redirect(url_for("processes.index"))
    with open(path, encoding="utf-8") as fh:
        dataframes = json.load(fh)
    return render_template("result.html", dataframes=dataframes)


@app.route("/download")
@login_required
@require_process
def download():
    """Download the groepsindeling for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "results.xlsx")
    if not os.path.exists(path):
        flash("Groepsindeling niet gevonden. Mogelijk nog aan het berekenen", "error")
        return render_template("result.html", dataframes={})

    return send_file(
        path,
        as_attachment=True,
        download_name="results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/done")
@login_required
def done():
    """Show done page"""
    return render_template("done.html")


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=env == "development")
