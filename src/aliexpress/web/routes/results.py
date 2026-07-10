"""Results blueprint: processing, status, sociogram, result, download, and done routes."""

import json
import logging
import os

from flask import (
    Blueprint,
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

from ..models import Process
from ..storage import get_file_path
from .auth import effective_school_id
from .processes import require_process

logger = logging.getLogger(__name__)

results_bp = Blueprint("results", __name__)


@results_bp.route("/processing")
@login_required
@require_process
def processing():
    """Display processing page"""
    return render_template("processing.html")


@results_bp.route("/status")
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


@results_bp.route("/handle-error", methods=["POST"])
@login_required
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    flash(data["message"], "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@results_bp.route("/sociogram")
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


@results_bp.route("/result")
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
    view_path = get_file_path(school_id, process_id, "groepsindeling_view.json")
    groepsindeling_view = None
    if os.path.exists(view_path):
        with open(view_path, encoding="utf-8") as fh:
            groepsindeling_view = json.load(fh)
    return render_template(
        "result.html",
        dataframes=dataframes,
        groepsindeling_view=groepsindeling_view,
    )


@results_bp.route("/download")
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


@results_bp.route("/done")
@login_required
def done():
    """Show done page"""
    return render_template("done.html")


@results_bp.route("/download_preferences")
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
