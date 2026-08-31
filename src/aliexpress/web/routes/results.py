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

from ...main import build_input_summary
from ...solver._balance import default_balance_maxima
from ..models import Process
from ..process_files import load_balance_maxima, load_groups, load_voorkeuren
from ..storage import get_file_path
from .auth import effective_school_id
from .processes import require_process

logger = logging.getLogger(__name__)

results_bp = Blueprint("results", __name__)


@results_bp.route("/processing")
@login_required
@require_process
def processing():
    """Display the processing page: an idle panel to start the solve, or its live progress.

    Branches on the process's Run status: "pending" or "running" shows the live progress
    view (the poll-driven stepper etc., unchanged) — "pending" is the brief window right
    after Start verdeling, before the background thread's first status write lands, and a
    fast solve can finish within it, so it must not fall back to the idle panel; "done"
    redirects straight to the result; anything else ("error", or no run yet at all) shows
    the idle panel, read-only — it writes nothing, so revisiting this page never has side
    effects.
    """
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    proc = Process.by_name(school_id, process_id)
    run_status = proc.run.status if proc and proc.run else None

    if run_status == "done":
        return redirect(url_for("results.result_page"))

    preference_data, _ = load_voorkeuren(school_id, process_id)
    target_groups = load_groups(school_id, process_id)
    summary = build_input_summary(
        target_groups.counts,
        preference_data.students_info,
        preference_data.stamgroep_display,
    )

    if run_status in ("pending", "running"):
        return render_template("processing.html", mode="running", summary=summary)

    maxima_path = get_file_path(school_id, process_id, "balance_limits.json")
    if run_status == "error" and os.path.exists(maxima_path):
        maxima = load_balance_maxima(school_id, process_id)
    else:
        maxima = default_balance_maxima(
            preference_data.students_info, target_groups.counts
        )
    return render_template(
        "processing.html", mode="idle", summary=summary, maxima=maxima
    )


@results_bp.route("/status")
@login_required
@require_process
def status():
    """Return the current process's run status and progress as JSON."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_name = session["process_id"]
    proc = Process.by_name(school_id, process_name)
    if proc is None or proc.run is None:
        return jsonify({"status_studentdistribution": "unknown"})
    run = proc.run
    payload = {
        "status_studentdistribution": run.status,
        "sociogram_ready": os.path.exists(
            get_file_path(school_id, process_name, "sociogram.html")
        ),
    }
    progress_path = get_file_path(school_id, process_name, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as fh:
            payload.update(json.load(fh))
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


@results_bp.route("/interim_result")
@login_required
@require_process
def interim_result():
    """Render the current interim group-card view while the solve is still running.

    Loads ``interim_result.json`` (written by :class:`~..progress_writer.ProgressWriter`
    on every solved stage boundary); returns 204 when none exists yet (nothing solved
    far enough to report). The processing page fetches this whenever ``/status`` reports
    a new ``interim_result_updated_at``.
    """
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "interim_result.json")
    if not os.path.exists(path):
        return "", 204
    with open(path, encoding="utf-8") as fh:
        view = json.load(fh)
    return render_template("partials/interim_result.html", view=view)


@results_bp.route("/result")
@login_required
@require_process
def result_page():
    """Display the result: the group-card view-model plus the three analysis tables.

    Loads the analysis tables from ``result_tables.json`` and the structured group cards +
    klassenoverzicht from ``groepsindeling_view.json`` (when present); the template renders the
    cards from the view-model and the three tables as tabs.
    """
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
