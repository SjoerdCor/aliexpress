"""Results blueprint: processing, status, sociogram, result, download, and done routes."""

# The route modules intentionally repeat the small active-process guard for HTTP clarity.
# pylint: disable=duplicate-code

import json
import logging
import os
from dataclasses import asdict, replace

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
from ...sociogram import build_sociogram_view
from ...solver._balance import default_balance_maxima
from ..models import Process
from ..process_files import (
    load_balance_maxima,
    load_groups,
    load_not_together,
    load_voorkeuren,
)
from ..storage import get_file_path, get_process_path
from ..wizard_steps import wizard_context
from .auth import effective_school_id
from .processes import get_process_mode, require_process

logger = logging.getLogger(__name__)

results_bp = Blueprint("results", __name__)

_PROCESSING_BALANCE_LABELS = {
    "Groepsgrootte per jaarlaag": "Maximaal verschil in groepsgrootte per jaarlaag",
    "Groepsgrootte totaal": "Maximaal verschil in groepsgrootte over de hele groep",
    "Jongens/meisjes per jaarlaag": (
        "Maximaal verschil tussen jongens en meisjes per jaarlaag"
    ),
    "Jongens/meisjes totaal": (
        "Maximaal verschil tussen jongens en meisjes over de hele groep"
    ),
    "Zelfde stamgroep totaal": (
        "Maximaal aantal leerlingen uit dezelfde huidige groep in één groep in de "
        "nieuwe indeling"
    ),
    "Zelfde stamgroep per sekse": (
        "Maximaal aantal jongens of meisjes uit dezelfde huidige groep in één groep in "
        "de nieuwe indeling"
    ),
}


def _n_students_with_preferences(preference_data) -> int:
    """Count students with a positive or negative preference for the summary."""
    preferences = preference_data.preferences
    if preferences.empty:
        return 0
    kinds = preferences.index.get_level_values("TypeWens")
    students = preferences.index.get_level_values("Leerling")
    return len(
        {
            student
            for student, kind in zip(students, kinds)
            if kind in {"Graag met", "Liever niet met"}
        }
    )


def _processing_summary(groups_to, preference_data, group_display):
    """Build the page summary while keeping source groups in input order."""
    summary = build_input_summary(
        groups_to,
        preference_data.students_info,
        preference_data.stamgroep_display,
    )
    source_groups = {}
    for student_info in preference_data.students_info.values():
        group_key = student_info["Stamgroep"]
        group_name = preference_data.stamgroep_display.get(group_key, group_key)
        source_groups[group_name] = source_groups.get(group_name, 0) + 1
    return replace(summary, source_groups=source_groups), list(group_display.values())


def _processing_error_message(message: str) -> str:
    """Expand abbreviated balance labels in the processing-page flash."""
    for short_label, full_label in _PROCESSING_BALANCE_LABELS.items():
        message = message.replace(f"‘{short_label}’", f"‘{full_label}’")
    return message


def _load_json_snapshot(path):
    """Read a complete snapshot while keeping its Windows file handle short-lived."""
    with open(path, "rb") as fh:
        snapshot = fh.read()
    return json.loads(snapshot)


@results_bp.route("/processing")
@login_required
@require_process
def processing():
    """Display the processing page: an idle panel to start the solve, or its live progress.

    Branches on the process's Run status: "pending" or "running" shows the live progress
    view (the poll-driven stepper etc., unchanged) — "pending" is the brief window right
    after Start verdeling, before the background thread's first status write lands, and a
    fast solve can finish within it, so it must not fall back to the idle panel; "done"
    redirects to the result only in explicit watch mode and otherwise shows the idle
    panel for a new indeling; "error" shows the idle panel with its saved limits open;
    and no run yet shows the idle panel with data-driven defaults and closed limits.
    Ordinary GETs are read-only, so revisiting this page never has side effects.
    """
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    process_mode = get_process_mode(get_process_path(school_id, process_id))
    proc = Process.by_name(school_id, process_id)
    run_status = proc.run.status if proc and proc.run else None

    if run_status == "done" and request.args.get("watch") == "1":
        return redirect(url_for("results.result_page"))

    preference_data, _ = load_voorkeuren(school_id, process_id)
    groups_to, group_display = load_groups(school_id, process_id)
    summary, target_group_names = _processing_summary(
        groups_to, preference_data, group_display
    )
    processing_data = {
        "n_preferences": _n_students_with_preferences(preference_data),
        "n_spread_rules": len(load_not_together(school_id, process_id)),
    }

    if run_status in ("pending", "running"):
        return render_template(
            "processing.html",
            mode="running",
            summary=summary,
            target_group_names=target_group_names,
            processing_data=processing_data,
            recalculation=False,
            balance_limits_open=False,
            **wizard_context(process_mode, "processing"),
        )

    maxima_path = get_file_path(school_id, process_id, "balance_limits.json")
    if run_status in ("error", "done") and os.path.exists(maxima_path):
        maxima = load_balance_maxima(school_id, process_id)
    else:
        maxima = default_balance_maxima(preference_data.students_info, groups_to)
    if run_status == "error" and proc.run.message:
        flash(_processing_error_message(proc.run.message), "error")
    return render_template(
        "processing.html",
        mode="idle",
        summary=summary,
        target_group_names=target_group_names,
        processing_data=processing_data,
        maxima=maxima,
        recalculation=run_status == "done",
        balance_limits_open=run_status == "error",
        **wizard_context(process_mode, "processing"),
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
    payload = {"status_studentdistribution": run.status}
    progress_path = get_file_path(school_id, process_name, "progress.json")
    if os.path.exists(progress_path):
        payload.update(_load_json_snapshot(progress_path))
    if run.status == "error" and run.message:
        payload["message"] = run.message
    return jsonify(payload)


@results_bp.route("/sociogram")
@login_required
@require_process
def show_sociogram():
    """Display the sociogram built from the current process's canonical preferences."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    try:
        preference_data, _ = load_voorkeuren(school_id, process_id)
        sociogram_view = asdict(build_sociogram_view(preference_data))
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        logger.exception(
            "Could not load sociogram preferences for process %s", process_id
        )
        flash(
            "Sociogram niet beschikbaar: geldige voorkeuren ontbreken of kunnen niet "
            "worden gelezen. Ga terug naar de voorkeuren en sla ze opnieuw op.",
            "error",
        )
        return render_template(
            "sociogram.html",
            sociogram_view=None,
        )
    return render_template("sociogram.html", sociogram_view=sociogram_view)


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
    view = _load_json_snapshot(path)
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
    process_mode = get_process_mode(get_process_path(school_id, process_id))
    return render_template(
        "result.html",
        dataframes=dataframes,
        groepsindeling_view=groepsindeling_view,
        **wizard_context(process_mode, "result"),
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
        process_mode = get_process_mode(get_process_path(school_id, process_id))
        return render_template(
            "result.html",
            dataframes={},
            groepsindeling_view=None,
            **wizard_context(process_mode, "result"),
        )

    return send_file(
        path,
        as_attachment=True,
        download_name="results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@results_bp.route("/done")
@login_required
@require_process
def done():
    """Show done page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    process_mode = get_process_mode(get_process_path(school_id, process_id))
    return render_template("done.html", **wizard_context(process_mode, "done"))


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
