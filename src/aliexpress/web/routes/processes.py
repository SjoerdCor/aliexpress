"""Processes blueprint: list, create, delete, and select distribution processes."""

import functools
import json
import logging
import os
import re
import shutil

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import login_required

from ..extensions import db
from ..models import Process
from ..storage import get_process_path
from .auth import effective_school_id

logger = logging.getLogger(__name__)

processes_bp = Blueprint("processes", __name__, url_prefix="/processes")


def require_process(f):
    """Route decorator: redirect to /processes when no active process is in session."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "process_id" not in session:
            flash("Geen actief proces geselecteerd.", "error")
            return redirect(url_for("processes.index"))
        return f(*args, **kwargs)

    return wrapper


def require_school(f):
    """Route decorator: resolve the effective school and pass it as ``school_id``.

    A logged-in beheerder without an impersonated school has no own school, so the route
    is redirected to the admin dashboard instead.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        school_id = effective_school_id()
        if school_id is None:
            return redirect(url_for("admin.dashboard"))
        return f(*args, school_id=school_id, **kwargs)

    return wrapper


def get_process_mode(path: str) -> str:
    """Return the distribution mode for a process: 'forward', 'redistribute' or
    'redistribute_and_forward'.

    Defaults to 'forward' when mode.json is absent (processes created before the mode
    field was introduced).
    """
    mode_path = os.path.join(path, "mode.json")
    if not os.path.exists(mode_path):
        return "forward"
    with open(mode_path, encoding="utf-8") as fh:
        return json.load(fh).get("mode", "forward")


def is_redistribute_mode(mode: str) -> bool:
    """True for both herindelen modes (in-place and redistribute-and-forward).

    Doorzetten ("forward") is the only non-redistribute mode; both herindelen
    variants share the same wizard branch (auto groups_to, roster back-links).
    """
    return mode in ("redistribute", "redistribute_and_forward")


def year_offset_for_mode(mode: str) -> int:
    """Return the year-layer offset (0 or 1) that results for this mode should be shown at.

    This is the "shifts-by-a-year" axis: forward modes move students to the Overgang
    (they land in the Nieuwe jaarlaag, one year up), so results should be displayed
    one year-layer higher. It is orthogonal to `is_redistribute_mode`, which groups
    modes by wizard branch instead: "forward" IS a forward mode (offset 1) but is NOT
    a redistribute mode, while "redistribute_and_forward" is both.
    """
    return 1 if mode in ("forward", "redistribute_and_forward") else 0


def _is_valid_process_name(name):
    """True when the name is a safe single path segment (no separators, no traversal)."""
    return bool(re.match(r"^[\w\- ]+$", name))


def _validate_process_name(school_id, process_name, must_exist=True):
    """Return an error message, or None when the name is valid."""
    if not process_name:
        return "Naam is verplicht"
    if not _is_valid_process_name(process_name):
        return "Alleen letters, cijfers, spaties, - en _ toegestaan"
    proc = Process.by_name(school_id, process_name)
    if must_exist and proc is None:
        return "Proces bestaat niet"
    if not must_exist and proc is not None:
        return "Proces bestaat al"
    return None


@processes_bp.route("", strict_slashes=False)
@login_required
def index():
    """Display page to create or choose process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    os.makedirs(
        os.path.join(current_app.config["STORAGE_DIR"], school_id), exist_ok=True
    )
    procs = (
        Process.query.filter_by(school_id=school_id).order_by(Process.created_at).all()
    )
    return render_template("processes.html", processes=[p.name for p in procs])


@processes_bp.route("/create", methods=["POST"])
@login_required
def create():
    """Create a new process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_name = request.form.get("process_name", "").strip()
    if error := _validate_process_name(school_id, process_name, must_exist=False):
        flash(error, "error")
        return redirect(url_for("processes.index"))
    mode = request.form.get("mode", "forward")
    if mode not in ("forward", "redistribute", "redistribute_and_forward"):
        mode = "forward"
    proc = Process(school_id=school_id, name=process_name)
    db.session.add(proc)
    db.session.commit()
    try:
        proc_path = get_process_path(school_id, process_name)
        os.makedirs(proc_path)
    except PermissionError:
        flash("Ongeldige procesinformatie.", "error")
        return redirect(url_for("processes.index"))
    with open(os.path.join(proc_path, "mode.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": mode}, fh)
    session["process_id"] = process_name
    return redirect(url_for("wizard.upload_edexml"))


@processes_bp.route("/delete/<process_name>", methods=["POST"])
@login_required
def delete(process_name):
    """Delete a process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    if error := _validate_process_name(school_id, process_name, must_exist=True):
        flash(error, "error")
        return redirect(url_for("processes.index"))
    proc = Process.by_name(school_id, process_name)
    db.session.delete(proc)
    db.session.commit()
    try:
        shutil.rmtree(get_process_path(school_id, process_name))
    except PermissionError:
        flash("Ongeldige procesinformatie.", "error")
        return redirect(url_for("processes.index"))
    return redirect(url_for("processes.index"))


def _preferences_url(path):
    """Return the preferences URL based on the saved input method, defaulting to the form."""
    method_path = os.path.join(path, "input_method.json")
    if os.path.exists(method_path):
        with open(method_path, encoding="utf-8") as fh:
            method = json.load(fh).get("method", "form")
        if method == "excel":
            return url_for("wizard.preferences_excel")
    return url_for("wizard.preferences_form")


def _resume_url(proc, path):
    """Return the URL where an existing process should resume.

    Checks run status first (done → result, running/error → processing), then
    falls back to the wizard step that matches which files are already present.
    """
    if proc.run is not None and proc.run.status == "done":
        return url_for("results.result_page")
    if proc.run is not None and proc.run.status in ("running", "error"):
        return url_for("results.processing")

    def has(*names):
        return any(os.path.exists(os.path.join(path, n)) for n in names)

    # redistribute_and_forward picks destination groups after the roster step (its
    # select_groups comes after roster, unlike the other two modes); every other mode
    # continues straight to groups_to.
    after_roster = (
        url_for("wizard.select_groups")
        if get_process_mode(path) == "redistribute_and_forward"
        else url_for("wizard.groups_to_page")
    )
    # Latest wizard step whose artifact is present wins; checked newest-first.
    # Step order (ADR 0006): EDEXML → roster → groups_to → preferences → not_together.
    steps = [
        (("voorkeuren.json", "preferences.xlsx"), url_for("wizard.not_together_page")),
        (("groups.xlsx",), _preferences_url(path)),
        (("roster.json",), after_roster),
        (("relevant_students_and_groups.json",), url_for("roster.roster_page")),
    ]
    for names, target in steps:
        if has(*names):
            return target
    return url_for("wizard.upload_edexml")


@processes_bp.route("/select/<process_id>")
@login_required
def select(process_id):
    """Select process"""
    if not _is_valid_process_name(process_id):
        abort(404)
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    proc = Process.by_name(school_id, process_id)
    if proc is None:
        abort(404)

    session["process_id"] = process_id
    try:
        path = get_process_path(school_id, process_id)
    except PermissionError:
        flash("Ongeldige procesinformatie.", "error")
        return redirect(url_for("processes.index"))
    return redirect(_resume_url(proc, path))
