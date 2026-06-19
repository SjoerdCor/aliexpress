"""Processes blueprint: list, create, delete, and select distribution processes."""

import functools
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

from aliexpress.extensions import db
from aliexpress.models import Process
from aliexpress.routes.auth import effective_school_id
from aliexpress.storage import get_process_path

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
    proc = Process(school_id=school_id, name=process_name)
    db.session.add(proc)
    db.session.commit()
    try:
        os.makedirs(get_process_path(school_id, process_name))
    except PermissionError:
        flash("Ongeldige procesinformatie.", "error")
        return redirect(url_for("processes.index"))
    session["process_id"] = process_name
    return redirect(url_for("upload_edexml"))


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
    preferences_path = os.path.join(path, "preferences.xlsx")
    if os.path.exists(preferences_path):
        return redirect(url_for("not_together_page"))
    groups_path = os.path.join(path, "groups.xlsx")
    if os.path.exists(groups_path):
        return redirect(url_for("student_preferences"))
    candidates_path = os.path.join(path, "relevant_students_and_groups.json")
    if os.path.exists(candidates_path):
        return redirect(url_for("groups_to_page"))
    return redirect(url_for("upload_edexml"))
