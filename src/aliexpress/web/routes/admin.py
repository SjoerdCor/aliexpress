"""Admin Blueprint: login, dashboard, and school impersonation routes.

Registered in ``app.py`` under the ``/admin`` prefix. All routes require the current
user to be an Admin (enforced by ``_admin_required``); impersonation stores the
selected schoolcode in the session so normal school routes can serve the admin as if
they were that school.
"""

import functools
import logging

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_user
from werkzeug.security import check_password_hash

from ..extensions import db, limiter
from ..models import Admin, Process, School

logger = logging.getLogger("aliexpress.admin")

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _admin_required(f):
    """Decorator: redirect to admin login when the current user is not an admin."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return redirect(url_for("admin.login"))
        return f(*args, **kwargs)

    return wrapper


@admin_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per minute", methods=["POST"])
def login():
    """Show and handle the admin login form."""
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for("admin.dashboard"))
    if request.method == "POST":
        password = request.form.get("wachtwoord", "")
        admin = Admin.query.first()
        if admin is None or not check_password_hash(admin.password_hash, password):
            flash("Ongeldig wachtwoord.", "error")
            return render_template("admin_login.html")
        session.clear()
        login_user(admin, remember=False)
        return redirect(url_for("admin.dashboard"))
    return render_template("admin_login.html")


@admin_bp.route("/")
@_admin_required
def dashboard():
    """Admin overview: all schools and their processes."""
    schools = School.query.order_by(School.naam).all()
    school_data = []
    for school in schools:
        procs = (
            Process.query.filter_by(school_id=school.schoolcode)
            .order_by(Process.created_at)
            .all()
        )
        school_data.append({"school": school, "processes": procs})
    impersonating = session.get("impersonating_school")
    return render_template(
        "admin_dashboard.html", school_data=school_data, impersonating=impersonating
    )


@admin_bp.route("/impersonate/<schoolcode>")
@_admin_required
def impersonate(schoolcode):
    """Start impersonating a school: all subsequent school routes act as that school."""
    school = db.session.get(School, schoolcode)
    if school is None:
        abort(404)
    session["impersonating_school"] = schoolcode
    logger.info("Admin started impersonating school '%s'", schoolcode)
    return redirect(url_for("processes.index"))


@admin_bp.route("/stop-impersonating")
@_admin_required
def stop_impersonating():
    """Stop impersonating and return to the admin dashboard."""
    session.pop("impersonating_school", None)
    return redirect(url_for("admin.dashboard"))
