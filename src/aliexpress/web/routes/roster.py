"""Roster blueprint: the shared "Leerlingen controleren" step (ADR 0005, reordered by ADR 0006).

Determines which leerlingen take part in this verdeling — confirming who goes (unticking
Verlengers) and, rarely, adding an incoming student. It is the first step after the EDEXML
upload and continues to "Groepen naartoe"; the choice of how to enter preferences (web form
or Excel) now lives on that next page, its immediate predecessor (ADR 0006). The resolved
population is persisted as ``roster.json`` and consumed by both preference routes.
"""

import logging

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import login_required

from ...data.form_parsers import build_participants, validate_new_students
from ...errors import ValidationError
from ..display import sorted_for_display
from ..flashing import warn_and_flash
from ..process_files import load_candidates, load_roster, save_roster
from ..storage import get_process_path
from ..validation_messages import to_validation_message
from .processes import get_process_mode, require_process, require_school

logger = logging.getLogger(__name__)

roster_bp = Blueprint("roster", __name__)


# ── Route ─────────────────────────────────────────────────────────────────────


@roster_bp.route("/roster", methods=["GET", "POST"])
@login_required
@require_process
@require_school
def roster_page(school_id):
    """Confirm the population, then continue to the next wizard step."""
    process_id = session["process_id"]

    try:
        orig_candidates, groups_from, jaargroep_options = load_candidates(
            school_id, process_id
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Could not read candidates for roster")
        flash(to_validation_message(exc), "error")
        return redirect(url_for("wizard.upload_edexml"))

    mode = get_process_mode(get_process_path(school_id, process_id))
    roster_context = {
        "groups_from": groups_from,
        "jaargroep_options": jaargroep_options,
        "mode": mode,
    }

    if request.method == "POST":
        return _handle_roster_post(
            school_id, process_id, orig_candidates, roster_context
        )

    saved = load_roster(school_id, process_id)
    checked_keys, new_students = _saved_roster_values(saved, orig_candidates)
    return _render_roster_page(
        orig_candidates,
        checked_keys,
        new_students,
        roster_context,
    )


def _saved_roster_values(saved, orig_candidates):
    """Return saved selections in the shape expected by the roster template."""
    orig_keys = {c["key"] for c in orig_candidates}
    if saved is None:
        return orig_keys, []  # first visit: everyone goes by default
    participants = saved["participants"]
    checked_keys = {p["key"] for p in participants if p["key"] in orig_keys}
    new_students = [p for p in participants if p["key"] not in orig_keys]
    return checked_keys, new_students


def _navigation(mode):
    """Return the route and labels for both visible navigation actions."""

    if mode == "redistribute":
        prev_url = url_for("wizard.select_groups")
        prev_label = "← Terug naar groepen kiezen"
        next_label = "Verder naar voorkeuren →"
    elif mode == "redistribute_and_forward":
        prev_url = url_for("wizard.upload_edexml")
        prev_label = "← Terug naar leerlinggegevens"
        next_label = "Verder naar nieuwe groepen →"
    else:
        prev_url = url_for("wizard.upload_edexml")
        prev_label = "← Terug naar leerlinggegevens"
        next_label = "Verder naar groepen controleren →"
    return prev_url, prev_label, next_label


def _render_roster_page(
    orig_candidates,
    checked_keys,
    new_students,
    roster_context,
):
    """Render saved values or the values from a rejected POST."""
    groups_from = roster_context["groups_from"]
    jaargroep_options = roster_context["jaargroep_options"]
    mode = roster_context["mode"]
    prev_url, prev_label, next_label = _navigation(mode)

    return render_template(
        "roster.html",
        candidates=sorted_for_display(orig_candidates),
        checked_keys=checked_keys,
        new_students=new_students,
        groups_from=groups_from,
        prev_url=prev_url,
        prev_label=prev_label,
        next_label=next_label,
        mode=mode,
        jaargroep_options=jaargroep_options,
    )


def _form_value(values, name, index):
    """Return one indexed form value, or an empty string for an omitted field."""
    items = values[name]
    return items[index] if index < len(items) else ""


def _submitted_new_students(form):
    """Copy new-student form values for a rejected POST without persisting them."""
    field_names = (
        "new_key[]",
        "new_voornaam[]",
        "new_achternaam[]",
        "new_geslacht[]",
        "new_groep[]",
        "new_jaargroep[]",
    )
    values = {name: form.getlist(name) for name in field_names}
    row_count = max((len(items) for items in values.values()), default=0)
    students = []
    for index in range(row_count):
        students.append(
            {
                "key": _form_value(values, "new_key[]", index) or f"new_{index}",
                "roepnaam": _form_value(values, "new_voornaam[]", index),
                "achternaam": _form_value(values, "new_achternaam[]", index),
                "geslacht": _form_value(values, "new_geslacht[]", index),
                "groepsnaam": _form_value(values, "new_groep[]", index),
                "jaargroep": _form_value(values, "new_jaargroep[]", index),
                "confirmed": False,
            }
        )
    return students


def _handle_roster_post(
    school_id,
    process_id,
    orig_candidates,
    roster_context,
):
    """Validate + persist the roster, then continue to the next wizard step."""
    groups_from = roster_context["groups_from"]
    mode = roster_context["mode"]
    try:
        validate_new_students(request.form, orig_candidates, mode, groups_from)
    except ValidationError as exc:
        warn_and_flash(to_validation_message(exc), log_detail=exc.code)
        orig_keys = {candidate["key"] for candidate in orig_candidates}
        checked_keys = {
            key for key in request.form.getlist("gaat_over") if key in orig_keys
        }
        return _render_roster_page(
            orig_candidates,
            checked_keys,
            _submitted_new_students(request.form),
            roster_context,
        )

    participants = build_participants(request.form, orig_candidates, groups_from, mode)
    if not participants:
        exc = ValidationError("no_students_selected")
        warn_and_flash(to_validation_message(exc), log_detail=exc.code)
        orig_keys = {candidate["key"] for candidate in orig_candidates}
        checked_keys = {
            key for key in request.form.getlist("gaat_over") if key in orig_keys
        }
        return _render_roster_page(
            orig_candidates,
            checked_keys,
            _submitted_new_students(request.form),
            roster_context,
        )
    save_roster(school_id, process_id, {"participants": participants})
    logger.info("Roster accepted: %d participants", len(participants))
    if mode == "redistribute_and_forward":
        return redirect(url_for("wizard.select_groups"))
    return redirect(url_for("wizard.groups_to_page"))
