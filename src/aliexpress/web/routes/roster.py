"""Roster blueprint: the shared "Wie gaat mee" step (ADR 0005, reordered by ADR 0006).

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
    """Shared "Wie gaat mee" step: confirm the population, then pick the preference route."""
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

    if request.method == "POST":
        return _handle_roster_post(
            school_id, process_id, orig_candidates, groups_from, mode
        )

    saved = load_roster(school_id, process_id)
    orig_keys = {c["key"] for c in orig_candidates}
    if saved is None:
        checked_keys = orig_keys  # first visit: everyone goes by default
        new_students = []
    else:
        participants = saved["participants"]
        checked_keys = {p["key"] for p in participants if p["key"] in orig_keys}
        new_students = [p for p in participants if p["key"] not in orig_keys]

    if mode == "redistribute":
        prev_url = url_for("wizard.select_groups")
        prev_label = "← Naar Groepskeuze"
        next_label = "Naar Voorkeuren →"
    elif mode == "redistribute_and_forward":
        prev_url = url_for("wizard.upload_edexml")
        prev_label = "← Naar Schoolinformatie uploaden"
        next_label = "Naar Groepskeuze →"
    else:
        prev_url = url_for("wizard.upload_edexml")
        prev_label = "← Naar Schoolinformatie uploaden"
        next_label = "Naar Groepen naartoe →"

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


def _handle_roster_post(school_id, process_id, orig_candidates, groups_from, mode):
    """Validate + persist the roster, then continue to "Groepen naartoe" (ADR 0006)."""
    try:
        validate_new_students(request.form, orig_candidates, mode)
    except ValidationError as exc:
        warn_and_flash(to_validation_message(exc), log_detail=exc.code)
        return redirect(url_for("roster.roster_page"))

    participants = build_participants(request.form, orig_candidates, groups_from, mode)
    save_roster(school_id, process_id, {"participants": participants})
    logger.info("Roster accepted: %d participants", len(participants))
    if mode == "redistribute_and_forward":
        return redirect(url_for("wizard.select_groups"))
    return redirect(url_for("wizard.groups_to_page"))
