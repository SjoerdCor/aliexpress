"""Roster blueprint: the shared "Wie gaat mee" step (ADR 0005, reordered by ADR 0006).

Determines which leerlingen take part in this verdeling — confirming who goes (unticking
Verlengers) and, rarely, adding an incoming student. It is the first step after the EDEXML
upload and continues to "Groepen naartoe"; the choice of how to enter preferences (web form
or Excel) now lives on that next page, its immediate predecessor (ADR 0006). The resolved
population is persisted as ``roster.json`` and consumed by both preference routes.

The pure form/data helpers below are free of Flask so they can be unit-tested and reused
by the wizard blueprint without an import cycle.
"""

import json
import logging
import os
from itertools import zip_longest

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import login_required

from ...data import datareader
from ...errors import ValidationError
from ..flashing import warn_and_flash
from ..storage import get_file_path, get_process_path
from ..validation_messages import to_validation_message
from .processes import get_process_mode, require_process, require_school

logger = logging.getLogger(__name__)

roster_bp = Blueprint("roster", __name__)


# ── Pure helpers ──────────────────────────────────────────────────────────────


def load_roster(roster_path):
    """Load the saved roster dict, or None when the step was not used yet."""
    if not os.path.exists(roster_path):
        return None
    with open(roster_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_candidates(candidates_path):
    """Load (candidate dicts, groups_from, jaargroepen) from relevant_students_and_groups.json.

    ``jaargroepen`` (herindelen only) is the set of jaargroepen settled by the group
    selection itself, not re-derived from the candidates — see ``_select_groups_post``.
    """
    with open(candidates_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return (
        raw.get("candidates", []),
        raw.get("groups_from", []),
        raw.get("jaargroepen", []),
    )


def sorted_for_display(candidates: list[dict]) -> list[dict]:
    """Order candidates per origin group, alphabetically by roepnaam within each group.

    The "Anders" group (students without a real origin group, e.g. new arrivals) sorts
    last regardless of its name, so it forms the final block on the page.
    """

    def key(candidate: dict):
        group = candidate.get("groepsnaam", "")
        anders_last = group.strip().lower() == "anders"
        return (anders_last, group, candidate.get("roepnaam", ""))

    return sorted(candidates, key=key)


def build_new_candidates(
    form, groups_from: list, default_jaargroep: int | None = None
) -> list[dict]:
    """Build candidate dicts for incoming students added via the form.

    Expects parallel lists ``new_key[]``, ``new_voornaam[]``, ``new_achternaam[]``,
    ``new_geslacht[]`` and optionally ``new_groep[]``, ``new_jaargroep[]``. Incomplete rows
    are skipped. A row's own ``new_jaargroep[]`` wins; otherwise ``default_jaargroep`` is
    used (the process's single shared jaargroep in doorzetten mode). No ``"jaargroep"`` key
    is added when neither is available, matching the Excel input path's None-cohort.
    """
    fallback = groups_from[0] if groups_from else ""
    candidates = []
    for key, vn, an, geslacht, groep, jaargroep in zip_longest(
        form.getlist("new_key[]"),
        form.getlist("new_voornaam[]"),
        form.getlist("new_achternaam[]"),
        form.getlist("new_geslacht[]"),
        form.getlist("new_groep[]"),
        form.getlist("new_jaargroep[]"),
        fillvalue="",
    ):
        vn, an = vn.strip(), an.strip()
        if vn and an and geslacht and key:
            candidate = {
                "key": key,
                "roepnaam": vn,
                "achternaam": an,
                "geslacht": geslacht,
                "groepsnaam": groep or fallback,
            }
            resolved_jaargroep = (
                int(jaargroep) if jaargroep.strip() else default_jaargroep
            )
            if resolved_jaargroep is not None:
                candidate["jaargroep"] = resolved_jaargroep
            candidates.append(candidate)
    return candidates


def validate_new_students(form, orig_candidates, mode: str) -> None:
    """Validate hand-added new students; raise ValidationError on the first problem.

    The form is best-effort client-side, so the server is the safety net: a row that was
    started but left incomplete, or whose name clashes (compared on matching keys, so
    spelling/case differences still collide) with an existing leerling or another new
    student, is rejected. Entirely empty rows are ignored. In herindelen mode (``mode ==
    "redistribute"``), candidates span several jaargroepen, so a new student must say which
    one explicitly; in doorzetten mode they all share one, so it can be assumed instead
    (see ``build_new_candidates``'s ``default_jaargroep``).
    """
    existing = {
        datareader.matching_key(f"{c['roepnaam']} {c['achternaam']}")
        for c in orig_candidates
    }
    seen = set()
    for vn, an, geslacht, jaargroep in zip_longest(
        form.getlist("new_voornaam[]"),
        form.getlist("new_achternaam[]"),
        form.getlist("new_geslacht[]"),
        form.getlist("new_jaargroep[]"),
        fillvalue="",
    ):
        vn, an = vn.strip(), an.strip()
        if not (vn or an or geslacht):
            continue  # untouched row
        if not (vn and an and geslacht):
            raise ValidationError(code="incomplete_new_student")
        if mode == "redistribute" and not jaargroep.strip():
            raise ValidationError(code="missing_jaargroep_new_student")
        key = datareader.matching_key(f"{vn} {an}")
        if key in existing or key in seen:
            raise ValidationError(
                code="duplicate_new_student", context={"naam": f"{vn} {an}"}
            )
        seen.add(key)


def build_participants(form, orig_candidates, groups_from, mode: str) -> list[dict]:
    """Resolve the population: ticked existing candidates plus hand-added new students."""
    checked_keys = set(form.getlist("gaat_over"))
    participants = [c for c in orig_candidates if c["key"] in checked_keys]
    default_jaargroep = (
        orig_candidates[0].get("jaargroep")
        if orig_candidates and mode != "redistribute"
        else None
    )
    participants.extend(build_new_candidates(form, groups_from, default_jaargroep))
    return participants


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
            get_file_path(school_id, process_id, "relevant_students_and_groups.json")
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

    saved = load_roster(get_file_path(school_id, process_id, "roster.json"))
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
    with open(
        get_file_path(school_id, process_id, "roster.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump({"participants": participants}, fh, ensure_ascii=False)
    logger.info("Roster accepted: %d participants", len(participants))
    return redirect(url_for("wizard.groups_to_page"))
