"""Wizard blueprint: step-by-step distribution setup routes and helpers."""

import json
import logging
from dataclasses import dataclass
from io import BytesIO
from threading import Thread
from typing import Any

import pandera as pa
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from flask_login import login_required

from ... import sociogram
from ...data import candidatedetermination, datareader, input_writer
from ...data.form_parsers import parse_groups_to_form
from ...data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from ...errors import (
    CouldNotReadFileError,
    DuplicateNameError,
    FeasibilityError,
    SolverError,
    ValidationError,
)
from ...logging_config import bind_log_context
from ...main import distribute_students_from_data
from ..extensions import db
from ..flashing import warn_and_flash
from ..models import LogLine, Process, Run
from ..process_files import (
    has_edexml,
    has_preferences_excel,
    load_candidates,
    load_edexml,
    load_groups,
    load_groups_to,
    load_groups_to_state,
    load_input_method,
    load_not_together,
    load_pref_form_state,
    load_roster,
    load_student_names,
    load_voorkeuren,
    reset_downstream_wizard_files,
    reset_result_files,
    save_candidates,
    save_edexml,
    save_groups_excel,
    save_groups_to_state,
    save_input_method,
    save_not_together,
    save_pref_form_state,
    save_preferences_excel,
    save_voorkeuren,
)
from ..storage import get_file_path, get_process_path
from ..validation_messages import to_validation_message
from .auth import effective_school_id
from .processes import get_process_mode, is_redistribute_mode, require_process
from .roster import sorted_for_display

logger = logging.getLogger(__name__)

wizard_bp = Blueprint("wizard", __name__)


@dataclass
class _ThreadContext:
    """Shared context passed to background solver/sociogram threads.

    Bundles the Flask app object (needed to open a thread-local app context) with the
    process identifiers required to locate files and append log lines.
    """

    app_obj: Any
    school_id: str
    process_name: str
    run_id: int


def file_to_io(uploaded_file) -> BytesIO:
    """Get file as BytesIO"""
    return BytesIO(uploaded_file.read())


def _write_result_files(school_id, process_name, result):
    """Persist the solver output as files in the process dir (download + rendered tables).

    Written before the status flips to "done" so the result page never polls ahead of the
    files it needs.
    """
    with open(get_file_path(school_id, process_name, "results.xlsx"), "wb") as fh:
        fh.write(result["download"].getbuffer())
    tables = {name: df.to_html(na_rep="") for name, df in result["dataframes"].items()}
    with open(
        get_file_path(school_id, process_name, "result_tables.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(tables, fh, ensure_ascii=False)


def _flash_upload_error(exc: Exception) -> None:
    """Log a rejected upload and flash a friendly Dutch message to the user."""
    logger.exception("Upload rejected")
    flash(to_validation_message(exc), "error")


def _handle_failure(exc, school_id, process_name):
    file_reading_errs = (
        pa.errors.SchemaError,
        ValidationError,
        CouldNotReadFileError,
    )
    if isinstance(exc, file_reading_errs):
        log_msg = "Files are incorrect"
    elif isinstance(exc, FeasibilityError):
        log_msg = "Problem is infeasible"
    elif isinstance(exc, SolverError):
        log_msg = "Solver could not solve the problem"
    else:
        log_msg = "Uncaught exception"
    logger.exception(log_msg)
    Process.by_name(school_id, process_name).run.set_status(
        "error", to_validation_message(exc)
    )


def _run_solve_thread(ctx: _ThreadContext, groups_to_path, not_together):
    """Background thread: run the solver and write result artifacts.

    Each call creates its own app context and DB session. ``ctx.run_id`` is the integer
    PK of the Run row so log lines can be appended without a school+name query per line.
    Reads preferences from ``voorkeuren.json`` (written by both input paths) so that the
    solver is independent of the original file format.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=ctx.run_id, text=message))
        db.session.commit()

    with ctx.app_obj.app_context():
        with bind_log_context(
            school=ctx.school_id,
            process=ctx.process_name,
            run=str(ctx.run_id),
            phase="solve",
        ):
            try:  # pylint: disable=broad-exception-caught
                Process.by_name(ctx.school_id, ctx.process_name).run.set_status(
                    "running"
                )
                preference_data, _ = load_voorkeuren(ctx.school_id, ctx.process_name)
                target_groups = datareader.read_groups_excel(groups_to_path)
                result = distribute_students_from_data(
                    preference_data,
                    target_groups,
                    not_together,
                    on_update=on_update,
                )
                logger.info("Distributing students finished successfully")
                # Write artifacts before flipping to "done" so the result page never
                # races ahead of the files it needs.
                _write_result_files(ctx.school_id, ctx.process_name, result)
                Process.by_name(ctx.school_id, ctx.process_name).run.set_status("done")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _handle_failure(exc, ctx.school_id, ctx.process_name)


def _create_sociogram_thread(ctx: _ThreadContext):
    """Background thread: build and write the Plotly sociogram HTML.

    Runs concurrently with the solver; log lines are appended via ``ctx.run_id`` just
    like the solver thread does. Reads preferences from ``voorkeuren.json`` (written by
    both input paths) via ``SociogramMaker.from_preference_data``, so the sociogram is
    available for both the Excel and web-form input paths.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=ctx.run_id, text=message))
        db.session.commit()

    with ctx.app_obj.app_context():
        with bind_log_context(
            school=ctx.school_id,
            process=ctx.process_name,
            run=str(ctx.run_id),
            phase="sociogram",
        ):
            try:  # pylint: disable=broad-exception-caught
                on_update("Sociogram tekenen...")
                preference_data, _ = load_voorkeuren(ctx.school_id, ctx.process_name)
                sg = sociogram.SociogramMaker.from_preference_data(preference_data)
                fig, g, pos = sg.plot_sociogram()
                logger.info("Sociogram created")
                fig = sociogram.networkx_to_plotly(g, pos)
                html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                logger.info("HTML created")
                with open(
                    get_file_path(ctx.school_id, ctx.process_name, "sociogram.html"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    fh.write(html)
                on_update(
                    '<a href=/sociogram target="_blank" class="button">'
                    "Bekijk het sociogram nu!</a>"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Could not create sociogram")


def _parse_preference_list(form, key, soort_field_value) -> list[Preference]:
    """Parse all preferences of one kind for a student from the submitted form."""
    kind = (
        PreferenceKind.APART
        if soort_field_value == "liever_niet_met"
        else PreferenceKind.TOGETHER
    )
    prefix = f"preference_{key}_{soort_field_value}"
    targets = form.getlist(f"{prefix}_target")
    weights = form.getlist(f"{prefix}_gewicht")
    result = []
    for target, gewicht_raw in zip(targets, weights):
        target = target.strip()
        if not target:
            continue
        try:
            gewicht = float(gewicht_raw)
        except ValueError:
            gewicht = 1.0
        if gewicht <= 0:
            gewicht = 1.0
        result.append(Preference(target=target, weight=gewicht, kind=kind))
    return result


def _parse_student_entry(candidate: dict, form) -> StudentEntry:
    """Build a StudentEntry from one candidate dict and the submitted form data.

    Graag-met preferences use ``preference_{key}_graag_met_target[]`` / ``_gewicht[]``.
    Liever-niet-met use ``preference_{key}_liever_niet_met_target[]`` / ``_gewicht[]``.
    Group exclusions use ``nieting_{key}[]``.
    Min. satisfaction uses ``min_sat_{key}``.
    """
    key = candidate["key"]
    name = f"{candidate['roepnaam']} {candidate['achternaam']}"

    preferences = _parse_preference_list(
        form, key, "graag_met"
    ) + _parse_preference_list(form, key, "liever_niet_met")

    excluded = [g.strip() for g in form.getlist(f"nieting_{key}") if g.strip()]

    raw_min_sat = form.get(f"min_sat_{key}", "").strip()
    try:
        min_satisfaction = float(raw_min_sat) / 100.0 if raw_min_sat else None
    except ValueError:
        min_satisfaction = None

    return StudentEntry(
        student=name,
        sex=candidate["geslacht"],
        origin_group=candidate["groepsnaam"],
        min_satisfaction=min_satisfaction,
        year_group=candidate.get("jaargroep"),
        preferences=preferences,
        excluded_groups=excluded,
    )


def _build_form_state(entries: list[StudentEntry], participants: list[dict]) -> dict:
    """Serialize submitted preferences to a dict for prefill on next GET.

    The population is already fixed by the roster step (every participant takes part), so
    this only carries each participant's preferences, keyed so the page can restore them.
    """
    entry_by_name = {e.student: e for e in entries}
    state_students = []
    for c in participants:
        name = f"{c['roepnaam']} {c['achternaam']}"
        entry = entry_by_name.get(name)
        state_students.append(
            {
                "key": c["key"],
                "roepnaam": c["roepnaam"],
                "achternaam": c["achternaam"],
                "groepsnaam": c.get("groepsnaam", ""),
                "geslacht": c.get("geslacht", ""),
                "min_satisfaction": entry.min_satisfaction if entry else None,
                "graag_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in (entry.preferences if entry else [])
                    if p.kind == PreferenceKind.TOGETHER
                ],
                "liever_niet_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in (entry.preferences if entry else [])
                    if p.kind == PreferenceKind.APART
                ],
                "niet_in": entry.excluded_groups if entry else [],
            }
        )
    return {"students": state_students}


def _write_pref_form_state(school_id, process_id, form, participants):
    """Parse the form and persist the intermediate draft (``preferences_form_state.json``).

    Returns the parsed ``StudentEntry`` list. Does not validate — it captures whatever is on
    the page so a reload (or a crash) restores it. The population is the settled roster, so
    every participant is parsed (there are no checkboxes and no new students here anymore).
    """
    entries = [_parse_student_entry(c, form) for c in participants]
    state = _build_form_state(entries, participants)
    save_pref_form_state(school_id, process_id, state)
    return entries


def _pref_form_post_data(school_id, process_id, form, participants, all_groups_to):
    """Parse + save the draft, then validate and return the resulting PreferenceData."""
    entries = _write_pref_form_state(school_id, process_id, form, participants)
    return build_preference_data(entries, all_groups_to)


def _reconcile_dangling(draft_state, participants, group_labels):
    """Drop draft preferences whose target no longer takes part; return friendly notices.

    A classmate target is valid only when that leerling is still a participant (the teacher
    may have removed them on the roster step). Group targets stay valid. Mutates
    ``draft_state`` in place and returns one Dutch message per removed preference.
    """
    valid_keys = {
        datareader.matching_key(f"{p['roepnaam']} {p['achternaam']}")
        for p in participants
    } | {datareader.matching_key(g) for g in group_labels}
    notices = []
    for student in draft_state["students"]:
        owner = f"{student['roepnaam']} {student['achternaam']}".strip()
        for kind in ("graag_met", "liever_niet_met"):
            kept = []
            for wish in student.get(kind, []):
                if datareader.matching_key(wish["target"]) in valid_keys:
                    kept.append(wish)
                else:
                    notices.append(
                        f"{wish['target']} gaat niet meer mee — de voorkeur van {owner} "
                        f"naar {wish['target']} is verwijderd."
                    )
            student[kind] = kept
    return notices


def _not_together_get_context(school_id, process_id):
    """Return (existing_rules, prev_url) for a GET to /not_together."""
    rules = load_not_together(school_id, process_id)
    existing_rules = [
        {"group": list(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in rules
    ]
    input_method = load_input_method(school_id, process_id)
    prev_url = (
        url_for("wizard.preferences_excel")
        if input_method == "excel"
        else url_for("wizard.preferences_form")
    )
    return existing_rules, prev_url


def _parse_not_together_form(form, n_rules):
    """Parse not-together form fields into rule dicts. Returns (rules, error_msg)."""
    rules = []
    for i in range(n_rules):
        names_raw = form.getlist(f"rule_students[{i}]")
        # Keep the names as entered for display; dedupe on the matching key so the same
        # student picked twice (in any spelling) is caught.
        cleaned = [datareader.display_name(n) for n in names_raw if n.strip()]
        if len({datareader.matching_key(n) for n in cleaned}) != len(cleaned):
            return (
                None,
                f"Niet-samen-regel {i + 1} bevat dezelfde leerling meerdere keren.",
            )
        max_samen_raw = form.get(f"rule_max[{i}]", "").strip()
        if not max_samen_raw:
            return None, f"Vul het maximale aantal samen in voor regel {i + 1}."
        try:
            max_samen = int(max_samen_raw)
        except ValueError:
            return (
                None,
                f"Maximale aantal samen moet een heel getal zijn (regel {i + 1}).",
            )
        if cleaned:
            rules.append({"group": set(cleaned), "Max_aantal_samen": max_samen})
    return rules, None


@wizard_bp.route("/input_templates/<path:filename>")
def download_template(filename):
    """Download the template sheets"""
    return send_from_directory("input_templates", filename, as_attachment=True)


def _redistribute_upload(process_id, edexml):
    """mode == 'redistribute' (in_place): validate the EDEXML, then pick the groups."""
    datareader.EdexReader(edexml).get_full_df()
    logger.info(
        "EDEXML accepted for redistribute mode in %s: redirecting to group selection",
        process_id,
    )
    return redirect(url_for("wizard.select_groups"))


def _redistribute_and_forward_upload(school_id, process_id, edexml):
    """mode == 'redistribute_and_forward': pick jaargroepen here, destinations later.

    Unlike plain herindelen, candidates are determined school-wide from the chosen
    jaargroepen (not from a group selection), and ``groups_to`` is left empty — the
    destination groups are only settled on the next step, /select_groups.
    """
    jaargroepen = [int(j) for j in request.form.getlist("jaargroepen")]
    if not jaargroepen:
        warn_and_flash(
            "Selecteer minimaal één jaargroep.",
            log_detail="no_jaargroepen_selected",
        )
        return redirect(url_for("wizard.upload_edexml"))
    df = datareader.EdexReader(edexml).get_full_df()
    candidates, groups_from, groups_to = (
        candidatedetermination.handle_edexml_upload_redistribute_and_forward(
            df, jaargroepen, []
        )
    )
    if not candidates:
        warn_and_flash(
            "Geen leerlingen gevonden in de gekozen jaargroepen.",
            log_detail="no_candidates_in_selected_jaargroepen",
        )
        return redirect(url_for("wizard.upload_edexml"))
    save_candidates(
        school_id,
        process_id,
        {
            "candidates": candidates,
            "groups_from": groups_from,
            "groups_to": groups_to,
            "jaargroepen": jaargroepen,
        },
    )
    missing = int(df["jaargroep"].isna().sum())
    if missing > 0:
        flash(
            f"Let op: {missing} leerling(en) hebben geen jaargroep in het bestand en "
            "doen niet mee.",
            "info",
        )
    logger.info(
        "EDEXML accepted for redistribute_and_forward mode in %s: %d candidates "
        "across jaargroepen %s",
        process_id,
        len(candidates),
        jaargroepen,
    )
    return redirect(url_for("roster.roster_page"))


def _forward_upload(school_id, process_id, edexml):
    """mode == 'forward' (doorzetten): candidates for one jaargroep, groups_to derived
    from the next jaargroep already present in the EDEXML."""
    jaargroep = int(request.form["jaargroep"])
    df = datareader.EdexReader(edexml).get_full_df()
    candidates, groups_from, groups_to = candidatedetermination.handle_edexml_upload(
        df, jaargroep
    )
    save_candidates(
        school_id,
        process_id,
        {"candidates": candidates, "groups_from": groups_from, "groups_to": groups_to},
    )
    logger.info("EDEXML accepted: %d candidates, %d groups", len(candidates), jaargroep)
    return redirect(url_for("roster.roster_page"))


def _dispatch_edexml_upload(mode, school_id, process_id, edexml):
    """Branch the EDEXML upload on mode, returning the resulting redirect response."""
    if mode == "redistribute":
        return _redistribute_upload(process_id, edexml)
    if mode == "redistribute_and_forward":
        return _redistribute_and_forward_upload(school_id, process_id, edexml)
    return _forward_upload(school_id, process_id, edexml)


@wizard_bp.route("/upload_edexml", methods=["GET", "POST"])
@login_required
def upload_edexml():
    """Route to upload edexml"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session.get("process_id")
    if request.method == "GET":
        mode = "forward"
        if process_id:
            try:
                mode = get_process_mode(get_process_path(school_id, process_id))
            except PermissionError:
                pass
        return render_template("upload_edexml.html", mode=mode)
    # POST
    if not process_id:
        flash("Geen actief proces geselecteerd.", "error")
        return redirect(url_for("processes.index"))
    mode = get_process_mode(get_process_path(school_id, process_id))
    try:
        edex_file = request.files["edexml"]
        save_edexml(school_id, process_id, edex_file)
        edex_file.stream.seek(0)
        edexml = file_to_io(edex_file)
        reset_downstream_wizard_files(school_id, process_id)
        return _dispatch_edexml_upload(mode, school_id, process_id, edexml)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.upload_edexml"))


def _select_groups_post_in_place(df, school_id, process_id, selected):
    """redistribute (in_place): candidates and origin groups are the selected groups
    themselves — students are redistributed within them."""
    candidates, groups_from, groups_to = (
        candidatedetermination.handle_edexml_upload_herindelen(df, selected)
    )
    if not candidates:
        warn_and_flash(
            "Geen leerlingen gevonden in de geselecteerde groepen.",
            log_detail="no_candidates_in_selected_groups",
        )
        return redirect(url_for("wizard.select_groups"))
    # Recorded here, from the full EDEXML data at selection time, rather than re-derived
    # from `candidates` later: the set of jaargroepen involved in this herindeling is
    # settled by this selection, independent of who ends up ticked on the roster page.
    jaargroepen = sorted(
        df.loc[df["groepsnaam"].isin(selected), "jaargroep"].dropna().unique().tolist()
    )
    save_candidates(
        school_id,
        process_id,
        {
            "candidates": candidates,
            "groups_from": groups_from,
            "groups_to": groups_to,
            "jaargroepen": jaargroepen,
        },
    )
    logger.info(
        "Groups selected for redistribution in %s: %s (%d candidates)",
        process_id,
        ", ".join(selected),
        len(candidates),
    )
    return redirect(url_for("roster.roster_page"))


def _select_groups_post_redistribute_and_forward(school_id, process_id, selected):
    """redistribute_and_forward: candidates and origin groups were already settled at
    upload time (school-wide, by jaargroep); this step only records the destinations."""
    candidates, groups_from, jaargroepen = load_candidates(school_id, process_id)
    save_candidates(
        school_id,
        process_id,
        {
            "candidates": candidates,
            "groups_from": groups_from,
            "groups_to": {g: [] for g in selected},
            "jaargroepen": jaargroepen,
        },
    )
    logger.info(
        "Destination groups selected for redistribute_and_forward in %s: %s",
        process_id,
        ", ".join(selected),
    )
    return redirect(url_for("wizard.groups_to_page"))


def _select_groups_post(df, school_id, process_id, mode):
    """Process a POST to /select_groups: validate the selection, then branch on mode."""
    selected = request.form.getlist("groups")
    if len(selected) < 2:
        warn_and_flash(
            "Selecteer minimaal twee groepen om te herindelen.",
            log_detail="too_few_groups_redistribute",
        )
        return redirect(url_for("wizard.select_groups"))
    if mode == "redistribute_and_forward":
        return _select_groups_post_redistribute_and_forward(
            school_id, process_id, selected
        )
    return _select_groups_post_in_place(df, school_id, process_id, selected)


@wizard_bp.route("/select_groups", methods=["GET", "POST"])
@login_required
@require_process
def select_groups():
    """Select groups: redistribute (which groups to shuffle) or redistribute_and_forward
    (destination groups for the jaargroepen chosen at upload time)."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    mode = get_process_mode(get_process_path(school_id, process_id))
    if not has_edexml(school_id, process_id):
        warn_and_flash(
            "Upload eerst het EDEXML-bestand.",
            log_detail="missing_edex_for_select_groups",
        )
        return redirect(url_for("wizard.upload_edexml"))
    try:
        edexml = load_edexml(school_id, process_id)
        df = datareader.EdexReader(edexml).get_full_df()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.upload_edexml"))
    if request.method == "GET":
        groups = sorted(df["groepsnaam"].unique().tolist())
        return render_template("select_groups.html", groups=groups, mode=mode)
    return _select_groups_post(df, school_id, process_id, mode)


def _groups_to_auto_redistribute(school_id, process_id, groups_to):
    """Write groups.xlsx (zero occupancy per group) and redirect straight to preferences_form."""
    distribution = {g: {"Jongens": 0, "Meisjes": 0} for g in groups_to}
    save_groups_excel(school_id, process_id, distribution)
    save_input_method(school_id, process_id, "form")
    logger.info(
        "Groups-to auto-written for redistribute process %s: %d groups, zero occupancy",
        process_id,
        len(distribution),
    )
    return redirect(url_for("wizard.preferences_form"))


@wizard_bp.route("/groups_to", methods=["GET", "POST"])
@login_required
@require_process
def groups_to_page():
    """Display and process the groups_to page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    groups_to = load_groups_to(school_id, process_id)
    mode = get_process_mode(get_process_path(school_id, process_id))

    if is_redistribute_mode(mode):
        return _groups_to_auto_redistribute(school_id, process_id, groups_to)

    if request.method == "GET":
        return render_template(
            "groups_to.html",
            groups_to=groups_to,
            state=load_groups_to_state(school_id, process_id),
        )

    submitted_names = request.form.getlist("group")
    seen, duplicates = set(), []
    for name in submitted_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        exc = ValidationError(
            "duplicate_group_names", {"duplicates": ", ".join(duplicates)}
        )
        warn_and_flash(to_validation_message(exc), log_detail=exc.code)
        return redirect(url_for("wizard.groups_to_page"))

    submission = parse_groups_to_form(request.form, groups_to)
    if len(submission.distribution) < 2:
        warn_and_flash(
            "Er moeten minsten twee groepen zijn om de leerlingen over te verdelen",
            log_detail="too_few_groups",
        )
        return redirect(url_for("wizard.groups_to_page"))

    save_groups_excel(school_id, process_id, submission.distribution)
    save_groups_to_state(school_id, process_id, submission.state)
    logger.info(
        "Groups-to saved for process %s: %d active group(s), %d disabled, %d new",
        process_id,
        len(submission.distribution),
        len(submission.state["disabled_groups"]),
        len(submission.state["new_groups"]),
    )
    # This is the last step before entering preferences, so the teacher picks here how to
    # enter them (web form or Excel); the choice is recorded for resume and the back link
    # of the not-together step (ADR 0006, reordering ADR 0005).
    method = "form" if request.form.get("action") == "form" else "excel"
    save_input_method(school_id, process_id, method)
    target = (
        "wizard.preferences_form" if method == "form" else "wizard.preferences_excel"
    )
    return redirect(url_for(target))


@wizard_bp.route("/preferences_excel", methods=["GET", "POST"])
@login_required
@require_process
def preferences_excel():
    """Excel input path: download a template prefilled with the roster, then upload it.

    The population is fixed by the shared roster step (ADR 0005), so this page no longer
    selects students; the download is built straight from ``roster.json``.
    """
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    saved_roster = load_roster(school_id, process_id)
    if saved_roster is None:
        # The population must be settled first; send the teacher to "Wie gaat mee".
        return redirect(url_for("roster.roster_page"))
    participants = saved_roster["participants"]

    if request.method == "GET":
        return render_template(
            "preferences_excel.html",
            preferences_uploaded=has_preferences_excel(school_id, process_id),
        )

    if not participants:
        warn_and_flash(
            "Er moet minsten één leerling aanwezig zijn", log_detail="no_participants"
        )
        return redirect(url_for("roster.roster_page"))
    try:
        df_total = candidatedetermination.students_df_from_records(participants)
    except DuplicateNameError as exc:
        logger.exception(exc)
        flash(f"Vond leerlingen dubbel: {exc.context['duplicate_names']}", "error")
        return redirect(url_for("roster.roster_page"))

    _, group_display = load_groups(school_id, process_id)
    buffer = input_writer.create_prefilled_excel(list(group_display.values()), df_total)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="voorkeuren.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@wizard_bp.route("/upload_preferences", methods=["POST"])
@login_required
@require_process
def upload_preferences():
    """Handle the upload of the preferences file, or continue with an earlier upload.

    Re-uploading is optional when going back and forth: if no new file is chosen but a
    valid preferences file was uploaded earlier, the teacher simply continues with it.
    """
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    upload = request.files.get("preferences")
    if not (upload and upload.filename):
        if has_preferences_excel(school_id, process_id):
            logger.info(
                "No new preferences upload for process %s; continuing with stored file",
                process_id,
            )
            return redirect(url_for("wizard.not_together_page"))
        warn_and_flash(
            "Upload eerst het ingevulde bestand om verder te gaan.",
            log_detail="no_file_uploaded",
        )
        return redirect(url_for("wizard.preferences_excel"))
    try:
        raw = upload.read()
        groups_to_data, _ = load_groups(school_id, process_id)
        groups_to = list(groups_to_data.keys())
        processor = datareader.VoorkeurenProcessor(BytesIO(raw))
        processor.process(all_to_groups=groups_to)  # validates; raises on invalid input
        # Save the raw upload directly so re-reading later preserves names as entered.
        # VoorkeurenProcessor normalises names to matching keys at read time anyway,
        # and storing the original ensures student_display maps correctly to display names.
        save_preferences_excel(school_id, process_id, raw)
        # Persist as voorkeuren.json so the solver and sociogram can load from a single
        # canonical format regardless of input path (Excel or web form).
        save_voorkeuren(
            school_id, process_id, processor.to_preference_data(), source="excel"
        )
        logger.info(
            "Preferences accepted for process %s: %d students",
            process_id,
            len(processor.input.index),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.preferences_excel"))
    return redirect(url_for("wizard.not_together_page"))


def _apply_draft_preferences(
    draft_state, participants, display_candidates, all_groups_to, group_display
):
    """Mutate display_candidates in-place with saved min_satisfaction; return notices to flash."""
    if not draft_state:
        return []
    group_labels = [group_display[g] for g in all_groups_to]
    ms_by_key = {s["key"]: s.get("min_satisfaction") for s in draft_state["students"]}
    for candidate in display_candidates:
        candidate["min_satisfaction"] = ms_by_key.get(candidate["key"])
    return list(_reconcile_dangling(draft_state, participants, group_labels))


def _handle_pref_form_post(school_id, process_id, participants, all_groups_to):
    """Process a POST to /preferences_form and return the response to send.

    Two actions: ``autosave`` saves only the draft (best effort, no validation — used by the
    modal's "Opslaan"); otherwise (``volgende``) build and persist ``voorkeuren.json`` and
    navigate. Validation errors are flashed and the form re-rendered — the draft is already
    saved, so nothing is lost.
    """
    if request.form.get("action") == "autosave":
        # Best-effort background save of the draft only (never voorkeuren.json, never
        # validated): a reload then restores the work via the normal GET prefill.
        _write_pref_form_state(school_id, process_id, request.form, participants)
        return ("", 204)
    try:
        preference_data = _pref_form_post_data(
            school_id, process_id, request.form, participants, all_groups_to
        )
    except (pa.errors.SchemaError, ValidationError, ValueError) as exc:
        # The form is novalidate + JS-best-effort, so the server must catch bad input.
        _flash_upload_error(exc)
        return redirect(url_for("wizard.preferences_form"))
    save_voorkeuren(school_id, process_id, preference_data, source="form")
    logger.info("Preferences form accepted: %d participants", len(participants))
    return redirect(url_for("wizard.not_together_page"))


@wizard_bp.route("/preferences_form", methods=["GET", "POST"])
@login_required
@require_process
def preferences_form():
    """Web-form input path: per-student preferences for the settled roster population."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]

    saved_roster = load_roster(school_id, process_id)
    if saved_roster is None:
        # The population must be settled first; send the teacher to "Wie gaat mee".
        return redirect(url_for("roster.roster_page"))
    try:
        groups_to, group_display = load_groups(school_id, process_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.groups_to_page"))

    participants = saved_roster["participants"]
    all_groups_to = list(groups_to.keys())

    if request.method == "POST":
        return _handle_pref_form_post(
            school_id, process_id, participants, all_groups_to
        )

    # GET — load saved preferences for prefill, dropping any that now dangle because their
    # target was removed from the roster, with a friendly notice about what was removed.
    draft_state = load_pref_form_state(school_id, process_id)
    display_candidates = sorted_for_display(participants)
    for notice in _apply_draft_preferences(
        draft_state, participants, display_candidates, all_groups_to, group_display
    ):
        flash(notice, "info")

    if is_redistribute_mode(get_process_mode(get_process_path(school_id, process_id))):
        prev_url = url_for("roster.roster_page")
        prev_label = "← Naar Wie gaat mee"
    else:
        prev_url = url_for("wizard.groups_to_page")
        prev_label = "← Naar Groepen naartoe"

    return render_template(
        "preferences_form.html",
        candidates=display_candidates,
        target_groups=all_groups_to,
        group_display=group_display,
        draft_state=draft_state,
        short_names=candidatedetermination.unique_display_names(participants),
        prev_url=prev_url,
        prev_label=prev_label,
    )


@wizard_bp.route("/not_together", methods=["GET", "POST"])
@login_required
@require_process
def not_together_page():
    """Display and process the not-together rules page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]

    try:
        groups_to, _ = load_groups(school_id, process_id)
        students = load_student_names(school_id, process_id, groups_to)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.preferences_excel"))
    n_groups = len(groups_to)

    if request.method == "GET":
        existing_rules, prev_url = _not_together_get_context(school_id, process_id)
        return render_template(
            "not_together.html",
            students=students,
            n_groups=n_groups,
            existing_rules=existing_rules,
            prev_preferences_url=prev_url,
        )

    n_rules = int(request.form.get("n_rules", 0))
    rules, error = _parse_not_together_form(request.form, n_rules)
    if error is None:
        try:
            datareader.validate_not_together(rules, students, n_groups)
        except ValidationError as exc:
            error = to_validation_message(exc)
    if error:
        warn_and_flash(error, log_detail="not_together_invalid")
        return redirect(url_for("wizard.not_together_page"))

    save_not_together(school_id, process_id, rules)
    logger.info("Not-together rules accepted: %d rules", len(rules))
    return redirect(url_for("wizard.start_distribution"))


@wizard_bp.route("/start_distribution", methods=["GET"])
@login_required
@require_process
def start_distribution():
    """Start the student distribution using stored input files"""
    logger.info("Starting distribution")
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_name = session["process_id"]
    groups_to_path = get_file_path(school_id, process_name, "groups.xlsx")

    not_together = load_not_together(school_id, process_name)

    proc = Process.by_name(school_id, process_name)
    Run.reset(proc.id)
    reset_result_files(school_id, process_name)
    # Capture the integer PK before spawning threads so they append log lines without
    # a school+name lookup on every on_update call.
    run_id = proc.id
    # Capture the real app object before spawning threads; current_app is a proxy that
    # cannot be used across thread boundaries.
    ctx = _ThreadContext(
        app_obj=current_app._get_current_object(),  # pylint: disable=protected-access
        school_id=school_id,
        process_name=process_name,
        run_id=run_id,
    )
    Thread(target=_create_sociogram_thread, args=(ctx,)).start()
    Thread(target=_run_solve_thread, args=(ctx, groups_to_path, not_together)).start()
    return redirect(url_for("results.processing"))
