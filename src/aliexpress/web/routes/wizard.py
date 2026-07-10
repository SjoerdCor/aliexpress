"""Wizard blueprint: step-by-step distribution setup routes and helpers."""

import logging
from io import BytesIO
from threading import Thread

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

from ...data import candidatedetermination, datareader, input_writer
from ...data.form_parsers import (
    build_form_state,
    parse_groups_to_form,
    parse_not_together_form,
    parse_student_entry,
    reconcile_dangling,
)
from ...data.preferences_form import build_preference_data
from ...errors import DuplicateNameError, ValidationError
from ..display import sorted_for_display
from ..flashing import warn_and_flash
from ..models import Process, Run
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
from ..storage import get_process_path
from ..tasks import ThreadContext, create_sociogram_thread, run_solve_thread
from ..validation_messages import to_validation_message
from .auth import effective_school_id
from .processes import get_process_mode, is_redistribute_mode, require_process

logger = logging.getLogger(__name__)

wizard_bp = Blueprint("wizard", __name__)


def file_to_io(uploaded_file) -> BytesIO:
    """Get file as BytesIO"""
    return BytesIO(uploaded_file.read())


def _flash_upload_error(exc: Exception) -> None:
    """Log a rejected upload and flash a friendly Dutch message to the user."""
    logger.exception("Upload rejected")
    flash(to_validation_message(exc), "error")


def _write_pref_form_state(school_id, process_id, form, participants):
    """Parse the form and persist the intermediate draft (``preferences_form_state.json``).

    Returns the parsed ``StudentEntry`` list. Does not validate — it captures whatever is on
    the page so a reload (or a crash) restores it. The population is the settled roster, so
    every participant is parsed (there are no checkboxes and no new students here anymore).
    """
    entries = [parse_student_entry(c, form) for c in participants]
    state = build_form_state(entries, participants)
    save_pref_form_state(school_id, process_id, state)
    return entries


def _pref_form_post_data(school_id, process_id, form, participants, all_groups_to):
    """Parse + save the draft, then validate and return the resulting PreferenceData."""
    entries = _write_pref_form_state(school_id, process_id, form, participants)
    # Short unique chip labels, re-keyed from full name -> short name to the same
    # matching keys used everywhere else on PreferenceData (student_display etc.).
    unique_name = {
        datareader.matching_key(full): short
        for full, short in candidatedetermination.unique_display_names(
            participants
        ).items()
    }
    return build_preference_data(entries, all_groups_to, unique_name)


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
    removed = reconcile_dangling(draft_state, participants, group_labels)
    return [
        f"{target} gaat niet meer mee — de voorkeur van {owner} "
        f"naar {target} is verwijderd."
        for owner, target in removed
    ]


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
    try:
        rules = parse_not_together_form(request.form, n_rules)
        datareader.validate_not_together(rules, students, n_groups)
    except ValidationError as exc:
        warn_and_flash(to_validation_message(exc), log_detail=exc.code)
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

    not_together = load_not_together(school_id, process_name)

    proc = Process.by_name(school_id, process_name)
    Run.reset(proc.id)
    reset_result_files(school_id, process_name)
    # Capture the integer PK before spawning threads so they append log lines without
    # a school+name lookup on every on_update call.
    run_id = proc.id
    # Capture the real app object before spawning threads; current_app is a proxy that
    # cannot be used across thread boundaries.
    ctx = ThreadContext(
        app_obj=current_app._get_current_object(),  # pylint: disable=protected-access
        school_id=school_id,
        process_name=process_name,
        run_id=run_id,
    )
    Thread(target=create_sociogram_thread, args=(ctx,)).start()
    Thread(target=run_solve_thread, args=(ctx, not_together)).start()
    return redirect(url_for("results.processing"))
