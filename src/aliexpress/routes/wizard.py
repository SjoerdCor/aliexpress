"""Wizard blueprint: step-by-step distribution setup routes and helpers."""

import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from threading import Thread
from typing import Any

import pandas as pd
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

from aliexpress import candidatedetermination, datareader, input_writer, sociogram
from aliexpress.errors import (
    CouldNotReadFileError,
    DuplicateNameError,
    FeasibilityError,
    ValidationError,
)
from aliexpress.extensions import db
from aliexpress.form_parsers import parse_groups_to_form
from aliexpress.main import distribute_students_from_data
from aliexpress.models import LogLine, Process, Run
from aliexpress.preferences_data import PreferenceData
from aliexpress.routes.auth import effective_school_id
from aliexpress.routes.processes import require_process
from aliexpress.storage import get_file_path
from aliexpress.validation_messages import to_validation_message

logger = logging.getLogger(__name__)

wizard_bp = Blueprint("wizard", __name__)


def _write_voorkeuren_json(
    path: str, preference_data: PreferenceData, source: str
) -> None:
    """Persist a PreferenceData as voorkeuren.json, tagged with its input source."""
    payload = json.loads(preference_data.to_json())
    payload["source"] = source
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _read_voorkeuren_json(path: str) -> tuple[PreferenceData, str]:
    """Load a PreferenceData and its source tag from voorkeuren.json."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    source = payload.pop("source", "form")
    return PreferenceData.from_json(json.dumps(payload)), source


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
        try:  # pylint: disable=broad-exception-caught
            voorkeuren_path = get_file_path(
                ctx.school_id, ctx.process_name, "voorkeuren.json"
            )
            Process.by_name(ctx.school_id, ctx.process_name).run.set_status("running")
            preference_data, _ = _read_voorkeuren_json(voorkeuren_path)
            target_groups = datareader.read_groups_excel(groups_to_path)
            result = distribute_students_from_data(
                preference_data, target_groups, not_together, on_update=on_update
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
        try:  # pylint: disable=broad-exception-caught
            on_update("Sociogram tekenen...")
            voorkeuren_path = get_file_path(
                ctx.school_id, ctx.process_name, "voorkeuren.json"
            )
            preference_data, _ = _read_voorkeuren_json(voorkeuren_path)
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


def _load_student_names(groups_to, voorkeuren_path, preferences_path) -> list[str]:
    """Return sorted display names of students to populate the not-together dropdown.

    Prefers ``voorkeuren.json`` (canonical, written by both input paths); falls back to
    reading the raw Excel for processes created before ``voorkeuren.json`` was introduced.
    """
    if os.path.exists(voorkeuren_path):
        preference_data, _ = _read_voorkeuren_json(voorkeuren_path)
        names = sorted(preference_data.student_display.values())
    else:
        processor = datareader.VoorkeurenProcessor(preferences_path)
        processor.process(all_to_groups=list(groups_to.keys()))
        logger.debug(processor.student_display)
        names = sorted(processor.student_display.values())
    logger.debug(", ".join(names))
    return names


def _load_groups_to(school_id, process_id) -> dict:
    """Load the groups-to mapping (groupname → students) from the candidates JSON."""
    path = get_file_path(school_id, process_id, "relevant_students_and_groups.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("groups_to", {})


def _load_groups_to_state(school_id, process_id):
    """Load the saved groups-to form state, or None when the page was not filled yet."""
    path = get_file_path(school_id, process_id, "groups_to_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_student_selection(school_id, process_id):
    """Load the saved student selection, or None when the page was not used yet."""
    path = get_file_path(school_id, process_id, "student_selection.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_student_selection(school_id, process_id, selected_ids, new_students):
    """Persist which candidates were ticked and which students were added by hand."""
    path = get_file_path(school_id, process_id, "student_selection.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"selected_ids": selected_ids, "new_students": new_students},
            fh,
            ensure_ascii=False,
        )


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


def _save_not_together(school_id, process_id, rules):
    """Persist not-together rules as JSON (sets serialised as lists)."""
    data = [
        {"group": list(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in rules
    ]
    path = get_file_path(school_id, process_id, "not_together.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def _extract_new_students(form):
    """Extract manually added students from form fields"""
    firstnames = form.getlist("new_firstname[]")
    lastnames = form.getlist("new_lastname[]")
    genders = form.getlist("new_gender[]")
    groups = form.getlist("new_group[]")

    return [
        {"roepnaam": fn, "achternaam": ln, "geslacht": sex, "groepsnaam": gr}
        for fn, ln, sex, gr in zip(firstnames, lastnames, genders, groups)
        if fn.strip() and ln.strip()
    ]


@wizard_bp.route("/input_templates/<path:filename>")
def download_template(filename):
    """Download the template sheets"""
    return send_from_directory("input_templates", filename, as_attachment=True)


@wizard_bp.route("/upload_edexml", methods=["GET", "POST"])
@login_required
def upload_edexml():
    """Route to upload edexml"""
    if request.method == "GET":
        return render_template("upload_edexml.html")
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    try:
        edex_file = request.files["edexml"]
        edex_path = get_file_path(school_id, process_id, "edex.xml")
        edex_file.save(edex_path)
        edex_file.stream.seek(0)

        edexml = file_to_io(edex_file)
        jaargroep = int(request.form["jaargroep"])
        df = datareader.EdexReader(edexml).get_full_df()
        candidates, groups_from, groups_to = (
            candidatedetermination.handle_edexml_upload(df, jaargroep)
        )
        data = {
            "candidates": candidates,
            "groups_from": groups_from,
            "groups_to": groups_to,
        }
        path = get_file_path(school_id, process_id, "relevant_students_and_groups.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.upload_edexml"))
    return redirect(url_for("wizard.groups_to_page"))


@wizard_bp.route("/groups_to", methods=["GET", "POST"])
@login_required
@require_process
def groups_to_page():
    """Display and process the groups_to page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    groups_to = _load_groups_to(school_id, process_id)

    if request.method == "GET":
        return render_template(
            "groups_to.html",
            groups_to=groups_to,
            state=_load_groups_to_state(school_id, process_id),
        )

    submission = parse_groups_to_form(request.form, groups_to)
    if len(submission.distribution) < 2:
        error = "Er moeten minsten twee groepen zijn om de leerlingen over te verdelen"
        flash(error, "error")
        return redirect(url_for("wizard.groups_to_page"))

    path = get_file_path(school_id, process_id, "groups.xlsx")
    pd.DataFrame(submission.distribution).transpose().to_excel(
        path, index_label="Groepen"
    )
    with open(
        get_file_path(school_id, process_id, "groups_to_state.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(submission.state, f, ensure_ascii=False)
    logger.info(
        "Groups-to saved for process %s: %d active group(s), %d disabled, %d new",
        process_id,
        len(submission.distribution),
        len(submission.state["disabled_groups"]),
        len(submission.state["new_groups"]),
    )
    return redirect(url_for("wizard.student_preferences"))


@wizard_bp.route("/student_preferences", methods=["GET", "POST"])
@login_required
@require_process
def student_preferences():
    """Display page where the teacher can add preferences for the student"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    with open(
        get_file_path(school_id, process_id, "relevant_students_and_groups.json"),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    candidates = data.get("candidates", [])
    groups_from = data.get("groups_from", {})

    if request.method == "GET":
        selection = _load_student_selection(school_id, process_id)
        return render_template(
            "student_preferences.html",
            candidates=candidates,
            groups_from=groups_from,
            preferences_uploaded=os.path.exists(
                get_file_path(school_id, process_id, "preferences.xlsx")
            ),
            # None means "first visit": default to all candidates ticked.
            selected_ids=set(selection["selected_ids"]) if selection else None,
            saved_new_students=selection["new_students"] if selection else [],
        )
    new_students = _extract_new_students(request.form)
    selected_ids = request.form.getlist("students")
    if len(new_students) + len(selected_ids) == 0:
        flash("Er moet minsten één leerling aanwezig zijn", "error")
        return redirect(url_for("wizard.student_preferences"))

    try:
        df_total = candidatedetermination.combine_students(
            candidates, selected_ids, new_students
        )
    except DuplicateNameError as exc:
        logger.exception(exc)
        flash(f"Vond leerlingen dubbel: {exc.context['duplicate_names']}", "error")
        return redirect(url_for("wizard.student_preferences"))

    # Remember exactly what the teacher selected so the page restores on return.
    _save_student_selection(school_id, process_id, selected_ids, new_students)

    groups_to = pd.read_excel(
        get_file_path(school_id, process_id, "groups.xlsx"), index_col=0
    ).index.tolist()
    buffer = input_writer.create_prefilled_excel(groups_to, df_total)

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
        if os.path.exists(get_file_path(school_id, process_id, "preferences.xlsx")):
            logger.info(
                "No new preferences upload for process %s; continuing with stored file",
                process_id,
            )
            return redirect(url_for("wizard.not_together_page"))
        flash("Upload eerst het ingevulde bestand om verder te gaan.", "error")
        return redirect(url_for("wizard.student_preferences"))
    try:
        raw = upload.read()
        groups_to_path = get_file_path(school_id, process_id, "groups.xlsx")
        groups_to_data, _ = datareader.read_groups_excel(groups_to_path)
        groups_to = list(groups_to_data.keys())
        processor = datareader.VoorkeurenProcessor(BytesIO(raw))
        processor.process(all_to_groups=groups_to)  # validates; raises on invalid input
        # Save the raw upload directly so re-reading later preserves names as entered.
        # VoorkeurenProcessor normalises names to matching keys at read time anyway,
        # and storing the original ensures student_display maps correctly to display names.
        preferences_path = get_file_path(school_id, process_id, "preferences.xlsx")
        with open(preferences_path, "wb") as fh:
            fh.write(raw)
        # Persist as voorkeuren.json so the solver and sociogram can load from a single
        # canonical format regardless of input path (Excel or web form).
        _write_voorkeuren_json(
            get_file_path(school_id, process_id, "voorkeuren.json"),
            processor.to_preference_data(),
            source="excel",
        )
        logger.info(
            "Preferences accepted for process %s: %d students",
            process_id,
            len(processor.input.index),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.student_preferences"))
    return redirect(url_for("wizard.not_together_page"))


@wizard_bp.route("/not_together", methods=["GET", "POST"])
@login_required
@require_process
def not_together_page():
    """Display and process the not-together rules page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    groups_to_path = get_file_path(school_id, process_id, "groups.xlsx")

    try:
        groups_to, _ = datareader.read_groups_excel(groups_to_path)
        students = _load_student_names(
            groups_to,
            get_file_path(school_id, process_id, "voorkeuren.json"),
            get_file_path(school_id, process_id, "preferences.xlsx"),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("wizard.student_preferences"))
    n_groups = len(groups_to)

    if request.method == "GET":
        nt_path = get_file_path(school_id, process_id, "not_together.json")
        if os.path.exists(nt_path):
            with open(nt_path, encoding="utf-8") as fh:
                existing_rules = json.load(fh)
        else:
            existing_rules = []
        return render_template(
            "not_together.html",
            students=students,
            n_groups=n_groups,
            existing_rules=existing_rules,
        )

    n_rules = int(request.form.get("n_rules", 0))
    rules, error = _parse_not_together_form(request.form, n_rules)
    if error is None:
        try:
            datareader.validate_not_together(rules, students, n_groups)
        except ValidationError as exc:
            error = to_validation_message(exc)
    if error:
        flash(error, "error")
        return redirect(url_for("wizard.not_together_page"))

    _save_not_together(school_id, process_id, rules)
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

    not_together_path = get_file_path(school_id, process_name, "not_together.json")
    if os.path.exists(not_together_path):
        with open(not_together_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        not_together = [
            {"group": set(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
            for r in raw
        ]
    else:
        not_together = []

    proc = Process.by_name(school_id, process_name)
    Run.reset(proc.id)
    for _stale in ("results.xlsx", "result_tables.json", "sociogram.html"):
        _path = get_file_path(school_id, process_name, _stale)
        if os.path.exists(_path):
            os.remove(_path)
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
