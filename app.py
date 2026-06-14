"""The flask server that governs the app"""

import functools
import json
import logging
import os
import re
import shutil
import webbrowser
from dataclasses import dataclass
from io import BytesIO
from threading import Thread

import numpy as np
import pandas as pd
import pandera as pa
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)

from aliexpress import candidatedetermination, datareader, input_writer, sociogram
from aliexpress.errors import (
    CouldNotReadFileError,
    DuplicateNameError,
    FeasibilityError,
    ValidationError,
)
from aliexpress.extensions import db
from aliexpress.logging_config import add_file_handler, configure_logging
from aliexpress.main import distribute_students_once
from aliexpress.models import LogLine, Run

configure_logging()
logger = logging.getLogger("aliexpress.app")  # file handler added below


load_dotenv()

env = os.getenv("FLASK_ENV", "production")
if env == "development":
    from aliexpress.appconfig import DevelopmentConfig as ConfigClass
else:
    from aliexpress.appconfig import ProductionConfig as ConfigClass

app = Flask(__name__)
app.config.from_object(ConfigClass)
LOG_DIR = os.path.join(app.instance_path, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
add_file_handler(os.path.join(LOG_DIR, "aliexpress.log"))
BASE_DIR = os.path.join(app.instance_path, "storage")
os.makedirs(BASE_DIR, exist_ok=True)
logger.debug("Created dir if not exists: %s", BASE_DIR)

# A relative SQLite URI is resolved against the instance folder, which must exist first.
os.makedirs(app.instance_path, exist_ok=True)
db.init_app(app)
with app.app_context():
    db.create_all()


def get_process_path(process_id):
    """Get directory for process"""
    return os.path.join(BASE_DIR, process_id)


def get_file_path(process_id, filename):
    """Get file for a certain process"""
    return os.path.join(get_process_path(process_id), filename)


def require_process(f):
    """Route decorator: redirect to /processes when no active process is in session."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "process_id" not in session:
            flash("Geen actief proces geselecteerd.", "error")
            return redirect(url_for("processes"))
        return f(*args, **kwargs)

    return wrapper


def _reset_run(process_id):
    """Start a fresh run for this process: reset its row, logs and stale result files.

    Runs in the request context; the committed row is then visible to the background
    threads (which open their own session in their own app context).
    """
    run = db.session.get(Run, process_id)
    if run is None:
        run = Run(process_id=process_id)
        db.session.add(run)
    run.status = "pending"
    run.message = None
    LogLine.query.filter_by(process_id=process_id).delete()
    db.session.commit()
    for filename in ("results.xlsx", "result_tables.json", "sociogram.html"):
        stale = get_file_path(process_id, filename)
        if os.path.exists(stale):
            os.remove(stale)


def _set_status(process_id, new_status, message=None):
    """Update the run's status (and optional error message) for this process."""
    run = db.session.get(Run, process_id)
    run.status = new_status
    run.message = message
    db.session.commit()


def _write_result_files(process_id, result):
    """Persist the solver output as files in the process dir (download + rendered tables).

    Written before the status flips to "done" so the result page never polls ahead of the
    files it needs.
    """
    with open(get_file_path(process_id, "results.xlsx"), "wb") as fh:
        fh.write(result["download"].getbuffer())
    tables = {name: df.to_html(na_rep="") for name, df in result["dataframes"].items()}
    with open(
        get_file_path(process_id, "result_tables.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(tables, fh, ensure_ascii=False)


@app.route("/")
def home():
    """Display home page"""
    return render_template("home.html")


@app.route("/processes")
def processes():
    """Display page to create or choose process"""
    existing_processes = [
        name
        for name in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, name))
    ]
    return render_template("processes.html", processes=existing_processes)


def _validate_process_name(process_name, must_exist=True):
    """Return an error message, or None when the name is valid."""
    if not process_name:
        return "Naam is verplicht"
    if not re.match(r"^[\w\- ]+$", process_name):
        return "Alleen letters, cijfers, spaties, - en _ toegestaan"
    path = os.path.join(BASE_DIR, process_name)
    if must_exist and not os.path.exists(path):
        return "Proces bestaat niet"
    if not must_exist and os.path.exists(path):
        return "Proces bestaat al"
    return None


@app.route("/processes/create", methods=["POST"])
def create_process():
    """Create a new process"""
    process_name = request.form.get("process_name", "").strip()
    if error := _validate_process_name(process_name, must_exist=False):
        flash(error, "error")
        return redirect(url_for("processes"))
    path = os.path.join(BASE_DIR, process_name)
    os.makedirs(path)
    session["process_id"] = process_name
    return redirect(url_for("upload_edexml"))


@app.route("/processes/delete/<process_name>", methods=["POST"])
def delete_process(process_name):
    """Delete a process"""
    if error := _validate_process_name(process_name, must_exist=True):
        flash(error, "error")
        return redirect(url_for("processes"))
    path = os.path.join(BASE_DIR, process_name)
    shutil.rmtree(path)
    return redirect(url_for("processes"))


@app.route("/processes/select/<process_id>")
def select_process(process_id):
    """Select process"""
    path = get_process_path(process_id)
    if not os.path.exists(path):
        abort(404)

    session["process_id"] = process_id
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


def file_to_io(uploaded_file) -> BytesIO:
    """Get file as BytesIO"""
    return BytesIO(uploaded_file.read())


@app.route("/input_templates/<path:filename>")
def download_template(filename):
    """Download the template sheets"""
    return send_from_directory("input_templates", filename, as_attachment=True)


@app.route("/download_preferences")
@require_process
def download_preferences():
    """Download the original, filled-in preferences file as the teacher uploaded it.

    This is the richly-formatted upload (with column help and dropdown validations), not
    the normalised processed file, so the teacher can keep editing it safely.
    """
    path = get_file_path(session["process_id"], "preferences_original.xlsx")
    if not os.path.exists(path):
        logger.warning(
            "Download of filled-in preferences requested but none stored for process %s",
            session["process_id"],
        )
        abort(404)
    logger.info(
        "Serving stored preferences upload for process %s", session["process_id"]
    )
    return send_file(
        path,
        as_attachment=True,
        download_name="voorkeuren (ingevuld).xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/upload_edexml", methods=["GET", "POST"])
def upload_edexml():
    """Route to upload edexml"""
    if request.method == "GET":
        return render_template("upload_edexml.html")
    try:
        edex_file = request.files["edexml"]
        edex_path = get_file_path(session["process_id"], "edex.xml")
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
        path = get_file_path(session["process_id"], "relevant_students_and_groups.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("upload_edexml"))
    return redirect(url_for("groups_to_page"))


def _load_groups_to(process_id) -> dict:
    """Load the groups-to mapping (groupname → students) from the candidates JSON."""
    path = get_file_path(process_id, "relevant_students_and_groups.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("groups_to", {})


def _load_groups_to_state(process_id):
    """Load the saved groups-to form state, or None when the page was not filled yet."""
    path = get_file_path(process_id, "groups_to_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/groups_to", methods=["GET", "POST"])
@require_process
def groups_to_page():
    """Display and process the groups_to page"""
    process_id = session["process_id"]
    groups_to = _load_groups_to(process_id)

    if request.method == "GET":
        return render_template(
            "groups_to.html",
            groups_to=groups_to,
            state=_load_groups_to_state(process_id),
        )

    submission = parse_groups_to_form(request.form, groups_to)
    if len(submission.distribution) < 2:
        error = "Er moeten minsten twee groepen zijn om de leerlingen over te verdelen"
        flash(error, "error")
        return redirect(url_for("groups_to_page"))

    path = get_file_path(process_id, "groups.xlsx")
    pd.DataFrame(submission.distribution).transpose().to_excel(
        path, index_label="Groepen"
    )
    with open(
        get_file_path(process_id, "groups_to_state.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(submission.state, f, ensure_ascii=False)
    logger.info(
        "Groups-to saved for process %s: %d active group(s), %d disabled, %d new",
        process_id,
        len(submission.distribution),
        len(submission.state["disabled_groups"]),
        len(submission.state["new_groups"]),
    )
    return redirect(url_for("student_preferences"))


def _load_student_selection(process_id):
    """Load the saved student selection, or None when the page was not used yet."""
    path = get_file_path(process_id, "student_selection.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_student_selection(process_id, selected_ids, new_students):
    """Persist which candidates were ticked and which students were added by hand."""
    path = get_file_path(process_id, "student_selection.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"selected_ids": selected_ids, "new_students": new_students},
            fh,
            ensure_ascii=False,
        )


@app.route("/student_preferences", methods=["GET", "POST"])
@require_process
def student_preferences():
    """Display page where the teacher can add preferences for the student"""
    process_id = session["process_id"]
    data_path = get_file_path(process_id, "relevant_students_and_groups.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = data.get("candidates", [])
    groups_from = data.get("groups_from", {})

    if request.method == "GET":
        selection = _load_student_selection(process_id)
        return render_template(
            "student_preferences.html",
            candidates=candidates,
            groups_from=groups_from,
            preferences_uploaded=os.path.exists(
                get_file_path(process_id, "preferences_original.xlsx")
            ),
            # None means "first visit": default to all candidates ticked.
            selected_ids=set(selection["selected_ids"]) if selection else None,
            saved_new_students=selection["new_students"] if selection else [],
        )
    new_students = _extract_new_students(request.form)
    selected_ids = request.form.getlist("students")
    if len(new_students) + len(selected_ids) == 0:
        error = "Er moet minsten één leerling aanwezig zijn"
        flash(error, "error")
        return redirect(url_for("student_preferences"))

    try:
        df_total = candidatedetermination.combine_students(
            candidates, selected_ids, new_students
        )
    except DuplicateNameError as exc:
        logger.exception(exc)
        flash(f"Vond leerlingen dubbel: {exc.context['duplicate_names']}", "error")
        return redirect(url_for("student_preferences"))

    # Remember exactly what the teacher selected so the page restores on return.
    _save_student_selection(process_id, selected_ids, new_students)

    path = get_file_path(process_id, "groups.xlsx")
    groups_to = pd.read_excel(path, index_col=0).index.tolist()
    buffer = input_writer.create_prefilled_excel(groups_to, df_total)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="voorkeuren.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/upload_preferences", methods=["POST"])
@require_process
def upload_preferences():
    """Handle the upload of the preferences file, or continue with an earlier upload.

    Re-uploading is optional when going back and forth: if no new file is chosen but a
    valid preferences file was uploaded earlier, the teacher simply continues with it.
    """
    process_id = session["process_id"]
    upload = request.files.get("preferences")
    if not (upload and upload.filename):
        if os.path.exists(get_file_path(process_id, "preferences.xlsx")):
            logger.info(
                "No new preferences upload for process %s; continuing with stored file",
                process_id,
            )
            return redirect(url_for("not_together_page"))
        flash("Upload eerst het ingevulde bestand om verder te gaan.", "error")
        return redirect(url_for("student_preferences"))
    try:
        raw = upload.read()
        groups_to_path = get_file_path(process_id, "groups.xlsx")
        groups_to_data, _ = datareader.read_groups_excel(groups_to_path)
        groups_to = list(groups_to_data.keys())
        processor = datareader.VoorkeurenProcessor(BytesIO(raw))
        processor.process(all_to_groups=groups_to)  # validates further
        preferences_path = get_file_path(process_id, "preferences.xlsx")
        input_writer.write_preferences_to_excel(
            processor.input.reset_index(), preferences_path
        )
        # Keep the original upload so the teacher can download and keep editing the
        # richly-formatted file with all wishes intact (the processed file above is
        # normalised and stripped of formatting/validations).
        original_path = get_file_path(process_id, "preferences_original.xlsx")
        with open(original_path, "wb") as fh:
            fh.write(raw)
        logger.info(
            "Preferences accepted for process %s: %d students; kept original upload",
            process_id,
            len(processor.input.index),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("student_preferences"))
    return redirect(url_for("not_together_page"))


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


@app.route("/not_together", methods=["GET", "POST"])
@require_process
def not_together_page():
    """Display and process the not-together rules page"""
    process_id = session["process_id"]
    preferences_path = get_file_path(process_id, "preferences.xlsx")
    groups_to_path = get_file_path(process_id, "groups.xlsx")

    try:
        groups_to, _ = datareader.read_groups_excel(groups_to_path)
        processor = datareader.VoorkeurenProcessor(preferences_path)
        processor.process(all_to_groups=list(groups_to.keys()))
        # Show names as entered in the dropdown; matching happens on the key on submit.
        students = sorted(processor.student_display.values())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("student_preferences"))
    n_groups = len(groups_to)

    if request.method == "GET":
        nt_path = get_file_path(process_id, "not_together.json")
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

    if request.form.get("action") == "skip":
        _save_not_together(process_id, [])
        return redirect(url_for("start_distribution"))

    n_rules = int(request.form.get("n_rules", 0))
    rules, error = _parse_not_together_form(request.form, n_rules)
    if error is None:
        try:
            datareader.validate_not_together(rules, students, n_groups)
        except ValidationError as exc:
            error = to_validation_message(exc)
    if error:
        flash(error, "error")
        return redirect(url_for("not_together_page"))

    _save_not_together(process_id, rules)
    return redirect(url_for("start_distribution"))


def _save_not_together(process_id, rules):
    """Persist not-together rules as JSON (sets serialised as lists)."""
    data = [
        {"group": list(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
        for r in rules
    ]
    path = get_file_path(process_id, "not_together.json")
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


@dataclass
class GroupsToSubmission:
    """Parsed groups-to form.

    ``distribution`` holds the retained boy/girl counts per group (written to
    ``groups.xlsx`` for the solver); ``state`` captures exactly what the teacher did so
    the page can be restored on return (written to ``groups_to_state.json``).
    """

    distribution: dict[str, dict[str, int]]
    state: dict


def _checked_indices(form, groupname: str, n_students: int) -> list[int]:
    """Return the submitted student indices for a group, bounded to the known students.

    Checkbox values are the student's position in the groups-to list (not the gender),
    so the server can both count genders and remember exactly who was ticked.
    """
    indices = []
    for raw in form.getlist(f"group_students[{groupname}]"):
        try:
            index = int(raw)
        except ValueError:
            continue
        if 0 <= index < n_students:
            indices.append(index)
    return indices


def parse_groups_to_form(form, groups_to: dict) -> GroupsToSubmission:
    """Turn the submitted groups-to form into retained counts and a restore state.

    Active groups come from the ``group`` fields. An original group missing from that
    list was switched off; a submitted name that is not an original group is a
    teacher-added empty group. Genders are looked up from ``groups_to`` by the submitted
    student indices, so the counts cannot drift from the source data.

    Every original group's ticks are remembered (their checkboxes submit even when the
    group is switched off), so switching a group back on restores exactly who was ticked.
    Only active groups contribute to ``distribution`` (and thus to ``groups.xlsx``).
    """
    submitted = form.getlist("group")
    # Remember the ticks of every original group, including switched-off ones.
    original_state = {
        name: {"checked_indices": _checked_indices(form, name, len(students))}
        for name, students in groups_to.items()
    }
    distribution: dict[str, dict[str, int]] = {}
    new_groups: list[str] = []

    for name in submitted:
        if name in groups_to:
            students = groups_to[name]
            indices = original_state[name]["checked_indices"]
            distribution[name] = {
                "Jongens": sum(students[i]["geslacht"] == "Jongen" for i in indices),
                "Meisjes": sum(students[i]["geslacht"] == "Meisje" for i in indices),
            }
        else:
            distribution[name] = {"Jongens": 0, "Meisjes": 0}
            new_groups.append(name)

    state = {
        "original_groups": original_state,
        "disabled_groups": [name for name in groups_to if name not in submitted],
        "new_groups": new_groups,
    }
    return GroupsToSubmission(distribution=distribution, state=state)


def to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
    if isinstance(exc, pa.errors.SchemaError):
        return schemaerror_to_validation_message(exc)
    if isinstance(exc, (ValidationError, CouldNotReadFileError, FeasibilityError)):
        return readableerror_to_validation_message(exc)
    return (
        "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
        "Laat de maker dit onderzoeken."
    )


def _flash_upload_error(exc: Exception) -> None:
    """Log a rejected upload and flash a friendly Dutch message to the user."""
    logger.exception("Upload rejected")
    flash(to_validation_message(exc), "error")


def readableerror_to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
    friendly_templates = {
        "wrong_columns_preferences": (
            "Het voorkeuren-bestand heeft de verkeerde kolommen. Controleer of je het goede"
            " bestand hebt geupload en het meest recente template hebt gebruikt. "
            "\n{wrong_columns}"
        ),
        "infeasible_problem": (
            "Met deze vereiste klassenbalans en verdeling van leerlingen die overgaan is het"
            "niet mogelijk. Overweeg de volgende versoepelingen om het probleem wel op te "
            "lossen:\n {possible_improvement}"
        ),
        "internal_error": (
            "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
            "Laat de maker dit onderzoeken."
        ),
        "too_few_students_not_together": (
            "Niet-samen-regel {rule_index} heeft minder dan 2 leerlingen. "
            "Voeg minstens 2 leerlingen toe."
        ),
        "invalid_max_samen_not_together": (
            "Niet-samen-regel {rule_index}: het maximale aantal samen moet minstens 1 zijn."
        ),
        "unknown_student_not_together": (
            "In de niet-samen-regels staan onbekende leerlingen: {unknown_students}. "
            "Controleer of de namen overeenkomen met het voorkeuren-bestand."
        ),
        "too_strict_not_together": (
            "Niet-samen-regel {rule_index}: met {n_groups} groepen is het niet mogelijk om "
            "{n_students} leerlingen te verdelen met maximaal {max_samen} bij elkaar."
        ),
    }

    template = friendly_templates.get(exc.code, None)
    if template:
        return template.format(**exc.context)
    return (
        "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
        "Laat de maker dit onderzoeken."
    )


# Deliberately overruling pylint here; we need a branch per validation
# pylint: disable=too-many-return-statements, too-many-branches
def schemaerror_to_validation_message(exc: pa.errors.SchemaError) -> str:
    """Convert a pandera SchemaError to a user-friendly message

    This SchemaError must have been modified to contain a 'filetype' attribute.
    """
    if exc.reason_code in (
        pa.errors.SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
        pa.errors.SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
    ):
        return (
            f"Het {exc.filetype}-bestand heeft de verkeerde kolommen. Controleer of je het goede"
            " bestand hebt geupload en het meest recente template hebt gebruikt. "
            f"\n{exc.failure_cases}"
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.DATATYPE_COERCION:
        return (
            f"Ongeldige waarden gevonden in kolom {exc.schema.name} "
            f"van het {exc.filetype}-bestand"
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS:
        students = getattr(exc, "offending_students", [])
        if students:
            return (
                f"In het {exc.filetype}-bestand mist een waarde bij: "
                f"{', '.join(students)}. Vul bij elke wens een naam of groep in, of haal "
                "het bijbehorende gewicht weg als er geen wens is."
            )
        return (
            f"In het {exc.filetype}-bestand zijn niet alle verplichte velden gevuld "
            f"(kolom {exc.column_name})."
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_DUPLICATES:
        if exc.filetype == "voorkeuren":
            duplicates = ", ".join(exc.failure_cases["failure_case"])
            return (
                f"In voorkeuren is de volgende naam/namen niet uniek: {duplicates}\n"
                "Voeg de eerste letter van de achternaam toe om de leerlingen van "
                "elkaar te onderscheiden."
            )
        return (
            f"In het {exc.filetype}-bestand zijn dubbelingen ingevuld "
            f"in kolom {exc.column_name}"
        )

    if exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK:
        if exc.check.name == "empty_df":
            return (
                f"Het {exc.filetype}-bestand was helemaal leeg. Daardoor kan er "
                "geen groepsindeling worden berekend"
            )
        if exc.column_name == ("Jongen/meisje", np.nan, np.nan):
            return f"Verkeerd ingevuld geslacht voor {', '.join(exc.failure_cases['index'])}"
        if exc.check.name == "greater_than" and "Gewicht" in exc.column_name:
            return "Er zijn negatieve gewichten in het voorkeurenbestand."
        if exc.check.name == "duplicated_values_preferences":
            students_with_duplicates = ", ".join(
                set(exc.failure_cases["index"].get_level_values("Leerling"))
            )
            return (
                "In het voorkeuren-bestand is voor "
                f"{students_with_duplicates} een leerling of groep gevonden die "
                "dubbel voorkomt. Tel ze op of streep ze tegen elkaar weg om "
                "dubbelingen te voorkomen."
            )
        if exc.check.name == "invalid_values_preferences":
            invalid_values = ", ".join(
                set(
                    exc.failure_cases.loc[
                        lambda df: df["column"] == "Waarde", "failure_case"
                    ]
                )
            )
            return f"Onbekende leerling of groep in categorie: {invalid_values}"
        if exc.check.name == "isin" and exc.filetype == "niet_samen":
            unknown_students = ", ".join(exc.failure_cases["failure_case"].astype(str))
            return (
                f"In het niet-samen-bestand komt {unknown_students} voor, "
                "die niet in het voorkeurenbestand voorkomt"
            )
        if exc.check.name == "duplicated_students_not_together":
            rows = ", ".join(set(exc.failure_cases["index"].add(1).astype(str)))
            duplicated_students = ", ".join(
                exc.failure_cases.groupby("index")["failure_case"].apply(
                    lambda s: s[s.duplicated()]
                )
            )
            return (
                f"In het niet-samen-bestand wordt in de {rows}e "
                f"groep dezelfde leerling meerdere keren genoemd: {duplicated_students}"
            )
        if exc.check.name == "too_strict_not_together":
            rows = ", ".join(set(exc.failure_cases["index"].add(1).astype(str)))
            max_samen = ", ".join(
                exc.failure_cases.loc[
                    lambda df: df["column"] == "Max aantal samen", "failure_case"
                ].astype(str)
            )
            nr_students = ", ".join(
                exc.failure_cases.groupby("index").size().sub(1).astype(str)
            )

            return (
                f"In het niet-samen-bestand op de {rows}e rij is de maximale "
                f"groepsgrootte te klein: met dit aantal groepen lukt het niet om {nr_students} "
                f"leerlingen te verdelen met maximaal {max_samen} bij elkaar."
            )

    return (
        f"Er is iets onverwachts misgegaan bij het lezen van {exc.filetype}. "
        "Controleer het bestand goed en of je het meest recente template hebt gebruikt. "
        "Als het probleem blijft bestaan, laat de maker dit onderzoeken."
    )


def _handle_failure(exc, process_id):
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
    _set_status(process_id, "error", to_validation_message(exc))


@app.route("/start_distribution", methods=["GET"])
@require_process
def start_distribution():
    """Start the student distribution using stored input files"""
    logger.info("Starting distribution")
    process_id = session["process_id"]
    preferences_path = get_file_path(process_id, "preferences.xlsx")
    groups_to_path = get_file_path(process_id, "groups.xlsx")

    not_together_path = get_file_path(process_id, "not_together.json")
    if os.path.exists(not_together_path):
        with open(not_together_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        not_together = [
            {"group": set(r["group"]), "Max_aantal_samen": r["Max_aantal_samen"]}
            for r in raw
        ]
    else:
        not_together = []

    with open(preferences_path, "rb") as fh:
        preferences_bytes = BytesIO(fh.read())

    _reset_run(process_id)

    def on_update(message):
        # Called from within a thread's app context (see run_task/create_sociogram).
        db.session.add(LogLine(process_id=process_id, text=message))
        db.session.commit()

    # pylint: disable=broad-exception-caught
    def run_task():
        # A background thread needs its own app context to get a DB session.
        with app.app_context():
            try:
                _set_status(process_id, "running")
                result = distribute_students_once(
                    preferences_path, groups_to_path, not_together, on_update=on_update
                )
                logger.info("Distributing students finished successfully")
                # Write the artifacts before flipping to "done", so the polling result
                # page never races ahead of the files it serves.
                _write_result_files(process_id, result)
                _set_status(process_id, "done")
            except Exception as exc:
                _handle_failure(exc, process_id)

    def create_sociogram(preferences, groups_to):
        with app.app_context():
            try:
                on_update("Sociogram tekenen...")
                groups_to_data, _ = datareader.read_groups_excel(groups_to)
                sg = sociogram.SociogramMaker(preferences, list(groups_to_data.keys()))
                fig, g, pos = sg.plot_sociogram()
                logger.info("Sociogram created")

                fig = sociogram.networkx_to_plotly(g, pos)
                html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                logger.info("HTML created")
                with open(
                    get_file_path(process_id, "sociogram.html"), "w", encoding="utf-8"
                ) as fh:
                    fh.write(html)
                on_update(
                    '<a href=/sociogram target="_blank" class="button">'
                    "Bekijk het sociogram nu!</a>"
                )
            except Exception:
                logger.exception("Could not create sociogram")

    # pylint: enable=broad-exception-caught
    Thread(target=create_sociogram, args=(preferences_bytes, groups_to_path)).start()
    Thread(target=run_task).start()

    return redirect(url_for("processing"))


@app.route("/status")
@require_process
def status():
    """Return the current process's run status and log lines as JSON."""
    run = db.session.get(Run, session["process_id"])
    if run is None:
        return jsonify({"status_studentdistribution": "unknown", "logs": []})
    payload = {
        "status_studentdistribution": run.status,
        "logs": [line.text for line in run.log_lines],
    }
    if run.status == "error" and run.message:
        payload["message"] = run.message
    return jsonify(payload)


@app.route("/processing")
@require_process
def processing():
    """Display processing page"""
    return render_template("processing.html")


@app.route("/handle-error", methods=["POST"])
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    flash(data["message"], "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@app.route("/sociogram")
@require_process
def show_sociogram():
    """Display the sociogram for the current process"""
    path = get_file_path(session["process_id"], "sociogram.html")
    if not os.path.exists(path):
        flash("Sociogram niet beschikbaar.", "error")
        return redirect(url_for("processes"))
    with open(path, encoding="utf-8") as fh:
        plotly_div = fh.read()
    return render_template("sociogram.html", plotly_div=plotly_div)


@app.route("/result")
@require_process
def result_page():
    """Display result for the current process"""
    path = get_file_path(session["process_id"], "result_tables.json")
    if not os.path.exists(path):
        flash("Resultaat niet beschikbaar.", "error")
        return redirect(url_for("processes"))
    with open(path, encoding="utf-8") as fh:
        dataframes = json.load(fh)
    return render_template("result.html", dataframes=dataframes)


@app.route("/download")
@require_process
def download():
    """Download the groepsindeling for the current process"""
    path = get_file_path(session["process_id"], "results.xlsx")
    if not os.path.exists(path):
        flash("Groepsindeling niet gevonden. Mogelijk nog aan het berekenen", "error")
        return render_template("result.html", dataframes={})

    return send_file(
        path,
        as_attachment=True,
        download_name="results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/done")
def done():
    """Show done page"""
    return render_template("done.html")


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=env == "development")
