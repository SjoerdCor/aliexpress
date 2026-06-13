"""The flask server that governs the app"""

import functools
import json
import os
import re
import shutil
import uuid
import webbrowser
from collections import Counter, defaultdict
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
from aliexpress.logging_config import add_file_handler, setup_logger
from aliexpress.main import distribute_students_once

logger = setup_logger(
    __name__
)  # file handler added below, after instance path is known


load_dotenv()

env = os.getenv("FLASK_ENV", "production")
if env == "development":
    from src.aliexpress.appconfig import DevelopmentConfig as ConfigClass
else:
    from src.aliexpress.appconfig import ProductionConfig as ConfigClass

app = Flask(__name__)
app.config.from_object(ConfigClass)
LOG_DIR = os.path.join(app.instance_path, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
add_file_handler(logger, os.path.join(LOG_DIR, "aliexpress.log"))
BASE_DIR = os.path.join(app.instance_path, "storage")
os.makedirs(BASE_DIR, exist_ok=True)
logger.debug("Created dir if not exists: %s", BASE_DIR)


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


temp_storage = {}
status_dct = defaultdict(
    lambda: {
        "status_studentdistribution": "pending",
        "status_sociogram": "pending",
        "logs": [],
    }
)


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


@app.route("/groups_to", methods=["GET", "POST"])
@require_process
def groups_to_page():
    """Display and process the groups_to page"""
    if request.method == "GET":
        data_path = get_file_path(
            session["process_id"], "relevant_students_and_groups.json"
        )
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        groups_to_data = data.get("groups_to", [])

        return render_template("groups_to.html", groups_to=groups_to_data)

    boy_girl_distribution = extract_selected_per_group(request.form)
    if len(boy_girl_distribution) < 2:
        error = "Er moeten minsten twee groepen zijn om de leerlingen over te verdelen"
        flash(error, "error")
        return redirect(url_for("groups_to_page"))

    path = get_file_path(session["process_id"], "groups.xlsx")
    pd.DataFrame(boy_girl_distribution).transpose().to_excel(
        path, index_label="Groepen"
    )
    return redirect(url_for("student_preferences"))


@app.route("/student_preferences", methods=["GET", "POST"])
@require_process
def student_preferences():
    """Display page where the teacher can add preferences for the student"""
    data_path = get_file_path(
        session["process_id"], "relevant_students_and_groups.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = data.get("candidates", [])
    groups_from = data.get("groups_from", {})

    if request.method == "GET":
        return render_template(
            "student_preferences.html", candidates=candidates, groups_from=groups_from
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

    path = get_file_path(session["process_id"], "groups.xlsx")
    groups_to = pd.read_excel(path, index_col=0).index.tolist()
    buffer = input_writer.create_prefilled_excel(groups_to, df_total)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="voorkeuren.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def write_preferences_to_excel(df, fname, **kwargs):
    """This is a challenge because of MultiLevel index with nans

    kwargs are passed to .to_excel()
    """
    df_header = pd.DataFrame(
        [
            (
                "Leerling",
                "MinimaleTevredenheid",
                "Jongen/meisje",
                "Stamgroep",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Liever niet met",
                "Liever niet met",
                "Niet in",
                "Niet in",
            ),
            (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                1,
                1,
                1,
                2,
            ),
            (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Waarde",
            ),
        ]
    )

    assert df_header.shape[1] == df.shape[1]
    concatted = pd.concat(
        [
            df_header.set_axis(range(df_header.shape[1]), axis="columns"),
            df.set_axis(range(df.shape[1]), axis="columns"),
        ],
        ignore_index=True,
    )
    return concatted.to_excel(fname, index=False, header=False, **kwargs)


@app.route("/upload_preferences", methods=["POST"])
def upload_preferences():
    """Handle the upload of the preferences files"""
    try:
        preferences = file_to_io(request.files["preferences"])
        groups_to_path = get_file_path(session["process_id"], "groups.xlsx")
        groups_to_data, _ = datareader.read_groups_excel(groups_to_path)
        groups_to = list(groups_to_data.keys())
        processor = datareader.VoorkeurenProcessor(preferences)
        processor.process(all_to_groups=groups_to)  # validates further
        preferences_path = get_file_path(session["process_id"], "preferences.xlsx")
        write_preferences_to_excel(processor.input.reset_index(), preferences_path)
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


def extract_selected_per_group(form) -> dict[str, list[str]]:
    """
    Find nr of Jongens and Meisjes in each key in the form that starts with "group_students[]"

    Args:
        form: werkzeug MultiDict (e.g., request.form)

    Returns:
        dict[str, list[str]]: mapping of groupname → list of values of the group
    """
    selected = defaultdict(list)
    for key in form:
        if key.startswith("group_students["):
            groupname = key[len("group_students[") : -1]  # text inside [ ]
            selected[groupname].extend(form.getlist(key))

    gender_distribution = {}
    for g, lst in selected.items():
        c = Counter(lst)
        gender_distribution[g] = {
            "Jongens": c.get("Jongen", 0),
            "Meisjes": c.get("Meisje", 0),
        }
    return gender_distribution


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
        return (
            f"In het {exc.filetype}-bestand zijn niet alle verplichte kolommen gevuld: "
            f"controleer {exc.column_name} bij regel "
            f"{', '.join(exc.failure_cases[', '].astype(str))}"
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


def _handle_failure(exc, task_id):
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
    message = to_validation_message(exc)
    status_dct[task_id]["status_studentdistribution"] = "error"
    status_dct[task_id]["message"] = message


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

    task_id = str(uuid.uuid4())
    temp_storage[task_id] = {}

    def on_update(message):
        status_dct[task_id]["logs"].append(message)

    # pylint: disable=broad-exception-caught
    def run_task():
        try:
            status_dct[task_id]["status_studentdistribution"] = "running"
            result = distribute_students_once(
                preferences_path, groups_to_path, not_together, on_update=on_update
            )
            logger.info("Distributing students finished successfully")
            status_dct[task_id]["status_studentdistribution"] = "done"
            temp_storage[task_id]["groepsindeling"] = result
        except Exception as exc:
            _handle_failure(exc, task_id)

    def create_sociogram(preferences, groups_to):
        try:
            on_update("Sociogram tekenen...")
            groups_to_data, _ = datareader.read_groups_excel(groups_to)
            sg = sociogram.SociogramMaker(preferences, list(groups_to_data.keys()))
            fig, g, pos = sg.plot_sociogram()
            logger.info("Sociogram created")

            fig = sociogram.networkx_to_plotly(g, pos)
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            logger.info("HTML created")
            on_update(
                f'<a href=/sociogram/{task_id} target="_blank" class="button">'
                "Bekijk het sociogram nu!</a>"
            )
            temp_storage[task_id]["sociogram"] = html
        except Exception:
            logger.exception("Could not create sociogram")

    # pylint: enable=broad-exception-caught
    Thread(target=create_sociogram, args=(preferences_bytes, groups_to_path)).start()
    Thread(target=run_task).start()

    return redirect(url_for("processing", task_id=task_id))


@app.route("/status/<task_id>")
def status(task_id):
    """Return status as json"""
    result = status_dct.get(task_id)
    if not result:
        return jsonify({"status_studentdistribution": "unknown"})
    return jsonify(result)


@app.route("/processing/<task_id>")
def processing(task_id):
    """Display processing page"""
    return render_template("processing.html", task_id=task_id)


@app.route("/handle-error", methods=["POST"])
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    flash(data["message"], "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@app.route("/sociogram/<task_id>")
def show_sociogram(task_id):
    """Display sociogram"""
    task = temp_storage.get(task_id, {})
    if "sociogram" not in task:
        flash("Sociogram niet beschikbaar.", "error")
        return redirect(url_for("processes"))
    return render_template("sociogram.html", plotly_div=task["sociogram"])


@app.route("/result/<task_id>")
def result_page(task_id):
    """Display result for single run"""
    task = temp_storage.get(task_id, {})
    if "groepsindeling" not in task:
        flash("Resultaat niet beschikbaar.", "error")
        return redirect(url_for("processes"))
    dataframes = {
        k: df.to_html(na_rep="")
        for k, df in task["groepsindeling"]["dataframes"].items()
    }
    return render_template("result.html", task_id=task_id, dataframes=dataframes)


@app.route("/download/<task_id>")
def download(task_id):
    """Download single groepsindeling"""
    logger.debug(task_id)
    task = temp_storage.get(task_id, {})
    if "groepsindeling" not in task:
        flash("Groepsindeling niet gevonden. Mogelijk nog aan het berekenen", "error")
        return render_template("result.html", task_id=task_id, dataframes={})

    return send_file(
        task["groepsindeling"]["download"],
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
    app.run(debug=True)
