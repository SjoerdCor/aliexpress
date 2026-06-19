"""The flask server that governs the app"""

import json
import logging
import os
import webbrowser
from io import BytesIO
from threading import Thread

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
from flask_login import login_required

from aliexpress import candidatedetermination, datareader, input_writer, sociogram
from aliexpress.admin import admin_bp
from aliexpress.cli import admins as admins_cli
from aliexpress.cli import schools as schools_cli
from aliexpress.errors import (
    CouldNotReadFileError,
    DuplicateNameError,
    FeasibilityError,
    ValidationError,
)
from aliexpress.extensions import db, limiter, login_manager
from aliexpress.form_parsers import parse_groups_to_form
from aliexpress.http_errors import register_error_handlers
from aliexpress.logging_config import add_file_handler, configure_logging
from aliexpress.main import distribute_students_once
from aliexpress.models import LogLine, Process, Run
from aliexpress.routes.auth import auth_bp, effective_school_id, load_user
from aliexpress.routes.processes import processes_bp, require_process
from aliexpress.storage import get_file_path
from aliexpress.validation_messages import to_validation_message

configure_logging()
logger = logging.getLogger("aliexpress.app")  # file handler added below


load_dotenv()

env = os.getenv("FLASK_ENV", "production")
if env == "development":
    from aliexpress.appconfig import DevelopmentConfig as ConfigClass
else:
    from aliexpress.appconfig import ProductionConfig as ConfigClass


def ensure_secret_key(flask_app):
    """Refuse to start without a signing key.

    An empty SECRET_KEY makes session cookies unsignable (and login state forgeable),
    so fail fast at startup rather than at the first request. Development supplies a
    fallback key in DevelopmentConfig, so this only bites a misconfigured production deploy.
    """
    if not flask_app.config.get("SECRET_KEY"):
        raise RuntimeError(
            "SECRET_KEY is not set; refusing to start. Set it in the environment (.env)."
        )


app = Flask(__name__)
app.config.from_object(ConfigClass)
ensure_secret_key(app)
LOG_DIR = os.path.join(app.instance_path, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
add_file_handler(os.path.join(LOG_DIR, "aliexpress.log"))
app.config["STORAGE_DIR"] = os.path.join(app.instance_path, "storage")
os.makedirs(app.config["STORAGE_DIR"], exist_ok=True)
logger.debug("Created dir if not exists: %s", app.config["STORAGE_DIR"])

# A relative SQLite URI is resolved against the instance folder, which must exist first.
os.makedirs(app.instance_path, exist_ok=True)
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message = "Je moet ingelogd zijn om deze pagina te bekijken."
login_manager.login_message_category = "error"

# Rate-limit the login route against brute force. In-memory storage suffices for a single
# process; a multi-process deployment (the future EU server) needs a shared backend (Redis).
limiter.init_app(app)

with app.app_context():
    db.create_all()


login_manager.user_loader(load_user)

app.cli.add_command(schools_cli, "schools")
app.cli.add_command(admins_cli, "admins")
app.register_blueprint(admin_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(processes_bp)


register_error_handlers(app)


@app.route("/")
def home():
    """Display home page"""
    return render_template("home.html")


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


def file_to_io(uploaded_file) -> BytesIO:
    """Get file as BytesIO"""
    return BytesIO(uploaded_file.read())


@app.route("/input_templates/<path:filename>")
def download_template(filename):
    """Download the template sheets"""
    return send_from_directory("input_templates", filename, as_attachment=True)


@app.route("/download_preferences")
@login_required
@require_process
def download_preferences():
    """Download the filled-in preferences file as the teacher uploaded it."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "preferences.xlsx")
    if not os.path.exists(path):
        logger.warning(
            "Download of filled-in preferences requested but none stored for process %s",
            process_id,
        )
        abort(404)
    logger.info("Serving stored preferences upload for process %s", process_id)
    return send_file(
        path,
        as_attachment=True,
        download_name="voorkeuren (ingevuld).xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/upload_edexml", methods=["GET", "POST"])
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
        return redirect(url_for("upload_edexml"))
    return redirect(url_for("groups_to_page"))


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


@app.route("/groups_to", methods=["GET", "POST"])
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
        return redirect(url_for("groups_to_page"))

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
    return redirect(url_for("student_preferences"))


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


@app.route("/student_preferences", methods=["GET", "POST"])
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


@app.route("/upload_preferences", methods=["POST"])
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
            return redirect(url_for("not_together_page"))
        flash("Upload eerst het ingevulde bestand om verder te gaan.", "error")
        return redirect(url_for("student_preferences"))
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
        logger.info(
            "Preferences accepted for process %s: %d students",
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
@login_required
@require_process
def not_together_page():
    """Display and process the not-together rules page"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    preferences_path = get_file_path(school_id, process_id, "preferences.xlsx")
    groups_to_path = get_file_path(school_id, process_id, "groups.xlsx")

    try:
        groups_to, _ = datareader.read_groups_excel(groups_to_path)
        processor = datareader.VoorkeurenProcessor(preferences_path)
        processor.process(all_to_groups=list(groups_to.keys()))
        # Show names as entered in the dropdown; matching happens on the key on submit.
        logger.debug(processor.student_display)
        students = sorted(processor.student_display.values())
        logger.debug(", ".join(students))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _flash_upload_error(exc)
        return redirect(url_for("student_preferences"))
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
        return redirect(url_for("not_together_page"))

    _save_not_together(school_id, process_id, rules)
    return redirect(url_for("start_distribution"))


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


def _run_solve_thread(school_id, process_name, groups_to_path, not_together, run_id):
    """Background thread: run the solver and write result artifacts.

    Each call creates its own app context and DB session. ``run_id`` is the integer PK
    of the Run row so log lines can be appended without a school+name query per line.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=run_id, text=message))
        db.session.commit()

    with app.app_context():
        try:  # pylint: disable=broad-exception-caught
            preferences_path = get_file_path(
                school_id, process_name, "preferences.xlsx"
            )
            Process.by_name(school_id, process_name).run.set_status("running")
            result = distribute_students_once(
                preferences_path, groups_to_path, not_together, on_update=on_update
            )
            logger.info("Distributing students finished successfully")
            # Write artifacts before flipping to "done" so the result page never
            # races ahead of the files it needs.
            _write_result_files(school_id, process_name, result)
            Process.by_name(school_id, process_name).run.set_status("done")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _handle_failure(exc, school_id, process_name)


def _create_sociogram_thread(preferences, groups_to, school_id, process_name, run_id):
    """Background thread: build and write the Plotly sociogram HTML.

    Runs concurrently with the solver; log lines are appended via ``run_id`` just like
    the solver thread does.
    """

    def on_update(message):
        db.session.add(LogLine(run_id=run_id, text=message))
        db.session.commit()

    with app.app_context():
        try:  # pylint: disable=broad-exception-caught
            on_update("Sociogram tekenen...")
            groups_to_data, _ = datareader.read_groups_excel(groups_to)
            sg = sociogram.SociogramMaker(preferences, list(groups_to_data.keys()))
            fig, g, pos = sg.plot_sociogram()
            logger.info("Sociogram created")
            fig = sociogram.networkx_to_plotly(g, pos)
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            logger.info("HTML created")
            with open(
                get_file_path(school_id, process_name, "sociogram.html"),
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


@app.route("/start_distribution", methods=["GET"])
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

    with open(get_file_path(school_id, process_name, "preferences.xlsx"), "rb") as fh:
        preferences_bytes = BytesIO(fh.read())

    proc = Process.by_name(school_id, process_name)
    Run.reset(proc.id)
    for _stale in ("results.xlsx", "result_tables.json", "sociogram.html"):
        _path = get_file_path(school_id, process_name, _stale)
        if os.path.exists(_path):
            os.remove(_path)
    # Capture the integer PK before spawning threads so they append log lines without
    # a school+name lookup on every on_update call.
    run_id = proc.id
    Thread(
        target=_create_sociogram_thread,
        args=(preferences_bytes, groups_to_path, school_id, process_name, run_id),
    ).start()
    Thread(
        target=_run_solve_thread,
        args=(school_id, process_name, groups_to_path, not_together, run_id),
    ).start()
    return redirect(url_for("processing"))


@app.route("/status")
@login_required
@require_process
def status():
    """Return the current process's run status and log lines as JSON."""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_name = session["process_id"]
    proc = Process.by_name(school_id, process_name)
    if proc is None or proc.run is None:
        return jsonify({"status_studentdistribution": "unknown", "logs": []})
    run = proc.run
    payload = {
        "status_studentdistribution": run.status,
        "logs": [line.text for line in run.log_lines],
    }
    if run.status == "error" and run.message:
        payload["message"] = run.message
    return jsonify(payload)


@app.route("/processing")
@login_required
@require_process
def processing():
    """Display processing page"""
    return render_template("processing.html")


@app.route("/handle-error", methods=["POST"])
@login_required
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    flash(data["message"], "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@app.route("/sociogram")
@login_required
@require_process
def show_sociogram():
    """Display the sociogram for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "sociogram.html")
    if not os.path.exists(path):
        flash("Sociogram niet beschikbaar.", "error")
        return redirect(url_for("processes.index"))
    with open(path, encoding="utf-8") as fh:
        plotly_div = fh.read()
    return render_template("sociogram.html", plotly_div=plotly_div)


@app.route("/result")
@login_required
@require_process
def result_page():
    """Display result for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "result_tables.json")
    if not os.path.exists(path):
        flash("Resultaat niet beschikbaar.", "error")
        return redirect(url_for("processes.index"))
    with open(path, encoding="utf-8") as fh:
        dataframes = json.load(fh)
    return render_template("result.html", dataframes=dataframes)


@app.route("/download")
@login_required
@require_process
def download():
    """Download the groepsindeling for the current process"""
    school_id = effective_school_id()
    if school_id is None:
        return redirect(url_for("admin.dashboard"))
    process_id = session["process_id"]
    path = get_file_path(school_id, process_id, "results.xlsx")
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
@login_required
def done():
    """Show done page"""
    return render_template("done.html")


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=env == "development")
