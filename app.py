"""The flask server that governs the app"""

from collections import defaultdict
from io import BytesIO
import logging
import os
from threading import Thread
import uuid
import webbrowser

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    send_from_directory,
    session,
    flash,
    jsonify,
)

from src.aliexpress.main import distribute_students_once
from src.aliexpress.errors import FeasibilityError, ValidationError


def setup_logger():
    """Create logging instance"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("aliexpress.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


load_dotenv()

env = os.getenv("FLASK_ENV", "production")
if env == "development":
    from src.aliexpress.appconfig import DevelopmentConfig as ConfigClass
else:
    from src.aliexpress.appconfig import ProductionConfig as ConfigClass

app = Flask(__name__)
app.config.from_object(ConfigClass)


temp_storage = {}
results = defaultdict(lambda: {"status": "pending", "logs": []})

FRIENDLY_TEMPLATES = {
    "duplicate_students_preferences": (
        "In voorkeuren is de volgende naam/namen niet uniek: {duplicated}\n"
        "Voeg de eerste letter van de achternaam toe om de leerlingen van elkaar te onderscheiden."
    ),
    "wrong_sex": "Verkeerd ingevuld geslacht voor {students_incorrect_sex}",
    "wrong_index_names_preferences": (
        "Het voorkeurenbestand kon niet worden verwerkt. Gebruik het meeste recente template"
    ),
    "wrong_column_names_preferences": (
        "Het voorkeurenbestand kon niet worden verwerkt. Gebruik het meeste recente template"
    ),
    "negative_weights_preferences": "Er zijn negatieve gewichten in het voorkeurenbestand.",
    "invalid_values_preferences": (
        "Onbekende leerling of groep in categorie {wishtype}: {invalid_values}"
    ),
    "duplicated_students_not_together": (
        "In het niet-samen-bestand wordt in de {row}e groep dezelfde leerling meerdere "
        "keren genoemd: {duplicated_students}"
    ),
    "unknown_students_not_together": (
        "In het niet-samen-bestand wordt in de {row}e groep komt {unknown_students} voor, "
        "die niet in het voorkeurenbestand voorkomt"
    ),
    "too_strict_not_together": (
        "In het niet-samen-bestand op de {row}e rij is de maximale groepsgrootte te klein: "
        "bij {n_students} leerlingen en {n_groups} groepen moeten er minmiaal "
        "{acceptabel_max_samen} bij elkaar mogen, niet {max_aantal_samen}"
    ),
    "wrong_columns_preferences": (
        "Het voorkeuren-bestand heeft de verkeerde kolommen. Controleer of je het goede"
        " bestand hebt geupload en het meest recente template hebt gebruikt. "
        "\n{wrong_columns}"
    ),
    "wrong_columns_not_together": (
        "Het niet-samen-bestand heeft de verkeerde kolommen. Controleer of je het goede"
        " bestand hebt geupload en het meest recente template hebt gebruikt. "
        "\n{wrong_columns}"
    ),
    "wrong_columns_groups_to": (
        "Het groepen-bestand heeft de verkeerde kolommen. Controleer of je het goede "
        "bestand hebt geupload en het meeste recente template hebt gebruikt. "
        "\n{wrong_columns}"
    ),
    "infeasible_problem": (
        "Met deze vereiste klassenbalans en verdeling van leerlingen die overgaan is het"
        "niet mogelijk. Overweeg de volgende versoepelingen om het probleem wel op te "
        "lossen:\n {possible_improvement}"
    ),
    "empty_mandatory_columns_preferences": (
        "In het voorkeuren-bestand zijn niet alle verplichte kolommen gevuld: controleer {failed_columns}"
    ),
    "empty_mandatory_columns_groups_to": (
        "In het groepen-bestand zijn niet alle verplichte kolommen gevuld: controleer {failed_columns}"
    ),
    "empty_mandatory_columns_not_together": (
        "In het niet-samen-bestand zijn niet alle verplichte kolommen gevuld: controleer {failed_columns}"
    ),
    "could_not_read": (
        "Het {filetype}-bestand kon niet worden ingelezen. Controleer of je het juiste bestand hebt geupload"
    ),
    "internal_error": (
        "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
        "Laat de maker dit onderzoeken."
    ),
}


def file_to_io(uploaded_file) -> BytesIO:
    """Get file as BytesIO"""
    return BytesIO(uploaded_file.read())


@app.route("/input_templates/<path:filename>")
def download_template(filename):
    """Download the template sheets"""
    return send_from_directory("input_templates", filename, as_attachment=True)


@app.route("/fillin")
def fillin():
    """Display the fillin page"""
    return render_template("fillin.html")


@app.route("/", methods=["GET", "POST"])
def upload_files():
    """Handle upload page, including form submission"""
    if request.method == "POST":
        logger.info("Submitted")
        preferences = file_to_io(request.files["preferences"])
        groups_to = file_to_io(request.files["groups_to"])
        not_together = file_to_io(request.files["not_together"])

        try:
            max_diff_n_students_total = int(request.form["max_diff_n_students_total"])
            max_diff_n_students_year = int(request.form["max_diff_n_students_year"])
            max_imbalance_boys_girls_total = int(
                request.form["max_imbalance_boys_girls_total"]
            )
            max_imbalance_boys_girls_year = int(
                request.form["max_imbalance_boys_girls_year"]
            )
            max_clique = int(request.form["max_clique"])
            max_clique_sex = int(request.form["max_clique_sex"])
        except (KeyError, ValueError):
            return "Alle parameters moeten positieve gehele getallen zijn", 400

        kwargs = {
            "max_diff_n_students_total": max_diff_n_students_total,
            "max_diff_n_students_year": max_diff_n_students_year,
            "max_imbalance_boys_girls_total": max_imbalance_boys_girls_total,
            "max_imbalance_boys_girls_year": max_imbalance_boys_girls_year,
            "max_clique": max_clique,
            "max_clique_sex": max_clique_sex,
        }
        session["config"] = kwargs

        def on_update(message):
            results[task_id]["logs"].append(message)

        logger.info("Starting distribution...")

        task_id = str(uuid.uuid4())

        def run_task(*args, **kwargs):
            try:
                results[task_id]["status"] = "running"
                result = distribute_students_once(*args, **kwargs, on_update=on_update)
                logger.info("Distributing students finished successfully")
                results[task_id]["status"] = "done"
                temp_storage[task_id] = result

            except ValidationError as e:
                logger.exception("Files are incorrect")
                results[task_id]["status"] = "error"
                results[task_id]["error_code"] = e.code
                results[task_id]["error_context"] = e.context
            except FeasibilityError as e:
                logger.exception("Problem is infeasible")
                results[task_id]["status"] = "error"
                results[task_id]["error_code"] = e.code
                results[task_id]["error_context"] = e.context
            except Exception as e:
                logger.exception("Uncaught exception")
                results[task_id]["status"] = "error"
                results[task_id]["error_code"] = "internal_error"
                results[task_id]["error_context"] = {"details": str(e)}

        Thread(
            target=run_task,
            args=(preferences, groups_to, not_together),
            kwargs=kwargs,
        ).start()

        return redirect(url_for("processing", task_id=task_id))
    logger.info("Showing upload page")
    return render_template("upload.html")


@app.route("/status/<task_id>")
def status(task_id):
    """Return status as json"""
    result = results.get(task_id)
    if not result:
        return jsonify({"status": "unknown"})
    return jsonify(result)


@app.route("/processing/<task_id>")
def processing(task_id):
    """Display processing page"""
    return render_template("processing.html", task_id=task_id)


@app.route("/handle-error", methods=["POST"])
def handle_error():
    """Show information about errors to user"""
    data = request.get_json()
    code = data.get("code")
    context = data.get("context", {})

    template = FRIENDLY_TEMPLATES.get(code, "Er ging iets fout.")
    message = template.format(**context)
    flash(message, "error")

    # By not redirecting here but in JS, this is more flexible
    return "", 204


@app.route("/result/<task_id>")
def result_page(task_id):
    """Display result for single run"""
    return render_template("result.html", task_id=task_id)


@app.route("/download/<task_id>")
def download(task_id):
    """Download single groepsindeling"""
    file_buffer = temp_storage.get(task_id)
    logger.debug(task_id)
    if file_buffer is None:
        flash("Groepsindeling niet gevonden. Mogelijk nog aan het berekenen", "error")
        return render_template("result.html", task_id=task_id)

    return send_file(
        file_buffer,
        as_attachment=True,
        download_name="results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run()
