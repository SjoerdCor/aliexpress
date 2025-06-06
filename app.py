from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import uuid
import logging

import webbrowser
from dotenv import load_dotenv

from src.aliexpress.main import distribute_students_once
from src.aliexpress.datareader import ValidationError


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("aliexpress.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
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

FRIENDLY_TEMPLATES = {
    "duplicate_students_preferences": "In voorkeuren is de volgende naam/namen niet uniek: {duplicated}\n Voeg de eerste letter van de achternaam toe om de leerlingen van elkaar te onderscheiden.",
    "wrong_sex": "Onbekende student(en) op rij {row}: {names}",
}


@app.route("/", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        preferences = request.files["preferences"]
        groups_to = request.files["groups_to"]
        not_together = request.files["not_together"]

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

        try:
            output_file = distribute_students_once(
                preferences, groups_to, not_together, **kwargs
            )
        except ValidationError as e:
            template = FRIENDLY_TEMPLATES.get(e.code)
            message = template.format(**e.context)

            base_msg = "Kon bestanden niet goed inlezen.\n"
            flash(base_msg + message, "error")
            return render_template("upload.html")
        except Exception as e:
            logger.exception("Uncaught exception")
            msg = "Er is iets onverwachts misgegaan. Het probleem is gelogd. laat de maker dit onderzoeken."
            flash(msg, "error")

            return render_template("upload.html", previous_data=request.form)
        file_id = str(uuid.uuid4())
        temp_storage[file_id] = output_file

        return redirect(url_for("result_page", file_id=file_id))

    return render_template("upload.html")


@app.route("/result/<file_id>")
def result_page(file_id):
    return render_template("result.html", file_id=file_id)


@app.route("/download/<file_id>")
def download(file_id):
    file_buffer = temp_storage.get(file_id)
    if file_buffer is None:
        return "File not found", 404

    return send_file(
        file_buffer,
        as_attachment=True,
        download_name="results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run()
