from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import uuid

import webbrowser

from src.aliexpress.main import distribute_students_once

temp_storage = {}

app = Flask(__name__)
temp_storage = {}
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


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

        output_file = distribute_students_once(
            preferences, groups_to, not_together, **kwargs
        )
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
