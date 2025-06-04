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
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        file3 = request.files["file3"]
        output_file = distribute_students_once(file1, file2, file3)
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
    app.run(debug=True)
