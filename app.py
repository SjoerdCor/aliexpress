"""Thin launcher: creates the Flask app and exposes it for WSGI servers."""

import webbrowser

from aliexpress import create_app

app = create_app()

if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=app.config.get("DEBUG", False))
