"""Thin launcher: creates the Flask app and exposes it for WSGI servers."""

from aliexpress import create_app
from aliexpress.main import serve_foreground

app = create_app()

if __name__ == "__main__":
    serve_foreground(app)
