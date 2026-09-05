"""Thin launcher for WSGI imports and direct local execution."""

if __name__ == "__main__":
    from aliexpress.main import serve_configured

    serve_configured()
else:
    from aliexpress import create_app

    app = create_app()
