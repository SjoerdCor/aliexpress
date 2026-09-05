"""Tests for the platform-neutral ``ali-express`` console command."""

import importlib.metadata
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from click.testing import CliRunner

import aliexpress.main as main_module
from aliexpress.main import main


def _registered_entry_point(name):
    return next(
        entry_point
        for entry_point in importlib.metadata.entry_points(group="console_scripts")
        if entry_point.name == name
    )


def test_registered_console_entrypoints_load_the_same_command():
    """Both published spellings must resolve to the canonical Click command."""
    assert _registered_entry_point("ali-express").load() is main
    assert _registered_entry_point("aliexpress").load() is main


def test_help_lists_the_solve_command():
    """The canonical command must be discoverable without touching application data."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "solve" in result.output
    assert "serve" in result.output


def test_solve_delegates_to_existing_solver(monkeypatch):
    """``solve`` must reuse the established file-reading/distribution function."""
    called = []

    def fake_distribute_students_once():
        called.append(True)

    monkeypatch.setattr(
        main_module, "distribute_students_once", fake_distribute_students_once
    )

    result = CliRunner().invoke(main, ["solve"])

    assert result.exit_code == 0, result.output
    assert called == [True]


def test_serve_passes_host_port_and_browser_choice_to_launcher(monkeypatch):
    """``serve`` must expose the portable launcher's user-facing options."""
    called = []

    monkeypatch.setattr(
        main_module,
        "serve_configured",
        lambda host, port, open_browser: called.append((host, port, open_browser)),
    )

    result = CliRunner().invoke(
        main,
        ["serve", "--host", "127.0.0.2", "--port", "43210", "--no-browser"],
    )

    assert result.exit_code == 0, result.output
    assert called == [("127.0.0.2", 43210, False)]


def test_serve_foreground_opens_browser_after_binding_without_debug_or_reloader(
    monkeypatch,
):
    """The browser must only open after the server is listening in local user mode."""
    events = []
    serving = threading.Event()
    release = threading.Event()

    class FakeApplication:  # pylint: disable=too-few-public-methods
        """Minimal Flask-like object used to inspect local-server configuration."""

        config = {"DEBUG": True, "TESTING": True}
        debug = True

        def run(self, **_kwargs):
            """Fail if the Flask development runner is used."""
            pytest.fail("serve must not use Flask's development runner")

    class FakeServer:
        """Minimal Werkzeug-like server with observable lifecycle events."""

        server_port = 43211

        def serve_forever(self):
            """Stay alive until the fake browser releases the server."""
            events.append("serving")
            serving.set()
            release.wait(timeout=2)

        def shutdown(self):
            """Release the fake server's serving loop."""
            release.set()

        def server_close(self):
            """Record that the listening socket would have been closed."""
            events.append("closed")

    def fake_make_server(host, port, application, **kwargs):
        events.append(("bound", host, port, application, kwargs))
        return FakeServer()

    def fake_browser(url):
        assert serving.is_set()
        events.append(("browser", url))
        release.set()
        return True

    monkeypatch.setattr(main_module, "make_server", fake_make_server)
    monkeypatch.setattr(main_module.webbrowser, "open", fake_browser)

    application = FakeApplication()
    main_module.serve_foreground(application, host="127.0.0.1", port=43210)

    assert application.config == {"DEBUG": False, "TESTING": False}
    assert application.debug is False
    assert events == [
        ("bound", "127.0.0.1", 43210, application, {"threaded": True}),
        "serving",
        ("browser", "http://127.0.0.1:43211"),
        "closed",
    ]


def test_serve_reports_a_busy_port_without_touching_browser(monkeypatch):
    """A bind failure must be a clear CLI error, not a process-management action."""
    browser_calls = []

    monkeypatch.setattr(
        main_module,
        "create_app",
        lambda: SimpleNamespace(config={}, debug=True),
    )

    def fake_make_server(*_args, **_kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(main_module, "make_server", fake_make_server)
    monkeypatch.setattr(
        main_module.webbrowser,
        "open",
        lambda _url: browser_calls.append(True),
    )

    result = CliRunner().invoke(main, ["serve", "--port", "43212", "--no-browser"])

    assert result.exit_code != 0
    assert "poort" in result.output.lower()
    assert "bezet" in result.output.lower()
    assert not browser_calls


def test_serve_subprocess_smoke_and_clean_stop(tmp_path):
    """The installed module serves HTTP from another working directory and stops cleanly."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
            "ALIEXPRESS_ENV": "local",
            "SECRET_KEY": "slice-three-secret",
            "ADMIN_PASSWORD": "A-long-random-slice-three-admin-password-42!",
            "PYTHONPATH": os.pathsep.join(
                [str(project_root / "src"), str(project_root)]
            ),
        }
    )
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from aliexpress.main import main; main()",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-browser",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    ) as process:
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read()
                    pytest.fail(f"serve exited before becoming ready: {output}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as response:
                        assert response.status == 200
                        break
                except (URLError, TimeoutError, OSError):
                    time.sleep(0.05)
            else:
                pytest.fail("serve did not become ready within ten seconds")

            if os.name == "nt":
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
            else:
                process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=5)
            assert process.returncode == 0, output
            assert "Server wordt gestopt" in output
        finally:
            if process.poll() is None:
                process.terminate()
