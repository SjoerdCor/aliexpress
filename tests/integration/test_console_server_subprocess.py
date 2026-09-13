"""Integration test for the installed ``ali-express serve`` subprocess."""

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

_SUBPROCESS_OUTPUT_EOF = object()
_SERVER_STARTUP_TIMEOUT = 30


def _read_subprocess_output(stream, output_queue):
    """Copy subprocess output to a queue without blocking the test thread."""
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    except (OSError, ValueError):
        # Cleanup may close the pipe to release a reader that outlived the process.
        pass
    finally:
        output_queue.put(_SUBPROCESS_OUTPUT_EOF)


def _drain_subprocess_output(output_queue, output_lines, timeout=0):
    """Append queued output until EOF or a bounded timeout is reached."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if timeout:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                line = output_queue.get(timeout=remaining)
            else:
                line = output_queue.get_nowait()
        except queue.Empty:
            return False
        if line is _SUBPROCESS_OUTPUT_EOF:
            return True
        output_lines.append(line)


def _stop_subprocess(process):
    """Stop a test subprocess with bounded, cross-platform cleanup waits."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Do not let test cleanup hang forever if the OS cannot reap the child.
            pass


def _server_environment(project_root, tmp_path):
    """Return isolated settings for the subprocess server."""
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
    return environment


def _server_command(tmp_path):
    """Return the command that starts the CLI through its installed module."""
    # Admin password hashing is covered separately. Keep this process-lifecycle smoke test
    # focused by avoiding the platform-dependent cost of the production scrypt default.
    bootstrap = (
        "from functools import partial; "
        "from werkzeug.security import generate_password_hash; "
        "import aliexpress; "
        "import aliexpress.web.admin_seed as admin_seed; "
        "admin_seed.generate_password_hash = partial("
        "generate_password_hash, method='pbkdf2:sha256:1'); "
        f"aliexpress.get_instance_path = lambda: {str(tmp_path)!r}; "
        "import aliexpress.main as main_module; "
        "main_module.main()"
    )
    return [
        sys.executable,
        "-c",
        bootstrap,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--no-browser",
    ]


def _start_output_reader(process):
    """Start continuous draining of the merged subprocess output."""
    output_queue = queue.Queue()
    output_lines = []
    reader = threading.Thread(
        target=_read_subprocess_output,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    reader.start()
    return output_queue, output_lines, reader


def _readiness_failure(output_queue, output_lines, message):
    """Fail with all output available when the server cannot become ready."""
    _drain_subprocess_output(output_queue, output_lines, timeout=1)
    pytest.fail(f"{message}:\n{''.join(output_lines)}")


def _wait_for_server_url(process, output_queue, output_lines):
    """Read startup output until the server reports its bound URL."""
    deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _readiness_failure(
                output_queue,
                output_lines,
                f"serve did not report readiness within {_SERVER_STARTUP_TIMEOUT} seconds",
            )
        try:
            line = output_queue.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            if process.poll() is not None:
                _readiness_failure(
                    output_queue,
                    output_lines,
                    "serve exited before becoming ready",
                )
            continue
        if line is _SUBPROCESS_OUTPUT_EOF:
            if process.poll() is not None:
                _readiness_failure(
                    output_queue,
                    output_lines,
                    "serve exited before becoming ready",
                )
            continue
        output_lines.append(line)
        match = re.search(r"Server gestart op (?P<url>https?://[^\s]+)", line)
        if match:
            return match.group("url")


def _assert_server_responds(server_url, output_queue, output_lines):
    """Verify that the announced URL serves a non-empty HTTP response."""
    try:
        with urlopen(server_url, timeout=2) as response:
            body = response.read()
            _drain_subprocess_output(output_queue, output_lines)
            assert response.status == 200, (
                f"unexpected HTTP status from {server_url}: "
                f"{response.status}\n{''.join(output_lines)}"
            )
            assert body, f"empty HTTP response body from {server_url}"
    except (URLError, TimeoutError, OSError) as exc:
        _drain_subprocess_output(output_queue, output_lines)
        pytest.fail(
            f"request to {server_url} failed: {exc}\n" f"{''.join(output_lines)}"
        )


def _interrupt_server(process):
    """Send the platform-appropriate interactive interrupt to the server."""
    if os.name == "nt":
        process.send_signal(getattr(signal, "CTRL_C_EVENT"))
    else:
        process.send_signal(signal.SIGINT)


def _wait_for_clean_stop(process, output_queue, output_lines):
    """Wait for graceful interrupt handling, reporting output on timeout."""
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _drain_subprocess_output(output_queue, output_lines)
        pytest.fail(
            "serve did not stop within five seconds after SIGINT:\n"
            f"{''.join(output_lines)}"
        )


def _finish_output_cleanup(process, output_queue, output_lines, reader, output_eof):
    """Bound cleanup of the process, reader thread, and remaining output."""
    _stop_subprocess(process)
    if not output_eof:
        output_eof = (
            _drain_subprocess_output(output_queue, output_lines, timeout=1)
            or output_eof
        )
    if reader.is_alive():
        process.stdout.close()
    reader.join(timeout=1)
    _drain_subprocess_output(output_queue, output_lines)
    return output_eof


def test_serve_subprocess_smoke_and_clean_stop(tmp_path):
    """The installed module serves HTTP from another working directory and stops cleanly."""
    project_root = Path(__file__).resolve().parents[2]
    environment = _server_environment(project_root, tmp_path)
    command = _server_command(tmp_path)
    with subprocess.Popen(
        command,
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    ) as process:
        output_queue, output_lines, reader = _start_output_reader(process)
        output_eof = False
        try:
            server_url = _wait_for_server_url(process, output_queue, output_lines)
            _assert_server_responds(server_url, output_queue, output_lines)
            _interrupt_server(process)
            _wait_for_clean_stop(process, output_queue, output_lines)
            output_eof = _drain_subprocess_output(output_queue, output_lines, timeout=1)
            if not output_eof:
                pytest.fail(
                    "serve output did not reach EOF within one second after stopping:\n"
                    f"{''.join(output_lines)}"
                )
            output = "".join(output_lines)
            assert process.returncode == 0, output
            assert "Server wordt gestopt" in output, output
        finally:
            _finish_output_cleanup(
                process, output_queue, output_lines, reader, output_eof
            )
