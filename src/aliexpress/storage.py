"""File-system path helpers for per-school, per-process storage.

All paths are derived from the ``STORAGE_DIR`` config key so that tests can
point at a temporary directory without touching the real instance folder.
"""

import os

from flask import current_app


def get_process_path(school_id, process_name):
    """Return the directory for a process, confined to the school's subdirectory.

    Raises ``PermissionError`` when the resolved path would escape the school
    directory (path-traversal guard).
    """
    base = current_app.config["STORAGE_DIR"]
    school_dir = os.path.normpath(os.path.join(base, school_id))
    path = os.path.normpath(os.path.join(school_dir, process_name))
    if os.path.commonpath([path, school_dir]) != school_dir:
        raise PermissionError(
            f"Path traversal detected: {path!r} escapes {school_dir!r}"
        )
    return path


def get_file_path(school_id, process_name, filename):
    """Return the absolute path to ``filename`` inside a process directory."""
    return os.path.join(get_process_path(school_id, process_name), filename)
