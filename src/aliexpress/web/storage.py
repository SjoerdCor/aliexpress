"""File-system path helpers for per-school, per-process storage.

All paths are derived from the ``STORAGE_DIR`` config key so that tests can
point at a temporary directory without touching the real instance folder.
"""

import os
from pathlib import Path

from flask import current_app

from .identifiers import validate_identifier


def _confined_child(parent, child, *, label):
    """Return ``child`` resolved below ``parent`` or raise ``PermissionError``."""
    parent = Path(parent).expanduser().resolve(strict=False)
    child = Path(child).expanduser()
    if not child.is_absolute():
        child = parent / child
    resolved = child.resolve(strict=False)
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise PermissionError(
            f"{label} escapes its configured directory: {resolved!s}"
        ) from exc
    return resolved


def get_school_path(school_id):
    """Return a school directory confined to ``STORAGE_DIR``."""
    try:
        school_id = validate_identifier(school_id, label="schoolcode")
    except ValueError as exc:
        raise PermissionError(str(exc)) from exc
    return os.fspath(
        _confined_child(
            current_app.config["STORAGE_DIR"], school_id, label="School storage"
        )
    )


def get_process_path(school_id, process_name):
    """Return the directory for a process, confined to the school's subdirectory.

    Raises ``PermissionError`` when an identifier is unsafe or the resolved path
    would escape the school directory, including through a symlink.
    """
    try:
        process_name = validate_identifier(process_name, label="procesnaam")
    except ValueError as exc:
        raise PermissionError(str(exc)) from exc
    school_dir = get_school_path(school_id)
    return os.fspath(_confined_child(school_dir, process_name, label="Process storage"))


def get_file_path(school_id, process_name, filename):
    """Return a path to ``filename`` confined inside a process directory."""
    process_dir = get_process_path(school_id, process_name)
    return os.fspath(_confined_child(process_dir, filename, label="Process file"))
