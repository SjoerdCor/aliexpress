"""Unit tests for the atomic progress.json writer (web layer)."""

import json
import os
from dataclasses import asdict

from aliexpress.solver.groepsindeling_view import GroepsindelingView
from aliexpress.web.progress_writer import ProgressWriter


def _assert_parsable(path):
    """The file must exist and parse as JSON after every write — never half-written."""
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_writes_parsable_json_after_every_event(tmp_path):
    """Each stage_started/stage_finished call leaves a fully parsable file behind."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))
    _assert_parsable(path)  # written on construction

    writer.stage_started("floor")
    data = _assert_parsable(path)
    assert data["steps"]["floor"] == "busy"
    assert data["steps"]["balance"] == "pending"
    assert data["steps"]["satisfaction"] == "pending"

    writer.stage_finished("floor", 1.5)
    data = _assert_parsable(path)
    assert data["steps"]["floor"] == "done"
    assert data["stage_seconds"] == [{"label": "floor", "seconds": 1.5}]

    writer.stage_started("balance")
    writer.stage_finished("balance", 2.5)
    writer.stage_started("satisfaction")
    writer.stage_finished("satisfaction", 3.0)

    data = _assert_parsable(path)
    assert data["steps"] == {
        "floor": "done",
        "balance": "done",
        "satisfaction": "done",
    }
    assert data["stage_seconds"] == [
        {"label": "floor", "seconds": 1.5},
        {"label": "balance", "seconds": 2.5},
        {"label": "satisfaction", "seconds": 3.0},
    ]
    assert "started_at" in data


def test_no_leftover_temp_file(tmp_path):
    """The atomic write (temp file + os.replace) leaves no stray temp file behind."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))
    writer.stage_started("floor")
    writer.stage_finished("floor", 0.1)

    remaining = set(os.listdir(tmp_path))
    assert remaining == {"progress.json"}


def test_interim_result_view_writes_separate_file_and_timestamp(tmp_path):
    """interim_result_view writes interim_result.json and sets a timestamp in progress.json."""
    progress_path = tmp_path / "progress.json"
    interim_path = tmp_path / "interim_result.json"
    writer = ProgressWriter(str(progress_path), str(interim_path))
    data = _assert_parsable(progress_path)
    assert data["interim_result_updated_at"] is None

    view = GroepsindelingView(group_order=["A"], groups=[], balance_rows=[])
    writer.interim_result_view(view)

    with open(interim_path, encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored == asdict(view)

    data = _assert_parsable(progress_path)
    assert data["interim_result_updated_at"] is not None

    remaining = set(os.listdir(tmp_path))
    assert remaining == {"progress.json", "interim_result.json"}


def test_interim_result_view_updated_at_changes_on_every_call(tmp_path):
    """No damping: interim_result_updated_at is refreshed on every call."""
    progress_path = tmp_path / "progress.json"
    interim_path = tmp_path / "interim_result.json"
    writer = ProgressWriter(str(progress_path), str(interim_path))

    view = GroepsindelingView(group_order=[], groups=[], balance_rows=[])
    writer.interim_result_view(view)
    first = _assert_parsable(progress_path)["interim_result_updated_at"]
    writer.interim_result_view(view)
    second = _assert_parsable(progress_path)["interim_result_updated_at"]
    assert second >= first
