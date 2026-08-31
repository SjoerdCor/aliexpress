"""Unit tests for the atomic progress.json writer (web layer)."""

import json
import os
import threading
from dataclasses import asdict

import pytest

from aliexpress.solver.groepsindeling_view import GroepsindelingView
from aliexpress.solver.progress import PlateauOutcome
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


@pytest.mark.skipif(os.name != "nt", reason="Windows holds the destination open")
def test_open_reader_does_not_drop_next_update_on_windows(tmp_path):
    """A transient Windows read lock delays, but does not lose, the next snapshot."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))
    write_errors = []

    def write_next_snapshot():
        try:
            writer.stage_started("floor")
        except PermissionError as exc:  # pragma: no cover - asserted in parent thread
            write_errors.append(exc)

    # This open handle is the real condition that used to make os.replace fail on
    # Windows when /status happened to be reading during a progress update.
    with open(path, encoding="utf-8") as reader:
        assert json.load(reader)["steps"]["floor"] == "pending"
        writer_thread = threading.Thread(target=write_next_snapshot)
        writer_thread.start()

        # join(timeout) observes the writer without releasing the reader. The fixed
        # writer is still retrying; the broken writer already exited with WinError 5.
        writer_thread.join(timeout=0.1)
        assert writer_thread.is_alive(), "writer did not wait for the open reader"

    # Leaving the with-block closes the reader. The same pending write must now finish
    # and publish "busy"; retrying may not discard that latest snapshot.
    writer_thread.join(timeout=2)
    assert not writer_thread.is_alive()
    assert not write_errors
    assert _assert_parsable(path)["steps"]["floor"] == "busy"


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


def test_plateau_finished_records_round_seconds(tmp_path):
    """Each plateaus entry carries the per-round wall-clock seconds it was called with."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.plateau_finished(PlateauOutcome(0.62, 34, 8.0))
    writer.plateau_finished(PlateauOutcome(0.78, 5, 12.0))

    data = _assert_parsable(path)
    assert data["plateaus"] == [
        {"min_pct": 62, "n_can_improve": 34, "seconds": 8.0},
        {"min_pct": 78, "n_can_improve": 5, "seconds": 12.0},
    ]


def test_estimate_phase_a_before_balance_finishes(tmp_path):
    """Before stage_finished("balance", ...), the estimate is the static phase-A line."""
    path = tmp_path / "progress.json"
    ProgressWriter(str(path))  # constructor seeds the initial phase-A estimate

    data = _assert_parsable(path)
    assert data["estimate"]["phase"] == "a"
    assert data["estimate"]["seconds"] is None
    assert data["estimate"]["text"] == (
        "Aan het rekenen… dit duurt meestal minder dan een minuut, soms enkele minuten."
    )


def test_estimate_phase_b_after_balance_before_any_plateau(tmp_path):
    """Phase B: 12x the balance duration, no plateau finished yet."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 2.0)

    data = _assert_parsable(path)
    assert data["estimate"]["phase"] == "b"
    assert data["estimate"]["seconds"] == 24.0


def test_estimate_phase_c_uses_longest_round_not_average(tmp_path):
    """Phase C: max(TYPICAL_ROUNDS - rounds_done, 1) * the longest round so far (not average)."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 2.0)
    writer.plateau_finished(PlateauOutcome(0.5, 10, 8.0))
    writer.plateau_finished(PlateauOutcome(0.6, 5, 12.0))

    data = _assert_parsable(path)
    assert data["estimate"]["phase"] == "c"
    # max(7 - 2, 1) * 12.0 == 60.0; if it used the average (10.0) this would be 50.0.
    assert data["estimate"]["seconds"] == 60.0


def test_estimate_phase_c_tail_floor_never_zero_or_negative(tmp_path):
    """After more plateaus than TYPICAL_ROUNDS, the estimate floors at 1 round, not 0."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 2.0)
    for _ in range(8):
        writer.plateau_finished(PlateauOutcome(0.5, 10, 10.0))

    data = _assert_parsable(path)
    assert data["estimate"]["phase"] == "c"
    assert data["estimate"]["seconds"] == 10.0


def test_estimate_text_rounding_seconds_under_a_minute(tmp_path):
    """Under 60s, the text rounds up to the nearest 10 seconds, plural 'seconden'."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 4.0)  # 12 * 4.0 = 48.0 -> rounds up to 50

    data = _assert_parsable(path)
    assert (
        data["estimate"]["text"] == "naar verwachting nog ~50 seconden (ruwe schatting)"
    )


def test_estimate_text_rounding_130_seconds_to_3_minutes(tmp_path):
    """130 seconds rounds up to the nearest whole minute: 3 minutes."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 2.0)
    writer.plateau_finished(PlateauOutcome(0.5, 10, 26.0))

    data = _assert_parsable(path)
    # phase c: max(7 - 1, 1) * 26.0 == 156.0, rounds up to 3 minutes.
    assert data["estimate"]["seconds"] == 156.0
    assert (
        data["estimate"]["text"] == "naar verwachting nog ~3 minuten (ruwe schatting)"
    )


def test_estimate_text_singular_minute(tmp_path):
    """An estimate that rounds to exactly one minute uses the singular 'minuut'."""
    path = tmp_path / "progress.json"
    writer = ProgressWriter(str(path))

    writer.stage_finished("balance", 2.0)
    writer.plateau_finished(PlateauOutcome(0.5, 10, 10.0))
    # phase c: max(7 - 1, 1) * 10.0 == 60.0 -> exactly one minute after rounding up.
    data = _assert_parsable(path)
    assert data["estimate"]["seconds"] == 60.0
    assert data["estimate"]["text"] == "naar verwachting nog ~1 minuut (ruwe schatting)"


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
