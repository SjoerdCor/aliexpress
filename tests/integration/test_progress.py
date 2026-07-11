"""Integration test for the solver-layer progress seam (solver/progress.py).

Runs the real CP-SAT solve on the small synthetic dataset with a recording
listener and asserts the three UI stages fire in order, each with a
plausible non-negative duration. This is the instrumentation slice 7's ETA
calibration will read from later; here it only proves the events are
observed correctly.
"""

import pytest

from aliexpress import main
from aliexpress.solver.progress import ProgressListener


class _RecordingListener(ProgressListener):
    """Appends every event as a tuple for order/value assertions."""

    def __init__(self):
        self.events = []

    def stage_started(self, stage):
        self.events.append(("started", stage))

    def stage_finished(self, stage, seconds):
        self.events.append(("finished", stage, seconds))

    def plateau_finished(self, min_satisfaction, n_can_improve):
        self.events.append(("plateau_finished", min_satisfaction, n_can_improve))

    def tiebreak_started(self):
        self.events.append(("tiebreak_started",))


def test_stages_fire_in_order_with_nonnegative_durations():
    """floor -> balance -> satisfaction, each started then finished, duration >= 0."""
    target_groups = main._read_groups(  # pylint: disable=protected-access
        "tests/integration/groepen_small.xlsx"
    )
    preference_data = main._read_preferences(  # pylint: disable=protected-access
        "tests/integration/voorkeuren_small.xlsx", target_groups.counts
    )
    listener = _RecordingListener()

    main.distribute_students_from_data(
        preference_data, target_groups, listener=listener
    )

    # This test only cares about the stage_started/stage_finished pair; the plateau
    # and tie-break events recorded on the same listener are asserted separately by
    # test_plateaus_fire_in_order_then_tiebreak.
    stage_events = [e for e in listener.events if e[0] in ("started", "finished")]
    stages = [e[1] for e in stage_events]
    assert stages == [
        "floor",
        "floor",
        "balance",
        "balance",
        "satisfaction",
        "satisfaction",
    ]
    kinds = [e[0] for e in stage_events]
    assert kinds == [
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
    ]
    for event in stage_events:
        if event[0] == "finished":
            seconds = event[2]
            assert isinstance(seconds, float)
            assert seconds >= 0


def test_plateaus_fire_in_order_then_tiebreak():
    """Each completed lexmaxmin level reports (min_satisfaction, n_can_improve);
    the tie-break fires exactly once, after the last plateau.

    The small fixture has a single plateau: level 0 pins the minimum at ~0.516 with
    4 students escaping above it; level 1's minimum then comes back above
    strategies.SATISFACTION_MAX (0.8), so lexmaxmin stops there without a second
    plateau_finished (no count stage ran for that level — see the "early return"
    note in strategies._lexmaxmin). The value is the deterministic optimum on the
    small synthetic dataset (voorkeuren_small.xlsx + groepen_small.xlsx) — pinned
    like the other integration tests: if this changes after a model edit, the
    model's behaviour changed.
    """
    target_groups = main._read_groups(  # pylint: disable=protected-access
        "tests/integration/groepen_small.xlsx"
    )
    preference_data = main._read_preferences(  # pylint: disable=protected-access
        "tests/integration/voorkeuren_small.xlsx", target_groups.counts
    )
    listener = _RecordingListener()

    main.distribute_students_from_data(
        preference_data, target_groups, listener=listener
    )

    plateaus = [e for e in listener.events if e[0] == "plateau_finished"]
    tiebreaks = [e for e in listener.events if e[0] == "tiebreak_started"]

    assert [e[1] for e in plateaus] == pytest.approx([0.516129], abs=1e-6)
    assert [e[2] for e in plateaus] == [4]
    assert tiebreaks == [("tiebreak_started",)]
    # The tie-break comes after every plateau is pinned.
    assert listener.events.index(tiebreaks[0]) > listener.events.index(plateaus[-1])
