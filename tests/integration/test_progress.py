"""Integration test for the solver-layer progress seam (solver/progress.py).

Runs the real CP-SAT solve on the small synthetic dataset with a recording
listener and asserts the three UI stages fire in order, each with a
plausible non-negative duration. This is the instrumentation slice 7's ETA
calibration will read from later; here it only proves the events are
observed correctly.
"""

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

    stages = [e[1] for e in listener.events]
    assert stages == [
        "floor",
        "floor",
        "balance",
        "balance",
        "satisfaction",
        "satisfaction",
    ]
    kinds = [e[0] for e in listener.events]
    assert kinds == [
        "started",
        "finished",
        "started",
        "finished",
        "started",
        "finished",
    ]
    for event in listener.events:
        if event[0] == "finished":
            seconds = event[2]
            assert isinstance(seconds, float)
            assert seconds >= 0
