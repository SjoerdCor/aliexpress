"""Integration test for the solver-layer progress seam (solver/progress.py).

Runs the real CP-SAT solve on the small synthetic dataset with a recording
listener and asserts the three UI stages fire in order, each with a
plausible non-negative duration. This is the instrumentation slice 7's ETA
calibration will read from later; here it only proves the events are
observed correctly.
"""

import pytest

from aliexpress import main
from aliexpress.solver import engine
from aliexpress.solver.progress import ProgressListener


class _RecordingListener(ProgressListener):
    """Appends every event as a tuple for order/value assertions."""

    def __init__(self):
        self.events = []

    def stage_started(self, stage):
        self.events.append(("started", stage))

    def stage_finished(self, stage, seconds):
        self.events.append(("finished", stage, seconds))

    def plateau_finished(self, outcome):
        self.events.append(("plateau_finished", outcome))

    def tiebreak_started(self):
        self.events.append(("tiebreak_started",))

    def interim_result(self, assignment, satisfied):
        self.events.append(("interim_result", assignment, satisfied))


def _read_small_fixtures():
    target_groups = main._read_groups(  # pylint: disable=protected-access
        "tests/integration/groepen_small.xlsx"
    )
    preference_data = main._read_preferences(  # pylint: disable=protected-access
        "tests/integration/voorkeuren_small.xlsx", target_groups.counts
    )
    return target_groups, preference_data


def test_stages_fire_in_order_with_nonnegative_durations():
    """floor -> balance -> satisfaction, each started then finished, duration >= 0."""
    target_groups, preference_data = _read_small_fixtures()
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
    target_groups, preference_data = _read_small_fixtures()
    listener = _RecordingListener()

    main.distribute_students_from_data(
        preference_data, target_groups, listener=listener
    )

    plateaus = [e for e in listener.events if e[0] == "plateau_finished"]
    tiebreaks = [e for e in listener.events if e[0] == "tiebreak_started"]

    outcomes = [e[1] for e in plateaus]
    assert [o.min_satisfaction for o in outcomes] == pytest.approx([0.516129], abs=1e-6)
    assert [o.n_can_improve for o in outcomes] == [4]
    assert tiebreaks == [("tiebreak_started",)]
    # The tie-break comes after every plateau is pinned.
    assert listener.events.index(tiebreaks[0]) > listener.events.index(plateaus[-1])
    # Every plateau carries the wall-clock duration of that lexmaxmin round.
    # On this small fixture a round can be sub-millisecond, so >= 0 rather than > 0.
    for outcome in outcomes:
        assert isinstance(outcome.seconds, float)
        assert outcome.seconds >= 0


def test_interim_result_fires_per_level_with_full_assignments():
    """The balance stage and each lexmaxmin level report a complete assignment.

    ``engine.solve_within_minimal_relaxation`` is called directly (no ``main``): the
    interim-result event is preference-free, so it is observable at the engine layer
    alone. The small fixture pins a single lexmaxmin level (see
    ``test_plateaus_fire_in_order_then_tiebreak``), so this expects at least the one
    balance-stage interim result plus that one level's — two in total — each covering
    every student.
    """
    target_groups, preference_data = _read_small_fixtures()
    listener = _RecordingListener()

    engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=[],
        listener=listener,
    )

    interim_results = [e for e in listener.events if e[0] == "interim_result"]
    assert len(interim_results) >= 2
    all_students = set(preference_data.students_info)
    for _, assignment, _satisfied in interim_results:
        assert set(assignment) == all_students


class _RecordingDownstream(ProgressListener):
    """Records only the display-space views forwarded by _InterimResultAdapter."""

    def __init__(self):
        self.views = []

    def interim_result_view(self, view):
        self.views.append(view)


def test_interim_result_adapter_translates_to_display_view():
    """main._InterimResultAdapter turns each interim_result into a GroepsindelingView.

    Proves the display-translation path: the view's ``group_order`` must be the
    target groups' display names (not the solver's internal matching keys), and at
    least one chip must carry a student's display name.
    """
    target_groups, preference_data = _read_small_fixtures()
    downstream = _RecordingDownstream()
    adapter = main._InterimResultAdapter(  # pylint: disable=protected-access
        downstream, preference_data, target_groups, year_offset=0
    )

    engine.solve_within_minimal_relaxation(
        preferences=preference_data.preferences,
        students=preference_data.students_info,
        groups_to=target_groups.counts,
        not_together=[],
        listener=adapter,
    )

    assert len(downstream.views) >= 2
    view = downstream.views[0]
    assert sorted(view.group_order) == sorted(target_groups.display.values())
    chip_names = {
        chip.full_name
        for card in view.groups
        for section in card.year_sections
        for chip in section.boys.students + section.girls.students
    }
    assert chip_names & set(preference_data.student_display.values())
