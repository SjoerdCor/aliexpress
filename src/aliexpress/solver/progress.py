"""Flask-free progress-reporting seam for the solve pipeline.

The solver package stays web-free: it only knows about the :class:`ProgressListener`
interface below. Callers that don't care about progress (the CLI, most tests) pass
``None``, and every emit site in the solver guards on ``if listener is not None``.
The web layer subclasses it to persist progress as JSON for the processing page to
poll (see ``web/progress_writer.py``).

``stage`` is one of the three UI steps shown on the processing page:

- ``"floor"`` — stage 1, finding the minimal relaxation floor (non-positive
  satisfaction count).
- ``"balance"`` — stage 2, minimizing the weighted balance relaxation within
  that floor.
- ``"satisfaction"`` — stage 3, maximizing satisfaction (all lexmaxmin levels
  and the tie-break) within the fixed balance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSummary:
    """Headline counts for the input overview at the top of the processing page.

    Built once, early in the solve (before stage 1), from the same data
    ``main._log_initial_state`` already derives its logging from. ``source_groups`` maps
    each origin group (display name as entered) to its student count, ordered most
    students first. ``years`` lists the distinct Jaarlagen present (empty when the input
    carries none).
    """

    n_students: int
    n_boys: int
    n_girls: int
    source_groups: dict[str, int]
    n_target_groups: int
    years: list[int]


class ProgressListener:
    """An observer interface for solve progress: override only what you care about.

    Every method has a no-op default body, so a subclass need only override the events
    it wants (e.g. only ``interim_result_view``). Call sites in the solver package hold
    an ``Optional[ProgressListener]`` and guard each emit with ``if listener is not
    None:`` — that guard, not this class, is what makes progress reporting optional:
    it lets a caller that doesn't care skip building the (sometimes non-trivial) event
    payload entirely, not just skip receiving it.
    """

    def stage_started(self, stage: str) -> None:
        """Called when ``stage`` begins."""

    def stage_finished(self, stage: str, seconds: float) -> None:
        """Called when ``stage`` completes, after ``seconds`` seconds."""

    def input_summary(self, summary: InputSummary) -> None:
        """Called once, early, with the headline counts of the problem being solved."""

    def plateau_finished(self, min_satisfaction: float, n_can_improve: int) -> None:
        """Called when a lexmaxmin level completes (``_lexmaxmin`` in strategies.py).

        ``min_satisfaction`` is the pinned plateau, already divided by the satisfaction
        scale (a fraction, e.g. 0.62); it can be negative (satisfaction can be negative,
        see ADR-0014) and must not be clamped. ``n_can_improve`` is how many students
        escaped this plateau and go on to the next level (0 on the terminal level).
        """

    def tiebreak_started(self) -> None:
        """Called once all lexmaxmin plateaus are pinned and the final tie-break begins."""

    def interim_result(self, assignment: dict, satisfied: dict) -> None:
        """Called with the best complete distribution at a solved stage boundary.

        Fired once after the balance stage (:func:`~.engine.solve_within_minimal_relaxation`)
        and once per completed lexmaxmin level (:func:`~.strategies._lexmaxmin`) — every
        stage boundary, with no damping or throttling. The payload is preference-free,
        read straight off that stage's ``CpSolver``: ``assignment`` maps each student to
        their assigned group, ``satisfied`` maps each ``(student, Nr)`` preference row to
        whether it was honored. A listener that needs the display-space view (chips,
        satisfaction, ...) translates this itself — see
        ``aliexpress.main._InterimResultAdapter``, which turns each call into an
        ``interim_result_view`` call on its downstream listener.
        """

    def interim_result_view(self, view) -> None:
        """Called by ``aliexpress.main._InterimResultAdapter`` with a translated interim view.

        ``view`` is a :class:`~.groepsindeling_view.GroepsindelingView` (left unannotated
        here so this module stays decoupled from the view package). Never called directly
        by the solver: only that adapter emits it, wrapping a downstream listener and
        turning each :meth:`interim_result` into a display-space view. Not fired on its own.
        """
