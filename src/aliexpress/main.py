"""Main module for distributing students into groups based on preferences.

It has one orchestrating function that can be called from the command line or app"""

import logging
from io import BytesIO

import pandas as pd
import pandera as pa

from . import errors
from .data import datareader
from .data.datareader import GroupCounts
from .data.preferences_data import PreferenceData
from .solver import engine, groepsindeling_view, results, solutions
from .solver._balance import GroupBalance
from .solver.groepsindeling_view import GroepsindelingView
from .solver.progress import InputSummary, ProgressListener
from .solver.results import SolutionResult

FILE_PREFERENCES = "voorkeuren.xlsx"
FILE_GROUPS_TO = "groepen.xlsx"


logger = logging.getLogger(__name__)


def _safe_read(fn, *, filetype, technical_message, catch=Exception):
    try:
        return fn()
    except (errors.ValidationError, pa.errors.SchemaError):
        raise
    except catch as e:
        raise errors.CouldNotReadFileError(
            "could_not_read",
            context={"filetype": filetype},
            technical_message=technical_message,
        ) from e


def _read_groups(path) -> GroupCounts:
    """Return a :class:`GroupCounts` for the target groups file."""
    return _safe_read(
        lambda: datareader.read_groups_excel(path),
        filetype="groepen",
        technical_message="Could not read groups_to",
    )


def _read_preferences(path, groups_to):
    """Read and validate the preferences xlsx into a :class:`PreferenceData`."""

    def _inner():
        processor = datareader.VoorkeurenProcessor(path)
        processor.process(all_to_groups=list(groups_to.keys()))
        return processor.to_preference_data()

    return _safe_read(
        _inner,
        filetype="voorkeuren",
        technical_message="Could not read preferences",
    )


def _log_initial_state(groups_to, students_info, stamgroep_display=None):
    # students_info is keyed by matching key; map the Stamgroep back to the name as
    # entered for the log messages.
    stamgroep_display = stamgroep_display or {}
    df_groups = pd.DataFrame.from_dict(groups_to, orient="index")
    logger.info(
        "Current groups:\n%s",
        df_groups.assign(Totaal=lambda df: df.sum("columns")),
    )

    df_students = pd.DataFrame.from_dict(students_info, orient="index")
    sex_dist = df_students[["Jongen/meisje"]].value_counts()
    logger.info("Current boy/girl distribution:\n%s", sex_dist)

    stamgroepen = df_students["Stamgroep"].map(lambda g: stamgroep_display.get(g, g))
    logger.info("Current stamgroep distribution:\n%s", stamgroepen.value_counts())


def _build_input_summary(
    groups_to, students_info, stamgroep_display=None
) -> InputSummary:
    """Derive the input-overview counts for the processing page.

    Same derivation as ``_log_initial_state``: ``students_info`` is keyed by matching
    key, so the Stamgroep is mapped to its display name before counting per source group
    (never the internal matching keys).
    """
    stamgroep_display = stamgroep_display or {}
    df_students = pd.DataFrame.from_dict(students_info, orient="index")
    sex_dist = df_students[["Jongen/meisje"]].value_counts()
    stamgroepen = df_students["Stamgroep"].map(lambda g: stamgroep_display.get(g, g))
    # Native str/int so the summary serialises to JSON (value_counts yields numpy ints),
    # ordered most students first.
    source_groups = {
        str(group): int(count) for group, count in stamgroepen.value_counts().items()
    }
    jaarlagen = df_students.get("Jaarlaag", pd.Series(dtype=object))
    years = sorted({int(y) for y in jaarlagen if pd.notna(y)})
    return InputSummary(
        n_students=len(df_students),
        n_boys=int(sex_dist.get("Jongen", 0)),
        n_girls=int(sex_dist.get("Meisje", 0)),
        source_groups=source_groups,
        n_target_groups=len(groups_to),
        years=years,
    )


def _build_groepsindeling_view(
    result: SolutionResult,
    preference_data: PreferenceData,
    target_groups: GroupCounts,
    year_offset: int = 0,
) -> GroepsindelingView:
    """Translate a solver-space result to display space and build its structured view.

    The single display-translation path from a matching-key-keyed :class:`SolutionResult`
    to the Flask-free :class:`GroepsindelingView` the result page renders. Used both for
    the final result (via :func:`_export`) and for each interim result reported during
    solving (via :class:`_InterimResultAdapter`) — both hold exactly these three solver-
    space artefacts. ``year_offset`` shifts only the displayed Nieuwe jaarlaag (see
    :func:`~.solver.groepsindeling_view.build`); 0 leaves it unchanged.
    """
    display_names = solutions.DisplayNames(
        student=preference_data.student_display,
        group=target_groups.display,
        stamgroep=preference_data.stamgroep_display,
    )
    # The solver works on matching keys; translate to names as entered before reporting.
    result, preferences, input_sheet, students_info = solutions.to_display_names(
        result,
        preference_data.preferences,
        preference_data.input_sheet,
        preference_data.students_info,
        display_names,
    )
    # unique_name is a matching-key -> short-name map; relabel it to display space
    # (full name -> short name) so the chip builder can key it by the display name.
    # Empty in the Excel/CLI path -> chips fall back to the full name.
    unique_display = {
        preference_data.student_display.get(k, k): short
        for k, short in preference_data.unique_name.items()
    }
    return groepsindeling_view.build(
        result, students_info, preferences, input_sheet, unique_display, year_offset
    )


def _export(result, preference_data, target_groups, year_offset: int = 0):
    """Build the download workbook, result tables and the structured Groepsindeling view.

    Returns ``(output, dfs, view)``: the download workbook, the three analysis tables
    (Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren), and the Flask-free
    :class:`GroepsindelingView` for the result page. The Groepsindeling and Klassenoverzicht
    now live in the view-model, not in ``dfs``; the full workbook (``to_excel``) still writes
    every sheet. ``year_offset`` shifts the displayed Nieuwe jaarlaag (see
    :class:`~.solver.solutions.SolutionAnalyzer`); 0 leaves it unchanged.
    """
    display_names = solutions.DisplayNames(
        student=preference_data.student_display,
        group=target_groups.display,
        stamgroep=preference_data.stamgroep_display,
    )
    # The solver works on matching keys; translate to names as entered before reporting.
    result_disp, preferences, input_sheet, students_info = solutions.to_display_names(
        result,
        preference_data.preferences,
        preference_data.input_sheet,
        preference_data.students_info,
        display_names,
    )
    sa = solutions.SolutionAnalyzer(
        result_disp, preferences, input_sheet, students_info, year_offset=year_offset
    )

    output = BytesIO()
    sa.to_excel(output)
    output.seek(0)

    dfs = {
        "Overgangsmatrix": sa.display_transition_matrix(),
        "Leerlingtevredenheid": sa.display_student_performance(),
        "VervuldeVoorkeuren": sa.display_satisfied_preferences(),
    }

    view = _build_groepsindeling_view(
        result, preference_data, target_groups, year_offset=year_offset
    )

    return output, dfs, view


class _InterimResultAdapter(ProgressListener):
    """Turns a solver-space ``interim_result`` into a display-space ``interim_result_view``.

    Why this exists — it bridges a layer gap that neither side can close alone:

    - The **solver** (``engine``/``strategies``) knows the assignment on *matching keys*
      only, not the names as entered: it never receives the display maps (student/group/
      Stamgroep display, ``unique_name``, ``input_sheet``), which live in
      :class:`~.data.preferences_data.PreferenceData` / :class:`~.data.datareader.GroupCounts`
      here in the main layer. So the solver can only emit a preference-free payload.
    - The **web layer** (``progress_writer``) must not do the translation either: it is
      pandas/display work above the web boundary, and the writer has no ``PreferenceData``.

    So this adapter sits in the middle (the main layer, which *does* hold the display
    maps): it wraps a ``downstream`` :class:`ProgressListener`, forwards every other event
    verbatim, and completes each ``interim_result`` (a stage-boundary ``assignment`` +
    honored-wish booleans, read straight off the solver) into a full
    :class:`~.solver.engine.Solution` and then a display-space
    :class:`~.solver.groepsindeling_view.GroepsindelingView` via
    :func:`_build_groepsindeling_view` — the same path :func:`_export` uses for the final
    result — before forwarding it as ``downstream.interim_result_view(view)``.

    Used by :func:`distribute_students_from_data`, which wraps the caller-supplied
    listener in this adapter before handing it to the solver. No damping/throttling:
    every stage boundary the solver emits is translated and forwarded as-is.
    """

    def __init__(
        self,
        downstream: ProgressListener,
        preference_data: PreferenceData,
        target_groups: GroupCounts,
        year_offset: int = 0,
    ):
        self.downstream = downstream
        self.preference_data = preference_data
        self.target_groups = target_groups
        self.year_offset = year_offset

    def stage_started(self, stage: str) -> None:
        self.downstream.stage_started(stage)

    def stage_finished(self, stage: str, seconds: float) -> None:
        self.downstream.stage_finished(stage, seconds)

    def input_summary(self, summary: InputSummary) -> None:
        self.downstream.input_summary(summary)

    def plateau_finished(self, min_satisfaction: float, n_can_improve: int) -> None:
        self.downstream.plateau_finished(min_satisfaction, n_can_improve)

    def tiebreak_started(self) -> None:
        self.downstream.tiebreak_started()

    def interim_result(self, assignment: dict, satisfied: dict) -> None:
        """Complete the preference-free payload into a view and forward it downstream."""
        preferences = self.preference_data.preferences
        students_info = self.preference_data.students_info
        solution = engine.Solution(
            assignment=assignment,
            satisfied=satisfied,
            student_satisfaction=engine.float_satisfaction(
                preferences, satisfied, list(assignment)
            ),
        )
        result = results.to_solution_result(
            solution, preferences, students_info, self.target_groups.counts
        )
        view = _build_groepsindeling_view(
            result, self.preference_data, self.target_groups, self.year_offset
        )
        self.downstream.interim_result_view(view)


def _log_solve_summary(
    result: SolutionResult, groupbalance: GroupBalance | None = None
) -> None:
    """Log anonymous headline metrics after a completed solve — no student names.

    ``groupbalance`` logs the class-balance limits that were actually used, when
    known to the caller (the manual path always knows them; the automatic path
    does not report the balance it settled on, so it is omitted there).
    """
    if groupbalance is not None:
        logger.info(
            "Balance used: clique=%d clique_sex=%d diff_year=%d diff_total=%d "
            "imbalance_year=%d imbalance_total=%d",
            groupbalance.max_clique,
            groupbalance.max_clique_sex,
            groupbalance.max_diff_n_students_year,
            groupbalance.max_diff_n_students_total,
            groupbalance.max_imbalance_boys_girls_year,
            groupbalance.max_imbalance_boys_girls_total,
        )
    sat = result.student_satisfaction
    values = list(sat.values())
    n = len(values)
    n_full = sum(1 for v in values if v >= 1.0)
    logger.info(
        "Satisfaction: n=%d min=%.3f fully_satisfied=%d unfulfilled=%d",
        n,
        min(values) if values else 0.0,
        n_full,
        n - n_full,
    )


def distribute_students_from_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Six independent inputs to the solve+export pipeline (data, rules, balance
    # override, display shift, structured progress listener); grouping them would
    # obscure the constraints being modelled, matching the style of
    # solutions.SolutionAnalyzer.__init__.
    preference_data: PreferenceData,
    target_groups: GroupCounts,
    not_together: list[dict] | None = None,
    groupbalance: GroupBalance | None = None,
    year_offset: int = 0,
    listener: ProgressListener | None = None,
):
    """Distribute all students over all groups with lexmaxmin — the pure data core.

    Reads no files: it takes pre-built ``preference_data`` and ``target_groups`` and
    feeds them straight through the solve + export pipeline. It is the shared core behind
    :func:`distribute_students_once`, and the entry point intended for callers that already
    hold these objects in memory — e.g. the web route once it loads the preferences from
    ``voorkeuren.json`` (wired up in a later step). To read both Excel files from disk
    first, use :func:`distribute_students_once`.

    Parameters
    ----------
    preference_data : PreferenceData
        The processed preferences, student meta info and display maps.
    target_groups : GroupCounts
        The destination groups; ``target_groups.counts`` is the solver's keyed group dict
        (current boy/girl occupancy) and ``target_groups.display`` maps those keys back to
        the names as entered.
    not_together : list[dict] | None
        Not-together rules built from web-form data or constructed in tests. Each dict has
        keys 'group' (set[str]) and 'Max_aantal_samen' (int). Pass None for no constraints.
    groupbalance : GroupBalance | None
        Class-balance constraints. When None (the default), the balance is determined
        automatically: satisfaction is maximized within the minimal relaxation that still
        lets every student fulfil a positive wish (see
        :func:`~.solver.engine.solve_within_minimal_relaxation`). Pass a
        GroupBalance to override this with fixed manual limits instead (see
        :func:`~.solver.engine.solve_with_fixed_balance`).
    year_offset : int
        Shifts the displayed Nieuwe jaarlaag in the result view and export (see
        :class:`~.solver.solutions.SolutionAnalyzer`). 0 (the default) for modes without
        an Overgang; 1 for the "forward"/"redistribute_and_forward" modes.
    listener : ProgressListener | None
        Notified of the three solve stages (see
        :func:`~.solver.engine.solve_within_minimal_relaxation`) when the balance is
        determined automatically. Not consulted on the fixed-balance path, which has
        no stepper. ``None`` (the default) means no one is watching; the input summary
        is then not even built.

    Returns
    -------
    dict
        ``{"download": <xlsx BytesIO>, "dataframes": {<3 analysis tables>},
        "groepsindeling_view": GroepsindelingView}`` — the workbook, the three analysis tables
        (Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren) and the structured
        group-card view-model for the result page.
    """
    preferences = preference_data.preferences
    students_info = preference_data.students_info
    if not_together is None:
        not_together = []
    datareader.validate_not_together(
        not_together, students_info.keys(), len(target_groups.counts)
    )
    # Rule groups hold names as entered; the solver matches on the same keys as students.
    not_together = [
        {**rule, "group": {datareader.matching_key(s) for s in rule["group"]}}
        for rule in not_together
    ]
    logger.info("All files read")

    if listener is not None:
        listener = _InterimResultAdapter(
            listener, preference_data, target_groups, year_offset
        )
        listener.input_summary(
            _build_input_summary(
                target_groups.counts, students_info, preference_data.stamgroep_display
            )
        )
    _log_initial_state(
        target_groups.counts,
        students_info,
        preference_data.stamgroep_display,
    )

    if groupbalance is None:
        logger.info("Solving within the minimal class-balance relaxation")
        solution = engine.solve_within_minimal_relaxation(
            preferences=preferences,
            students=students_info,
            groups_to=target_groups.counts,
            not_together=not_together,
            optimize="lexmaxmin",
            listener=listener,
        )
        result = results.to_solution_result(
            solution, preferences, students_info, target_groups.counts
        )
        _log_solve_summary(result)
    else:
        logger.info("Solving with a fixed class balance")
        try:
            solution = engine.solve_with_fixed_balance(
                preferences=preferences,
                students=students_info,
                groups_to=target_groups.counts,
                not_together=not_together,
                groupbalance=groupbalance,
                optimize="lexmaxmin",
            )
        except errors.StageInfeasible as exc:
            raise errors.FeasibilityError(
                "infeasible_problem",
                context={
                    "possible_improvement": "Kies een ruimere klassenbalans; met de "
                    "huidige vaste instellingen is geen geldige verdeling mogelijk."
                },
                technical_message="Fixed class balance admits no valid assignment",
            ) from exc
        result = results.to_solution_result(
            solution, preferences, students_info, target_groups.counts
        )
        _log_solve_summary(result, groupbalance)

    output, dfs, view = _export(
        result, preference_data, target_groups, year_offset=year_offset
    )
    logger.info("Done!")
    return {"download": output, "dataframes": dfs, "groepsindeling_view": view}


def distribute_students_once(
    path_preferences=FILE_PREFERENCES,
    path_groups_to=FILE_GROUPS_TO,
    not_together: list[dict] | None = None,
    groupbalance: GroupBalance | None = None,
):
    """Convenience wrapper for the CLI and tests: read both Excel files, then distribute.

    Reads the preferences from ``path_preferences`` and the target groups from
    ``path_groups_to``, then delegates to :func:`distribute_students_from_data`, which is
    the pure data core. See that function for the ``not_together`` and ``groupbalance``
    parameters.
    """
    target_groups = _read_groups(path_groups_to)
    preference_data = _read_preferences(path_preferences, target_groups.counts)
    return distribute_students_from_data(
        preference_data, target_groups, not_together, groupbalance
    )


if __name__ == "__main__":
    distribute_students_once()
