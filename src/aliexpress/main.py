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
from .solver import feasibility, problemsolver, solutions
from .solver.problemsolver import GroupBalance

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


def _log_initial_state(groups_to, students_info, on_update, stamgroep_display=None):
    # students_info is keyed by matching key; map the Stamgroep back to the name as
    # entered for the user-facing messages (logs may keep the internal keys).
    stamgroep_display = stamgroep_display or {}
    df_groups = pd.DataFrame.from_dict(groups_to, orient="index")
    logger.info(
        "Current groups:\n%s",
        df_groups.assign(Totaal=lambda df: df.sum("columns")),
    )

    df_students = pd.DataFrame.from_dict(students_info, orient="index")
    sex_dist = df_students[["Jongen/meisje"]].value_counts()

    on_update(
        f"{len(df_students)} leerlingen te verdelen, "
        f"waarvan {sex_dist.loc['Jongen'].squeeze()} jongens "
        f"en {sex_dist.loc['Meisje'].squeeze()} meisjes"
    )
    logger.info("Current boy/girl distribution:\n%s", sex_dist)

    on_update("Komen uit de volgende groepen:")
    stamgroepen = df_students["Stamgroep"].map(lambda g: stamgroep_display.get(g, g))
    for group, value in stamgroepen.value_counts().items():
        on_update(f"{group}: {value}")


def _check_feasibility(ps):
    feas_prob = feasibility.check_balance_feasibility(ps)
    if feas_prob.objective.value() <= 0:
        return

    slack_info = {
        "SLACK_balanced_boys_girls_total": (
            "Maximale verschil jongens/meisjes totale groep",
            ps.groupbalance.max_imbalance_boys_girls_total,
        ),
        "SLACK_balanced_boys_girls_year": (
            "Maximale verschil jongens/meisjes nieuwe jaarlaag",
            ps.groupbalance.max_imbalance_boys_girls_year,
        ),
        "SLACK_diff_n_students_total": (
            "Maximale verschil totale groepsgrootte",
            ps.groupbalance.max_diff_n_students_total,
        ),
        "SLACK_diff_n_students_year": (
            "Maximale verschil groepsgrootte nieuwe jaarlaag",
            ps.groupbalance.max_diff_n_students_year,
        ),
        "SLACK_max_clique": (
            "Maximale groep vanuit eerdere groep",
            ps.groupbalance.max_clique,
        ),
        "SLACK_max_clique_sex": (
            "Maximale groep jongens/meisjes vanuit eerdere groep",
            ps.groupbalance.max_clique_sex,
        ),
    }

    msg = []
    variables = feas_prob.variablesDict()

    for name, (label, base_value) in slack_info.items():
        val = variables[name].value()
        if val > 0:
            msg.append(f"{label}: {round(base_value + val)} (+ {round(val)})")

    raise errors.FeasibilityError(
        "infeasible_problem",
        context={"possible_improvement": "\n".join(msg)},
        technical_message="Can not solve the problem for this class imbalance",
    )


def _export(result, preference_data, target_groups):
    """Build the download workbook and result tables from the already-solved result."""
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
    sa = solutions.SolutionAnalyzer(result, preferences, input_sheet, students_info)

    output = BytesIO()
    sa.to_excel(output)
    output.seek(0)

    dfs = {
        "Groepsindeling": sa.display_groepsindeling(),
        "Klassenoverzicht": sa.group_report,
        "Overgangsmatrix": sa.display_transition_matrix(),
        "Leerlingtevredenheid": sa.display_student_performance(),
        "VervuldeVoorkeuren": sa.display_satisfied_preferences(),
    }

    return output, dfs


def _log_solve_summary(
    result: problemsolver.SolutionResult, groupbalance: GroupBalance | None = None
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


def distribute_students_from_data(
    preference_data: PreferenceData,
    target_groups: GroupCounts,
    not_together: list[dict] | None = None,
    on_update=lambda msg: None,
    groupbalance: GroupBalance | None = None,
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
    on_update : func
        Takes a user-friendly message and decides what to do with it for the calling
        function. By default, ignores them.
    groupbalance : GroupBalance | None
        Class-balance constraints. When None (the default), the balance is determined
        automatically: satisfaction is maximized within the minimal relaxation that still
        lets every student fulfil a positive wish (see
        :meth:`ProblemSolver.solve_within_minimal_relaxation`). Pass a GroupBalance to
        override this with fixed manual limits instead.
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
    on_update("Alle bestanden zijn gevalideerd!")
    logger.info("All files read")

    _log_initial_state(
        target_groups.counts,
        students_info,
        on_update,
        preference_data.stamgroep_display,
    )

    ps = problemsolver.ProblemSolver(
        preferences,
        students_info,
        target_groups.counts,
        not_together,
        groupbalance=groupbalance,
        optimize="lexmaxmin",
    )
    on_update("Aan de slag! Groepen indelen...")
    if groupbalance is None:
        logger.info("Solving within the minimal class-balance relaxation")
        try:
            ps.solve_within_minimal_relaxation()
        except errors.FeasibilityError as exc:
            if exc.code == "infeasible_preferences":
                exc.context = {"case": feasibility.diagnose(ps)}
                logger.warning("Infeasible preferences: case=%s", exc.context["case"])
            raise
        result = ps.extract_solution()
        _log_solve_summary(result, ps.groupbalance)
    else:
        _check_feasibility(ps)
        on_update("Bepaald dat probleem oplosbaar is!")
        logger.info("Finding first solution... lexmaxmin")
        ps.run()
        result = ps.extract_solution()
        _log_solve_summary(result, groupbalance)

    output, dfs = _export(result, preference_data, target_groups)
    logger.info("Done!")
    on_update("Klaar!")
    return {"download": output, "dataframes": dfs}


def distribute_students_once(
    path_preferences=FILE_PREFERENCES,
    path_groups_to=FILE_GROUPS_TO,
    not_together: list[dict] | None = None,
    on_update=lambda msg: None,
    groupbalance: GroupBalance | None = None,
):
    """Convenience wrapper for the CLI and tests: read both Excel files, then distribute.

    Reads the preferences from ``path_preferences`` and the target groups from
    ``path_groups_to``, then delegates to :func:`distribute_students_from_data`, which is
    the pure data core. See that function for the ``not_together``, ``on_update`` and
    ``groupbalance`` parameters.
    """
    target_groups = _read_groups(path_groups_to)
    preference_data = _read_preferences(path_preferences, target_groups.counts)
    return distribute_students_from_data(
        preference_data, target_groups, not_together, on_update, groupbalance
    )


if __name__ == "__main__":
    distribute_students_once()
