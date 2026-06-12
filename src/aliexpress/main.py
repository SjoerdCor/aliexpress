"""Main module for distributing students into groups based on preferences.

It has one orchestrating function that can be called from the command line or app"""

import os
import tempfile
from io import BytesIO

import pandas as pd
import pandera as pa

from . import datareader, errors, problemsolver, solutions
from .logging_config import setup_logger
from .problemsolver import GroupBalance

FILE_PREFERENCES = "voorkeuren.xlsx"
FILE_GROUPS_TO = "groepen.xlsx"


logger = setup_logger(__name__)


def jsons_to_excel(folder, preferences, input_sheet, students_info):
    """Write all solution-jsons in folder to comprehensible excel overview"""
    for file in os.listdir(folder):
        if file.endswith(".json"):
            fname = os.path.join(folder, file)
            sa = solutions.SolutionAnalyzer(
                fname, preferences, input_sheet, students_info
            )
            sa.to_excel()


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


def _read_groups(path):
    return _safe_read(
        lambda: datareader.read_groups_excel(path),
        filetype="groepen",
        technical_message="Could not read groups_to",
    )


def _read_preferences(path, groups_to):
    def _inner():
        processor = datareader.VoorkeurenProcessor(path)
        preferences = processor.process(all_to_groups=list(groups_to.keys()))
        return processor, preferences

    return _safe_read(
        _inner,
        filetype="voorkeuren",
        technical_message="Could not read preferences",
    )


def _log_initial_state(groups_to, students_info, on_update):
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
    for group, value in df_students["Stamgroep"].value_counts().items():
        on_update(f"{group}: {value}")


def _check_feasibility(ps):
    feas_prob = ps.calculate_feasibility()
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


def _export(ps, preferences, processor, students_info):
    """Build the download workbook and result tables from the already-solved problem."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        ps.prob.toJson(tmp.name)
        tmp.flush()
        sa = solutions.SolutionAnalyzer(
            tmp.name, preferences, processor.input, students_info
        )

    output = BytesIO()
    sa.to_excel(output)
    output.seek(0)

    dfs = {
        "Groepsindeling": sa.display_groepsindeling(),
        "Klassenoverzicht": sa.group_report,
        "Overgangsmatrix": sa.display_transition_matrix(),
        "Leerlingtevredenheid": sa.display_student_performance(),
        "VervuldeWensen": sa.display_satisfied_preferences(),
    }

    return output, dfs


def distribute_students_once(
    path_preferences=FILE_PREFERENCES,
    path_groups_to=FILE_GROUPS_TO,
    not_together: list[dict] | None = None,
    on_update=lambda msg: None,
    groupbalance: GroupBalance | None = None,
):
    """Distribute all students with preferences over all groups with lexmaxmin.

    Parameters:
        not_together : list[dict] | None
            Not-together rules built from web-form data or constructed in tests.
            Each dict has keys 'group' (set[str]) and
            'Max_aantal_samen' (int). Pass None or omit for no constraints.
        on_update : func
            Takes a user friendly message and decides what to do with it for the calling
            function. By default, ignores them
        groupbalance : GroupBalance | None
            Class-balance constraints. When None (the default), the balance is determined
            automatically: satisfaction is maximized within the minimal relaxation that
            still lets every student fulfil a positive wish (see
            :meth:`ProblemSolver.solve_within_minimal_relaxation`). Pass a GroupBalance to
            override this with fixed manual limits instead.
    """
    groups_to = _read_groups(path_groups_to)
    processor, preferences = _read_preferences(path_preferences, groups_to)

    students_info = processor.get_students_meta_info()
    if not_together is None:
        not_together = []
    datareader.validate_not_together(not_together, students_info.keys(), len(groups_to))
    on_update("Alle bestanden zijn gevalideerd!")
    logger.info("All files read")

    _log_initial_state(groups_to, students_info, on_update)

    ps = problemsolver.ProblemSolver(
        preferences,
        students_info,
        groups_to,
        not_together,
        groupbalance=groupbalance,
        optimize="lexmaxmin",
    )
    on_update("Aan de slag! Groepen indelen...")
    if groupbalance is None:
        logger.info("Solving within the minimal class-balance relaxation")
        ps.solve_within_minimal_relaxation()
    else:
        _check_feasibility(ps)
        on_update("Bepaald dat probleem oplosbaar is!")
        logger.info("Finding first solution... lexmaxmin")
        ps.run(save=False)

    output, dfs = _export(ps, preferences, processor, students_info)
    logger.info("Done!")
    on_update("Klaar!")
    return {"download": output, "dataframes": dfs}


if __name__ == "__main__":
    distribute_students_once()
