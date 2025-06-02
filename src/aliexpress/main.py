import argparse
import logging
import os

import pandas as pd


from . import datareader
from . import problemsolver
from . import solutions

FILE_PREFERENCES = "voorkeuren.xlsx"
FILE_GROUPS_TO = "groepen.xlsx"
FILE_NOT_TOGETHER = "niet_samen.xlsx"


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


def jsons_to_excel(folder, preferences, input_sheet, students_info):
    """Write all solution-jsons in folder to comprehensible excel overview"""
    for file in os.listdir(folder):
        if file.endswith(".json"):
            fname = os.path.join(folder, file)
            sa = solutions.SolutionAnalyzer(
                fname, preferences, input_sheet, students_info
            )
            sa.to_excel()


def distribute_students(**kwargs):
    """Distribute all students with preferences over all groups to

    Kwargs are passed to problemsolver
    """
    groups_to = pd.read_excel(FILE_GROUPS_TO, index_col=0).to_dict(orient="index")
    processor = datareader.VoorkeurenProcessor(FILE_PREFERENCES)
    preferences = processor.process(all_to_groups=list(groups_to.keys()))
    students_info = processor.get_students_meta_info()
    not_together = datareader.read_not_together(
        FILE_NOT_TOGETHER, students_info.keys(), len(groups_to)
    )
    logger.info("All files read")

    df_groups_to = pd.DataFrame.from_dict(groups_to, orient="index")
    logger.info(
        "Current groups:\n%s", df_groups_to.assign(Totaal=lambda df: df.sum("columns"))
    )

    df_students = pd.DataFrame.from_dict(students_info, orient="index")

    logger.info(
        "Current boy/girl distribution:\n%s",
        df_students[["Jongen/meisje"]].value_counts(),
    )
    logger.info("Coming from groups:\n%s", df_students["Stamgroep"].value_counts())

    defaults_problemsolver = {"max_imbalance_boys_girls_total": 6}
    kwargs_problemsolver = {**defaults_problemsolver, **kwargs}
    ps_lexmaxmin = problemsolver.ProblemSolver(
        preferences,
        students_info,
        groups_to,
        not_together,
        optimize="lexmaxmin",
        **kwargs_problemsolver,
    )
    feas_prob = ps_lexmaxmin.calculate_feasibility()
    if feas_prob.objective.value() > 0:
        raise RuntimeError("Can not solve the problem for this class imbalance")

    logger.info("Finding first solution... lexmaxmin")
    ps_lexmaxmin.run(overwrite=True)
    jsons_to_excel(
        ps_lexmaxmin.get_solution_name(), preferences, processor.input, students_info
    )
    logger.info("Lexmaxmin done!")

    logger.info("Finding 2 solutions... least satisfied")
    ps_least_satisfied = problemsolver.ProblemSolver(
        preferences,
        students_info,
        groups_to,
        not_together,
        optimize="least_satisfied",
        **kwargs_problemsolver,
    )

    ps_least_satisfied.run(n_solutions=2, overwrite=True)
    jsons_to_excel(
        ps_least_satisfied.get_solution_name(),
        preferences,
        processor.input,
        students_info,
    )
    logger.info("Least satisfied: 2 solutions found")

    logger.info("Finding different solution...")
    ps_least_satisfied.run(distance=10, overwrite=True)
    jsons_to_excel(
        ps_least_satisfied.get_solution_name(),
        preferences,
        processor.input,
        students_info,
    )
    logger.info("Least satisfied: different solution found")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Distribute student.")

    parser.add_argument(
        "--max_clique",
        type=int,
        default=5,
        help="The number of students that can go to the same group (default = 5)",
    )
    parser.add_argument(
        "--max_clique_sex",
        type=int,
        default=3,
        help="The number of students from an original group of the same sex that can go to the same group (default = 3)",
    )
    parser.add_argument(
        "--max_diff_n_students_year",
        type=int,
        default=2,
        help="The maximum difference between assigned students to the largest group and the smallest group (default = 2)",
    )
    parser.add_argument(
        "--max_diff_n_students_total",
        type=int,
        default=3,
        help="The maximum difference between largest group and the smallest group in total (default = 3)",
    )

    parser.add_argument(
        "--max_imbalance_boys_girls_year",
        type=int,
        default=2,
        help="The maximum difference between number of boys and girls in each year in a group (default = 2)",
    )

    parser.add_argument(
        "--max_imbalance_boys_girls_total",
        type=int,
        default=3,
        help="The maximum difference between number of boys and girls in the total group (default = 3)",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    distribute_students(**kwargs)


if __name__ == "__main__":
    main()
