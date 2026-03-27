"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import logging
import math
import os
import warnings
from dataclasses import dataclass

import pandas as pd
import pulp

from aliexpress import optimizationstrategies, preferences_utils, pulp_logical


def setup_logger():
    """Setup a logger for the module"""
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


logger = setup_logger()


@dataclass
class GroupBalance:
    """
    Constraints controlling how students are distributed across groups.

    All values must be non-negative integers.
    """

    max_clique: int = 1
    """The number of students that can go to the same group"""

    max_clique_sex: int = 1
    """Maximum number of students of the same sex from the same original group in a group."""

    max_diff_n_students_year: int = 1
    """Max difference between largest and smallest group per year."""

    max_diff_n_students_total: int = 1
    """Max difference between largest and smallest group overall."""

    max_imbalance_boys_girls_year: int = 1
    """Max difference between boys and girls per year in a group."""

    max_imbalance_boys_girls_total: int = 1
    """Max difference between boys and girls in total per group."""

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
class ProblemSolver:
    """
    Create a problem to distribute students over groups

    Parameters
    ----------
    preferences: pd.DataFrame
        A DataFrame with as MultiIndex with (Leerling, Type, Nr) and a value, where
        Leerling is the Name, Type is either "Graag met", "Niet in" or "Liever niet"
        Waarde is then a column with either a Student or Group name. In combination with
        Niet In only a Group name is allowed

    students : dict
        Each student as key, and as value a dictionary that contains at least the
        "Stamgroep" and "Jongen/meisje". Used to make balanced new groups

    groups_to: dict
        A dictionary that contains the groups to which the students can be sent as keys,
        and as values a dictionary with characteristics: the number of boys and the
        number of girls
    not_together: list[dict]
        A list where each element is a dictionary containing a group of students and
        a max_aantal_samen, defining how many can at most be together in a new group

    constraints : GroupBalance
        Configuration of group balancing constraints.

    optimize, str (default = "studentsatisfaction")
        What to optimize for: "studentsatisfaction" (total satisfaction of the students,
        where satisfaction is dominated by getting at least 1 preferences),
        "least_satisfied" (formally, the least satisfied student), "n_preferences"
        or "weighted_preferences"
    """

    def _validate_not_together_students_exist(self):
        for i, rule in enumerate(self.not_together, start=1):
            group = rule["group"]
            for student in group:
                if student not in self.students:
                    raise ValueError(
                        f"Student {student!r} from group {i} in not together not found as student"
                    )

    def __init__(
        self,
        preferences: pd.DataFrame,
        students: dict,
        groups_to: dict,
        not_together: list[dict],
        groupbalance: GroupBalance = GroupBalance(),
        optimize="studentsatisfaction",
    ):
        self.preferences = preferences
        self.students = students
        self.groups_to = groups_to
        self.not_together = not_together
        self._validate_not_together_students_exist()

        self.groupbalance = groupbalance
        self.optimize = optimize
        self.prob = pulp.LpProblem("studentdistribution", pulp.LpMaximize)
        self.in_group = pulp.LpVariable.dicts(
            "group",
            itertools.product(self.students.keys(), self.groups_to.keys()),
            cat="Binary",
        )
        self.studentsatisfaction = pulp.LpVariable.dict(
            "studentsatisfaction", self.students.keys(), cat="Continuous"
        )
        self.known_solutions = []

        self.calculate_feasibility()

    def get_solution_name(self):
        """Create name from config to identify the solution"""
        attrs = [
            self.optimize,
            self.groupbalance.max_clique,
            self.groupbalance.max_clique_sex,
            self.groupbalance.max_diff_n_students_total,
            self.groupbalance.max_diff_n_students_year,
            self.groupbalance.max_imbalance_boys_girls_total,
            self.groupbalance.max_imbalance_boys_girls_year,
        ]
        return "".join(str(s) for s in attrs)

    def _constraint_student_to_exactly_one_group(self, prob):
        for student in self.students:
            prob += (
                pulp.lpSum([self.in_group[(student, gr)] for gr in self.groups_to]) == 1
            )

    def _constraint_equal_new_students(self, prob, incl_slack=True):
        """Every group should have an approximately equal number of new students"""

        slack_var = pulp.LpVariable(
            "SLACK_diff_n_students_year",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )
        min_in_group_year = pulp.LpVariable("min_in_group_year", cat="Integer")
        max_in_group_year = pulp.LpVariable("max_in_group_year", cat="Integer")

        new_students_in_group = pulp.LpVariable.dict(
            "new_students_in_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to in self.groups_to:
            prob += new_students_in_group[group_to] == pulp.lpSum(
                [self.in_group[(student, group_to)] for student in self.students]
            )

            prob += new_students_in_group[group_to] <= max_in_group_year
            prob += new_students_in_group[group_to] >= min_in_group_year
        prob += (
            max_in_group_year - min_in_group_year
            <= self.groupbalance.max_diff_n_students_year + slack_var
        )

    def _constraint_equal_total_students(self, prob, incl_slack=True):
        current_per_group = {
            gr: self.groups_to[gr]["Jongens"] + self.groups_to[gr]["Meisjes"]
            for gr in self.groups_to
        }

        slack_var = pulp.LpVariable(
            "SLACK_diff_n_students_total",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )
        min_in_group_total = pulp.LpVariable("min_in_group_total", cat="Integer")
        max_in_group_total = pulp.LpVariable("max_in_group_total", cat="Integer")
        total_in_group = pulp.LpVariable.dict(
            "total_in_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to in self.groups_to:

            prob += total_in_group[group_to] == (
                pulp.lpSum(
                    [self.in_group[(student, group_to)] for student in self.students]
                )
                + current_per_group[group_to]
            )

            prob += total_in_group[group_to] <= max_in_group_total
            prob += total_in_group[group_to] >= min_in_group_total
        prob += (
            max_in_group_total - min_in_group_total
            <= self.groupbalance.max_diff_n_students_total + slack_var
        )

    def _constraint_equal_students_from_previous_group(self, prob, incl_slack=False):
        """Every group can have a max number of students from an earlier group (no cliques)"""
        groups_from = {self.students[student]["Stamgroep"] for student in self.students}
        from_group_to_group = pulp.LpVariable.dicts(
            "from_group_to_group",
            itertools.product(groups_from, self.groups_to.keys()),
            cat="Integer",
        )
        slack_var = pulp.LpVariable(
            "SLACK_max_clique",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )

        for group_to in self.groups_to:
            for group_from in groups_from:
                prob += from_group_to_group[(group_from, group_to)] == pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Stamgroep"] == group_from
                    ]
                )

                prob += (
                    from_group_to_group[(group_from, group_to)]
                    <= self.groupbalance.max_clique + slack_var
                )

    def _constraint_clique_sex_group(self, prob, incl_slack=False):
        """Every group can have a max number of students of the samen sex
        from an earlier group (no cliques)"""
        groups_from = {self.students[student]["Stamgroep"] for student in self.students}
        sexes = {self.students[student]["Jongen/meisje"] for student in self.students}
        slack_var = pulp.LpVariable(
            "SLACK_max_clique_sex",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )

        for group_to in self.groups_to:
            for group_from in groups_from:
                for sex in sexes:
                    this_clique = [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Stamgroep"] == group_from
                        and self.students[student]["Jongen/meisje"] == sex
                    ]

                    prob += (
                        pulp.lpSum(this_clique)
                        <= self.groupbalance.max_clique_sex + slack_var
                    )

    def _constraint_equal_boys_girls(self, prob, incl_slack=False):
        boys_to_group = pulp.LpVariable.dicts(
            "boys_to_group", self.groups_to.keys(), cat="Integer"
        )
        girls_to_group = pulp.LpVariable.dicts(
            "girls_to_group", self.groups_to.keys(), cat="Integer"
        )

        slack_var = pulp.LpVariable(
            "SLACK_balanced_boys_girls_year",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )

        for group_to in self.groups_to:
            prob += boys_to_group[group_to] == pulp.lpSum(
                [
                    self.in_group[(student, group_to)]
                    for student in self.students
                    if self.students[student]["Jongen/meisje"] == "Jongen"
                ]
            )
            prob += girls_to_group[group_to] == pulp.lpSum(
                [
                    self.in_group[(student, group_to)]
                    for student in self.students
                    if self.students[student]["Jongen/meisje"] == "Meisje"
                ]
            )
            prob += (
                girls_to_group[group_to] - boys_to_group[group_to]
                <= self.groupbalance.max_imbalance_boys_girls_year + slack_var
            )
            prob += (
                boys_to_group[group_to] - girls_to_group[group_to]
                <= self.groupbalance.max_imbalance_boys_girls_year + slack_var
            )

    def _constraint_balanced_boys_girls_total(self, prob, incl_slack=False):
        boys_in_group = pulp.LpVariable.dicts(
            "boys_in_group", self.groups_to.keys(), cat="Integer"
        )
        girls_in_group = pulp.LpVariable.dicts(
            "girls_in_group", self.groups_to.keys(), cat="Integer"
        )

        slack_var = pulp.LpVariable(
            "SLACK_balanced_boys_girls_total",
            lowBound=0,
            upBound=None if incl_slack else 0,
            cat="Integer",
        )

        for group_to, n_boys_girls in self.groups_to.items():
            prob += boys_in_group[group_to] == (
                n_boys_girls["Jongens"]
                + pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Jongen/meisje"] == "Jongen"
                    ]
                )
            )
            prob += girls_in_group[group_to] == (
                n_boys_girls["Meisjes"]
                + pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Jongen/meisje"] == "Meisje"
                    ]
                )
            )
            prob += (
                girls_in_group[group_to] - boys_in_group[group_to]
                <= self.groupbalance.max_imbalance_boys_girls_total + slack_var
            )
            prob += (
                boys_in_group[group_to] - girls_in_group[group_to]
                <= self.groupbalance.max_imbalance_boys_girls_total + slack_var
            )

    def _constraint_not_in_forbidden_group(self, prob):
        """Some students can not move int other groups (e.g. a brother/sister is already there)"""
        for i, row in self.preferences.query('TypeWens == "Niet in"').iterrows():
            student, _, _ = i
            gr = row["Waarde"]
            prob += self.in_group[(student, gr)] == 0

    def _constraint_not_together(self, prob):
        """Enforces constraint of difficult students not being together"""
        for dct in self.not_together:
            for group_to in self.groups_to:
                prob += (
                    pulp.lpSum(
                        [
                            self.in_group[(student, group_to)]
                            for student in self.students
                            if student in dct["group"]
                        ]
                    )
                    <= dct["Max_aantal_samen"]
                )

    def _constraint_minimal_satisfaction(self, prob):
        for student, info in self.students.items():
            if not math.isnan(info["MinimaleTevredenheid"]):
                prob += (
                    self.studentsatisfaction[student] >= info["MinimaleTevredenheid"]
                ), f"MinimalSatisfaction{student}"

    def add_fundamental_constraints(self, prob):
        """Add constraints fundamental to a solution"""
        self._constraint_student_to_exactly_one_group(prob)
        self._constraint_not_in_forbidden_group(prob)

    def add_class_balance_constraints(self, prob, incl_slack=False):
        """Add constraints to force good class balance in next groups"""
        self._constraint_equal_new_students(prob, incl_slack)
        self._constraint_equal_total_students(prob, incl_slack)
        self._constraint_equal_boys_girls(prob, incl_slack)
        self._constraint_balanced_boys_girls_total(prob, incl_slack)
        self._constraint_equal_students_from_previous_group(prob, incl_slack)
        self._constraint_clique_sex_group(prob, incl_slack)

    def add_satisfaction_constraints(self, prob):
        """Add constraints about social dynamics"""
        self._constraint_not_together(prob)
        self._constraint_minimal_satisfaction(prob)

    def add_constraints(self, prob=None, incl_slack=False):
        """Add all hard constraints via the functions per constraint"""

        prob = prob or self.prob
        self.add_fundamental_constraints(prob)
        self.add_class_balance_constraints(prob, incl_slack)
        self.add_satisfaction_constraints(prob)

    def set_minimal_feasible_parameters(self):
        """Set class balance so that the problem is feasible, with optimal balance

        Changes the class balance parameters, weighting for the current year heavier
        """

        feas_prob = pulp.LpProblem("MinimumRelaxationFeasibility", pulp.LpMinimize)
        self.add_constraints(feas_prob, incl_slack=True)
        slack_vars = [v for v in feas_prob.variables() if "SLACK" in v.name]

        # weight historic indifferences lower
        slack_info = {
            "SLACK_diff_n_students_year": {
                "weight": 1,
                "attr": "max_diff_n_students_year",
            },
            "SLACK_diff_n_students_total": {
                "weight": 0.49,
                "attr": "max_diff_n_students_total",
            },
            "SLACK_max_clique": {"weight": 1, "attr": "max_clique"},
            "SLACK_max_clique_sex": {
                "weight": 1,
                "attr": "max_clique_sex",
            },
            "SLACK_balanced_boys_girls_year": {
                "weight": 1,
                "attr": "max_imbalance_boys_girls_year",
            },
            "SLACK_balanced_boys_girls_total": {
                "weight": 0.49,
                "attr": "max_imbalance_boys_girls_total",
            },
        }

        for var in slack_vars:
            if var.name in slack_info:
                slack_info[var.name]["slack_var"] = var

        feas_prob.setObjective(
            pulp.lpSum(dct["weight"] * dct["slack_var"] for dct in slack_info.values())
        )

        solver = self._get_solver()
        status = feas_prob.solve(solver=solver)
        if pulp.LpStatus[status] != "Optimal":
            raise ValueError("Feasibility problem could not be solved")

        for dct in slack_info.values():
            slack_var = dct.get("slack_var", None)
            if slack_var is not None:
                current_val = getattr(self.groupbalance, dct["attr"])
                setattr(
                    self.groupbalance,
                    dct["attr"],
                    int(current_val + slack_var.varValue),
                )

    def calculate_feasibility(self) -> pulp.LpProblem:
        """Calculates whether the constraints for class imbalance are feasible

        Takes current groups and students into account, and suggests smallest possible
        relaxation.

        Returns
        -------
        pulp.LpProblem
            The relaxation problem, for further inspection
        """
        feas_prob = pulp.LpProblem("MinimumRelaxationFeasibility", pulp.LpMinimize)
        self.add_constraints(feas_prob, incl_slack=True)

        slack_vars = [v for v in feas_prob.variables() if "SLACK" in v.name]
        solver = self._get_solver()

        feas_prob.setObjective(pulp.lpSum(slack_vars))
        feas_prob.solve(solver=solver)

        if feas_prob.objective.value() == 0:
            logger.info("Problem feasible. Continue")
        else:
            msg = (
                "Problem infeasible. Consider changing variables to make it possible:\n"
            )
            for v in slack_vars:
                if v.value() > 0:
                    msg += f'{v.name.lstrip("SLACK_")}: +{round(v.value())}\n'
            logger.error(msg)
        return feas_prob

    def _add_variable_in_same_group(
        self, student1: str, student2: str
    ) -> pulp.LpVariable:
        """Returns variable that contains wether student1 and student2 are in the same group

        Parameters
        ----------
        student1 : str
            Name of the first student
        student2 : str
            Name of the second student

        Returns
        -------
        pulp.LpVariable
            The variable that contains whether the two students are in the same group
        """
        group_vars = []
        for gr in self.groups_to:
            # Together in one group
            satisfied_per_group = pulp_logical.AND(
                self.prob,
                self.in_group[(student1, gr)],
                self.in_group[(student2, gr)],
            )
            group_vars.append(satisfied_per_group)
        # Theyare in the same group if it is correct for one group
        return pulp_logical.OR(self.prob, *group_vars)

    def add_variables_which_preferences_satisfied(self) -> dict:
        """Add all preferences to the LP-problem, so we can optimize how many we can fulfill

        Returns
        -------
        dict
            Dictionary of type pulp.LpVariable.dicts
            Contains for each preference wether it is satisfied or not
        """
        graag_met = self.preferences.xs("Graag met", level="TypeWens")
        satisfied = pulp.LpVariable.dicts(
            "Satisfied", graag_met.index.to_list(), cat="Binary"
        )

        for key, row in graag_met.iterrows():
            student, _ = key
            other = row["Waarde"]
            if other in self.groups_to:
                in_same_group = self.in_group[(student, other)]
            else:
                in_same_group = self._add_variable_in_same_group(student, other)

            if row["Gewicht"] > 0:
                self.prob += satisfied[key] == in_same_group
            else:
                self.prob += satisfied[key] == 1 - in_same_group
        return satisfied

    def _calculate_n_satisfied_optimization(self, satisfied: dict) -> pulp.LpVariable:
        """Calculate the total number of satisfied preferences."""
        return pulp.lpSum(satisfied)

    def _calculate_weighted_preferences(self, satisfied: dict) -> pulp.LpVariable:
        """Calculate the weighted sum of satisfied preferences."""
        graag_met = self.preferences.xs("Graag met", level="TypeWens")
        weights = graag_met["Gewicht"].to_dict()
        weights_pulp = pulp.LpVariable.dicts(
            "Weights_preferences", graag_met.index.to_list(), cat="Continuous"
        )
        weighted_satisfied = pulp.LpVariable.dicts(
            "WeightedSatisfied", graag_met.index.to_list(), cat="Continuous"
        )

        for key, weight in weights.items():
            self.prob += weights_pulp[key] == weight
            if weight > 0:
                # Weight is positive: you get points for getting it right
                self.prob += weighted_satisfied[key] == (satisfied[key] * weight)
            else:
                # Weight is negative: you get deduction if you do it wrong
                self.prob += weighted_satisfied[key] == ((1 - satisfied[key]) * weight)

        return weighted_satisfied

    def _calculate_weighted_preference_optimization(
        self, satisfied: dict
    ) -> pulp.LpVariable:
        weighted_satisfied = self._calculate_weighted_preferences(satisfied)
        return pulp.lpSum(weighted_satisfied)

    def _calculate_student_satisfaction(self, satisfied: dict) -> pulp.LpVariable:
        added_satisfaction = preferences_utils.calculate_added_satisfaction(
            self.preferences
        )
        weighted_satisfied = self._calculate_weighted_preferences(satisfied)

        for student in self.students:
            student_weighted = [
                weighted_satisfied.get((student, i), 0)
                for i in range(1, len(added_satisfaction) + 1)
            ]
            wp_satisfied = pulp.lpSum(student_weighted)

            wp_satisfied_per_student = pulp.LpVariable.dicts(
                f"{student}_weighted_preferences_accountend",
                added_satisfaction.keys(),
                cat="Binary",
            )

            preferences_utils.apply_threshold_constraints(
                self.prob,
                wp_satisfied,
                added_satisfaction.keys(),
                wp_satisfied_per_student,
                eps=1e-3,  # Necessary to run lexmaxmin without errors; I dont know why
            )

            satisfaction_current_student = pulp.lpSum(
                val * wp_satisfied_per_student[n_wp]
                for n_wp, val in added_satisfaction.items()
            )

            with warnings.catch_warnings(
                action="ignore", category=pd.errors.PerformanceWarning
            ):
                # Add base satisfaction if no (positive) preferences, so maxmin optimizes
                # for student with actual preferences
                try:
                    preferences = self.preferences.loc[(student, "Graag met")]
                except KeyError:
                    satisfaction_current_student = 1
                else:
                    positive_preferences = preferences.query("Gewicht > 0")
                    if positive_preferences.empty:
                        satisfaction_current_student += 1
                    else:
                        max_wishes = positive_preferences["Gewicht"].sum()
                        max_satisfaction = preferences_utils.get_satisfaction_integral(
                            0, max_wishes
                        )
                        satisfaction_current_student /= max_satisfaction
            self.prob += (
                self.studentsatisfaction[student] == satisfaction_current_student
            )
        return self.studentsatisfaction

    def _get_solver(self):
        kwargs = {"logPath": "solver.log", "msg": False}
        if pulp.HiGHS_CMD().available():
            solver = pulp.HiGHS_CMD(**kwargs, gapRel=0)
        else:
            logger.warning("Falling back to CBC solver. Might be very slow!")
            solver = pulp.PULP_CBC_CMD(**kwargs)
        return solver

    def set_optimization_target(self, studentsatisfaction: dict) -> None:
        """Calculate the variables which can be directly optimized

        For each option of the class, this calculates the variable from the underlying
        (possibly weighted) preferences or satisfaction

        Parameters
        ----------
        studentsatisfaction : dict
            Dictionary of type pulp.LpVariable.dicts
            Contains satisfaction for each student

        Returns
        -------
        dict
            Keys the possible optimization strategies of the class
            Values the LpVariables which sum the underlying (satisfied) preferences

        """

        if self.optimize == "studentsatisfaction":
            optimization_target = optimizationstrategies.total(studentsatisfaction)
        elif self.optimize == "least_satisfied":
            optimization_target = optimizationstrategies.lowest_score(
                studentsatisfaction, self.prob
            )
        elif self.optimize == "lexmaxmin":
            optimization_target = optimizationstrategies.plateaud_lexmaxmin(
                studentsatisfaction,
                self.prob,
                satisfaction_max=0.8,
                solver=self._get_solver(),
            )
        else:
            raise ValueError(f"Unknown optimization strategy {self.optimize!r}")
        self.prob += optimization_target

    def _constraint_not_solution(self, solution, distance=1):
        """Add constraint that solution is not allowed

        Parameters
        ----------
        solution : dictionary
            of shape .in_group, with fixed values
        distance : int, optional
            how many values must at least be different, by default 1
        """
        self.prob += (
            pulp.lpSum(
                [
                    self.in_group[key] if solution[key] == 0 else 1 - self.in_group[key]
                    for key in solution.keys()
                ]
            )
            >= distance
        )

    def solve(self, solutions_to_ignore=None) -> None:
        """Mathematically solve the problem

        Parameters
        ----------
        solutions_to_ignore : Iterable of tuples
            Iterable of 2-tuples. First element must be a dictionary for solutions which
            should not be allowed (e.g. because they are already known). Should be
            dictionaries with key (student, group) and value 0 or . The second element
            must be an int that declares the distance

        Raises
        ------
        RuntimeError
            If the problem is infeasible
        """
        if solutions_to_ignore is not None:
            for solution, dist in solutions_to_ignore:
                self._constraint_not_solution(solution, distance=dist)

        solver = self._get_solver()
        self.prob.solve(solver)
        if pulp.LpStatus[self.prob.status] != "Optimal":
            raise RuntimeError(
                f"Could not solve LP-problem, status {pulp.LpStatus[self.prob.status]!r}"
            )
        self.known_solutions.append(
            {k: round(v.value()) for k, v in self.in_group.items()}
        )

    def run(
        self, save=True, overwrite=False, n_solutions=1, distance=1
    ) -> pulp.LpProblem:
        """Set up and solve the LpProblem

        Parameters
        ----------
        save : bool (default = True)
            Whether to save the outcomes
        overwrite : bool
            Whether to allow overwriting previous solution file
        n_solutions : int, (default = 1)
            The number of solutions to find.
        distance : int
            The distance that must be held from each known solution

        Returns
        -------
        pulp.LpProblem
            The solved LpProblem
        """
        if self.optimize == "lexmaxmin" and n_solutions > 1:
            raise NotImplementedError(
                "Can not generate multiple solutions for lexmaxmin"
            )
        if save:
            os.makedirs(self.get_solution_name(), exist_ok=True)
        if not self.prob.constraints and self.prob.objective is None:
            self.add_constraints()
            satisfied = self.add_variables_which_preferences_satisfied()
            studentsatisfaction = self._calculate_student_satisfaction(satisfied)
            self.set_optimization_target(studentsatisfaction)

        for i in range(n_solutions):
            solutions_to_ignore = [(sol, distance) for sol in self.known_solutions]
            try:
                self.solve(solutions_to_ignore=solutions_to_ignore)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to find {i + 1} solution(s)") from e
            if save:
                fname = os.path.join(
                    self.get_solution_name(), f"{len(self.known_solutions)}.json"
                )
                self.save(fname, overwrite=overwrite)
        return self.prob

    def save(self, fname: str, overwrite=False) -> None:
        """
        Save variables and model to a json file

        Parameters
        ----------

        fname : str
            The file name to write to
        overwrite : bool
            Whether to allow overwriting previous solution file
        Raises
        ------
            FileExistsError
            If overwrite isn't allowed, but file exists
        """
        if not overwrite and os.path.exists(fname):
            raise FileExistsError(
                f"The file '{fname}' already exists. Operation aborted."
            )
        if pulp.LpStatus[self.prob.status] != "Optimal":
            warnings.warn("Writing non-optimal solution")
        self.prob.to_json(fname)
