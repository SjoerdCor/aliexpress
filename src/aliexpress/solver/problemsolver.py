"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import logging
import math
import warnings
from dataclasses import dataclass

import pandas as pd
import pulp

from . import feasibility, optimizationstrategies, preferences_utils, pulp_logical
from ._balance import STRICTEST_BALANCE, GroupBalance, get_solver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupComposition:
    """Boys/girls counts for one target group: total and for the new cohort (year)."""

    boys_total: int
    girls_total: int
    boys_year: int
    girls_year: int


@dataclass(frozen=True)
class SolutionResult:
    """Structured outcome of a solved distribution, read straight from the solver.

    Consumed by :class:`~aliexpress.solutions.SolutionAnalyzer`; it replaces parsing the
    solution back out of pulp variable names. Every field holds plain Python values (no
    pulp objects), so the result is straightforward to serialise once a persistence route
    is needed.

    The ``(student, Nr)`` keys index the positive ("Graag met") wishes: ``Nr`` is the
    wish's sequence number within that student's wishes; its target (a classmate or group)
    lives in ``preferences.loc[(student, "Graag met", Nr), "Waarde"]``.
    """

    assignment: dict[str, str]  # student -> assigned group
    student_satisfaction: dict[str, float]  # student -> relative satisfaction (0..1)
    satisfied: dict[tuple[str, int], bool]  # (student, Nr) -> wish fulfilled
    weighted_satisfied: dict[tuple[str, int], float]  # (student, Nr) -> weighted value
    weights: dict[tuple[str, int], float]  # (student, Nr) -> wish weight (signed)
    group_composition: dict[str, GroupComposition]  # group -> boys/girls counts


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
        where satisfaction is dominated by getting at least 1 preference),
        "least_satisfied" (formally, the least satisfied student), or "lexmaxmin"
        (plateaud lexicographic max-min over student satisfaction)
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
        groupbalance: GroupBalance | None = None,
        optimize="studentsatisfaction",
    ):
        self.preferences = preferences
        self.students = students
        self.groups_to = groups_to
        self.not_together = not_together
        self._validate_not_together_students_exist()

        # A mutable default (GroupBalance()) would be created once and shared by every
        # instance; use None so each ProblemSolver gets its own default constraints.
        self.groupbalance = groupbalance if groupbalance is not None else GroupBalance()
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

        # Solver outputs captured during the main solve, read back by extract_solution.
        # They stay None until a main solve runs (the subproblems do not set them).
        self.satisfied = None
        self.weighted_satisfied = None
        self.weights = None
        self.boys_in_group = None
        self.girls_in_group = None
        self.boys_to_group = None
        self.girls_to_group = None

    def _constraint_student_to_exactly_one_group(self, prob):
        for student in self.students:
            prob += (
                pulp.lpSum([self.in_group[(student, gr)] for gr in self.groups_to]) == 1
            )

    def _constraint_equal_new_students(self, prob, make_soft=True):
        """Every group should have an approximately equal number of new students"""

        slack_var = pulp.LpVariable(
            "SLACK_diff_n_students_year",
            lowBound=0,
            upBound=None if make_soft else 0,
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

    def _constraint_equal_total_students(self, prob, make_soft=True):
        current_per_group = {
            gr: self.groups_to[gr]["Jongens"] + self.groups_to[gr]["Meisjes"]
            for gr in self.groups_to
        }

        slack_var = pulp.LpVariable(
            "SLACK_diff_n_students_total",
            lowBound=0,
            upBound=None if make_soft else 0,
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

    def _constraint_equal_students_from_previous_group(self, prob, make_soft=False):
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
            upBound=None if make_soft else 0,
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

    def _constraint_clique_sex_group(self, prob, make_soft=False):
        """Every group can have a max number of students of the samen sex
        from an earlier group (no cliques)"""
        groups_from = {self.students[student]["Stamgroep"] for student in self.students}
        sexes = {self.students[student]["Jongen/meisje"] for student in self.students}
        slack_var = pulp.LpVariable(
            "SLACK_max_clique_sex",
            lowBound=0,
            upBound=None if make_soft else 0,
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

    def _constraint_equal_boys_girls(self, prob, make_soft=False):
        boys_to_group = pulp.LpVariable.dicts(
            "boys_to_group", self.groups_to.keys(), cat="Integer"
        )
        girls_to_group = pulp.LpVariable.dicts(
            "girls_to_group", self.groups_to.keys(), cat="Integer"
        )

        slack_var = pulp.LpVariable(
            "SLACK_balanced_boys_girls_year",
            lowBound=0,
            upBound=None if make_soft else 0,
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

        # Keep the new-cohort (year) counts of the main problem for the solution report.
        if prob is self.prob:
            self.boys_to_group = boys_to_group
            self.girls_to_group = girls_to_group

    def _constraint_balanced_boys_girls_total(self, prob, make_soft=False):
        boys_in_group = pulp.LpVariable.dicts(
            "boys_in_group", self.groups_to.keys(), cat="Integer"
        )
        girls_in_group = pulp.LpVariable.dicts(
            "girls_in_group", self.groups_to.keys(), cat="Integer"
        )

        slack_var = pulp.LpVariable(
            "SLACK_balanced_boys_girls_total",
            lowBound=0,
            upBound=None if make_soft else 0,
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

        # Keep the total (current + new) counts of the main problem for the solution report.
        if prob is self.prob:
            self.boys_in_group = boys_in_group
            self.girls_in_group = girls_in_group

    def _constraint_not_in_forbidden_group(self, prob):
        """Some students can not move int other groups (e.g. a brother/sister is already there)"""
        for i, row in self.preferences.query('TypeWens == "Niet in"').iterrows():
            student, _, _ = i
            gr = row["Waarde"]
            prob += self.in_group[(student, gr)] == 0

    def constraint_not_together(self, prob, make_soft=False):
        """Enforces constraint of difficult students not being together.

        With ``make_soft`` each rule gets a slack variable (``sum <= max + slack``) so a
        diagnosis can relax this family; returns the slack variables (empty list when hard).
        """
        slacks = []
        for i, dct in enumerate(self.not_together):
            if make_soft:
                slack = pulp.LpVariable(f"SLACK_not_together_{i}", lowBound=0)
                slacks.append(slack)
            else:
                slack = 0
            for group_to in self.groups_to:
                prob += (
                    pulp.lpSum(
                        [
                            self.in_group[(student, group_to)]
                            for student in self.students
                            if student in dct["group"]
                        ]
                    )
                    <= dct["Max_aantal_samen"] + slack
                )
        return slacks

    def constraint_minimal_satisfaction(self, prob, make_soft=False):
        """Force each student's satisfaction to its minimum (UI: "Extra zekerheid").

        With ``make_soft`` each floor gets a slack variable so a diagnosis can relax this
        family; returns the slack variables (empty list when hard).
        """
        slacks = []
        for student, info in self.students.items():
            if math.isnan(info["MinimaleTevredenheid"]):
                continue
            floor = info["MinimaleTevredenheid"]
            if make_soft:
                slack = pulp.LpVariable(f"SLACK_min_satisfaction_{student}", lowBound=0)
                slacks.append(slack)
                prob += self.studentsatisfaction[student] + slack >= floor
            else:
                prob += (
                    self.studentsatisfaction[student] >= floor
                ), f"MinimalSatisfaction{student}"
        return slacks

    def add_fundamental_constraints(self, prob):
        """Add constraints fundamental to a solution"""
        self._constraint_student_to_exactly_one_group(prob)
        self._constraint_not_in_forbidden_group(prob)

    def add_class_balance_constraints(self, prob, make_soft=False):
        """Add constraints to force good class balance in next groups"""
        self._constraint_equal_new_students(prob, make_soft)
        self._constraint_equal_total_students(prob, make_soft)
        self._constraint_equal_boys_girls(prob, make_soft)
        self._constraint_balanced_boys_girls_total(prob, make_soft)
        self._constraint_equal_students_from_previous_group(prob, make_soft)
        self._constraint_clique_sex_group(prob, make_soft)

    def add_satisfaction_constraints(self, prob):
        """Add constraints about social dynamics"""
        self.constraint_not_together(prob)
        self.constraint_minimal_satisfaction(prob)

    def add_constraints(self, prob=None, make_soft=False):
        """Add all hard constraints via the functions per constraint"""

        prob = prob or self.prob
        self.add_fundamental_constraints(prob)
        self.add_class_balance_constraints(prob, make_soft)
        self.add_satisfaction_constraints(prob)

    def solve_within_minimal_relaxation(self):
        """Solve, maximizing satisfaction within the *minimal* class-balance relaxation that
        still lets every student fulfil at least one positive wish.

        Picking one concrete "tightest" balance is ill-defined: several balances share the
        same minimal relaxation yet lead to different satisfaction, so a solver would pick
        one arbitrarily. Instead this works in two stages:

        1. Compute the minimal relaxation budget ``R*`` (a unique value): the smallest
           weighted balance relaxation under which every student can still reach a positive
           wish. The per-student wish requirement lives *only* here, where it shapes ``R*``.
        2. Run the normal lexmaxmin solve with the balance limits kept *soft* and their total
           weighted relaxation capped at ``R*``.

        Maximizing satisfaction over the whole minimal-relaxation region (rather than over a
        single arbitrarily chosen balance) makes the satisfaction the unique optimum, so the
        outcome is well-defined and solver-independent (the in-process and CLI HiGHS solvers
        agree). Within that budget lexmaxmin maximizes the lowest satisfaction and therefore
        already gives every student a positive wish, so no explicit wish floor is needed in
        this stage - it belongs only to stage 1.
        """
        budget = feasibility.minimal_relaxation_budget(self)
        self.groupbalance = STRICTEST_BALANCE  # main solve also uses the strict base

        self.add_fundamental_constraints(self.prob)
        self.add_class_balance_constraints(self.prob, make_soft=True)
        self.add_satisfaction_constraints(self.prob)
        satisfied = self.add_variables_which_preferences_satisfied()
        self.satisfied = satisfied
        studentsatisfaction = self.calculate_student_satisfaction(satisfied)
        self.prob += feasibility.weighted_relaxation(self.prob) <= budget + 1e-6
        self.set_optimization_target(studentsatisfaction)
        self.solve()

    def _add_variable_in_same_group(
        self, student1: str, student2: str, prob: pulp.LpProblem = None
    ) -> pulp.LpVariable:
        """Returns variable that contains wether student1 and student2 are in the same group

        Parameters
        ----------
        student1 : str
            Name of the first student
        student2 : str
            Name of the second student
        prob : pulp.LpProblem, optional
            Problem to add the constraints to. Defaults to ``self.prob``.

        Returns
        -------
        pulp.LpVariable
            The variable that contains whether the two students are in the same group
        """
        prob = prob or self.prob
        group_vars = []
        for gr in self.groups_to:
            # Together in one group
            satisfied_per_group = pulp_logical.AND(
                prob,
                self.in_group[(student1, gr)],
                self.in_group[(student2, gr)],
            )
            group_vars.append(satisfied_per_group)
        # Theyare in the same group if it is correct for one group
        return pulp_logical.OR(prob, *group_vars)

    def add_variables_which_preferences_satisfied(
        self, prob: pulp.LpProblem = None
    ) -> dict:
        """Add all preferences to the LP-problem, so we can optimize how many we can fulfill

        Parameters
        ----------
        prob : pulp.LpProblem, optional
            Problem to add the constraints to. Defaults to ``self.prob``.

        Returns
        -------
        dict
            Dictionary of type pulp.LpVariable.dicts
            Contains for each preference wether it is satisfied or not
        """
        prob = prob or self.prob
        graag_met = preferences_utils.get_graag_met(self.preferences)
        satisfied = pulp.LpVariable.dicts(
            "Satisfied", graag_met.index.to_list(), cat="Binary"
        )

        for key, row in graag_met.iterrows():
            student, _ = key
            other = row["Waarde"]
            if other in self.groups_to:
                in_same_group = self.in_group[(student, other)]
            else:
                in_same_group = self._add_variable_in_same_group(
                    student, other, prob=prob
                )

            if row["Gewicht"] > 0:
                prob += satisfied[key] == in_same_group
            else:
                prob += satisfied[key] == 1 - in_same_group
        return satisfied

    def _calculate_weighted_preferences(
        self, satisfied: dict, prob: pulp.LpProblem = None
    ) -> pulp.LpVariable:
        """Calculate the weighted sum of satisfied preferences."""
        prob = prob or self.prob
        graag_met = preferences_utils.get_graag_met(self.preferences)
        weights = graag_met["Gewicht"].to_dict()
        weights_pulp = pulp.LpVariable.dicts(
            "Weights_preferences", graag_met.index.to_list(), cat="Continuous"
        )
        weighted_satisfied = pulp.LpVariable.dicts(
            "WeightedSatisfied", graag_met.index.to_list(), cat="Continuous"
        )

        for key, weight in weights.items():
            prob += weights_pulp[key] == weight
            if weight > 0:
                # Weight is positive: you get points for getting it right
                prob += weighted_satisfied[key] == (satisfied[key] * weight)
            else:
                # Weight is negative: you get deduction if you do it wrong
                prob += weighted_satisfied[key] == ((1 - satisfied[key]) * weight)

        # Keep the main problem's weighted preferences and their (signed) weights for
        # the solution report.
        if prob is self.prob:
            self.weighted_satisfied = weighted_satisfied
            self.weights = weights

        return weighted_satisfied

    def calculate_student_satisfaction(
        self, satisfied: dict, prob: pulp.LpProblem = None
    ) -> pulp.LpVariable:
        """Compute per-student satisfaction variables and add them to ``prob``."""
        prob = prob or self.prob
        added_satisfaction = preferences_utils.calculate_added_satisfaction(
            self.preferences
        )
        weighted_satisfied = self._calculate_weighted_preferences(satisfied, prob=prob)

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
                prob,
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
            prob += self.studentsatisfaction[student] == satisfaction_current_student
        return self.studentsatisfaction

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
                solver=get_solver(),
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

        solver = get_solver()
        self.prob.solve(solver)
        if pulp.LpStatus[self.prob.status] != "Optimal":
            raise RuntimeError(
                f"Could not solve LP-problem, status {pulp.LpStatus[self.prob.status]!r}"
            )
        self.known_solutions.append(
            {k: round(v.value()) for k, v in self.in_group.items()}
        )

    def run(self, n_solutions=1, distance=1) -> pulp.LpProblem:
        """Set up and solve the LpProblem

        Parameters
        ----------
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
        if not self.prob.constraints and self.prob.objective is None:
            self.add_constraints()
            satisfied = self.add_variables_which_preferences_satisfied()
            self.satisfied = satisfied
            studentsatisfaction = self.calculate_student_satisfaction(satisfied)
            self.set_optimization_target(studentsatisfaction)

        for i in range(n_solutions):
            solutions_to_ignore = [(sol, distance) for sol in self.known_solutions]
            try:
                self.solve(solutions_to_ignore=solutions_to_ignore)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to find {i + 1} solution(s)") from e
        return self.prob

    def extract_solution(self) -> SolutionResult:
        """Read the solved problem into a structured :class:`SolutionResult`.

        Reads ``.value()`` straight off the solver variables captured during the main
        solve - no serialisation and no parsing of variable names. Must be called after
        a solve (``run`` or ``solve_within_minimal_relaxation``).

        Raises
        ------
        RuntimeError
            If the problem has not been solved to optimality.
        """
        if pulp.LpStatus[self.prob.status] != "Optimal":
            raise RuntimeError(
                f"Can not extract a solution, status {pulp.LpStatus[self.prob.status]!r}"
            )

        assignment = {
            student: group
            for (student, group), var in self.in_group.items()
            if round(var.value()) == 1
        }
        return SolutionResult(
            assignment=assignment,
            student_satisfaction={
                student: var.value()
                for student, var in self.studentsatisfaction.items()
            },
            satisfied={
                key: bool(round(var.value())) for key, var in self.satisfied.items()
            },
            weighted_satisfied={
                key: var.value() for key, var in self.weighted_satisfied.items()
            },
            weights=dict(self.weights),
            group_composition=self._group_composition(),
        )

    def _group_composition(self) -> dict[str, GroupComposition]:
        """Per target group, the boys/girls counts read from the solved count variables."""
        return {
            group: GroupComposition(
                boys_total=round(self.boys_in_group[group].value()),
                girls_total=round(self.girls_in_group[group].value()),
                boys_year=round(self.boys_to_group[group].value()),
                girls_year=round(self.girls_to_group[group].value()),
            )
            for group in self.groups_to
        }
