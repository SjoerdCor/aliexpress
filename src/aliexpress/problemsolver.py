"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import math
import os
import warnings

import pandas as pd
import pulp

from . import pulp_logical


def _apply_threshold_constraints(
    prob, value, thresholds, threshold_vars, M=1_000_000, eps=1e-6
):
    """
    Adds threshold-based indicator constraints to a PuLP problem.

    This function ensures that each binary variable in `threshold_vars` correctly
    tracks whether `value` has met or exceeded a given threshold. It enforces
    logical conditions using big-M constraints to approximate indicator behavior.

    Parameters
    ----------
    prob : pulp.LpProblem
        The linear programming problem to which constraints are added.
    value : pulp.LpVariable
        The continuous decision variable being compared to thresholds.
    thresholds : iterable of float
        The threshold values that determine activation of binary variables.
    threshold_vars : dict of {float: pulp.LpVariable}
        A dictionary mapping each threshold to a corresponding binary variable.
    """

    for threshold in thresholds:
        if threshold > 0:
            prob += threshold_vars[threshold] <= value * (1 / threshold) + eps
            prob += threshold_vars[threshold] >= (value - (threshold - eps)) / M
        else:
            prob += threshold_vars[threshold] >= value * (1 / threshold) - eps
            prob += threshold_vars[threshold] <= (value - (threshold + eps)) / M


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def all_unique_sums(iterable):
    """Calculate all possible sums from sublists from the iterable"""
    return {sum(l) for l in powerset(iterable)}


def get_possible_weighted_preferences(preferences: pd.DataFrame) -> set:
    """
    Get all the possible number of weighted preferences

    This will be used to know for which values a satisfaction score must be calculated
    and which dictionary values must be calculated per student. By minimizing this number,
    we make the problem calculation as fast as possible, while allowing for arbitrary precision

    Parameters
    ----------
    preferences: pd.DataFrame
        The DataFrame containing the preferences of the students, must have a MultiIndex
        with levels ("Leerling", "TypeWens") with columns ("Waarde" & "Gewicht")
    """
    unique_weighted_preferences_per_student = (
        preferences.xs("Graag met", level="TypeWens")
        .groupby("Leerling")["Gewicht"]
        .apply(all_unique_sums)
    )

    unique_weighted_preferences = set()
    for wp in unique_weighted_preferences_per_student:
        unique_weighted_preferences.update(wp)
    return unique_weighted_preferences


def get_satisfaction_integral(x_a: float, x_b: float) -> float:
    """
    Calculate the extra satisfaction from granting x_b preferences instead of x_a

    This is the (scaled) integral of 0.5**x. This satisfaction function ensures everybody
    first gets their first preference, then everybody their second preference, etc.

    Parameters
    ----------
    x_a: float
        The number of (weighted) preferences as the basic satisfaction of the student
    x_b: float
        The number of (weighted) preferences as the goal satisfaction of the student

    Returns
    -------
        The added satisfaction score of the student
    """
    # In principle, we should probably only specify the satisfaction function and
    # then have this just be a numerical integration for optimal flexibility, but since
    # this flexibility isn't required yet, we're using a analytical integration.

    return (-(0.5**x_b)) - (-(0.5**x_a))


def calculate_added_satisfaction(preferences) -> dict:
    """
    Calculate the score of getting all possible weighted preferences values accounted for
    """

    possible_weighted_preferences = get_possible_weighted_preferences(preferences)

    # Sorting is important since we're going to difference!
    positive_values = sorted(v for v in possible_weighted_preferences if v >= 0)
    negative_values = sorted(
        (v for v in possible_weighted_preferences if v <= 0), reverse=True
    )

    preference_value = {}
    for values in (negative_values, positive_values):
        # The 0 value is deliberately not taken into account!
        # This would lead to ZeroDivisionErrors
        for last_wp, wp in zip(values[:-1], values[1:]):
            preference_value[wp] = get_satisfaction_integral(last_wp, wp)
    return preference_value


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

    max_clique, int (default = 5)
        The number of students that can go to the same group

    max_clique_sex, int (default = 3)
        The number of students from an original group of the same sex that can go
        to the same group

    max_diff_n_students_year, float (default = 2)
        The maximum difference between assigned students to the largest group
        and the smallest group

    max_diff_n_students_total, float (default = 3)
        The maximum difference between largest group and the smallest group (in total)

    max_imbalance_boys_girls_year, float (default = 2)
        The maximum difference between number of boys and girls in each year in a group

    max_imbalance_boys_girls_total, float (default = 3)
        The maximum difference between number of boys and girls in the total group

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
        max_clique=5,
        max_clique_sex=3,
        max_diff_n_students_year=2,
        max_diff_n_students_total=3,
        max_imbalance_boys_girls_year=2,
        max_imbalance_boys_girls_total=3,
        optimize="studentsatisfaction",
    ):
        self.preferences = preferences
        self.students = students
        self.groups_to = groups_to
        self.not_together = not_together
        self._validate_not_together_students_exist()

        self.max_clique = max_clique
        self.max_clique_sex = max_clique_sex
        self.max_diff_n_students_year = max_diff_n_students_year
        self.max_diff_n_students_total = max_diff_n_students_total
        self.max_imbalance_boys_girls_year = max_imbalance_boys_girls_year
        self.max_imbalance_boys_girls_total = max_imbalance_boys_girls_total
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

    def get_solution_name(self):
        """Create name from config to identify the solution"""
        attrs = [
            self.optimize,
            self.max_clique,
            self.max_clique_sex,
            self.max_diff_n_students_total,
            self.max_diff_n_students_year,
            self.max_imbalance_boys_girls_total,
            self.max_imbalance_boys_girls_year,
        ]
        return "".join(str(s) for s in attrs)

    def _constraint_student_to_exactly_one_group(self):
        for student in self.students:
            self.prob += (
                pulp.lpSum([self.in_group[(student, gr)] for gr in self.groups_to]) == 1
            )

    def _constraint_equal_new_students(self):
        """Every group should have an approximately equal number of new students"""
        avg_new_per_group = len(self.students) / len(self.groups_to)
        min_in_group = int(avg_new_per_group - self.max_diff_n_students_year / 2)
        max_in_group = int(avg_new_per_group + self.max_diff_n_students_year / 2)

        new_students_in_group = pulp.LpVariable.dict(
            "new_students_in_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to in self.groups_to:

            self.prob += new_students_in_group[group_to] == pulp.lpSum(
                [self.in_group[(student, group_to)] for student in self.students]
            )

            self.prob += new_students_in_group[group_to] <= max_in_group
            self.prob += new_students_in_group[group_to] >= min_in_group

    def _constraint_equal_total_students(self):
        current_per_group = {
            gr: self.groups_to[gr]["Jongens"] + self.groups_to[gr]["Meisjes"]
            for gr in self.groups_to
        }
        avg_per_group = (len(self.students) + sum(current_per_group.values())) / len(
            current_per_group
        )
        min_in_group = int(avg_per_group - self.max_diff_n_students_total / 2)
        max_in_group = int(avg_per_group + self.max_diff_n_students_total / 2)
        total_in_group = pulp.LpVariable.dict(
            "total_in_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to in self.groups_to:

            self.prob += total_in_group[group_to] == (
                pulp.lpSum(
                    [self.in_group[(student, group_to)] for student in self.students]
                )
                + current_per_group[group_to]
            )

            self.prob += total_in_group[group_to] <= max_in_group
            self.prob += total_in_group[group_to] >= min_in_group

    def _constraint_equal_students_from_previous_group(self):
        """Every group can have a max number of students from an earlier group (no cliques)"""
        groups_from = {self.students[student]["Stamgroep"] for student in self.students}
        from_group_to_group = pulp.LpVariable.dicts(
            "from_group_to_group",
            itertools.product(groups_from, self.groups_to.keys()),
            cat="Integer",
        )

        for group_to in self.groups_to:
            for group_from in groups_from:
                self.prob += from_group_to_group[(group_from, group_to)] == pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Stamgroep"] == group_from
                    ]
                )

                self.prob += (
                    from_group_to_group[(group_from, group_to)] <= self.max_clique
                )

    def _constraint_clique_sex_group(self):
        """Every group can have a max number of students of the samen sex
        from an earlier group (no cliques)"""
        groups_from = {self.students[student]["Stamgroep"] for student in self.students}
        sexes = {self.students[student]["Jongen/meisje"] for student in self.students}

        for group_to in self.groups_to:
            for group_from in groups_from:
                for sex in sexes:
                    this_clique = [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Stamgroep"] == group_from
                        and self.students[student]["Jongen/meisje"] == sex
                    ]

                    self.prob += pulp.lpSum(this_clique) <= self.max_clique_sex

    def _constraint_equal_boys_girls(self):
        boys_to_group = pulp.LpVariable.dicts(
            "boys_to_group", self.groups_to.keys(), cat="Integer"
        )
        girls_to_group = pulp.LpVariable.dicts(
            "girls_to_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to in self.groups_to:
            self.prob += boys_to_group[group_to] == pulp.lpSum(
                [
                    self.in_group[(student, group_to)]
                    for student in self.students
                    if self.students[student]["Jongen/meisje"] == "Jongen"
                ]
            )
            self.prob += girls_to_group[group_to] == pulp.lpSum(
                [
                    self.in_group[(student, group_to)]
                    for student in self.students
                    if self.students[student]["Jongen/meisje"] == "Meisje"
                ]
            )
            self.prob += (
                girls_to_group[group_to] - boys_to_group[group_to]
                <= self.max_imbalance_boys_girls_year
            )
            self.prob += (
                boys_to_group[group_to] - girls_to_group[group_to]
                <= self.max_imbalance_boys_girls_year
            )

    def _constraint_balanced_boys_girls_total(self):
        boys_in_group = pulp.LpVariable.dicts(
            "boys_in_group", self.groups_to.keys(), cat="Integer"
        )
        girls_in_group = pulp.LpVariable.dicts(
            "girls_in_group", self.groups_to.keys(), cat="Integer"
        )

        for group_to, n_boys_girls in self.groups_to.items():
            self.prob += boys_in_group[group_to] == (
                n_boys_girls["Jongens"]
                + pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Jongen/meisje"] == "Jongen"
                    ]
                )
            )
            self.prob += girls_in_group[group_to] == (
                n_boys_girls["Meisjes"]
                + pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students
                        if self.students[student]["Jongen/meisje"] == "Meisje"
                    ]
                )
            )
            self.prob += (
                girls_in_group[group_to] - boys_in_group[group_to]
                <= self.max_imbalance_boys_girls_total
            )
            self.prob += (
                boys_in_group[group_to] - girls_in_group[group_to]
                <= self.max_imbalance_boys_girls_total
            )

    def _constraint_not_in_forbidden_group(self):
        """Some students can not move int other groups (e.g. a brother/sister is already there)"""
        for i, row in self.preferences.query('TypeWens == "Niet in"').iterrows():
            student, _, _ = i
            gr = row["Waarde"]
            self.prob += self.in_group[(student, gr)] == 0

    def _constraint_not_together(self):
        """Enforces constraint of difficult students not being together"""
        for dct in self.not_together:
            for group_to in self.groups_to:
                self.prob += (
                    pulp.lpSum(
                        [
                            self.in_group[(student, group_to)]
                            for student in self.students
                            if student in dct["group"]
                        ]
                    )
                    <= dct["Max_aantal_samen"]
                )

    def _constraint_minimal_satisfaction(self):
        for student, info in self.students.items():
            if not math.isnan(info["MinimaleTevredenheid"]):
                self.prob += (
                    self.studentsatisfaction[student] >= info["MinimaleTevredenheid"]
                ), f"MinimalSatisfaction{student}"

    def add_constraints(self):
        """Add all hard constraints via the functions per constraint"""
        self._constraint_student_to_exactly_one_group()
        self._constraint_not_in_forbidden_group()

        self._constraint_equal_new_students()
        self._constraint_equal_total_students()
        self._constraint_equal_students_from_previous_group()
        self._constraint_equal_boys_girls()
        self._constraint_balanced_boys_girls_total()
        self._constraint_not_together()
        self._constraint_minimal_satisfaction()
        self._constraint_clique_sex_group()

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
        weighted_satisfied = pulp.LpVariable.dicts(
            "WeightedSatisfied", graag_met.index.to_list(), cat="Continuous"
        )

        for key, weight in weights.items():
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
        added_satisfaction = calculate_added_satisfaction(self.preferences)
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

            _apply_threshold_constraints(
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
                positive_preferences = self.preferences.loc[
                    (student, "Graag met")
                ].query("Gewicht > 0")
                if positive_preferences.empty:
                    # Add base satisfaction if no positive preferences, so maxmin optimizes
                    # for student with actual preferences
                    satisfaction_current_student += 1
                else:
                    max_wishes = positive_preferences["Gewicht"].sum()
                    max_satisfaction = get_satisfaction_integral(0, max_wishes)
                    satisfaction_current_student /= max_satisfaction
            self.prob += (
                self.studentsatisfaction[student] == satisfaction_current_student
            )
        return self.studentsatisfaction

    def _calculate_total_student_satisfaction(self, satisfied: dict) -> pulp.LpVariable:
        self._calculate_student_satisfaction(satisfied)
        return pulp.lpSum(self.studentsatisfaction)

    def _least_satisfied_student(self, satisfied: dict) -> pulp.LpVariable:
        self._calculate_student_satisfaction(satisfied)

        minimal_satisfaction = pulp.LpVariable("MinimalSatisfaction")
        for satisfaction in self.studentsatisfaction.values():
            self.prob += minimal_satisfaction <= satisfaction
        M = 1_000_000  # Large enough so min dominates sum
        return M * minimal_satisfaction + pulp.lpSum(self.studentsatisfaction.values())

    def _lex_max_min(self, satisfied: dict, n_levels=10) -> pulp.LpVariable:
        """
        Solve the approximate lexmaxmin problem for student satisfaction

        Uses an iterative solve, making use of the fact that student satisfaction is
        often plateaud: there are multiple students at the same level. Level by level,
        first the next lowest plateau is determined, and then the number of students
        on that plateau. When each number is found, it is then added as a constraint and
        continues solving. Total student satisfaction is the ultimate tie breaker

        Parameters
        ----------
        n_levels : int
            The number of plateaus to use. Higher means more precision, but slightly slower,
            although the last levels are usually very quick, when the solution is already
            fixed. Too high might result in an Infeasible problem
        """
        M = 100
        eps = 1e-6
        solver = self._get_solver()

        self._calculate_student_satisfaction(satisfied)
        for level in range(n_levels):

            # Step 1: maximize minimal satisfaction
            minimal_satisfaction = pulp.LpVariable(f"MinimalSatisfaction_{level}")
            if level == 0:
                for satisfaction in self.studentsatisfaction.values():
                    self.prob += minimal_satisfaction <= satisfaction
            else:
                self.prob += minimal_satisfaction >= m_val + eps
                for student, satisfaction in self.studentsatisfaction.items():
                    self.prob += (
                        minimal_satisfaction
                        <= satisfaction + (1 - has_this_level[student]) * M + eps
                    ), f"MinimalSatisfactionLT{student}_{level}"

            self.prob.sense = pulp.LpMaximize
            self.prob.setObjective(minimal_satisfaction)
            self.prob.solve(solver)
            m_val = minimal_satisfaction.value()
            print(f"Level {level}, step 1 done, {m_val}")
            # Add as constraint
            if level == 0:
                for student in self.students:
                    self.prob += self.studentsatisfaction[student] >= m_val
            else:
                for student in self.students:
                    self.prob += (
                        self.studentsatisfaction[student]
                        >= m_val * has_this_level[student] - eps
                    ), f"MinimalSatisfaction_{student}_{level}"

            # Useful for debugging - usually from numerical errors
            # if level > 0:
            #     self.prob.solve(solver)

            # Step 2: minimize its occurrence
            has_this_level = pulp.LpVariable.dicts(
                f"HasThisLevel_{level}", self.students.keys(), cat="Binary"
            )
            delta = 1e-5
            for student in self.students:
                has_this_level_student = pulp.LpVariable.dicts(
                    f"HasLevel_{level}_{student}", [m_val + delta], cat="Binary"
                )
                _apply_threshold_constraints(
                    self.prob,
                    self.studentsatisfaction[student],
                    [m_val + delta],
                    has_this_level_student,
                    M=100,
                )
                self.prob += (
                    has_this_level[student] == has_this_level_student[m_val + delta]
                )
            self.prob.sense = pulp.LpMaximize
            self.prob.setObjective(pulp.lpSum(has_this_level.values()))
            self.prob.solve(solver)

            count_at_level = sum(
                1
                for student in self.students
                if pulp.value(has_this_level[student]) > 0.5
            )

            print(f"Level {level}, step 2 done, {count_at_level}")

            # Add as constraint
            self.prob += pulp.lpSum(has_this_level.values()) == count_at_level

        return pulp.lpSum(self.studentsatisfaction.values())

    def _get_solver(self):
        kwargs = {"logPath": "solver.log", "msg": False}
        if pulp.HiGHS_CMD().available():
            solver = pulp.HiGHS_CMD(**kwargs, gapRel=0)
        else:
            solver = pulp.PULP_CBC_CMD(**kwargs)
        return solver

    def set_optimization_target(self, satisfied: dict) -> None:
        """Calculate the variables which can be directly optimized

        For each option of the class, this calculates the variable from the underlying
        (possibly weighted) preferences or satisfaction

        Parameters
        ----------
        satisfied : dict
            Dictionary of type pulp.LpVariable.dicts
            Contains for each preference wether it is satisfied or not

        Returns
        -------
        dict
            Keys the possible optimization strategies of the class
            Values the LpVariables which sum the underlying (satisfied) preferences

        """
        optimization_funcs = {
            "n_preferences": self._calculate_n_satisfied_optimization,
            "weighted_preferences": self._calculate_weighted_preference_optimization,
            "studentsatisfaction": self._calculate_total_student_satisfaction,
            "least_satisfied": self._least_satisfied_student,
            "lexmaxmin": self._lex_max_min,
        }
        optimization_func = optimization_funcs[self.optimize]
        optimization_target = optimization_func(satisfied)
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
            self.set_optimization_target(satisfied)

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
