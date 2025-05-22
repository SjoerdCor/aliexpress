"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import pandas as pd
import pulp

import pulp_logical


def _apply_threshold_constraints(prob, value, thresholds, threshold_vars):
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

    M = 1_000_000  # A very big number, so that constraints are never larger than 1
    EPS = 0.001  # A small number to correct for numerical inaccuracies

    for threshold in thresholds:
        if threshold > 0:
            prob += threshold_vars[threshold] <= value / threshold + EPS
            prob += threshold_vars[threshold] >= (value - (threshold - EPS)) / M
        else:
            prob += threshold_vars[threshold] >= value / threshold - EPS
            prob += threshold_vars[threshold] <= (value - (threshold + EPS)) / M


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

    max_clique, int (default = 5)
        The number of students that can go to the same group

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

    def __init__(
        self,
        preferences: pd.DataFrame,
        students: dict,
        groups_to: dict,
        max_clique=5,
        max_diff_n_students_year=2,
        max_diff_n_students_total=3,
        max_imbalance_boys_girls_year=2,
        max_imbalance_boys_girls_total=3,
        optimize="studentsatisfaction",
    ):
        self.preferences = preferences
        self.students = students
        self.groups_to = groups_to
        self.max_clique = max_clique
        self.max_diff_n_students_year = max_diff_n_students_year
        self.max_diff_n_students_total = max_diff_n_students_total
        self.max_imbalance_boys_girls_year = max_imbalance_boys_girls_year
        self.max_imbalance_boys_girls_total = max_imbalance_boys_girls_total
        self.optimize = optimize
        self.prob = pulp.LpProblem("studentdistribution", pulp.LpMaximize)
        self.in_group = self._define_variables()

    def _define_variables(self):
        return pulp.LpVariable.dicts(
            "group",
            itertools.product(self.students.keys(), self.groups_to.keys()),
            cat="Binary",
        )

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

    def add_constraints(self):
        """Add all hard constraints via the functions per constraint"""
        self._constraint_student_to_exactly_one_group()
        self._constraint_not_in_forbidden_group()

        self._constraint_equal_new_students()
        self._constraint_equal_total_students()
        self._constraint_equal_students_from_previous_group()
        self._constraint_equal_boys_girls()
        self._constraint_balanced_boys_girls_total()

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
        satisfaction_per_student = pulp.LpVariable.dict(
            "studentsatisfaction", self.students.keys(), cat="Continuous"
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

            _apply_threshold_constraints(
                self.prob,
                wp_satisfied,
                added_satisfaction.keys(),
                wp_satisfied_per_student,
            )

            satisfaction_per_student[student] = pulp.lpSum(
                val * wp_satisfied_per_student[n_wp]
                for n_wp, val in added_satisfaction.items()
            )
        return satisfaction_per_student

    def _calculate_total_student_satisfaction(self, satisfied: dict) -> pulp.LpVariable:
        satisfaction_per_student = self._calculate_student_satisfaction(satisfied)
        return pulp.lpSum(satisfaction_per_student)

    def _least_satisfied_student(self, satisfied: dict) -> pulp.LpVariable:
        satisfaction_per_student = self._calculate_student_satisfaction(satisfied)

        minimal_satisfaction = pulp.LpVariable("MinimalSatisfaction")
        for satisfaction in satisfaction_per_student.values():
            self.prob += minimal_satisfaction <= satisfaction
        M = 1_000_000  # Large enough so min dominates sum
        return M * minimal_satisfaction + pulp.lpSum(satisfaction_per_student.values())

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
        }
        optimization_func = optimization_funcs[self.optimize]
        optimization_target = optimization_func(satisfied)
        self.prob += optimization_target

    def solve(self) -> None:
        """Mathematically solve the problem

        Parameters
        ----------
        optimization_targets : dict
            Dictionary containing the possible optimization targets. Which one is chosen
            depends on the class setup

        Raises
        ------
        RuntimeError
            If the problem is infeasible
        """
        solver = pulp.PULP_CBC_CMD(logPath="debug.txt")
        self.prob.solve(solver)
        if pulp.LpStatus[self.prob.status] != "Optimal":
            raise RuntimeError(
                f"Could not solve LP-problem, status {pulp.LpStatus[self.prob.status]!r}"
            )

    def run(self) -> pulp.LpProblem:
        """Set up and solve the LpProblem

        Returns
        -------
        pulp.LpProblem
            The solved LpProblem
        """
        self.add_constraints()
        satisfied = self.add_variables_which_preferences_satisfied()
        self.set_optimization_target(satisfied)
        self.solve()
        return self.prob
