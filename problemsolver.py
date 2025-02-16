"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import pandas as pd
import pulp

import pulp_logical

M = 1_000_000  # A very big number, so that constraints are never larger than 1
EPS = 0.001  # A small number to correct for numerical inaccuracies


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

    students_per_group_from : dict
        Key: the name of the previous group. Value: list of students

    groups_to: Iterable
        An interable containing all names of the groups to which students can be sent

    max_kliekje, int (default = 5)
        The number of students that can go to the same group

    max_diff_n_students_per_group, float (default = 3)
        The maximum difference between assigned students to the largest group
        and the smallest group

    optimize, str (default = "studentsatisfaction")
        What to optimize for: "studentsatisfaction" (basically, the least happy student
        is the most happy),
        "n_preferences" or "weighted_preferences"
    """

    def __init__(
        self,
        preferences: pd.DataFrame,
        students_per_group_from,
        groups_to,
        max_kliekje=5,
        max_diff_n_students_per_group=3,
        optimize="studentsatisfaction",
    ):
        self.preferences = preferences
        self.students_per_group_from = students_per_group_from
        self.students = sum(self.students_per_group_from.values(), [])
        self.groups_to = groups_to
        self.max_kliekje = max_kliekje
        self.max_diff_n_students_per_group = max_diff_n_students_per_group
        self.optimize = optimize
        self.prob = pulp.LpProblem("studentdistribution", pulp.LpMaximize)
        self.in_group = self._define_variables()

    def _define_variables(self):
        return pulp.LpVariable.dicts(
            "group",
            itertools.product(self.students, self.groups_to),
            cat="Binary",
        )

    def _constraint_student_to_exactly_one_group(self):
        for student in self.students:
            self.prob += (
                pulp.lpSum([self.in_group[(student, gr)] for gr in self.groups_to]) == 1
            )

    def _constraint_equal_new_students(self):
        """ "Every group can have a max number of students from an earlier group (no kliekjes)"""
        avg_new_per_group = len(self.students) / len(self.groups_to)
        min_in_group = int(avg_new_per_group - self.max_diff_n_students_per_group / 2)
        max_in_group = int(avg_new_per_group + self.max_diff_n_students_per_group / 2)

        new_students_in_group = pulp.LpVariable.dict(
            "new_students_in_group", self.groups_to, cat="Integer"
        )

        for group_to in self.groups_to:

            self.prob += new_students_in_group[group_to] == pulp.lpSum(
                [self.in_group[(student, group_to)] for student in self.students]
            )

            self.prob += new_students_in_group[group_to] <= max_in_group
            self.prob += new_students_in_group[group_to] >= min_in_group

    def _constraint_equal_students_from_previous_group(self):
        """Every group can have a max number of students from an earlier group (no kliekjes)"""
        groups_from = list(self.students_per_group_from.keys())
        from_group_to_group = pulp.LpVariable.dicts(
            "from_group_to_group",
            itertools.product(groups_from, self.groups_to),
            cat="Integer",
        )

        for group_to in self.groups_to:
            for group_from in groups_from:
                self.prob += from_group_to_group[(group_from, group_to)] == pulp.lpSum(
                    [
                        self.in_group[(student, group_to)]
                        for student in self.students_per_group_from[group_from]
                    ]
                )

                self.prob += (
                    from_group_to_group[(group_from, group_to)] <= self.max_kliekje
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

        self._constraint_equal_new_students()
        self._constraint_equal_students_from_previous_group()

        self._constraint_not_in_forbidden_group()

    def add_variables_which_preferences_satisfied(self) -> dict:
        """Add all preferences to the LP-problem, so we can optimize how many we can fulfill

        Returns
        -------
        dict
            Dictionary of type pulp.LpVariable.dicts
            Contains for each preference wether it is satisfied or not
        """
        graag_met = self.preferences.xs("Graag met", level="TypeWens")
        weights = graag_met["Gewicht"].to_dict()

        satisfied = pulp.LpVariable.dicts(
            "Satisfied", graag_met.index.to_list(), cat="Binary"
        )
        # Auxiliary variables to check whether a preference is satisfied for a single
        # group. This can then be combined to check whether that tis true for all groups
        pref_per_group = list(
            itertools.chain(
                *[
                    [(student, nr, gr) for gr in self.groups_to]
                    for student, nr in graag_met.index
                ]
            )
        )
        satisfied_per_group = pulp.LpVariable.dicts(
            "Satisfied_per_group", pref_per_group, cat="Binary"
        )

        for i, row in graag_met.iterrows():
            student, nr = i
            if row["Waarde"] not in self.groups_to:
                other_student = row["Waarde"]
                if weights[i] > 0:
                    for gr in self.groups_to:
                        # Matching preferences are an XNOR problem: if for every group
                        # either both or none are in them, they are in the same group
                        satisfied_per_group[(student, nr, gr)] = pulp_logical.XNOR(
                            self.prob,
                            self.in_group[(student, gr)],
                            self.in_group[(other_student, gr)],
                        )
                    # The preference is satisfied if it is correct for every group
                    group_vars = [
                        satisfied_per_group[(student, nr, gr)] for gr in self.groups_to
                    ]
                    pulp_logical.AND(self.prob, *group_vars, result_var=satisfied[i])

                else:
                    for gr in self.groups_to:
                        # This is the NAND variant, for when two students should _not_
                        # be in the same group
                        satisfied_per_group[(student, nr, gr)] = pulp_logical.NAND(
                            self.prob,
                            self.in_group[(student, gr)],
                            self.in_group[(other_student, gr)],
                        )
                    # The preference is satisfied if it is correct for every group
                    group_vars = [
                        satisfied_per_group[(student, nr, gr)] for gr in self.groups_to
                    ]
                    pulp_logical.AND(self.prob, *group_vars, result_var=satisfied[i])
            else:
                group = row["Waarde"]
                self.prob += self.in_group[(student, group)] >= satisfied[i]
                self.prob += self.in_group[(student, group)] <= satisfied[i]
        return satisfied

    def calculate_optimization_targets(self, satisfied: dict) -> dict:
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
        added_satisfaction = calculate_added_satisfaction(self.preferences)
        satisfaction_per_student = pulp.LpVariable.dict(
            "studentsatisfaction", self.students, cat="Continuous"
        )

        n_preferences_max = (
            self.preferences.xs("Graag met", level="TypeWens")
            .index.get_level_values("Nr")
            .max()
        )
        # Per student whether at least i preferences are satisfied
        n_satisfied_per_student = pulp.LpVariable.dicts(
            "studentassignedprefs",
            itertools.product(
                self.students, (i for i in range(1, n_preferences_max + 1))
            ),
            cat="Binary",
        )
        wp_satisfied_per_student = pulp.LpVariable.dicts(
            "studentassignedweights",
            itertools.product(self.students, added_satisfaction.keys()),
            cat="Binary",
        )

        for student in self.students:
            student_prefs = []
            student_weighted = []
            for i in range(1, n_preferences_max + 1):
                try:
                    student_prefs.append(satisfied[(student, i)])
                    student_weighted.append(weighted_satisfied[(student, i)])
                except KeyError:
                    break
            n_satisfied = pulp.lpSum(student_prefs)
            wp_satisfied = pulp.lpSum(student_weighted)

            for i in range(1, n_preferences_max + 1):
                # n_satisfied(i) for each student is 0 if less than `i` preferences are satisfied
                # The division works because n_true_per_student is binary, so can never
                # be larger than 1
                self.prob += n_satisfied_per_student[(student, i)] <= n_satisfied / i
                # n_satisfied(i) for each student is 1 if at least i preferences are satisfied
                # M ensures the constraint is never larger than 1
                self.prob += (
                    n_satisfied_per_student[(student, i)]
                    >= (n_satisfied - (i - 1) - EPS) / M
                )

            for n_wp in added_satisfaction:
                if n_wp > 0:
                    # wp_satisfied_per_student(i) for each student is 0 if less than `weights`
                    # are satisfied
                    # The division works because wp_satisfied_per_student is binary, so can
                    # never be larger than 1
                    self.prob += (
                        wp_satisfied_per_student[(student, n_wp)]
                        <= wp_satisfied / n_wp + EPS
                    )
                    # wp_satisfied_per_student(i) for each student is 1 if at least n_wp
                    # preferences are satisfied
                    self.prob += (
                        wp_satisfied_per_student[(student, n_wp)]
                        >= (wp_satisfied - (n_wp - EPS)) / M
                    )  # M ensures the constraint is never larger than 1
                else:
                    self.prob += (
                        wp_satisfied_per_student[(student, n_wp)]
                        >= wp_satisfied / n_wp - EPS
                    )
                    self.prob += wp_satisfied_per_student[(student, n_wp)] <= (
                        wp_satisfied - (n_wp + EPS) / M
                    )

            satisfaction_per_student[student] = sum(
                val * wp_satisfied_per_student[(student, n_wp)]
                for n_wp, val in added_satisfaction.items()
            )
        optimization_targets = {
            "n_preferences": pulp.lpSum(satisfied),
            "weighted_preferences": pulp.lpSum(weighted_satisfied),
            "studentsatisfaction": pulp.lpSum(satisfaction_per_student),
        }
        return optimization_targets

    def solve(self, optimization_targets: dict) -> None:
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
        self.prob += optimization_targets[self.optimize]

        self.prob.solve()
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
        optimization_targets = self.calculate_optimization_targets(satisfied)
        self.solve(optimization_targets)
        return self.prob
