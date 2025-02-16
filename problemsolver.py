"""Module which implements the problem as a Linear Programming problem in pulp and
implements different optimization targets (also known as satisfaction metrics).
"""

import itertools
import pandas as pd
import pulp

import pulp_bitwise_operations as pbo

M = 1_000_000  # A very big number, so that constraints are never larger than 1
EPS = 0.001  # A small number to correct for numerical inaccuracies


# TODO: fix naming. voorkeuren, wishes ==> preferences. leerlingen/ll ==> students. etc.
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def all_unique_sums(iterable):
    """Calculate all possible sums from sublists from the iterable"""
    return {sum(l) for l in powerset(iterable)}


def get_possible_weighted_wishes(voorkeuren: pd.DataFrame) -> set:
    """
    Get all the possible number of weighted wishes

    This will be used to know for which values a satisfaction score must be calculated
    and which dictionary values must be calculated per leerling. By minimizing this number,
    we make the problem calculation as fast as possible, while allowing for arbitrary precision

    Parameters
    ----------
    voorkeuren: pd.DataFrame
        The DataFrame containing the preferences of the leerlingen, must have a MultiIndex
        with levels ("Leerling", "TypeWens") with columns ("Waarde" & "Gewicht")
    """
    unique_weighted_wishes_per_ll = (
        voorkeuren.xs("Graag met", level="TypeWens")
        .groupby("Leerling")["Gewicht"]
        .apply(all_unique_sums)
    )

    unique_weighted_wishes = set()
    for ww in unique_weighted_wishes_per_ll:
        unique_weighted_wishes.update(ww)
    return unique_weighted_wishes


def get_satisfaction_integral(x_a: float, x_b: float) -> float:
    """
    Calculate the extra satisfaction from granting x_b wishes instead of x_a

    This is the (scaled) integral of 0.5**x. This satisfaction function ensures everybody
    first gets their first wish, then everybody their second wish, etc.

    Parameters
    ----------
    x_a: float
        The number of (weighted) wishes as the basic satisfaction of the leerling
    x_b: float
        The number of (weighted) wishes as the goal satisfaction of the leerling

    Returns
    -------
        The added satisfaction score of the leerling
    """
    # In principle, we should probably only specify the satisfaction function and
    # then have this just be a numerical integration for optimal flexibility, but since
    # this flexibility isn't required yet, we're using a analytical integration.

    return (-(0.5**x_b)) - (-(0.5**x_a))


def calculate_added_satisfaction(voorkeuren) -> dict:
    """
    Calculate the score of getting all possible weighted_wishes values accounted for
    """

    possible_weighted_wishes = get_possible_weighted_wishes(voorkeuren)

    # Sorting is important since we're going to difference!
    positive_values = sorted(v for v in possible_weighted_wishes if v >= 0)
    negative_values = sorted(
        (v for v in possible_weighted_wishes if v <= 0), reverse=True
    )

    preference_value = {}
    for values in (negative_values, positive_values):
        # The 0 value is deliberately not taken into account!
        # This would lead to ZeroDivisionErrors
        for last_ww, ww in zip(values[:-1], values[1:]):
            preference_value[ww] = get_satisfaction_integral(last_ww, ww)
    return preference_value


class ProblemSolver:
    """
    Create a problem to distribute leerlingen over groepen

    Parameters
    ----------
    voorkeuren: pd.DataFrame
        A DataFrame with as MultiIndex with (Leerling, Type, Nr) and a value, where
        Leerling is the Name, Type is either "Graag met", "Niet in" or "Liever niet"
        Waarde is then a column with either a Leerling or Group name. In combination with
        Niet In only a Group name is allowed

    leerlingen: Iterable
        An Iterable containing all names of the leerlingen
    leerling_per_obgroep : dict
        Key: the name of the previous group. Value: list of students
    groepen: Iterable
        An interable containing all names of the groepen to which leerlingen can be sent

    max_kliekje, int (default = 5)
        The number of leerlingen that can go to the same group

    max_diff_n_ll_per_group, float (default = 3)
        The maximum difference between assigned leerlingen to the largest group
        and the smallest group

    optimize, str (default = "llsatisfaction")
        What to optimize for: "llsatisfaction" (basically, the least happy student
        is the most happy),
        "n_wishes" or "weighted_wishes"
    """

    def __init__(
        self,
        voorkeuren: pd.DataFrame,
        leerling_per_obgroep,
        groepen,
        max_kliekje=5,
        max_diff_n_ll_per_group=3,
        optimize="llsatisfaction",
    ):
        self.voorkeuren = voorkeuren
        self.leerlingen_per_obgroep = leerling_per_obgroep
        self.leerlingen = sum(self.leerlingen_per_obgroep.values(), [])
        self.groepen = groepen
        self.max_kliekje = max_kliekje
        self.max_diff_n_ll_per_group = max_diff_n_ll_per_group
        self.optimize = optimize
        self.prob = pulp.LpProblem("leerlingindeling", pulp.LpMaximize)
        self.in_group = self._define_variables()

    def _define_variables(self):
        return pulp.LpVariable.dicts(
            "group",
            itertools.product(self.leerlingen, self.groepen),
            cat="Binary",
        )

    def _constraint_student_to_exactly_one_group(self):
        for ll in self.leerlingen:
            self.prob += (
                pulp.lpSum([self.in_group[(ll, gr)] for gr in self.groepen]) == 1
            )

    def _constraint_equal_new_students(self):
        """ "Every group can have a max number of students from an earlier group (no kliekjes)"""
        avg_new_per_group = len(self.leerlingen) / len(self.groepen)
        min_in_group = int(avg_new_per_group - self.max_diff_n_ll_per_group / 2)
        max_in_group = int(avg_new_per_group + self.max_diff_n_ll_per_group / 2)

        new_students_in_group = pulp.LpVariable.dict(
            "new_students_in_group", self.groepen, cat="Integer"
        )

        for mbgroep in self.groepen:

            self.prob += new_students_in_group[mbgroep] == pulp.lpSum(
                [self.in_group[(ll, mbgroep)] for ll in self.leerlingen]
            )

            self.prob += new_students_in_group[mbgroep] <= max_in_group
            self.prob += new_students_in_group[mbgroep] >= min_in_group

    def _constraint_equal_students_from_previous_group(self):
        """Every group can have a max number of students from an earlier group (no kliekjes)"""
        obgroepen = list(self.leerlingen_per_obgroep.keys())
        from_group_to_group = pulp.LpVariable.dicts(
            "from_group_to_group",
            itertools.product(obgroepen, self.groepen),
            cat="Integer",
        )

        for mbgroep in self.groepen:
            for obgroep in obgroepen:
                self.prob += from_group_to_group[(obgroep, mbgroep)] == pulp.lpSum(
                    [
                        self.in_group[(ll, mbgroep)]
                        for ll in self.leerlingen_per_obgroep[obgroep]
                    ]
                )

                self.prob += from_group_to_group[(obgroep, mbgroep)] <= self.max_kliekje

    def _constraint_not_in_forbidden_group(self):
        """Some students can not move int other groups (e.g. a brother/sister is already there)"""
        for i, row in self.voorkeuren.query('TypeWens == "Niet in"').iterrows():
            ll, _, _ = i
            gr = row["Waarde"]
            self.prob += self.in_group[(ll, gr)] == 0

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
        graag_met = self.voorkeuren.xs("Graag met", level="TypeWens")
        weights = graag_met["Gewicht"].to_dict()

        satisfied = pulp.LpVariable.dicts(
            "Satisfied", graag_met.index.to_list(), cat="Binary"
        )
        # Auxiliary variables to check whether a preference is satisfied for a single
        # group. This can then be combined to check whether that tis true for all groups
        pref_per_group = list(
            itertools.chain(
                *[[(ll, nr, gr) for gr in self.groepen] for ll, nr in graag_met.index]
            )
        )
        satisfied_per_group = pulp.LpVariable.dicts(
            "Satisfied_per_group", pref_per_group, cat="Binary"
        )

        # The following link provides the bitwise operators as inequalities for a
        # LP-problem. XNOR, NAND and AND are used in this case
        # https://yetanothermathprogrammingconsultant.blogspot.com/2022/06/xnor-as-linear-inequalities.html
        # TODO: move the bitwise operators to a different module
        for i, row in graag_met.iterrows():
            ll, nr = i
            if row["Waarde"] not in self.groepen:
                other_ll = row["Waarde"]
                for gr in self.groepen:
                    if weights[i] > 0:
                        # Matching preferences are an XNOR problem: if for every group
                        # either both or none are in them, they are in the same group

                        satisfied_per_group[(ll, nr, gr)] = pbo.xnor(
                            self.prob,
                            self.in_group[(ll, gr)],
                            self.in_group[(other_ll, gr)],
                        )
                    else:
                        # This is the NAND variant, for when two leerlingen shout _not_
                        # be in the same group
                        self.prob += (
                            satisfied_per_group[(ll, nr, gr)]
                            >= 1 - self.in_group[(ll, gr)]
                        )  # Als ll niet in deze groep: geen probleem (satisfied = 1)
                        self.prob += (
                            satisfied_per_group[(ll, nr, gr)]
                            >= 1 - self.in_group[(other_ll, gr)]
                        )  # Als andere ll niet in groep, geen probleem (satisfied = 1)

                        self.prob += (
                            satisfied_per_group[(ll, nr, gr)]
                            <= 2
                            - self.in_group[(ll, gr)]
                            - self.in_group[(other_ll, gr)]
                        )  # allebei in deze groep ==> satisfied = 0

                    # Using the AND-definition. The total preference is only satisfied
                    # if it is at least correct for this group
                    # https://yetanothermathprogrammingconsultant.blogspot.com/2022/06/xnor-as-linear-inequalities.html
                    self.prob += satisfied[i] <= satisfied_per_group[(ll, nr, gr)]

                # The preference is satisfied if it is correct for every group
                self.prob += (
                    satisfied[i]
                    >= pulp.lpSum(
                        [satisfied_per_group[(ll, nr, gr)] for gr in self.groepen]
                    )
                    - len(self.groepen)
                    + 1
                )
            else:
                group = row["Waarde"]
                self.prob += self.in_group[(ll, group)] >= satisfied[i]
                self.prob += self.in_group[(ll, group)] <= satisfied[i]
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
        graag_met = self.voorkeuren.xs("Graag met", level="TypeWens")
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
        added_satisfaction = calculate_added_satisfaction(self.voorkeuren)
        satisfaction_per_ll = pulp.LpVariable.dict(
            "LLSatisfaction", self.leerlingen, cat="Continuous"
        )

        n_wishes_max = (
            self.voorkeuren.xs("Graag met", level="TypeWens")
            .index.get_level_values("Nr")
            .max()
        )
        # Per ll whether at least i preferences are satisfied
        n_satisfied_per_ll = pulp.LpVariable.dicts(
            "llassignedprefs",
            itertools.product(self.leerlingen, (i for i in range(1, n_wishes_max + 1))),
            cat="Binary",
        )
        ww_satisfied_per_ll = pulp.LpVariable.dicts(
            "llassignedweights",
            itertools.product(self.leerlingen, added_satisfaction.keys()),
            cat="Binary",
        )

        for ll in self.leerlingen:
            ll_prefs = []
            ll_weighted = []
            for i in range(1, n_wishes_max + 1):
                try:
                    ll_prefs.append(satisfied[(ll, i)])
                    ll_weighted.append(weighted_satisfied[(ll, i)])
                except KeyError:
                    break
            n_satisfied = pulp.lpSum(ll_prefs)
            ww_satisfied = pulp.lpSum(ll_weighted)

            for i in range(1, n_wishes_max + 1):
                # n_satisfied(i) for each leerling is 0 if less than `i` preferences are satisfied
                # The division works because n_true_per_ll is binary, so can never be larger than 1
                self.prob += n_satisfied_per_ll[(ll, i)] <= n_satisfied / i
                # n_satisfied(i) for each leerling is 1 if at least i preferences are satisfied
                # M ensures the constraint is never larger than 1
                self.prob += (
                    n_satisfied_per_ll[(ll, i)] >= (n_satisfied - (i - 1) - EPS) / M
                )

            for n_ww in added_satisfaction:
                if n_ww > 0:
                    # ww_satisfied_per_ll(i) for each leerling is 0 if less than `weights`
                    # are satisfied
                    # The division works because ww_satisfied_per_ll is binary,so can
                    # never be larger than 1
                    self.prob += (
                        ww_satisfied_per_ll[(ll, n_ww)] <= ww_satisfied / n_ww + EPS
                    )
                    # ww_satisfied_per_ll(i) for each leerling is 1 if at least n_ww
                    # preferences are satisfied
                    self.prob += (
                        ww_satisfied_per_ll[(ll, n_ww)]
                        >= (ww_satisfied - (n_ww - EPS)) / M
                    )  # M ensures the constraint is never larger than 1
                else:
                    self.prob += (
                        ww_satisfied_per_ll[(ll, n_ww)] >= ww_satisfied / n_ww - EPS
                    )
                    self.prob += ww_satisfied_per_ll[(ll, n_ww)] <= (
                        ww_satisfied - (n_ww + EPS) / M
                    )

            satisfaction_per_ll[ll] = sum(
                val * ww_satisfied_per_ll[(ll, n_ww)]
                for n_ww, val in added_satisfaction.items()
            )
        optimization_targets = {
            "n_wishes": pulp.lpSum(satisfied),
            "weighted_wishes": pulp.lpSum(weighted_satisfied),
            "llsatisfaction": pulp.lpSum(satisfaction_per_ll),
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
