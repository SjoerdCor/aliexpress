"""The six class-balance constraint families, in CP-SAT form.

All six are counting constraints over the boolean assignment variables:

- equal new students per year cohort: per-group count variables, their max and
  min (``AddMaxEquality``/``AddMinEquality``), and ``max - min <= limit``;
- equal total students: the same spread, now over current occupancy + new;
- gender balance per year cohort and in total: ``|boys - girls| <= limit`` per
  group as two linear inequalities (total includes occupancy, per-cohort not);
- clique limits: students from one previous group per target group, overall
  and per sex, as plain sums with an upper bound.

Cohorts follow ``Jaarlaag``; students without it fall into one ``None``
cohort, which is the doorzetten case where per-year and whole-group
constraints coincide.
"""


def add_balance_constraints(model, in_group, students, groups_to, groupbalance):
    """Add all six balance families with the hard limits from ``groupbalance``."""
    _BalanceFamilies(model, in_group, students, groups_to).add_all(groupbalance)


# A stateful builder with a single entry point (add_all); the families share the
# model context (model, assignment vars, students) as instance attributes, which
# is exactly what a class is for here.
# pylint: disable=too-few-public-methods
class _BalanceFamilies:
    """Builder for the balance families; holds the shared model context."""

    def __init__(self, model, in_group, students, groups_to):
        self.model = model
        self.in_group = in_group
        self.students = students
        self.groups_to = groups_to

    def add_all(self, groupbalance):
        """Add all six families with the hard limits from ``groupbalance``."""
        for cohort in self._cohorts().values():
            self._count_spread(
                cohort, limit=groupbalance.max_diff_n_students_year, occupancy=None
            )
            self._gender_balance(
                cohort,
                limit=groupbalance.max_imbalance_boys_girls_year,
                with_occupancy=False,
            )

        everyone = list(self.students)
        occupancy = {
            group: counts["Jongens"] + counts["Meisjes"]
            for group, counts in self.groups_to.items()
        }
        self._count_spread(
            everyone,
            limit=groupbalance.max_diff_n_students_total,
            occupancy=occupancy,
        )
        self._gender_balance(
            everyone,
            limit=groupbalance.max_imbalance_boys_girls_total,
            with_occupancy=True,
        )
        self._cliques(groupbalance)

    def _cohorts(self) -> dict:
        """Students grouped by ``Jaarlaag``; one ``None`` cohort when absent."""
        result: dict = {}
        for student, info in self.students.items():
            result.setdefault(info.get("Jaarlaag"), []).append(student)
        return result

    def _count_spread(self, members, *, limit, occupancy):
        """Max-min spread of member counts over groups stays within ``limit``.

        With ``occupancy`` the spread is over total group sizes (current + new);
        without it over the new members only (the per-cohort family).
        """
        n = len(members)
        top = n + (max(occupancy.values()) if occupancy else 0)
        counts = []
        for group in self.groups_to:
            current = occupancy[group] if occupancy else 0
            count = self.model.NewIntVar(
                current, current + n, f"count_{id(members)}_{group}"
            )
            self.model.Add(
                count == sum(self.in_group[s, group] for s in members) + current
            )
            counts.append(count)
        largest = self.model.NewIntVar(0, top, f"max_{id(members)}")
        smallest = self.model.NewIntVar(0, top, f"min_{id(members)}")
        self.model.AddMaxEquality(largest, counts)
        self.model.AddMinEquality(smallest, counts)
        self.model.Add(largest - smallest <= limit)

    def _gender_balance(self, members, *, limit, with_occupancy):
        """|boys - girls| per group stays within ``limit`` for these members.

        ``with_occupancy`` adds the current boy/girl counts, as in the
        whole-group family; the per-cohort family counts new students only.
        """
        boys = [s for s in members if self.students[s]["Jongen/meisje"] == "Jongen"]
        girls = [s for s in members if self.students[s]["Jongen/meisje"] == "Meisje"]
        for group, counts in self.groups_to.items():
            current_boys = counts["Jongens"] if with_occupancy else 0
            current_girls = counts["Meisjes"] if with_occupancy else 0
            difference = (
                sum(self.in_group[s, group] for s in boys)
                + current_boys
                - sum(self.in_group[s, group] for s in girls)
                - current_girls
            )
            self.model.Add(difference <= limit)
            self.model.Add(-difference <= limit)

    def _cliques(self, groupbalance):
        """Limit students from one previous group per target group, and per sex."""
        previous_groups = {info["Stamgroep"] for info in self.students.values()}
        sexes = {info["Jongen/meisje"] for info in self.students.values()}
        for previous in previous_groups:
            clique = [
                s for s, info in self.students.items() if info["Stamgroep"] == previous
            ]
            for group in self.groups_to:
                self.model.Add(
                    sum(self.in_group[s, group] for s in clique)
                    <= groupbalance.max_clique
                )
                for sex in sexes:
                    same_sex = [
                        s for s in clique if self.students[s]["Jongen/meisje"] == sex
                    ]
                    self.model.Add(
                        sum(self.in_group[s, group] for s in same_sex)
                        <= groupbalance.max_clique_sex
                    )
