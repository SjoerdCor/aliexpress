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

Each family can also be added *soft*: the limit becomes the strictest possible
value (:data:`STRICTEST_LIMIT`) plus a shared per-family slack, so a caller can
relax class balance just as far as some other objective (e.g. a wish
requirement) demands, instead of fixing it upfront.
"""

from ortools.sat.python import cp_model

from ._balance import BalanceMaxima

#: The tightest possible value for every balance limit; the soft families
#: relax outward from here via their slack.
STRICTEST_LIMIT = 1

#: Per balance-slack family, its weight in a relaxation objective (scaled x100
#: to integers, so the objective stays exact). Whole-group families
#: (``_total``) weigh less than their per-year counterpart: spreading students
#: unevenly across the whole group is less disruptive than an uneven single
#: year cohort, so it is cheaper to relax first.
SLACK_WEIGHTS: dict[str, int] = {
    "diff_year": 100,
    "diff_total": 49,
    "clique": 100,
    "clique_sex": 100,
    "gender_year": 100,
    "gender_total": 49,
}

#: The six family names, in the order the slacks are created.
FAMILY_NAMES: tuple[str, ...] = tuple(SLACK_WEIGHTS)

#: Maps each family name to the ``BalanceMaxima`` field that caps its slack.
#: The same family <-> limit correspondence as ``_BalanceFamilies.add_all``:
#: a per-family maximum bounds how far *that* family may relax, so the soft
#: path must translate a family name back to the matching ``BalanceMaxima``
#: attribute.
_MAXIMA_FIELD_BY_FAMILY: dict[str, str] = {
    "diff_year": "max_diff_n_students_year",
    "diff_total": "max_diff_n_students_total",
    "clique": "max_clique",
    "clique_sex": "max_clique_sex",
    "gender_year": "max_imbalance_boys_girls_year",
    "gender_total": "max_imbalance_boys_girls_total",
}


def _slack_upper(name: str, maxima: BalanceMaxima | None, uncapped_bound: int) -> int:
    """The upper bound for family ``name``'s slack.

    The ``uncapped_bound`` unless ``maxima`` caps this family, in which case the
    slack tops out at ``cap - STRICTEST_LIMIT`` so the family's limit
    (``STRICTEST_LIMIT + slack``) can reach ``cap`` but no further. A cap equal
    to ``STRICTEST_LIMIT`` yields upper bound 0, pinning the family at its
    strictest value.
    """
    if maxima is None:
        return uncapped_bound
    cap = getattr(maxima, _MAXIMA_FIELD_BY_FAMILY[name])
    if cap is None:
        return uncapped_bound
    return cap - STRICTEST_LIMIT


def uncapped_slack_bound(students: dict, groups_to: dict) -> int:
    """Everyone who will ever sit in a group — every new student plus all
    groups' current occupancy — as a safe upper bound for any balance slack.

    The whole-group families (``diff_total``, ``gender_total``) measure counts
    and imbalances that include current occupancy, not just the new students —
    so bounding the slack by the new-student count alone can cut off the
    minimal relaxation an instance actually needs (e.g. a handful of new
    students distributed over groups whose *existing* occupancy is already
    lopsided). Every family's count or imbalance is a sub-quantity of
    "everyone who will ever be in a group": current occupancy across all
    groups, plus every new student.

    Parameters
    ----------
    students : dict
        Per-student info; only the count of students is used here.
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.

    Returns
    -------
    int
        The shared upper bound for every soft-family slack, and for a caller's
        own max-of-slacks variable (e.g. the automatic path's ``max_slack``).
    """
    total_occupancy = sum(
        counts["Jongens"] + counts["Meisjes"] for counts in groups_to.values()
    )
    return len(students) + total_occupancy


def add_balance_constraints(
    model: cp_model.CpModel,
    in_group: dict[tuple[str, str], cp_model.IntVar],
    students: dict,
    groups_to: dict,
    groupbalance,
) -> None:
    """Add all six balance families with the hard limits from ``groupbalance``.

    Parameters
    ----------
    model : cp_model.CpModel
        The model the constraints are added to.
    in_group : dict[tuple[str, str], cp_model.IntVar]
        Assignment booleans, keyed by ``(student, group)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    groupbalance : aliexpress.solver._balance.GroupBalance
        The hard limit for each of the six families.
    """
    _BalanceFamilies(model, in_group, students, groups_to).add_all(groupbalance)


def add_soft_balance_constraints(
    model: cp_model.CpModel,
    in_group: dict[tuple[str, str], cp_model.IntVar],
    students: dict,
    groups_to: dict,
    maxima: BalanceMaxima | None = None,
) -> dict[str, cp_model.IntVar]:
    """Add all six balance families with limit ``STRICTEST_LIMIT + slack``.

    Parameters
    ----------
    model : cp_model.CpModel
        The model the constraints are added to.
    in_group : dict[tuple[str, str], cp_model.IntVar]
        Assignment booleans, keyed by ``(student, group)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    maxima : BalanceMaxima | None
        Per-family ceilings on the relaxation. For any family whose maximum is
        not ``None``, its slack upper bound drops from the generous
        :func:`uncapped_slack_bound` to ``cap - STRICTEST_LIMIT`` — so that family's
        limit can never exceed ``cap``. ``None`` (whole object or a single
        field) leaves that family uncapped, as before.

    Returns
    -------
    dict[str, cp_model.IntVar]
        The six shared slacks, keyed by :data:`FAMILY_NAMES`, for the caller to
        weight into a relaxation objective (see :data:`SLACK_WEIGHTS`).
    """
    return _BalanceFamilies(model, in_group, students, groups_to).add_all_soft(maxima)


# A stateful builder with a single entry point (add_all); the families share the
# model context (model, assignment vars, students) as instance attributes, which
# is exactly what a class is for here.
# pylint: disable=too-few-public-methods
class _BalanceFamilies:
    """Builder for the balance families; holds the shared model context."""

    def __init__(
        self,
        model: cp_model.CpModel,
        in_group: dict[tuple[str, str], cp_model.IntVar],
        students: dict,
        groups_to: dict,
    ):
        self.model = model
        self.in_group = in_group
        self.students = students
        self.groups_to = groups_to

    def add_all(self, groupbalance) -> None:
        """Add all six families with the hard limits from ``groupbalance``.

        Parameters
        ----------
        groupbalance : aliexpress.solver._balance.GroupBalance
            The hard limit for each of the six families.
        """
        self._add_families(
            {
                "diff_year": groupbalance.max_diff_n_students_year,
                "diff_total": groupbalance.max_diff_n_students_total,
                "clique": groupbalance.max_clique,
                "clique_sex": groupbalance.max_clique_sex,
                "gender_year": groupbalance.max_imbalance_boys_girls_year,
                "gender_total": groupbalance.max_imbalance_boys_girls_total,
            }
        )

    def add_all_soft(
        self, maxima: BalanceMaxima | None = None
    ) -> dict[str, cp_model.IntVar]:
        """Add all six families with limit ``STRICTEST_LIMIT + slack``.

        Parameters
        ----------
        maxima : BalanceMaxima | None
            Per-family ceilings. A non-``None`` field caps the matching slack at
            ``cap - STRICTEST_LIMIT`` instead of the uncapped bound.

        Returns
        -------
        dict[str, cp_model.IntVar]
            The six shared slacks, keyed by :data:`FAMILY_NAMES`.
        """
        uncapped_bound = uncapped_slack_bound(self.students, self.groups_to)
        slacks = {
            name: self.model.NewIntVar(
                0, _slack_upper(name, maxima, uncapped_bound), f"slack_{name}"
            )
            for name in FAMILY_NAMES
        }
        self._add_families(
            {name: STRICTEST_LIMIT + slack for name, slack in slacks.items()}
        )
        return slacks

    def _add_families(self, limits: dict[str, cp_model.LinearExprT]) -> None:
        """Add all six families, each with its limit from ``limits``.

        Parameters
        ----------
        limits : dict[str, cp_model.LinearExprT]
            The limit for each of :data:`FAMILY_NAMES`: a plain int for a hard
            limit, or ``STRICTEST_LIMIT + slack`` for a soft one.
        """
        for cohort_key, cohort in self._cohorts().items():
            self._count_spread(
                cohort,
                limit=limits["diff_year"],
                occupancy=None,
                label=f"year_{cohort_key if cohort_key is not None else 'none'}",
            )
            self._gender_balance(
                cohort, limit=limits["gender_year"], with_occupancy=False
            )

        everyone = list(self.students)
        occupancy = {
            group: counts["Jongens"] + counts["Meisjes"]
            for group, counts in self.groups_to.items()
        }
        self._count_spread(
            everyone, limit=limits["diff_total"], occupancy=occupancy, label="total"
        )
        self._gender_balance(
            everyone, limit=limits["gender_total"], with_occupancy=True
        )
        self._cliques(limits["clique"], limits["clique_sex"])

    def _cohorts(self) -> dict[object, list[str]]:
        """Students grouped by ``Jaarlaag``; one ``None`` cohort when absent."""
        result: dict[object, list[str]] = {}
        for student, info in self.students.items():
            result.setdefault(info.get("Jaarlaag"), []).append(student)
        return result

    def _count_spread(
        self,
        members: list[str],
        *,
        limit: cp_model.LinearExprT,
        occupancy: dict[str, int] | None,
        label: str,
    ) -> None:
        """Max-min spread of member counts over groups stays within ``limit``.

        Parameters
        ----------
        members : list[str]
            The students the spread is computed over.
        limit : cp_model.LinearExprT
            The maximum allowed spread: a plain int for a hard limit, or
            ``STRICTEST_LIMIT + slack`` for a soft one.
        occupancy : dict[str, int] | None
            Current per-group occupancy to add to the count, or ``None`` to
            count only the new ``members`` (the per-cohort family).
        label : str
            Identifies this call's variables (e.g. ``"year_6"`` or ``"total"``)
            so variable names are deterministic across runs.
        """
        n = len(members)
        top = n + (max(occupancy.values()) if occupancy else 0)
        counts = []
        for group in self.groups_to:
            current = occupancy[group] if occupancy else 0
            count = self.model.NewIntVar(current, current + n, f"count_{label}_{group}")
            self.model.Add(
                count == sum(self.in_group[s, group] for s in members) + current
            )
            counts.append(count)
        largest = self.model.NewIntVar(0, top, f"max_{label}")
        smallest = self.model.NewIntVar(0, top, f"min_{label}")
        self.model.AddMaxEquality(largest, counts)
        self.model.AddMinEquality(smallest, counts)
        self.model.Add(largest - smallest <= limit)

    def _gender_balance(
        self,
        members: list[str],
        *,
        limit: cp_model.LinearExprT,
        with_occupancy: bool,
    ) -> None:
        """|boys - girls| per group stays within ``limit`` for these members.

        Parameters
        ----------
        members : list[str]
            The students the balance is computed over.
        limit : cp_model.LinearExprT
            The maximum allowed imbalance: a plain int for a hard limit, or
            ``STRICTEST_LIMIT + slack`` for a soft one.
        with_occupancy : bool
            Whether to add the current boy/girl occupancy (the whole-group
            family) or count only the new ``members`` (the per-cohort family).
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

    def _cliques(
        self,
        clique_limit: cp_model.LinearExprT,
        clique_sex_limit: cp_model.LinearExprT,
    ) -> None:
        """Limit students from one previous group per target group, and per sex.

        Parameters
        ----------
        clique_limit : cp_model.LinearExprT
            The maximum students from one ``Stamgroep`` per target group.
        clique_sex_limit : cp_model.LinearExprT
            The maximum same-sex students from one ``Stamgroep`` per target
            group.
        """
        previous_groups = {info["Stamgroep"] for info in self.students.values()}
        sexes = {info["Jongen/meisje"] for info in self.students.values()}
        for previous in previous_groups:
            clique = [
                s for s, info in self.students.items() if info["Stamgroep"] == previous
            ]
            for group in self.groups_to:
                self.model.Add(
                    sum(self.in_group[s, group] for s in clique) <= clique_limit
                )
                for sex in sexes:
                    same_sex = [
                        s for s in clique if self.students[s]["Jongen/meisje"] == sex
                    ]
                    self.model.Add(
                        sum(self.in_group[s, group] for s in same_sex)
                        <= clique_sex_limit
                    )
