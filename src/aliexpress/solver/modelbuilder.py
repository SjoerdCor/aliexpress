"""The CP-SAT model of the student distribution problem.

The model is built from boolean assignment variables ``in_group[student, group]``
(exactly one true per student) plus the hard constraints: forbidden groups,
not-together rules, satisfaction floors and the class-balance families
(:mod:`._balance_families`).

The satisfaction metric is the one defined in :mod:`.satisfaction` (the
integral of 0.5^x over the weighted honored-preference sum). CP-SAT reasons
over integers only, and the metric is non-linear — so per student it enters
the model as a lookup table (``AddElement``): every reachable weighted sum is
evaluated in Python floats and stored as a scaled integer. Weights are scaled
by the smallest exact factor (:mod:`.scaling`); satisfaction values by
:data:`SATISFACTION_SCALE`.

CP-SAT variables belong to exactly one
:class:`~ortools.sat.python.cp_model.CpModel`, so every analysis builds a
fresh model via :func:`build_problem` instead of sharing variables.
"""

import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..data import preferences_data
from ._balance import UNCAPPED, BalanceMaxima
from ._balance_families import add_balance_constraints, add_soft_balance_constraints
from .satisfaction import _normalize_and_bound
from .scaling import weight_scale

SATISFACTION_SCALE = 10**6
"""Satisfaction values as integers: exact to 1e-6, matching the 6-decimal
rounding the integration tests pin."""


@dataclass
class Problem:
    """A built CP-SAT model plus the variables the pipeline reads back.

    ``satisfied`` maps each ``(student, Nr)`` preference row to a boolean literal
    that is true when the wish is honored ("Graag met": together; negative weight:
    apart). ``satisfaction`` holds the per-student scaled-integer satisfaction.
    ``satisfaction_bounds`` holds each of those variables' own ``(low, high)``
    domain, computed in Python alongside them: CP-SAT's own domain reflection
    (``var.Proto().domain``) is unsafe to call on a constant variable (see
    :func:`_add_satisfaction`), so callers that need the domain (the lexmaxmin
    strategy's ``minimum`` bounds) read it from here instead.
    """

    model: cp_model.CpModel
    in_group: dict  # (student, group) -> BoolVar
    satisfied: dict  # (student, Nr) -> boolean literal (honored)
    satisfaction: dict  # student -> IntVar, scaled by SATISFACTION_SCALE
    satisfaction_bounds: dict  # student -> (low, high), same scale

    def read_solution(self, solver: cp_model.CpSolver) -> tuple[dict, dict]:
        """This problem's ``(assignment, satisfied)`` read off a solved ``solver``.

        See :func:`_read_solution`, the shared reader both problem types delegate to.
        """
        return _read_solution(self, solver)


@dataclass
class SoftProblem:
    """A built CP-SAT model with class balance left relaxable, plus the extra
    variables that shape how far it may relax.

    ``slacks`` holds the six shared class-balance slacks (see
    :func:`._balance_families.add_soft_balance_constraints`). ``nonpositive``
    maps each student whose satisfaction *can* reach zero or below to a
    literal that is true exactly when it does, so a caller can require every
    such student to reach strictly positive satisfaction instead.
    ``satisfaction_bounds`` is the same per-student domain as
    :attr:`Problem.satisfaction_bounds`.
    """

    model: cp_model.CpModel
    in_group: dict  # (student, group) -> BoolVar
    satisfied: dict  # (student, Nr) -> boolean literal (honored)
    satisfaction: dict  # student -> IntVar, scaled by SATISFACTION_SCALE
    satisfaction_bounds: dict  # student -> (low, high), same scale
    slacks: dict  # family name -> IntVar
    nonpositive: dict  # student -> BoolVar

    def read_solution(self, solver: cp_model.CpSolver) -> tuple[dict, dict]:
        """This problem's ``(assignment, satisfied)`` read off a solved ``solver``.

        See :func:`_read_solution`, the shared reader both problem types delegate to.
        """
        return _read_solution(self, solver)


@dataclass
class _AssignmentModel:
    """Equivalent one-hot/integer assignment representations and their cache."""

    model: cp_model.CpModel
    in_group: dict
    group_index: dict
    groups_to: dict
    together_by_pair: dict


def _read_solution(problem, solver: cp_model.CpSolver) -> tuple[dict, dict]:
    """Read the assignment and honored-wish booleans off a solved ``(problem, solver)`` pair.

    The shared body behind :meth:`Problem.read_solution` / :meth:`SoftProblem.read_solution`
    (a plain function so the two identical readers don't duplicate it). Preference-free: it
    only reads the model's own ``in_group``/``satisfied`` variables, so it works from a
    stage that has no ``preferences`` in scope (the lexmaxmin per-level loop in
    :mod:`.strategies`) as well as from the final extraction in :mod:`.engine`.

    Parameters
    ----------
    problem : Problem | SoftProblem
        The built model, for its ``in_group``/``satisfied`` variables.
    solver : cp_model.CpSolver
        A solver holding a solved (not necessarily final-stage) assignment.

    Returns
    -------
    tuple[dict, dict]
        ``(assignment, satisfied)``: ``assignment`` maps each student to their assigned
        group; ``satisfied`` maps each ``(student, Nr)`` preference row to whether it was
        honored.
    """
    assignment = {
        student: group
        for (student, group), var in problem.in_group.items()
        if solver.BooleanValue(var)
    }
    satisfied = {
        key: solver.BooleanValue(literal) for key, literal in problem.satisfied.items()
    }
    return assignment, satisfied


def build_problem(
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    groupbalance,
) -> Problem:
    """Build the full CP-SAT model with hard class-balance limits.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``,
        ``MinimaleTevredenheid``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    not_together : list
        Rules of the form ``{"group": {student, ...}, "Max_aantal_samen": int}``.
    groupbalance : aliexpress.solver._balance.GroupBalance
        The hard limit for each of the six class-balance families.

    Returns
    -------
    Problem
        The built model plus the variables the pipeline reads back.
    """
    assignment = _build_assignment(students, groups_to)
    model, in_group = assignment.model, assignment.in_group
    satisfied, satisfaction, satisfaction_bounds = _add_satisfaction(
        assignment, preferences, students
    )
    _constrain_forbidden_groups(model, in_group, preferences)
    _constrain_not_together(model, in_group, not_together, groups_to)
    _constrain_minimal_satisfaction(model, satisfaction, students)
    add_balance_constraints(model, in_group, students, groups_to, groupbalance)
    return Problem(
        model=model,
        in_group=in_group,
        satisfied=satisfied,
        satisfaction=satisfaction,
        satisfaction_bounds=satisfaction_bounds,
    )


def build_soft_problem(
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    maxima: BalanceMaxima = UNCAPPED,
) -> SoftProblem:
    """Build the CP-SAT model with class balance left relaxable.

    Forbidden groups, not-together rules and satisfaction floors stay hard —
    only the class-balance limits become soft, each as ``STRICTEST_LIMIT +
    slack`` (see :func:`._balance_families.add_soft_balance_constraints`).
    Each student whose satisfaction can reach zero or below also gets a
    ``nonpositive`` literal (:func:`_constrain_positive_satisfaction`), so a
    caller can look for the smallest balance relaxation under which every
    such student still reaches strictly positive satisfaction.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``,
        ``MinimaleTevredenheid``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    not_together : list
        Rules of the form ``{"group": {student, ...}, "Max_aantal_samen": int}``.
    maxima : aliexpress.solver._balance.BalanceMaxima
        Per-family ceilings on the relaxation. A non-``None`` family maximum
        bounds how far that family may relax; an empty (all-unlimited)
        ``BalanceMaxima`` (the default) leaves the balance fully relaxable.

    Returns
    -------
    SoftProblem
        The built model plus the variables the pipeline reads back.
    """
    assignment = _build_assignment(students, groups_to)
    model, in_group = assignment.model, assignment.in_group
    satisfied, satisfaction, satisfaction_bounds = _add_satisfaction(
        assignment, preferences, students
    )
    _constrain_forbidden_groups(model, in_group, preferences)
    _constrain_not_together(model, in_group, not_together, groups_to)
    _constrain_minimal_satisfaction(model, satisfaction, students)
    slacks = add_soft_balance_constraints(
        model, in_group, students, groups_to, maxima=maxima
    )
    nonpositive = _constrain_positive_satisfaction(
        model, satisfaction, satisfaction_bounds
    )
    return SoftProblem(
        model=model,
        in_group=in_group,
        satisfied=satisfied,
        satisfaction=satisfaction,
        satisfaction_bounds=satisfaction_bounds,
        slacks=slacks,
        nonpositive=nonpositive,
    )


# Lives here, not in feasibility.py: it is model assembly, composing the same
# private constraint helpers as build_problem/build_soft_problem. feasibility.py
# owns the reasoning on top and keeps those helpers private to this module.
def build_feasibility_problem(  # pylint: disable=too-many-arguments
    # Each argument is a distinct input to the model (raw data, rules, which
    # relaxable families stay hard); grouping them would obscure the
    # function's public interface rather than simplify it.
    preferences,
    students: dict,
    groups_to: dict,
    not_together: list,
    *,
    min_satisfaction_hard: bool,
    not_together_hard: bool,
) -> cp_model.CpModel:
    """Build a bare feasibility model: does any valid assignment exist at all?

    No objective is set — this is a pure SAT question, not an optimization.
    "Niet in" stays hard (it is fundamental) and class balance stays soft (the
    real solve always relaxes it, so it can never be the infeasibility cause).
    The two relaxable preference families — minimal satisfaction and
    not-together — are each hard or omitted per the caller's flags, so
    :mod:`.feasibility` can test whether leaving one (or both) soft turns an
    infeasible instance feasible again.

    Parameters
    ----------
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.
    students : dict
        Per-student info (``Jaarlaag``, ``Jongen/meisje``, ``Stamgroep``,
        ``MinimaleTevredenheid``).
    groups_to : dict
        Target groups, keyed by group name, with current ``Jongens``/``Meisjes``
        occupancy.
    not_together : list
        Rules of the form ``{"group": {student, ...}, "Max_aantal_samen": int}``.
    min_satisfaction_hard : bool
        Whether the per-student satisfaction floors are enforced.
    not_together_hard : bool
        Whether the not-together rules are enforced.

    Returns
    -------
    cp_model.CpModel
        The built model, with no objective set.
    """
    assignment = _build_assignment(students, groups_to)
    model, in_group = assignment.model, assignment.in_group
    _constrain_forbidden_groups(model, in_group, preferences)
    add_soft_balance_constraints(model, in_group, students, groups_to)
    if min_satisfaction_hard:
        _, satisfaction, _ = _add_satisfaction(assignment, preferences, students)
        _constrain_minimal_satisfaction(model, satisfaction, students)
    if not_together_hard:
        _constrain_not_together(model, in_group, not_together, groups_to)
    return model


def _build_assignment(students: dict, groups_to: dict) -> _AssignmentModel:
    """A fresh model plus equivalent one-hot and integer group assignments.

    The one-hot ``in_group`` booleans remain the natural representation for
    balance counts.  ``group_index`` channels that same choice into one integer
    per student, so a student-to-student preference can be represented directly
    as equality of two group indices instead of four per-group conjunctions and
    a max constraint.  ``AddExactlyOne`` makes the one-way channel constraints
    below sufficient: the single true literal fixes the integer to its index.

    Parameters
    ----------
    students : dict
        Per-student info; only the keys (student names) are used here.
    groups_to : dict
        Target groups; only the keys (group names) are used here.

    Returns
    -------
    _AssignmentModel
        The fresh model, its equivalent assignment variables, target groups
        and the cache shared by reciprocal student preferences.
    """
    model = cp_model.CpModel()
    in_group = {
        (student, group): model.NewBoolVar(f"in_{student}_{group}")
        for student in students
        for group in groups_to
    }
    for student in students:
        model.AddExactlyOne(in_group[student, group] for group in groups_to)
    group_index = {
        student: model.NewIntVar(0, len(groups_to) - 1, f"group_index_{student}")
        for student in students
    }
    for student in students:
        for index, group in enumerate(groups_to):
            model.Add(group_index[student] == index).OnlyEnforceIf(
                in_group[student, group]
            )
    return _AssignmentModel(model, in_group, group_index, groups_to, {})


def _constrain_positive_satisfaction(
    model: cp_model.CpModel, satisfaction: dict, satisfaction_bounds: dict
) -> dict:
    """Per-student ``nonpositive`` literal: true iff satisfaction is <= 0.

    Only students whose satisfaction variable *can* reach zero or below get a
    literal: their domain's low bound is <= 0. This excludes exactly the
    students with no preferences, whose satisfaction is a constant 1.0 (bounds
    ``(SATISFACTION_SCALE, SATISFACTION_SCALE)``) and so can never be
    nonpositive; it keeps pure-positive students in (low bound 0), since their
    satisfaction can still be exactly 0.

    The literal is a full reification (both directions), not a one-sided
    ``AddBoolOr``: minimizing the count of true ``nonpositive`` literals would
    otherwise let the solver push a literal false while the student's
    satisfaction is actually <= 0, since nothing forces the literal true in
    that case.

    Parameters
    ----------
    model : cp_model.CpModel
        The model the literals are added to.
    satisfaction : dict
        Per-student satisfaction IntVar, scaled by
        :data:`SATISFACTION_SCALE` (from :func:`_add_satisfaction`).
    satisfaction_bounds : dict
        Per-student ``(low, high)`` domain of the satisfaction variable, same
        scale (from :func:`_add_satisfaction`).

    Returns
    -------
    dict[str, cp_model.IntVar]
        The ``nonpositive`` literal per student whose satisfaction can be
        <= 0.
    """
    nonpositive = {}
    for student, (low, _high) in satisfaction_bounds.items():
        if low > 0:
            continue
        literal = model.NewBoolVar(f"nonpositive_{student}")
        model.Add(satisfaction[student] <= 0).OnlyEnforceIf(literal)
        model.Add(satisfaction[student] >= 1).OnlyEnforceIf(literal.Not())
        nonpositive[student] = literal
    return nonpositive


def _constrain_forbidden_groups(model, in_group, preferences):
    """'Niet in': the student cannot be placed in the named group (hard)."""
    for index, row in preferences.query('TypeWens == "Niet in"').iterrows():
        student = index[0]
        model.Add(in_group[student, row["Waarde"]] == 0)


def _constrain_not_together(model, in_group, not_together, groups_to):
    """At most ``Max_aantal_samen`` students of each rule group per target group."""
    for rule in not_together:
        for group in groups_to:
            model.Add(
                sum(in_group[student, group] for student in rule["group"])
                <= rule["Max_aantal_samen"]
            )


def _constrain_minimal_satisfaction(model, satisfaction, students):
    """Per-student satisfaction floors (UI: "Extra zekerheid").

    Floors are rounded *down* to the integer scale so a floor that is met
    exactly in float terms cannot become infeasible through rounding.
    """
    for student, info in students.items():
        floor = info["MinimaleTevredenheid"]
        if math.isnan(floor):
            continue
        model.Add(satisfaction[student] >= math.floor(floor * SATISFACTION_SCALE))


def _together_literal(assignment: _AssignmentModel, student, target):
    """Boolean literal: ``student`` ends up with ``target`` (a group or a classmate)."""
    if target in assignment.groups_to:
        return assignment.in_group[student, target]
    pair = tuple(sorted((student, target)))
    if pair not in assignment.together_by_pair:
        together = assignment.model.NewBoolVar(f"together_{pair[0]}_{pair[1]}")
        assignment.model.Add(
            assignment.group_index[student] == assignment.group_index[target]
        ).OnlyEnforceIf(together)
        assignment.model.Add(
            assignment.group_index[student] != assignment.group_index[target]
        ).OnlyEnforceIf(together.Not())
        assignment.together_by_pair[pair] = together
    return assignment.together_by_pair[pair]


def _add_satisfaction(assignment: _AssignmentModel, preferences, students):
    """Add per-preference honored-literals and per-student satisfaction integers.

    The weighted honored sum of a student is ``sum(weight * together)`` over all
    their preference rows: an honored positive wish contributes its weight, a
    violated negative wish its (negative) weight, anything else 0. The
    satisfaction integer follows from that sum through an element lookup over
    the staircase of F values (F = integral of 0.5^x), normalized as in
    :func:`..satisfaction._normalize_and_bound`.

    Returns ``(satisfied, satisfaction, satisfaction_bounds)``: the last maps
    each student to their satisfaction variable's own ``(low, high)`` domain,
    computed here in plain Python rather than read back from CP-SAT — a
    student with no preferences gets a constant variable
    (``model.NewConstant``), and this ortools build crashes (a Windows fatal
    exception, not a Python exception) when ``.Proto().domain`` is read from a
    constant IntVar.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    scale = weight_scale(graag_met["Gewicht"]) if not graag_met.empty else 1
    satisfied, weighted_terms, weight_range = _honored_terms(
        assignment, graag_met, scale
    )

    satisfaction = {}
    satisfaction_bounds = {}
    for student in students:
        if student not in weighted_terms:  # no preferences: constant baseline 1
            satisfaction[student] = assignment.model.NewConstant(SATISFACTION_SCALE)
            satisfaction_bounds[student] = (SATISFACTION_SCALE, SATISFACTION_SCALE)
            continue
        satisfaction[student], satisfaction_bounds[student] = _satisfaction_variable(
            assignment.model,
            student,
            weighted_terms[student],
            weight_range[student],
            scale,
        )
    return satisfied, satisfaction, satisfaction_bounds


def _honored_terms(assignment: _AssignmentModel, graag_met, scale):
    """Honored-literals plus each student's weighted terms and their bounds.

    Returns ``(satisfied, weighted_terms, weight_range)``: the honored literal
    per preference row, the scaled ``weight * together`` terms per student, and
    the per-student ``[negative sum, positive sum]`` bounds of their sum.
    """
    satisfied = {}
    weighted_terms: dict[str, list] = {}
    weight_range: dict[str, list[int]] = {}
    for key, row in graag_met.iterrows():
        student = key[0]
        weight = round(row["Gewicht"] * scale)
        together = _together_literal(assignment, student, row["Waarde"])
        satisfied[key] = together if weight > 0 else together.Not()
        weighted_terms.setdefault(student, []).append(weight * together)
        low, high = weight_range.setdefault(student, [0, 0])
        weight_range[student] = [low + min(weight, 0), high + max(weight, 0)]
    return satisfied, weighted_terms, weight_range


def _satisfaction_variable(model, student, terms, bounds, scale):
    """The student's satisfaction integer, plus its own ``(low, high)`` domain.

    Element lookup over the F staircase; ``min(table)``/``max(table)`` are
    already computed here to size the variable, so returning them as the
    domain costs nothing extra and avoids reading it back via CP-SAT
    reflection (see :func:`_add_satisfaction`).
    """
    low, high = bounds
    weighted_sum = model.NewIntVar(low, high, f"w_{student}")
    model.Add(weighted_sum == sum(terms))

    table = [
        _scaled_satisfaction(value / scale, high / scale, low / scale)
        for value in range(low, high + 1)
    ]
    satisfaction_low, satisfaction_high = min(table), max(table)
    satisfaction_var = model.NewIntVar(
        satisfaction_low, satisfaction_high, f"sat_{student}"
    )
    index = model.NewIntVar(0, high - low, f"wi_{student}")
    model.Add(index == weighted_sum - low)
    model.AddElement(index, table, satisfaction_var)
    return satisfaction_var, (satisfaction_low, satisfaction_high)


def _scaled_satisfaction(weighted: float, best: float, worst: float) -> int:
    """Normalized satisfaction at weighted level ``weighted``, as a scaled integer.

    ``best`` is the student's maximal positive weight sum, ``worst`` the
    minimal (most negative) weight sum. Delegates to
    :func:`~.satisfaction._normalize_and_bound` so the model's integer table
    and the reported float agree by construction.
    """
    return round(_normalize_and_bound(weighted, best, worst) * SATISFACTION_SCALE)
