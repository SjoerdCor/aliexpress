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


@dataclass
class SoftProblem:
    """A built CP-SAT model with class balance left relaxable, plus the extra
    variables that shape how far it may relax.

    ``slacks`` holds the six shared class-balance slacks (see
    :func:`._balance_families.add_soft_balance_constraints`). ``unmet`` maps
    each student with at least one positive wish to a literal that is true only
    when none of their positive wishes is honored, so a caller can require
    every such student to reach at least one. ``satisfaction_bounds`` is the
    same per-student domain as :attr:`Problem.satisfaction_bounds`.
    """

    model: cp_model.CpModel
    in_group: dict  # (student, group) -> BoolVar
    satisfied: dict  # (student, Nr) -> boolean literal (honored)
    satisfaction: dict  # student -> IntVar, scaled by SATISFACTION_SCALE
    satisfaction_bounds: dict  # student -> (low, high), same scale
    slacks: dict  # family name -> IntVar
    unmet: dict  # student -> BoolVar


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
    model, in_group = _build_assignment(students, groups_to)
    satisfied, satisfaction, satisfaction_bounds = _add_satisfaction(
        model, in_group, preferences, students, groups_to
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
    preferences, students: dict, groups_to: dict, not_together: list
) -> SoftProblem:
    """Build the CP-SAT model with class balance left relaxable.

    Forbidden groups, not-together rules and satisfaction floors stay hard —
    only the class-balance limits become soft, each as ``STRICTEST_LIMIT +
    slack`` (see :func:`._balance_families.add_soft_balance_constraints`). Each
    student with a positive wish also gets an ``unmet`` literal
    (:func:`_constrain_wish_requirement`), so a caller can look for the
    smallest balance relaxation under which every such student can still be
    given at least one positive wish.

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

    Returns
    -------
    SoftProblem
        The built model plus the variables the pipeline reads back.
    """
    model, in_group = _build_assignment(students, groups_to)
    satisfied, satisfaction, satisfaction_bounds = _add_satisfaction(
        model, in_group, preferences, students, groups_to
    )
    _constrain_forbidden_groups(model, in_group, preferences)
    _constrain_not_together(model, in_group, not_together, groups_to)
    _constrain_minimal_satisfaction(model, satisfaction, students)
    slacks = add_soft_balance_constraints(model, in_group, students, groups_to)
    unmet = _constrain_wish_requirement(model, satisfied, preferences)
    return SoftProblem(
        model=model,
        in_group=in_group,
        satisfied=satisfied,
        satisfaction=satisfaction,
        satisfaction_bounds=satisfaction_bounds,
        slacks=slacks,
        unmet=unmet,
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
    model, in_group = _build_assignment(students, groups_to)
    _constrain_forbidden_groups(model, in_group, preferences)
    add_soft_balance_constraints(model, in_group, students, groups_to)
    if min_satisfaction_hard:
        _, satisfaction, _ = _add_satisfaction(
            model, in_group, preferences, students, groups_to
        )
        _constrain_minimal_satisfaction(model, satisfaction, students)
    if not_together_hard:
        _constrain_not_together(model, in_group, not_together, groups_to)
    return model


def _build_assignment(
    students: dict, groups_to: dict
) -> tuple[cp_model.CpModel, dict[tuple[str, str], cp_model.IntVar]]:
    """A fresh model plus its assignment booleans (exactly one group per student).

    Parameters
    ----------
    students : dict
        Per-student info; only the keys (student names) are used here.
    groups_to : dict
        Target groups; only the keys (group names) are used here.

    Returns
    -------
    tuple[cp_model.CpModel, dict[tuple[str, str], cp_model.IntVar]]
        The fresh model and its ``in_group[student, group]`` booleans.
    """
    model = cp_model.CpModel()
    in_group = {
        (student, group): model.NewBoolVar(f"in_{student}_{group}")
        for student in students
        for group in groups_to
    }
    for student in students:
        model.AddExactlyOne(in_group[student, group] for group in groups_to)
    return model, in_group


def _constrain_wish_requirement(
    model: cp_model.CpModel, satisfied: dict, preferences
) -> dict:
    """Per-student ``unmet`` literal: true only if no positive wish is honored.

    Only students with at least one positive wish get a literal: a student with
    none cannot structurally satisfy the requirement, so including them would
    force ``unmet`` to always be true and add nothing to reason about.

    Parameters
    ----------
    model : cp_model.CpModel
        The model the literals are added to.
    satisfied : dict
        Honored-literal per ``(student, Nr)`` preference row (from
        :func:`_add_satisfaction`).
    preferences : pandas.DataFrame
        Long-format preference rows, indexed by ``(student, TypeWens, Nr)``.

    Returns
    -------
    dict[str, cp_model.IntVar]
        The ``unmet`` literal per student with at least one positive wish.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    positive_per_student: dict[str, list] = {}
    for key, row in graag_met.iterrows():
        if row["Gewicht"] > 0:
            positive_per_student.setdefault(key[0], []).append(satisfied[key])

    unmet = {}
    for student, honored in positive_per_student.items():
        literal = model.NewBoolVar(f"unmet_{student}")
        model.AddBoolOr(honored + [literal])
        unmet[student] = literal
    return unmet


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


def _together_literal(model, in_group, student, target, groups_to):
    """Boolean literal: ``student`` ends up with ``target`` (a group or a classmate)."""
    if target in groups_to:
        return in_group[student, target]
    per_group = []
    for group in groups_to:
        both = model.NewBoolVar(f"and_{student}_{target}_{group}")
        model.AddBoolAnd(
            in_group[student, group], in_group[target, group]
        ).OnlyEnforceIf(both)
        model.AddBoolOr(
            in_group[student, group].Not(), in_group[target, group].Not(), both
        )
        per_group.append(both)
    together = model.NewBoolVar(f"together_{student}_{target}")
    model.AddMaxEquality(together, per_group)
    return together


def _add_satisfaction(model, in_group, preferences, students, groups_to):
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
        model, in_group, graag_met, groups_to, scale
    )

    satisfaction = {}
    satisfaction_bounds = {}
    for student in students:
        if student not in weighted_terms:  # no preferences: constant baseline 1
            satisfaction[student] = model.NewConstant(SATISFACTION_SCALE)
            satisfaction_bounds[student] = (SATISFACTION_SCALE, SATISFACTION_SCALE)
            continue
        satisfaction[student], satisfaction_bounds[student] = _satisfaction_variable(
            model, student, weighted_terms[student], weight_range[student], scale
        )
    return satisfied, satisfaction, satisfaction_bounds


def _honored_terms(model, in_group, graag_met, groups_to, scale):
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
        together = _together_literal(model, in_group, student, row["Waarde"], groups_to)
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
