"""The CP-SAT model of the student distribution problem.

Mirrors the constraint families of :mod:`..problemsolver`, translated to integer
form. The satisfaction metric stays the one defined in :mod:`..satisfaction`
(the integral of 0.5^x over the weighted honored-preference sum); here it is
tabulated per student as scaled integers, because CP-SAT reasons over integers
only. Preference weights are scaled by the smallest exact factor (see
:mod:`.scaling`); satisfaction values by :data:`SATISFACTION_SCALE`.

Unlike the pulp substrate, CP-SAT variables belong to exactly one
:class:`~ortools.sat.python.cp_model.CpModel`, so every analysis builds a fresh
model via :func:`build_problem` instead of sharing variables.
"""

import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ...data import preferences_data
from ..satisfaction import get_satisfaction_integral
from ._balance_families import add_balance_constraints
from .scaling import weight_scale

SATISFACTION_SCALE = 10**6
"""Satisfaction values as integers: exact to 1e-6, matching the 6-decimal
rounding the integration tests pin."""


@dataclass
class CpSatProblem:
    """A built CP-SAT model plus the variables the pipeline reads back.

    ``satisfied`` maps each ``(student, Nr)`` preference row to a boolean literal
    that is true when the wish is honored ("Graag met": together; negative weight:
    apart). ``satisfaction`` holds the per-student scaled-integer satisfaction.
    """

    model: cp_model.CpModel
    in_group: dict  # (student, group) -> BoolVar
    satisfied: dict  # (student, Nr) -> boolean literal (honored)
    satisfaction: dict  # student -> IntVar, scaled by SATISFACTION_SCALE


def build_problem(preferences, students, groups_to, not_together, groupbalance):
    """Build the full CP-SAT model with hard class-balance limits.

    Parameters mirror :class:`~aliexpress.solver.problemsolver.ProblemSolver`:
    the long-format ``preferences`` frame, the ``students`` info dict, the target
    ``groups_to`` with current occupancy, the ``not_together`` rules and a
    :class:`~aliexpress.solver._balance.GroupBalance`.
    """
    model = cp_model.CpModel()
    in_group = {
        (student, group): model.NewBoolVar(f"in_{student}_{group}")
        for student in students
        for group in groups_to
    }
    for student in students:
        model.AddExactlyOne(in_group[student, group] for group in groups_to)

    _constrain_forbidden_groups(model, in_group, preferences)
    satisfied, satisfaction = _add_satisfaction(
        model, in_group, preferences, students, groups_to
    )
    _constrain_not_together(model, in_group, not_together, groups_to)
    _constrain_minimal_satisfaction(model, satisfaction, students)
    add_balance_constraints(model, in_group, students, groups_to, groupbalance)
    return CpSatProblem(
        model=model,
        in_group=in_group,
        satisfied=satisfied,
        satisfaction=satisfaction,
    )


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
    their preference rows: an honored negative wish contributes 0 (apart), a
    violated one its (negative) weight — identical to the pulp formulation. The
    satisfaction integer follows from that sum through an element lookup over
    the staircase of F values (F = integral of 0.5^x), normalized exactly as in
    :func:`..satisfaction._normalize_and_bound`.
    """
    graag_met = preferences_data.get_graag_met(preferences)
    scale = weight_scale(graag_met["Gewicht"]) if not graag_met.empty else 1
    satisfied, weighted_terms, weight_range = _honored_terms(
        model, in_group, graag_met, groups_to, scale
    )

    satisfaction = {}
    for student in students:
        if student not in weighted_terms:  # no preferences: constant baseline 1
            satisfaction[student] = model.NewConstant(SATISFACTION_SCALE)
            continue
        satisfaction[student] = _satisfaction_variable(
            model, student, weighted_terms[student], weight_range[student], scale
        )
    return satisfied, satisfaction


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
    """The student's satisfaction integer: element lookup over the F staircase."""
    low, high = bounds
    weighted_sum = model.NewIntVar(low, high, f"w_{student}")
    model.Add(weighted_sum == sum(terms))

    table = [
        _scaled_satisfaction(value / scale, high / scale)
        for value in range(low, high + 1)
    ]
    satisfaction_var = model.NewIntVar(min(table), max(table), f"sat_{student}")
    index = model.NewIntVar(0, high - low, f"wi_{student}")
    model.Add(index == weighted_sum - low)
    model.AddElement(index, table, satisfaction_var)
    return satisfaction_var


def _scaled_satisfaction(weighted: float, best: float) -> int:
    """Normalized satisfaction at weighted level ``weighted``, as a scaled integer.

    ``best`` is the student's maximal positive weight sum. Mirrors
    ``satisfaction._normalize_and_bound``: with positive wishes the score is
    F(weighted)/F(best); without them the baseline 1 plus the (non-positive)
    F(weighted) of violated negative wishes.
    """
    raw = get_satisfaction_integral(0, weighted)
    if best > 0:
        value = raw / get_satisfaction_integral(0, best)
    else:
        value = 1.0 + raw
    return round(value * SATISFACTION_SCALE)
