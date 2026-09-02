"""Focused tests for modelbuilder's shared student-pair literals."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
from ortools.sat.python import cp_model

from aliexpress.solver import modelbuilder
from aliexpress.solver.conflicts import (
    Conflict,
    ForbiddenGroup,
    MinimumSatisfaction,
    NotTogetherRule,
    PreferenceContext,
)

# The test intentionally exercises private modelbuilder helpers at their seam.
# pylint: disable=protected-access


def test_reciprocal_student_pair_reuses_same_group_literal():
    """Reciprocal wishes use one reified equality for the unordered pair."""
    assignment_model = modelbuilder._build_assignment_model(
        students={"alice": {}, "bob": {}},
        groups_to={"red": {}, "blue": {}},
    )

    forward = modelbuilder._same_group_literal(assignment_model, "alice", "bob")
    reverse = modelbuilder._same_group_literal(assignment_model, "bob", "alice")

    assert forward is reverse
    assert assignment_model.same_group_literal_by_student_pair == {
        ("alice", "bob"): forward
    }


def test_conflict_conditions_are_immutable_and_serializable():
    """A detailed conflict keeps user conditions distinct from preference context."""
    minimum = MinimumSatisfaction(
        "piet",
        1.0,
        preferences=(PreferenceContext("Graag met", "sam", 1.0),),
    )
    conflict = Conflict(
        (
            ForbiddenGroup("piet", "blauw"),
            minimum,
            NotTogetherRule(2, ("piet", "sam"), 1),
        )
    )

    assert conflict.to_context() == {
        "conditions": [
            {"type": "forbidden_group", "student": "piet", "group": "blauw"},
            {
                "type": "minimum_satisfaction",
                "student": "piet",
                "floor": 1.0,
                "preferences": [{"kind": "Graag met", "target": "sam", "weight": 1.0}],
            },
            {
                "type": "not_together",
                "rule_index": 2,
                "students": ["piet", "sam"],
                "max_together": 1,
            },
        ]
    }

    with pytest.raises(FrozenInstanceError):
        minimum.floor = 0.5


def test_diagnostic_builder_has_one_assumption_per_user_condition():
    """All diagnostic assumptions reproduce the existing hard feasibility result."""
    preferences = pd.DataFrame(
        [
            {
                "Leerling": "piet",
                "TypeWens": "Graag met",
                "Nr": 1.0,
                "Waarde": "sam",
                "Gewicht": 1.0,
            },
            {
                "Leerling": "piet",
                "TypeWens": "Niet in",
                "Nr": 1.0,
                "Waarde": "blauw",
                "Gewicht": 1.0,
            },
        ]
    ).set_index(["Leerling", "TypeWens", "Nr"])
    students = {
        "piet": {
            "MinimaleTevredenheid": 1.0,
            "Jongen/meisje": "Jongen",
            "Stamgroep": "a",
        },
        "sam": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "a",
        },
    }
    groups_to = {
        "blauw": {"Jongens": 0, "Meisjes": 0},
        "rood": {"Jongens": 0, "Meisjes": 0},
    }
    not_together = [{"group": {"piet", "sam"}, "Max_aantal_samen": 1}]

    diagnostic = modelbuilder.build_diagnostic_problem(
        preferences, students, groups_to, not_together
    )
    conditions = list(diagnostic.condition_by_index.values())

    assert len(conditions) == 3
    assert sum(isinstance(c, ForbiddenGroup) for c in conditions) == 1
    assert sum(isinstance(c, MinimumSatisfaction) for c in conditions) == 1
    assert sum(isinstance(c, NotTogetherRule) for c in conditions) == 1

    diagnostic_solver = cp_model.CpSolver()
    assert diagnostic_solver.Solve(diagnostic.model) == cp_model.INFEASIBLE

    existing = modelbuilder.build_feasibility_problem(
        preferences,
        students,
        groups_to,
        not_together,
        min_satisfaction_hard=True,
        not_together_hard=True,
    )
    existing_solver = cp_model.CpSolver()
    assert existing_solver.Solve(existing) == cp_model.INFEASIBLE


def test_diagnostic_builder_matches_feasible_hard_feasibility_model():
    """With no conflicting conditions, both model builders prove feasibility."""
    preferences = pd.DataFrame(
        columns=["Waarde", "Gewicht"],
        index=pd.MultiIndex.from_tuples([], names=["Leerling", "TypeWens", "Nr"]),
    )
    students = {
        "piet": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Jongen",
            "Stamgroep": "a",
        }
    }
    groups_to = {"blauw": {"Jongens": 0, "Meisjes": 0}}

    diagnostic = modelbuilder.build_diagnostic_problem(
        preferences, students, groups_to, []
    )
    existing = modelbuilder.build_feasibility_problem(
        preferences,
        students,
        groups_to,
        [],
        min_satisfaction_hard=True,
        not_together_hard=True,
    )

    diagnostic_solver = cp_model.CpSolver()
    existing_solver = cp_model.CpSolver()
    assert diagnostic_solver.Solve(diagnostic.model) in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    )
    assert existing_solver.Solve(existing) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_diagnostic_context_restores_negative_preference_label():
    """A negative canonical weight is shown as the form's avoid-preference type."""
    preferences = pd.DataFrame(
        [
            {
                "Leerling": "piet",
                "TypeWens": "Graag met",
                "Nr": 1.0,
                "Waarde": "sam",
                "Gewicht": -2.0,
            }
        ]
    ).set_index(["Leerling", "TypeWens", "Nr"])
    students = {
        "piet": {
            "MinimaleTevredenheid": 0.5,
            "Jongen/meisje": "Jongen",
            "Stamgroep": "a",
        },
        "sam": {
            "MinimaleTevredenheid": float("nan"),
            "Jongen/meisje": "Meisje",
            "Stamgroep": "a",
        },
    }
    diagnostic = modelbuilder.build_diagnostic_problem(
        preferences,
        students,
        {"blauw": {"Jongens": 0, "Meisjes": 0}},
        [],
    )

    minimum = next(
        condition
        for condition in diagnostic.condition_by_index.values()
        if isinstance(condition, MinimumSatisfaction)
    )

    assert minimum.preferences == (PreferenceContext("Liever niet met", "sam", 2.0),)
