"""Immutable, solver-space values used to explain hard-condition conflicts.

The solver stores matching keys rather than names as entered.  Keeping these values in
this small module lets the model and feasibility code describe a conflict without taking
on presentation concerns from :mod:`aliexpress.main` or the web layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceContext:
    """One ordinary preference shown as context for an extra-zekerheid condition."""

    kind: str
    target: str
    weight: float

    def to_context(self) -> dict:
        """Return a JSON-serializable representation."""
        return {"kind": self.kind, "target": self.target, "weight": self.weight}


@dataclass(frozen=True)
class ForbiddenGroup:
    """One hard ``Niet in`` exclusion."""

    student: str
    group: str

    def to_context(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "type": "forbidden_group",
            "student": self.student,
            "group": self.group,
        }


@dataclass(frozen=True)
class MinimumSatisfaction:
    """One hard extra-zekerheid floor and its non-hard preference context."""

    student: str
    floor: float
    preferences: tuple[PreferenceContext, ...] = ()

    def __post_init__(self):
        preferences = tuple(
            (
                preference
                if isinstance(preference, PreferenceContext)
                else PreferenceContext(
                    preference["kind"], preference["target"], preference["weight"]
                )
            )
            for preference in self.preferences
        )
        object.__setattr__(self, "preferences", preferences)

    def to_context(self) -> dict:
        """Return a JSON-serializable representation.

        ``preferences`` deliberately lives below this condition.  It explains how the
        floor is calculated but is not a separate hard condition in the conflict core.
        """
        return {
            "type": "minimum_satisfaction",
            "student": self.student,
            "floor": self.floor,
            "preferences": [preference.to_context() for preference in self.preferences],
        }


@dataclass(frozen=True)
class NotTogetherRule:
    """One complete user-entered not-together rule."""

    rule_index: int
    students: tuple[str, ...]
    max_together: int

    def __post_init__(self):
        object.__setattr__(self, "students", tuple(self.students))

    def to_context(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "type": "not_together",
            "rule_index": self.rule_index,
            "students": list(self.students),
            "max_together": self.max_together,
        }


ConflictCondition = ForbiddenGroup | MinimumSatisfaction | NotTogetherRule


@dataclass(frozen=True)
class Conflict:
    """One subset-minimal collection of hard user conditions."""

    conditions: tuple[ConflictCondition, ...]

    def __post_init__(self):
        object.__setattr__(self, "conditions", tuple(self.conditions))

    def to_context(self) -> dict:
        """Return the serializable payload stored in ``FeasibilityError.context``."""
        return {"conditions": [condition.to_context() for condition in self.conditions]}

    def to_error_context(self) -> dict:
        """Return the complete context shape used by ``FeasibilityError``."""
        return {"case": "detailed", "conflict": self.to_context()}
