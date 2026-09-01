"""Focused tests for modelbuilder's shared student-pair literals."""

# The test intentionally exercises private modelbuilder helpers at their seam.
# pylint: disable=protected-access

from aliexpress.solver import modelbuilder


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
