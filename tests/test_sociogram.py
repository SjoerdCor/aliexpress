"""Tests for the sociogram view model and its browser-only implementation."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
# pylint: disable=duplicate-code  # the canonical helper mirrors browser setup

import pandas as pd
import pytest

from aliexpress import sociogram
from aliexpress.data import datareader
from aliexpress.data.preferences_data import PreferenceData

GROUPS = ["blauw", "groen", "geel", "oranje"]
PREFS_PATH = "testdata/voorkeuren_klein.xlsx"


@pytest.fixture()
def preference_data():
    """PreferenceData built from the small test Excel."""
    processor = datareader.VoorkeurenProcessor(PREFS_PATH)
    processor.process(all_to_groups=GROUPS)
    return processor.to_preference_data()


def test_build_sociogram_view_contains_students_and_only_student_preferences(
    preference_data,
):
    """The first view contains every student and preserves visible arrow data."""
    view = sociogram.build_sociogram_view(preference_data)

    assert [node.id for node in view.nodes] == list(preference_data.students_info)
    assert [node.label for node in view.nodes] == list(
        preference_data.student_display.values()
    )
    assert [
        (edge.source, edge.target, edge.weight, edge.kind) for edge in view.preferences
    ] == [("harrie", "maya", -1.0, "negative")]

    received_by_id = {node.id: node.received_preference_score for node in view.nodes}
    assert received_by_id["maya"] == -1.0
    assert received_by_id["charlie"] == 0.0
    sizes = [node.size for node in view.nodes]
    assert max(sizes) - min(sizes) < 10


def test_build_sociogram_view_keeps_raw_edge_weight_but_clips_received_score(
    preference_data,
):
    """Display data keeps the input weight; the node score bounds each contribution."""
    preference_data.preferences.loc[("harrie", "Graag met", 2), "Gewicht"] = -7.0

    view = sociogram.build_sociogram_view(preference_data)

    assert view.preferences[0].weight == -7.0
    received_by_id = {node.id: node.received_preference_score for node in view.nodes}
    assert received_by_id["maya"] == -2.0


def test_build_sociogram_view_scales_edge_width_by_absolute_weight():
    """Stronger preferences get thicker, bounded lines regardless of their sign."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges(
            [
                ("alice", "bob", 2),
                ("charlie", "dave", 5),
                ("bob", "alice", -2),
                ("dave", "charlie", -5),
            ],
            students=("alice", "bob", "charlie", "dave"),
        )
    )

    widths = {edge.weight: edge.line_width for edge in view.preferences}

    assert widths[5.0] > widths[2.0]
    assert widths[-5.0] == pytest.approx(widths[5.0])
    assert widths[-2.0] == pytest.approx(widths[2.0])
    assert widths[5.0] / widths[2.0] < 2.5


def _preference_data_from_edges(edges, students=("alice", "bob", "charlie")):
    """Build the smallest canonical input needed for a layout-relation test."""
    records = []
    index = []
    for number, (source, target, weight) in enumerate(edges, start=1):
        index.append(
            (source, "Graag met" if weight > 0 else "Liever niet met", float(number))
        )
        records.append({"Waarde": target, "Gewicht": float(weight)})

    preferences = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(index, names=["Leerling", "TypeWens", "Nr"]),
        columns=["Waarde", "Gewicht"],
    )
    return PreferenceData(
        preferences=preferences,
        students_info={student: {} for student in students},
        student_display={student: student.title() for student in students},
        stamgroep_display={},
        input_sheet=pd.DataFrame(),
    )


def test_build_sociogram_view_uses_average_strength_for_mutual_positive_relation():
    """Mutual positive choices become one relation with their average strength."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges([("alice", "bob", 2), ("bob", "alice", 4)])
    )

    assert len(view.layout_relations) == 1
    assert [(edge.source, edge.target) for edge in view.preferences] == [
        ("alice", "bob"),
        ("bob", "alice"),
    ]
    relation = view.layout_relations[0]
    assert {relation.source, relation.target} == {"alice", "bob"}
    assert relation.kind == "mutual_positive"
    assert relation.strength == pytest.approx(3.0)


def test_build_sociogram_view_keeps_one_sided_positive_strength():
    """A one-sided positive choice becomes one positive layout relation."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges([("alice", "bob", 3)])
    )

    assert len(view.layout_relations) == 1
    relation = view.layout_relations[0]
    assert relation.kind == "positive"
    assert relation.strength == pytest.approx(3.0)


def test_build_sociogram_view_lets_negative_preference_win_mixed_sign_pair():
    """A positive and negative choice become a negative relation at max abs weight."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges([("alice", "bob", 4), ("bob", "alice", -2)])
    )

    relation = view.layout_relations[0]
    assert relation.kind == "negative"
    assert relation.strength == pytest.approx(2.0)


def test_build_sociogram_view_uses_strongest_negative_for_mutual_negative_pair():
    """Mutual negative choices stay negative and use the strongest absolute weight."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges([("alice", "bob", -1), ("bob", "alice", -5)])
    )

    relation = view.layout_relations[0]
    assert relation.kind == "negative"
    assert relation.strength == pytest.approx(5.0)


def test_build_sociogram_view_keeps_relation_bands_above_weight_nuance():
    """Mutual positive, positive and negative relations occupy ordered distance bands."""
    view = sociogram.build_sociogram_view(
        _preference_data_from_edges(
            [
                ("alice", "bob", 0.01),
                ("bob", "alice", 0.01),
                ("alice", "charlie", 100),
                ("alice", "dave", -0.01),
            ],
            students=("alice", "bob", "charlie", "dave"),
        )
    )

    relations = {
        (relation.source, relation.target): relation
        for relation in view.layout_relations
    }
    assert (
        relations["alice", "bob"].ideal_distance
        < relations["alice", "charlie"].ideal_distance
    )
    assert (
        relations["alice", "charlie"].ideal_distance
        < relations["alice", "dave"].ideal_distance
    )
    assert ("bob", "charlie") not in relations
    assert {relation.source for relation in view.layout_relations} | {
        relation.target for relation in view.layout_relations
    } == {"alice", "bob", "charlie", "dave"}
    assert len(view.layout_relations) == 3
