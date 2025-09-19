"""Test for candidate determination"""

import pandas as pd
import pytest

from aliexpress.candidatedetermination import (
    _build_groups_summary,
    _combine_students,
    create_unique_name,
    get_candidates,
    get_groups_from,
    get_groups_to,
    handle_edexml_upload,
    handle_form_submission,
)


# doesn't work with fixtures
# pylint: disable=redefined-outer-name
@pytest.fixture
def sample_df():
    """Generate sample df"""
    return pd.DataFrame(
        [
            {
                "key": "L1",
                "roepnaam": "Anna",
                "achternaam": "Bakker",
                "groepsnaam": "3A",
                "geslacht": "Meisje",
                "jaargroep": 3,
            },
            {
                "key": "L2",
                "roepnaam": "Ben",
                "achternaam": "Jansen",
                "groepsnaam": "3B",
                "geslacht": "Jongen",
                "jaargroep": 3,
            },
            {
                "key": "L3",
                "roepnaam": "Carl",
                "achternaam": "Visser",
                "groepsnaam": "3A",
                "geslacht": "Jongen",
                "jaargroep": 3,
            },
            {
                "key": "L4",
                "roepnaam": "Daan",
                "achternaam": "Smits",
                "groepsnaam": "4A",
                "geslacht": "Jongen",
                "jaargroep": 4,
            },
            {
                "key": "L5",
                "roepnaam": "Emma",
                "achternaam": "Bos",
                "groepsnaam": "4A",
                "geslacht": "Meisje",
                "jaargroep": 4,
            },
            {
                "key": "L6",
                "roepnaam": "Finn",
                "achternaam": "Dekker",
                "groepsnaam": "5A",
                "geslacht": "Jongen",
                "jaargroep": 5,
            },
        ]
    )


def test_get_candidates_sorted(sample_df):
    """Test candidates are sorted correctly"""
    result = get_candidates(sample_df, 3)
    # moet gesorteerd zijn op groepsnaam, roepnaam, achternaam
    names = [r["roepnaam"] for r in result]
    assert names == ["Anna", "Carl", "Ben"]
    # bevat juiste kolommen
    assert set(result[0].keys()) == {
        "key",
        "roepnaam",
        "achternaam",
        "groepsnaam",
        "geslacht",
    }


def test_get_candidates_empty(sample_df):
    """Test candidates empty list works"""
    result = get_candidates(sample_df, 99)
    assert result == []


def test_get_groups_from(sample_df):
    """Test original groups are returned correctly"""
    result = get_groups_from(sample_df, 3)
    assert set(result) == {"3A", "3B", "Anders"}


def test_get_groups_from_empty(sample_df):
    """Test no groups works correctly"""
    result = get_groups_from(sample_df, 42)
    assert result == ["Anders"]


def test_get_groups_to_normal(sample_df):
    """Test get_groups_to works for regular file"""
    result = get_groups_to(sample_df, 3)
    assert list(result.keys()) == ["4A"]
    members = result["4A"]
    assert all(not r["blijft_in_groep"] for r in members)


def test_get_groups_to_no_next(sample_df):
    """Test get_groups_to correctly gives nothing"""
    result = get_groups_to(sample_df, 5)
    assert result == {}


def test_handle_edexml_upload(sample_df):
    """Test orchestration still works"""
    candidates, groups_from, groups_to = handle_edexml_upload(sample_df, 3)
    assert isinstance(candidates, list)
    assert "Anders" in groups_from
    assert isinstance(groups_to, dict)


def test_build_groups_summary_counts():
    """ " Test correct groups are returned"""
    existing_groups = {
        "G1": ["Jongen", "Meisje", "Jongen"],
        "G2": ["Meisje"],
    }
    new_groups = ["G3"]

    result = _build_groups_summary(existing_groups, new_groups)
    expected_result = {
        "G1": {"Jongens": 2, "Meisjes": 1},
        "G2": {"Jongens": 0, "Meisjes": 1},
        "G3": {"Jongens": 0, "Meisjes": 0},
    }
    assert result == expected_result


def test_combine_students_and_unique_names():
    """Test new and old groups work correctly"""
    candidates = [
        {
            "key": "1",
            "roepnaam": "Anna",
            "achternaam": "Bakker",
            "groepsnaam": "X",
            "geslacht": "Meisje",
        },
        {
            "key": "2",
            "roepnaam": "Anna",
            "achternaam": "Bos",
            "groepsnaam": "X",
            "geslacht": "Meisje",
        },
    ]
    selected_ids = ["1", "2"]
    new_students = [
        {
            "roepnaam": "Chris",
            "achternaam": "Visser",
            "groepsnaam": "Y",
            "geslacht": "Jongen",
        }
    ]

    df_total = _combine_students(candidates, selected_ids, new_students)
    df_total_expected = pd.DataFrame(
        {
            "roepnaam": {"1": "Anna", "2": "Anna", 0: "Chris"},
            "achternaam": {"1": "Bakker", "2": "Bos", 0: "Visser"},
            "groepsnaam": {"1": "X", "2": "X", 0: "Y"},
            "geslacht": {"1": "Meisje", "2": "Meisje", 0: "Jongen"},
            "uniekenaam": {"1": "Anna Ba", "2": "Anna Bo", 0: "Chris"},
        }
    )
    pd.testing.assert_frame_equal(df_total, df_total_expected)


def test_handle_form_submission_integrated():
    """Small integration test"""
    existing_groups = {"A": ["Jongen", "Meisje"], "B": ["Jongen"]}
    new_groups = ["C"]
    candidates = [
        {
            "key": "1",
            "roepnaam": "Eva",
            "achternaam": "Jong",
            "groepsnaam": "A",
            "geslacht": "Meisje",
        },
        {
            "key": "2",
            "roepnaam": "Mark",
            "achternaam": "Bakker",
            "groepsnaam": "B",
            "geslacht": "Jongen",
        },
    ]
    selected_ids = ["1"]
    new_students = [
        {
            "roepnaam": "Lars",
            "achternaam": "Bos",
            "groepsnaam": "Anders",
            "geslacht": "Jongen",
        }
    ]

    groups_to, df_total = handle_form_submission(
        existing_groups, new_groups, candidates, new_students, selected_ids
    )
    expected_groups_to = {
        "A": {"Jongens": 1, "Meisjes": 1},
        "B": {"Jongens": 1, "Meisjes": 0},
        "C": {"Jongens": 0, "Meisjes": 0},
    }
    assert expected_groups_to == groups_to
    df_total_expected = pd.DataFrame(
        {
            "roepnaam": ["Eva", "Lars"],
            "achternaam": ["Jong", "Bos"],
            "groepsnaam": ["A", "Anders"],
            "geslacht": ["Meisje", "Jongen"],
            "uniekenaam": ["Eva", "Lars"],
        },
        index=["1", 0],
    )
    pd.testing.assert_frame_equal(df_total, df_total_expected)
    assert set(df_total["roepnaam"]) == {"Eva", "Lars"}


def test_create_unique_name_handles_duplicates():
    """Test unique names generates unique names"""
    df = pd.DataFrame(
        [
            {"roepnaam": "Sam", "achternaam": "Jansen"},
            {"roepnaam": "Sam", "achternaam": "Bos"},
            {"roepnaam": "Sam", "achternaam": "Bak"},
        ]
    )
    unique_names = create_unique_name(df)
    assert set(unique_names) == {"Sam J", "Sam Bo", "Sam Ba"}


def test_create_unique_name_no_duplicates():
    """Test unique names stay original if possible"""
    df = pd.DataFrame(
        [
            {"roepnaam": "Lars", "achternaam": "Bos"},
            {"roepnaam": "Eva", "achternaam": "Janssen"},
        ]
    )
    unique_names = create_unique_name(df)
    assert set(unique_names) == {"Lars", "Eva"}
