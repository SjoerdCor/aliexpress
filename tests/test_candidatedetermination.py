"""Test for candidate determination"""

import pandas as pd
import pytest

from aliexpress.data.candidatedetermination import (
    create_unique_name,
    get_candidates,
    get_groups_from,
    get_groups_to,
    handle_edexml_upload,
    students_df_from_records,
    unique_display_names,
)


def test_unique_display_names_maps_full_to_short():
    """Each participant's full display name maps to a short unique name: just the roepnaam
    when that is unique, extended with surname letters only to break a tie."""
    students = [
        {"roepnaam": "Sanne", "achternaam": "Klaassen"},
        {"roepnaam": "Sanne", "achternaam": "Kuipers"},
        {"roepnaam": "Tim", "achternaam": "de Vries"},
    ]
    result = unique_display_names(students)
    assert result["Tim de Vries"] == "Tim"  # unique roepnaam → no surname needed
    assert result["Sanne Klaassen"] == "Sanne Kl"  # tie broken with minimal letters
    assert result["Sanne Kuipers"] == "Sanne Ku"


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


def test_students_df_from_records_assigns_unique_names():
    """Roster participants get unique display names and are sorted for the Excel template."""
    participants = [
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
        {
            "key": "3",
            "roepnaam": "Chris",
            "achternaam": "Visser",
            "groepsnaam": "Y",
            "geslacht": "Jongen",
        },
    ]
    df_total = students_df_from_records(participants)
    assert df_total["uniekenaam"].tolist() == ["Anna Ba", "Anna Bo", "Chris"]
    assert df_total["groepsnaam"].tolist() == ["X", "X", "Y"]


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
