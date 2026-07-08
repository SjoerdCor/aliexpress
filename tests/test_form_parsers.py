"""Tests for data/form_parsers.py (pure form-to-dataclass conversions)."""

from werkzeug.datastructures import MultiDict

from aliexpress.data import form_parsers
from tests.helpers import make_students


class TestParseGroupsToForm:
    """Tests for the form-parsing helper parse_groups_to_form."""

    def test_counts_genders_and_keeps_empty_group(self):
        """Genders are looked up by index; a submitted group without ticks stays 0/0."""
        groups_to = {
            "Klas A": make_students("Jongen", "Meisje", "Jongen"),
            "Klas B": make_students("Jongen"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Klas B"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "1"),
                ("group_students[Klas A]", "2"),
            ]
        )
        result = form_parsers.parse_groups_to_form(form, groups_to)
        assert result.distribution == {
            "Klas A": {"Jongens": 2, "Meisjes": 1},
            "Klas B": {"Jongens": 0, "Meisjes": 0},
        }
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0, 1, 2]
        assert result.state["disabled_groups"] == []
        assert result.state["new_groups"] == []

    def test_disabled_and_new_groups_are_recorded(self):
        """An original group absent from 'group' is disabled; an unknown name is new."""
        groups_to = {
            "Klas A": make_students("Jongen", "Meisje"),
            "Klas B": make_students("Jongen"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group", "Nieuwe groep 1"),
                ("group_students[Klas A]", "0"),
            ]
        )
        result = form_parsers.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["new_groups"] == ["Nieuwe groep 1"]
        assert result.distribution["Nieuwe groep 1"] == {"Jongens": 0, "Meisjes": 0}

    def test_switched_off_group_keeps_its_ticks(self):
        """A switched-off group still submits its boxes, so its ticks are remembered."""
        groups_to = {
            "Klas A": make_students("Jongen"),
            "Klas B": make_students("Jongen", "Meisje"),
        }
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                # Klas B is switched off (absent from 'group') but its boxes still submit.
                ("group_students[Klas B]", "1"),
            ]
        )
        result = form_parsers.parse_groups_to_form(form, groups_to)
        assert result.state["disabled_groups"] == ["Klas B"]
        assert result.state["original_groups"]["Klas B"]["checked_indices"] == [1]
        # Switched-off groups must not reach groups.xlsx.
        assert "Klas B" not in result.distribution

    def test_out_of_range_or_non_numeric_indices_are_ignored(self):
        """Tampered indices that fall outside the student list are dropped safely."""
        groups_to = {"Klas A": make_students("Jongen")}
        form = MultiDict(
            [
                ("group", "Klas A"),
                ("group_students[Klas A]", "0"),
                ("group_students[Klas A]", "9"),
                ("group_students[Klas A]", "x"),
            ]
        )
        result = form_parsers.parse_groups_to_form(form, groups_to)
        assert result.distribution["Klas A"] == {"Jongens": 1, "Meisjes": 0}
        assert result.state["original_groups"]["Klas A"]["checked_indices"] == [0]
