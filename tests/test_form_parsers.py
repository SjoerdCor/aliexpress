"""Tests for data/form_parsers.py (pure form-to-dataclass conversions)."""

import pytest
from werkzeug.datastructures import MultiDict

from aliexpress.data import form_parsers
from aliexpress.errors import ValidationError
from aliexpress.solver._balance import BalanceMaxima
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


class TestParseBalanceMaximaForm:
    """Tests for the form-parsing helper parse_balance_maxima_form."""

    ALL_NUMBERS = MultiDict(
        [
            ("maxima_max_diff_n_students_year", "2"),
            ("maxima_max_diff_n_students_total", "3"),
            ("maxima_max_imbalance_boys_girls_year", "2"),
            ("maxima_max_imbalance_boys_girls_total", "3"),
            ("maxima_max_clique", "5"),
            ("maxima_max_clique_sex", "3"),
        ]
    )

    def test_all_fields_filled_in_gives_matching_maxima(self):
        """Six filled-in number fields produce a BalanceMaxima with those ints."""
        result = form_parsers.parse_balance_maxima_form(self.ALL_NUMBERS)
        assert result == BalanceMaxima(
            max_diff_n_students_year=2,
            max_diff_n_students_total=3,
            max_imbalance_boys_girls_year=2,
            max_imbalance_boys_girls_total=3,
            max_clique=5,
            max_clique_sex=3,
        )

    def test_unlimited_checkbox_overrides_an_ignored_number(self):
        """A checked unlimited checkbox makes that family None, even with a stray number."""
        form = MultiDict(self.ALL_NUMBERS.items(multi=True))
        form["maxima_max_clique_unlimited"] = "on"
        result = form_parsers.parse_balance_maxima_form(form)
        assert result.max_clique is None
        assert result.max_diff_n_students_year == 2

    def test_empty_number_without_unlimited_raises_missing(self):
        """A blank number field (no unlimited checkbox) raises with the missing code."""
        form = MultiDict(self.ALL_NUMBERS.items(multi=True))
        form["maxima_max_clique"] = ""
        with pytest.raises(ValidationError) as exc_info:
            form_parsers.parse_balance_maxima_form(form)
        assert exc_info.value.code == "missing_balance_maximum"

    def test_non_integer_value_raises_invalid(self):
        """A non-integer number field raises with the invalid code."""
        form = MultiDict(self.ALL_NUMBERS.items(multi=True))
        form["maxima_max_clique"] = "abc"
        with pytest.raises(ValidationError) as exc_info:
            form_parsers.parse_balance_maxima_form(form)
        assert exc_info.value.code == "invalid_balance_maximum"

    def test_zero_value_raises_invalid(self):
        """A value of 0 raises with the invalid code (minimum is 1)."""
        form = MultiDict(self.ALL_NUMBERS.items(multi=True))
        form["maxima_max_clique"] = "0"
        with pytest.raises(ValidationError) as exc_info:
            form_parsers.parse_balance_maxima_form(form)
        assert exc_info.value.code == "invalid_balance_maximum"
