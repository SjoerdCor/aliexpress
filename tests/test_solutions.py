"""Unit tests for solutions.py"""

import pandas as pd

from aliexpress.solver.results import GroupComposition, SexCounts, SolutionResult
from aliexpress.solver.solutions import DisplayNames, SolutionAnalyzer, to_display_names


def _analyzer() -> SolutionAnalyzer:
    """A SolutionAnalyzer built from a small, hand-made SolutionResult."""
    result = SolutionResult(
        assignment={"Anna": "A", "Bram": "B"},
        student_satisfaction={"Anna": 1.0, "Bram": 0.5},
        satisfied={("Anna", 1): True, ("Bram", 1): False},
        weighted_satisfied={("Anna", 1): 1.0, ("Bram", 1): 0.0},
        weights={("Anna", 1): 1.0, ("Bram", 1): 2.0},
        group_composition={
            "A": GroupComposition(
                boys_total=2, girls_total=1, per_year={None: SexCounts(1, 0)}
            ),
            "B": GroupComposition(
                boys_total=1, girls_total=2, per_year={None: SexCounts(0, 1)}
            ),
        },
    )
    students_info = {
        "Anna": {"Stamgroep": "X", "Jongen/meisje": "Meisje"},
        "Bram": {"Stamgroep": "Y", "Jongen/meisje": "Jongen"},
    }
    # preferences/input_sheet are only used by the styled-display methods, not here.
    return SolutionAnalyzer(result, pd.DataFrame(), pd.DataFrame(), students_info)


def test_indexed_series_empty_returns_named_multiindex():
    """Empty mapping keeps the (student, Nr) index names the aggregations rely on."""
    # pylint: disable-next=protected-access  # testing an internal static helper directly
    result = SolutionAnalyzer._indexed_series({}, "Satisfied")
    assert isinstance(result, pd.Series)
    assert len(result) == 0
    assert result.index.names == ["student", "Nr"]


def test_get_outcome_reflects_assignment():
    """The groepsindeling table mirrors the student -> group assignment."""
    groepsindeling = _analyzer().groepsindeling.set_index("Naam")["Group"]
    assert groepsindeling.to_dict() == {"Anna": "A", "Bram": "B"}


def test_group_report_counts_and_derived_columns():
    """Group report carries the composition counts plus difference and size."""
    report = _analyzer().group_report
    assert report.loc[("A", "Totaal"), "Jongen"] == 2
    assert report.loc[("A", "Totaal"), "Meisje"] == 1
    assert report.loc[("A", "Totaal"), "VerschilJongensMeisjes"] == 1
    assert report.loc[("A", "Totaal"), "Groepsgrootte"] == 3
    assert report.loc[("B", "Jaarlaag"), "Meisje"] == 1


def _result_with_composition(group_composition: dict) -> SolutionResult:
    """A SolutionResult with one placeholder wish, so the eagerly-computed performance
    views (unrelated to group_report) don't divide by zero on empty preference data."""
    return SolutionResult(
        assignment={},
        student_satisfaction={"X": 1.0},
        satisfied={("X", 1): True},
        weighted_satisfied={("X", 1): 1.0},
        weights={("X", 1): 1.0},
        group_composition=group_composition,
    )


def test_group_report_multi_year_rows_numerically_ordered():
    """Herindelen: each jaarlaag gets its own row, ordered numerically (not
    alphabetically — "Jaarlaag 10" must sort after "Jaarlaag 6", not before it)."""
    result = _result_with_composition(
        {
            "A": GroupComposition(
                boys_total=3,
                girls_total=2,
                per_year={
                    10: SexCounts(1, 1),
                    6: SexCounts(1, 0),
                    7: SexCounts(1, 1),
                },
            ),
        }
    )
    report = SolutionAnalyzer(result, pd.DataFrame(), pd.DataFrame(), {}).group_report

    assert list(report.loc["A"].index) == [
        "Totaal",
        "Jaarlaag 6",
        "Jaarlaag 7",
        "Jaarlaag 10",
    ]
    assert report.loc[("A", "Jaarlaag 6"), "Jongen"] == 1
    assert report.loc[("A", "Jaarlaag 6"), "Meisje"] == 0
    assert report.loc[("A", "Jaarlaag 7"), "Groepsgrootte"] == 2
    assert report.loc[("A", "Jaarlaag 10"), "Groepsgrootte"] == 2


def test_group_report_shifts_year_for_forward():
    """year_offset shifts the Groepsindeling sheet's jaarlaag row labels (Overgang mode)."""
    result = _result_with_composition(
        {
            "A": GroupComposition(
                boys_total=1,
                girls_total=0,
                per_year={5: SexCounts(1, 0)},
            ),
        }
    )
    shifted = SolutionAnalyzer(
        result, pd.DataFrame(), pd.DataFrame(), {}, year_offset=1
    ).group_report
    labels = list(shifted.loc["A"].index)
    assert "Jaarlaag 6" in labels
    assert "Jaarlaag 5" not in labels

    unshifted = SolutionAnalyzer(
        result, pd.DataFrame(), pd.DataFrame(), {}
    ).group_report
    labels_default = list(unshifted.loc["A"].index)
    assert "Jaarlaag 5" in labels_default


def test_group_report_mixed_none_and_numbered_cohorts():
    """A None cohort (no jaarlaag) can coexist with numbered ones and keeps its bare
    "Jaarlaag" label, positioned right after "Totaal"."""
    result = _result_with_composition(
        {
            "A": GroupComposition(
                boys_total=1,
                girls_total=1,
                per_year={6: SexCounts(0, 1), None: SexCounts(1, 0)},
            ),
        }
    )
    report = SolutionAnalyzer(result, pd.DataFrame(), pd.DataFrame(), {}).group_report

    assert list(report.loc["A"].index) == ["Totaal", "Jaarlaag", "Jaarlaag 6"]
    assert report.loc[("A", "Jaarlaag"), "Jongen"] == 1
    assert report.loc[("A", "Jaarlaag 6"), "Meisje"] == 1


def test_group_report_different_cohorts_per_group_fills_zero():
    """A group without a particular jaarlaag cohort gets a 0 row, not a crash/NaN."""
    result = _result_with_composition(
        {
            "A": GroupComposition(
                boys_total=1, girls_total=0, per_year={6: SexCounts(1, 0)}
            ),
            "B": GroupComposition(
                boys_total=0, girls_total=1, per_year={7: SexCounts(0, 1)}
            ),
        }
    )
    report = SolutionAnalyzer(result, pd.DataFrame(), pd.DataFrame(), {}).group_report

    assert report.loc[("A", "Jaarlaag 7"), "Groepsgrootte"] == 0
    assert report.loc[("B", "Jaarlaag 6"), "Groepsgrootte"] == 0


def test_student_performance_from_result():
    """Per-student satisfaction and wish counts come straight from the result."""
    perf = _analyzer().student_performance
    assert perf.loc["Anna", "RelativeSatisfaction"] == 1.0
    assert perf.loc["Bram", "RelativeSatisfaction"] == 0.5
    # Only positive weights count towards the number of wishes.
    assert perf.loc["Anna", "NrWeightedPreferences"] == 1.0
    assert perf.loc["Bram", "NrWeightedPreferences"] == 2.0
    assert perf.loc["Anna", "AccountedPreferences"] == 1
    assert perf.loc["Bram", "AccountedPreferences"] == 0


def _result_for_students(assignment: dict, students: list) -> SolutionResult:
    """A minimal SolutionResult keyed by matching keys, one fulfilled wish per student."""
    return SolutionResult(
        assignment=assignment,
        student_satisfaction={s: 1.0 for s in students},
        satisfied={(s, 1): True for s in students},
        weighted_satisfied={(s, 1): 1.0 for s in students},
        weights={(s, 1): 1.0 for s in students},
        group_composition={
            g: GroupComposition(
                boys_total=1, girls_total=1, per_year={None: SexCounts(1, 1)}
            )
            for g in set(assignment.values())
        },
    )


def test_display_names_shown_as_entered():
    """to_display_names maps the matching keys back to the names as entered."""
    result = _result_for_students(
        {"AnneClaire": "Groen", "Obrien": "Groen"}, ["AnneClaire", "Obrien"]
    )
    display = DisplayNames(
        student={"AnneClaire": "Anne Claire", "Obrien": "O'Brien"},
        group={"Groen": "Groen"},
    )
    result, preferences, input_sheet, students_info = to_display_names(
        result,
        pd.DataFrame(),
        pd.DataFrame(),
        {"AnneClaire": {"Stamgroep": "X"}, "Obrien": {"Stamgroep": "X"}},
        display,
    )
    analyzer = SolutionAnalyzer(result, preferences, input_sheet, students_info)
    naam_to_group = analyzer.groepsindeling.set_index("Naam")["Group"].to_dict()
    assert naam_to_group == {"Anne Claire": "Groen", "O'Brien": "Groen"}
    assert set(analyzer.student_performance.index) == {"Anne Claire", "O'Brien"}


def test_display_student_performance_escapes_html():
    """Student names are HTML-escaped in the table (it is rendered raw via | safe)."""
    result = _result_for_students({"Boef": "Groen"}, ["Boef"])
    display = DisplayNames(student={"Boef": "<b>Boef</b>"}, group={"Groen": "Groen"})
    result, preferences, input_sheet, students_info = to_display_names(
        result, pd.DataFrame(), pd.DataFrame(), {"Boef": {"Stamgroep": "X"}}, display
    )
    analyzer = SolutionAnalyzer(result, preferences, input_sheet, students_info)
    html = analyzer.display_student_performance().to_html()
    assert "<b>Boef</b>" not in html
    assert "&lt;b&gt;Boef&lt;/b&gt;" in html
