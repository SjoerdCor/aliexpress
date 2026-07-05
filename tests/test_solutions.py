"""Unit tests for solutions.py"""

import pandas as pd

from aliexpress.solver.results import GroupComposition, SolutionResult
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
                boys_total=2, girls_total=1, boys_year=1, girls_year=0
            ),
            "B": GroupComposition(
                boys_total=1, girls_total=2, boys_year=0, girls_year=1
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
            g: GroupComposition(boys_total=1, girls_total=1, boys_year=1, girls_year=1)
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
