"""Unit tests for the structured Groepsindeling view-model builder.

Two scenarios are exercised end to end from a hand-made ``SolutionResult``:

* **Doorzetten** — one mover jaarlaag with sitting occupancy (``*_total`` exceeds the
  sum of movers), so occupancy chips land on the single year section.
* **Herindelen** — several mover jaarlagen, occupancy 0 everywhere.
"""

import dataclasses

import pandas as pd

from aliexpress.solver.groepsindeling_view import GroepsindelingView, Preference
from aliexpress.solver.results import GroupComposition, SexCounts, SolutionResult
from aliexpress.solver.solutions import SolutionAnalyzer


def _preferences(rows: list[tuple]) -> pd.DataFrame:
    """Build a post-negation long-format preference frame.

    ``rows`` are ``(leerling, nr, waarde, gewicht)`` tuples, all under the folded
    "Graag met" TypeWens (negative ``gewicht`` = a "Liever niet met" preference).
    """
    if not rows:
        index = pd.MultiIndex.from_arrays(
            [[], [], []], names=["Leerling", "TypeWens", "Nr"]
        )
        return pd.DataFrame({"Waarde": [], "Gewicht": []}, index=index)
    index = pd.MultiIndex.from_tuples(
        [(leerling, "Graag met", nr) for leerling, nr, _, _ in rows],
        names=["Leerling", "TypeWens", "Nr"],
    )
    return pd.DataFrame(
        {
            "Waarde": [waarde for _, _, waarde, _ in rows],
            "Gewicht": [gewicht for _, _, _, gewicht in rows],
        },
        index=index,
    )


def _input_sheet(not_in: dict[str, list[str]]) -> pd.DataFrame:
    """A wide input sheet carrying only the two "Niet in" target columns per student."""
    columns = pd.MultiIndex.from_tuples(
        [("Niet in", 1.0, "Waarde"), ("Niet in", 2.0, "Waarde")],
        names=["TypeWens", "Nr", "TypeWaarde"],
    )
    students = list(not_in)
    data = []
    for student in students:
        targets = not_in[student] + [float("nan"), float("nan")]
        data.append(targets[:2])
    return pd.DataFrame(
        data, index=pd.Index(students, name="Leerling"), columns=columns
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Doorzetten: one mover jaarlaag, sitting occupancy > 0
# ---------------------------------------------------------------------------


def _doorzetten_analyzer() -> SolutionAnalyzer:
    """Group A: jaarlaag-5 movers (Anna♀, Bram♂, Cas♂) on top of 3 sitting boys / 2 girls.

    Group B: jaarlaag-5 movers (Daan♂, Eva♀) with no sitting occupancy.
    """
    result = SolutionResult(
        assignment={"Anna": "A", "Bram": "A", "Cas": "A", "Daan": "B", "Eva": "B"},
        # Anna: no preferences -> None; Bram/Cas/Daan have some; Eva only "Niet in".
        student_satisfaction={
            "Anna": 1.0,
            "Bram": -3.0,  # clamps to -1.0
            "Cas": 1.0,
            "Daan": 0.5,
            "Eva": 1.0,
        },
        satisfied={
            ("Bram", 1): True,  # liever-niet honoured (kept apart)
            ("Cas", 1): True,  # graag-met honoured
            ("Daan", 1): False,  # graag-met not honoured
        },
        weighted_satisfied={("Bram", 1): 0.0, ("Cas", 1): 1.0, ("Daan", 1): 0.0},
        weights={("Bram", 1): -2.0, ("Cas", 1): 1.0, ("Daan", 1): 1.0},
        group_composition={
            "A": GroupComposition(
                boys_total=5,  # 2 movers (Bram, Cas) + 3 sitting boys
                girls_total=3,  # 1 mover (Anna) + 2 sitting girls
                per_year={5: SexCounts(boys=2, girls=1)},
            ),
            "B": GroupComposition(
                boys_total=1,  # Daan
                girls_total=1,  # Eva
                per_year={5: SexCounts(boys=1, girls=1)},
            ),
        },
    )
    students_info = {
        "Anna": {
            "Stamgroep": "Kikkers",
            "Jongen/meisje": "Meisje",
            "Jaarlaag": 5,
            "MinimaleTevredenheid": float("nan"),
        },
        "Bram": {
            "Stamgroep": "Bevers",
            "Jongen/meisje": "Jongen",
            "Jaarlaag": 5,
            "MinimaleTevredenheid": 0.5,  # partial
        },
        "Cas": {
            "Stamgroep": "Bevers",
            "Jongen/meisje": "Jongen",
            "Jaarlaag": 5,
            "MinimaleTevredenheid": 1.0,  # full
        },
        "Daan": {
            "Stamgroep": "Adelaars",
            "Jongen/meisje": "Jongen",
            "Jaarlaag": 5,
            "MinimaleTevredenheid": float("nan"),
        },
        "Eva": {
            "Stamgroep": "Adelaars",
            "Jongen/meisje": "Meisje",
            "Jaarlaag": 5,
            "MinimaleTevredenheid": float("nan"),
        },
    }
    preferences = _preferences(
        [
            ("Bram", 1, "Cas", -2.0),  # liever niet met Cas
            ("Cas", 1, "Bram", 1.0),  # graag met Bram
            ("Daan", 1, "Eva", 1.0),  # graag met Eva
        ]
    )
    input_sheet = _input_sheet(
        {
            "Anna": [],
            "Bram": [],
            "Cas": [],
            "Daan": ["Groep C"],
            "Eva": [],
        }
    )
    return SolutionAnalyzer(result, preferences, input_sheet, students_info)


def test_doorzetten_group_card_totals_include_occupancy():
    """Card totals reflect boys_total/girls_total including the sitting occupancy."""
    view = _doorzetten_analyzer().groepsindeling_view()
    assert view.group_order == ["A", "B"]
    card_a = view.groups[0]
    assert (card_a.name, card_a.boys_total, card_a.girls_total, card_a.total) == (
        "A",
        5,
        3,
        8,
    )


def test_doorzetten_single_year_section_size_is_movers_only():
    """The one YearSection's size counts movers only, excluding sitting occupancy."""
    card_a = _doorzetten_analyzer().groepsindeling_view().groups[0]
    assert len(card_a.year_sections) == 1
    section = card_a.year_sections[0]
    assert section.year == 5
    assert section.label == "Jaarlaag 5"
    assert section.size == 3  # 2 boys + 1 girl movers, not the 5 sitting


def test_doorzetten_sex_columns_count_movers_only():
    """SexColumn.new_count counts the movers of that sex (occupancy is not shown here)."""
    section = _doorzetten_analyzer().groepsindeling_view().groups[0].year_sections[0]
    assert section.boys.new_count == 2  # movers, not the 3 sitting boys
    assert len(section.boys.students) == 2
    assert section.girls.new_count == 1
    assert len(section.girls.students) == 1


def test_doorzetten_chip_name_uses_unique_name_with_fallback():
    """chip_name comes from unique_name; a missing entry falls back to the full name."""
    view = _doorzetten_analyzer().groepsindeling_view(unique_name={"Bram": "Bra"})
    boys = view.groups[0].year_sections[0].boys
    by_full = {chip.full_name: chip for chip in boys.students}
    assert by_full["Bram"].chip_name == "Bra"  # from the map
    assert by_full["Cas"].chip_name == "Cas"  # fallback to full name
    assert by_full["Bram"].origin_abbrev == "Bev"
    assert by_full["Bram"].origin_full == "Bevers"


def test_doorzetten_satisfaction_clamped_and_none_without_preferences():
    """Satisfaction is clamped to [-1, 1]; a student without preferences shows None."""
    view = _doorzetten_analyzer().groepsindeling_view()
    girls = view.groups[0].year_sections[0].girls
    anna = next(c for c in girls.students if c.full_name == "Anna")
    assert anna.satisfaction is None  # no graag-met rows at all

    boys = view.groups[0].year_sections[0].boys
    bram = next(c for c in boys.students if c.full_name == "Bram")
    assert bram.satisfaction == -1.0  # clamped from -3.0


def test_doorzetten_liever_niet_preference_and_fulfilled():
    """A negative-weight preference surfaces as kind 'liever_niet_met' with its fulfilled flag."""
    view = _doorzetten_analyzer().groepsindeling_view()
    boys = view.groups[0].year_sections[0].boys
    bram = next(c for c in boys.students if c.full_name == "Bram")
    assert bram.preferences == [
        Preference(
            kind="liever_niet_met",
            target="Cas",
            fulfilled=True,
            target_is_group=False,
            weight=2.0,
        )
    ]
    cas = next(c for c in boys.students if c.full_name == "Cas")
    assert cas.preferences == [
        Preference(
            kind="graag_met",
            target="Bram",
            fulfilled=True,
            target_is_group=False,
            weight=1.0,
        )
    ]


def test_preference_target_group_uses_in_and_classmate_uses_short_name():
    """A group target is flagged (rendered "Graag in"); a classmate shows their short name."""
    result = SolutionResult(
        assignment={"Tess de Wit": "A", "Tim de Vries": "A"},
        student_satisfaction={"Tess de Wit": 1.0, "Tim de Vries": 1.0},
        # Tess: graag met classmate Tim (nr 1) and graag in group A (nr 2).
        satisfied={("Tess de Wit", 1): True, ("Tess de Wit", 2): True},
        weighted_satisfied={("Tess de Wit", 1): 1.0, ("Tess de Wit", 2): 1.0},
        weights={("Tess de Wit", 1): 1.0, ("Tess de Wit", 2): 1.0},
        group_composition={
            "A": GroupComposition(2, 0, {5: SexCounts(boys=2, girls=0)}),
        },
    )
    students_info = {
        "Tess de Wit": {
            "Stamgroep": "Kikkers",
            "Jongen/meisje": "Jongen",
            "Jaarlaag": 5,
        },
        "Tim de Vries": {
            "Stamgroep": "Bevers",
            "Jongen/meisje": "Jongen",
            "Jaarlaag": 5,
        },
    }
    preferences = _preferences(
        [
            ("Tess de Wit", 1, "Tim de Vries", 1.0),  # classmate
            ("Tess de Wit", 2, "A", 1.0),  # group
        ]
    )
    analyzer = SolutionAnalyzer(result, preferences, _input_sheet({}), students_info)
    view = analyzer.groepsindeling_view(unique_name={"Tim de Vries": "Tim d"})
    tess = next(
        c
        for c in view.groups[0].year_sections[0].boys.students
        if c.full_name == "Tess de Wit"
    )
    assert (
        Preference(
            kind="graag_met",
            target="Tim d",
            fulfilled=True,
            target_is_group=False,
            weight=1.0,
        )
        in tess.preferences
    )
    assert (
        Preference(
            kind="graag_met",
            target="A",
            fulfilled=True,
            target_is_group=True,
            weight=1.0,
        )
        in tess.preferences
    )


def test_preferences_sorted_graag_then_liever_heaviest_first():
    """Within a chip, graag-met comes before liever-niet-met, each heaviest weight first."""
    result = SolutionResult(
        assignment={"Sam": "A"},
        student_satisfaction={"Sam": 1.0},
        satisfied={("Sam", nr): True for nr in range(1, 5)},
        weighted_satisfied={("Sam", nr): 1.0 for nr in range(1, 5)},
        weights={("Sam", 1): 1.0, ("Sam", 2): 5.0, ("Sam", 3): -1.0, ("Sam", 4): -2.0},
        group_composition={
            "A": GroupComposition(1, 0, {5: SexCounts(boys=1, girls=0)})
        },
    )
    students_info = {
        "Sam": {"Stamgroep": "Kikkers", "Jongen/meisje": "Jongen", "Jaarlaag": 5}
    }
    preferences = _preferences(
        [
            ("Sam", 1, "Anna", 1.0),  # graag, light
            ("Sam", 2, "Bob", 5.0),  # graag, hartsvriend
            ("Sam", 3, "Cor", -1.0),  # liever niet, light
            ("Sam", 4, "Dex", -2.0),  # liever niet, strong
        ]
    )
    analyzer = SolutionAnalyzer(result, preferences, _input_sheet({}), students_info)
    sam = analyzer.groepsindeling_view().groups[0].year_sections[0].boys.students[0]
    assert [(p.kind, p.target, p.weight) for p in sam.preferences] == [
        ("graag_met", "Bob", 5.0),
        ("graag_met", "Anna", 1.0),
        ("liever_niet_met", "Dex", 2.0),
        ("liever_niet_met", "Cor", 1.0),
    ]


def test_doorzetten_not_in_targets_and_min_satisfaction_levels():
    """not_in lists the hard 'Niet in' targets; min_satisfaction maps 0.5/1.0/NaN."""
    view = _doorzetten_analyzer().groepsindeling_view()
    chips = {
        c.full_name: c
        for card in view.groups
        for section in card.year_sections
        for col in (section.boys, section.girls)
        for c in col.students
    }
    assert chips["Daan"].not_in == ["Groep C"]
    assert chips["Anna"].not_in == []
    assert chips["Bram"].min_satisfaction == "partial"
    assert chips["Cas"].min_satisfaction == "full"
    assert chips["Anna"].min_satisfaction is None


def test_doorzetten_balance_rows_order_and_metrics():
    """balance_rows: Totaal first (incl. occupancy), then the jaarlaag rows (movers)."""
    view = _doorzetten_analyzer().groepsindeling_view()
    labels = [(row.label, row.is_total) for row in view.balance_rows]
    assert labels == [("Totaal", True), ("Jaarlaag 5", False)]

    total = view.balance_rows[0]
    assert total.per_group == {"A": (8, 5, 3), "B": (2, 1, 1)}
    assert total.size_diff == 8 - 2  # 6
    assert total.sex_imbalance == max(abs(5 - 3), abs(1 - 1))  # 2

    year = view.balance_rows[1]
    assert year.per_group == {"A": (3, 2, 1), "B": (2, 1, 1)}
    assert year.size_diff == 3 - 2  # 1
    assert year.sex_imbalance == max(abs(2 - 1), abs(1 - 1))  # 1


def test_doorzetten_chip_is_asdict_able():
    """The view is a tree of plain dataclasses (serialisable via dataclasses.asdict)."""
    view = _doorzetten_analyzer().groepsindeling_view()
    dumped = dataclasses.asdict(view)
    assert isinstance(dumped, dict)
    assert dumped["group_order"] == ["A", "B"]


# ---------------------------------------------------------------------------
# Scenario 2 — Herindelen: several jaarlagen, occupancy 0
# ---------------------------------------------------------------------------


def _herindelen_analyzer() -> SolutionAnalyzer:
    """Two groups, movers spread over jaarlaag 6, 7 and the None cohort; no occupancy."""
    result = SolutionResult(
        assignment={
            "Finn": "A",
            "Gijs": "A",
            "Hugo": "A",
            "Iris": "B",
            "Jip": "B",
        },
        student_satisfaction={s: 1.0 for s in ["Finn", "Gijs", "Hugo", "Iris", "Jip"]},
        satisfied={},
        weighted_satisfied={},
        weights={},
        group_composition={
            "A": GroupComposition(
                boys_total=2,
                girls_total=1,
                per_year={
                    6: SexCounts(boys=1, girls=0),
                    7: SexCounts(boys=1, girls=0),
                    None: SexCounts(boys=0, girls=1),
                },
            ),
            "B": GroupComposition(
                boys_total=1,
                girls_total=1,
                per_year={
                    6: SexCounts(boys=0, girls=1),
                    7: SexCounts(boys=1, girls=0),
                },
            ),
        },
    )
    students_info = {
        "Finn": {"Stamgroep": "P", "Jongen/meisje": "Jongen", "Jaarlaag": 6},
        "Gijs": {"Stamgroep": "P", "Jongen/meisje": "Jongen", "Jaarlaag": 7},
        "Hugo": {"Stamgroep": "Q", "Jongen/meisje": "Meisje", "Jaarlaag": None},
        "Iris": {"Stamgroep": "Q", "Jongen/meisje": "Meisje", "Jaarlaag": 6},
        "Jip": {"Stamgroep": "R", "Jongen/meisje": "Jongen", "Jaarlaag": 7},
    }
    return SolutionAnalyzer(result, _preferences([]), _input_sheet({}), students_info)


def test_herindelen_multiple_year_sections():
    """Each card gets several YearSections; the section size counts its movers."""
    view = _herindelen_analyzer().groepsindeling_view()
    card_a = view.groups[0]
    # None cohort first (sort key), then jaarlaag 6, 7.
    assert [s.year for s in card_a.year_sections] == [None, 6, 7]
    assert [s.label for s in card_a.year_sections] == [
        "Jaarlaag",
        "Jaarlaag 6",
        "Jaarlaag 7",
    ]
    for card in view.groups:
        for section in card.year_sections:
            assert section.size == section.boys.new_count + section.girls.new_count
            assert section.boys.new_count == len(section.boys.students)


def test_herindelen_balance_rows_total_then_year_rows():
    """balance_rows: Totaal, then None-cohort ('Jaarlaag') directly after, then 6, 7."""
    view = _herindelen_analyzer().groepsindeling_view()
    assert [(r.label, r.is_total) for r in view.balance_rows] == [
        ("Totaal", True),
        ("Jaarlaag", False),
        ("Jaarlaag 6", False),
        ("Jaarlaag 7", False),
    ]
    none_row = view.balance_rows[1]
    # Only group A has a None-cohort mover (Hugo, a girl); B is missing it -> (0,0,0).
    assert none_row.per_group == {"A": (1, 0, 1), "B": (0, 0, 0)}
    assert none_row.size_diff == 1 - 0
    assert none_row.sex_imbalance == max(abs(0 - 1), abs(0 - 0))


def test_herindelen_min_satisfaction_missing_key_is_none():
    """A student_info without MinimaleTevredenheid yields min_satisfaction None."""
    view = _herindelen_analyzer().groepsindeling_view()
    chips = {
        c.full_name: c
        for card in view.groups
        for section in card.year_sections
        for col in (section.boys, section.girls)
        for c in col.students
    }
    assert all(chip.min_satisfaction is None for chip in chips.values())
    # And a student with no preferences still gets satisfaction None.
    assert chips["Finn"].satisfaction is None


def test_view_type_is_groepsindeling_view():
    """The builder returns the top-level dataclass."""
    assert isinstance(_herindelen_analyzer().groepsindeling_view(), GroepsindelingView)
