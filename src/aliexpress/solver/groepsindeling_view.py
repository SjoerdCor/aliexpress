"""Structured, Flask-free view-model for the Groepsindeling result page, and its builder.

:class:`GroepsindelingView` is a plain tree of frozen dataclasses (``dataclasses.asdict``-able),
built by :func:`build` from a solved :class:`~aliexpress.solver.results.SolutionResult` in display
space (names as entered). The web layer serialises it to JSON and a Jinja macro renders it; keeping
it free of Flask and of pandas objects keeps that seam clean and lets an intermediate solution reuse
the same builder later.

The dataclasses and the code that fills them live together here on purpose;
:class:`~aliexpress.solver.solutions.SolutionAnalyzer` exposes it through a thin
``groepsindeling_view`` method that delegates to :func:`build`.

All user-visible strings on these objects (jaarlaag labels, sex) are Dutch; the field names are
English, matching the rest of the solver package.
"""

import math
from dataclasses import dataclass

import pandas as pd

from ..data import preferences_data
from .results import SolutionResult


def shift_year(year: int | None, offset: int) -> int | None:
    """Shift a jaarlaag number by ``offset``, the Nieuwe-jaarlaag display shift.

    ``offset`` is 0 for distribution modes without an Overgang (the stored jaarlaag is
    already the one to display) and 1 for forward modes (Doorzetten/Overgang), where
    students move up one jaarlaag and the result should show that new jaarlaag, even
    though the stored data still reflects the current one. ``None`` is passed through
    unchanged: the None cohort only occurs in the bare-Excel/CLI path, which has no
    Overgang and therefore always uses offset 0.
    """
    return None if year is None else year + offset


def year_label(year: int | None, offset: int = 0) -> str:
    """Row label for a jaarlaag cohort, shifted by ``offset``.

    Bare "Jaarlaag" for the None cohort; "Jaarlaag N" otherwise, where N is ``year``
    shifted by ``offset`` (see :func:`shift_year`).
    """
    shifted = shift_year(year, offset)
    return "Jaarlaag" if shifted is None else f"Jaarlaag {shifted}"


def year_sort_key(year: int | None) -> tuple:
    """Sort key placing the None cohort right after "Totaal", then years numerically."""
    return (0,) if year is None else (1, year)


@dataclass(frozen=True)
class Preference:
    """One tevredenheid preference of a student and whether it was honoured.

    ``kind`` is ``"graag_met"`` (positive weight) or ``"liever_niet_met"`` (a folded
    negative-weight preference). ``fulfilled`` True means the preference was honoured: the
    target sits together for "graag_met", or was kept apart for "liever_niet_met". ``target``
    is the display label — a classmate's short unique name, or a group name when
    ``target_is_group`` (then the template phrases it as "Graag/Liever niet *in*").
    ``weight`` is the preference's magnitude (``abs`` of the signed weight): the template maps
    it to the same colour/glyph the preferences form uses, and lists the heaviest first.
    """

    kind: str
    target: str
    fulfilled: bool
    target_is_group: bool
    weight: float


# A flat DTO: every field is a distinct datum the chip/popover renders, and the view stays
# dataclasses.asdict-able. Splitting it would only add indirection.
@dataclass(frozen=True)
class StudentChip:  # pylint: disable=too-many-instance-attributes
    """A single student chip: short label plus everything the popover shows.

    ``chip_name`` is the short unique display name (web path) or the full name (Excel/CLI
    fallback); ``full_name`` is always the name as entered. ``satisfaction`` is clamped to
    [-1.0, 1.0] or ``None`` when the student has no preferences at all. ``min_satisfaction``
    is ``None`` / ``"partial"`` / ``"full"`` (the Extra-zekerheid badge level).
    """

    chip_name: str
    full_name: str
    origin_abbrev: str
    origin_full: str
    year_group: int | None
    satisfaction: float | None
    preferences: list[Preference]
    not_in: list[str]
    min_satisfaction: str | None


@dataclass(frozen=True)
class SexColumn:
    """One sex column within a jaarlaag section: the newly assigned movers of this sex.

    ``new_count`` counts them (``== len(students)``); the sitting occupancy is not shown per
    section — it only feeds the card's totals (``GroupCard.boys_total``/``girls_total``) and
    the Klassenoverzicht Totaal row.
    """

    sex: str
    new_count: int
    students: list[StudentChip]


@dataclass(frozen=True)
class YearSection:
    """One jaarlaag section of a group card: a Jongens and a Meisjes column.

    ``size`` counts movers only (both sexes' ``new_count``), excluding sitting occupancy.
    ``year`` is the jaarlaag number or ``None`` (Excel input without a jaarlaag); ``label``
    is the Dutch row label ("Jaarlaag" / "Jaarlaag N").
    """

    year: int | None
    label: str
    size: int
    boys: SexColumn
    girls: SexColumn


@dataclass(frozen=True)
class GroupCard:
    """One target group's card: totals (incl. occupancy) and its jaarlaag sections."""

    name: str
    total: int
    boys_total: int
    girls_total: int
    year_sections: list[YearSection]


@dataclass(frozen=True)
class BalanceRow:
    """One row of the Klassenoverzicht: per-group ``(count, boys, girls)`` plus balance.

    The Totaal row (``is_total``) includes sitting occupancy; jaarlaag rows count movers
    only. ``size_diff`` is the largest-minus-smallest group count; ``sex_imbalance`` is the
    largest ``|boys - girls|`` over the groups, both for this row.
    """

    label: str
    is_total: bool
    per_group: dict[str, tuple[int, int, int]]
    size_diff: int
    sex_imbalance: int


@dataclass(frozen=True)
class GroepsindelingView:
    """The full structured view of one solution: the cards plus the Klassenoverzicht."""

    group_order: list[str]
    groups: list[GroupCard]
    balance_rows: list[BalanceRow]


def build(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Four required display-space artefacts plus two optional display options
    # (unique_name, year_offset); they do not bundle into one object without a one-off
    # dataclass, so the wide-but-flat signature is the readable choice here.
    result: SolutionResult,
    students_info: dict,
    preferences: pd.DataFrame,
    input_sheet: pd.DataFrame,
    unique_name: dict[str, str] | None = None,
    year_offset: int = 0,
) -> GroepsindelingView:
    """Build the structured Groepsindeling view-model from a display-keyed solution.

    A Flask-free tree of plain dataclasses: one card per target group with jaarlaag sections
    and student chips, plus the Klassenoverzicht balance rows. ``unique_name`` maps a full
    name to a short chip label (web path); an absent entry falls back to the full name
    (Excel/CLI). Everything is read from the already display-keyed ``result`` /
    ``students_info`` / ``preferences`` / ``input_sheet``. ``year_offset`` shifts only the
    *displayed* jaarlaag (section/label/chip year, see :func:`shift_year`) for forward modes
    (Doorzetten/Overgang); grouping and sorting stay keyed on the raw, current jaarlaag.
    """
    return _ViewBuilder(
        result, students_info, preferences, input_sheet, year_offset
    ).build(unique_name)


@dataclass(frozen=True)
class _ViewBuilder:
    """Turns one display-keyed solution into a :class:`GroepsindelingView`.

    Groups the derivation into focused helpers; it holds the four display-space artefacts a
    :class:`~aliexpress.solver.solutions.SolutionAnalyzer` already owns, plus the display-only
    ``year_offset`` (0 unless a forward mode is shifting the shown jaarlaag).
    """

    result: SolutionResult
    students_info: dict
    preferences: pd.DataFrame
    input_sheet: pd.DataFrame
    year_offset: int = 0

    def build(self, unique_name: dict[str, str] | None = None) -> GroepsindelingView:
        """Assemble the cards and the Klassenoverzicht rows."""
        unique_name = unique_name or {}
        chips = self._build_chips(unique_name)
        group_order = sorted(self.result.group_composition)
        groups = [self._build_group_card(group, chips) for group in group_order]
        balance_rows = self._build_balance_rows(group_order)
        return GroepsindelingView(
            group_order=group_order, groups=groups, balance_rows=balance_rows
        )

    def _build_chips(
        self, unique_name: dict[str, str]
    ) -> dict[tuple[str, int | None, str], list[StudentChip]]:
        """All student chips, grouped and sorted by ``(group, jaarlaag, sex)``.

        Chips within a cell are ordered by Stamgroep, then by chip label (as in
        ``SolutionAnalyzer.display_groepsindeling``).
        """
        graag_met = preferences_data.get_graag_met(self.preferences)
        students_with_prefs = (
            set(graag_met.index.get_level_values("Leerling"))
            if len(graag_met)
            else set()
        )
        preferences_by_student = self._preferences_by_student(graag_met, unique_name)

        grouped: dict[tuple[str, int | None, str], list[StudentChip]] = {}
        for student, group in self.result.assignment.items():
            info = self.students_info[student]
            chip = self._build_chip(
                student, unique_name, students_with_prefs, preferences_by_student
            )
            key = (group, info.get("Jaarlaag"), info["Jongen/meisje"])
            grouped.setdefault(key, []).append(chip)
        for cell in grouped.values():
            cell.sort(key=lambda chip: (chip.origin_full, chip.chip_name))
        return grouped

    def _preferences_by_student(
        self, graag_met: pd.DataFrame, unique_name: dict[str, str]
    ) -> dict[str, list[Preference]]:
        """Per student, the folded "Graag met" preferences with their fulfilled flag.

        A positive weight is a "graag_met" preference, a negative one a folded
        "liever_niet_met"; ``satisfied`` True means honoured in both cases (sat together /
        kept apart). The target is a group when it names a destination group; otherwise it is
        a classmate, shown by their short unique name (``unique_name``, else the full name).
        Each student's list is ordered graag-met before liever-niet-met, heaviest first.
        """
        groups = set(self.result.group_composition)
        by_student: dict[str, list[Preference]] = {}
        for (student, nr), row in graag_met.iterrows():
            kind = "graag_met" if row["Gewicht"] > 0 else "liever_niet_met"
            fulfilled = bool(self.result.satisfied[(student, nr)])
            target = row["Waarde"]
            is_group = target in groups
            label = target if is_group else unique_name.get(target, target)
            by_student.setdefault(student, []).append(
                Preference(
                    kind=kind,
                    target=label,
                    fulfilled=fulfilled,
                    target_is_group=is_group,
                    weight=abs(row["Gewicht"]),
                )
            )
        for prefs in by_student.values():
            prefs.sort(key=lambda p: (p.kind != "graag_met", -p.weight))
        return by_student

    def _build_chip(
        self,
        full_name: str,
        unique_name: dict[str, str],
        students_with_prefs: set,
        preferences_by_student: dict[str, list[Preference]],
    ) -> StudentChip:
        """One student's chip in display space (full name is the student key)."""
        info = self.students_info[full_name]
        origin_full = info["Stamgroep"]
        satisfaction = None
        if full_name in students_with_prefs:
            satisfaction = max(
                -1.0, min(1.0, self.result.student_satisfaction[full_name])
            )
        return StudentChip(
            chip_name=unique_name.get(full_name, full_name),
            full_name=full_name,
            origin_abbrev=origin_full[:3],
            origin_full=origin_full,
            year_group=shift_year(info.get("Jaarlaag"), self.year_offset),
            satisfaction=satisfaction,
            preferences=preferences_by_student.get(full_name, []),
            not_in=self._not_in_targets(full_name),
            min_satisfaction=self._min_satisfaction(info),
        )

    def _not_in_targets(self, full_name: str) -> list[str]:
        """The hard "Niet in" target groups of a student (always respected)."""
        if full_name not in self.input_sheet.index:
            return []
        targets = []
        for col in self.input_sheet.columns:
            is_niet_in = (
                isinstance(col, tuple)
                and len(col) > 2
                and col[0] == "Niet in"
                and col[2] == "Waarde"
            )
            if not is_niet_in:
                continue
            value = self.input_sheet.loc[full_name, col]
            if pd.notna(value) and value != "":
                targets.append(value)
        return targets

    @staticmethod
    def _min_satisfaction(info: dict) -> str | None:
        """Extra-zekerheid badge level from ``MinimaleTevredenheid``: None/partial/full."""
        value = info.get("MinimaleTevredenheid", float("nan"))
        if math.isnan(value):
            return None
        if value >= 1.0:
            return "full"
        if value > 0:
            return "partial"
        return None

    def _build_group_card(
        self, group: str, chips: dict[tuple[str, int | None, str], list[StudentChip]]
    ) -> GroupCard:
        """One target group's card: totals plus a section per jaarlaag cohort."""
        comp = self.result.group_composition[group]
        years = sorted(comp.per_year, key=year_sort_key)

        sections = []
        for year in years:
            counts = comp.per_year[year]
            boys = self._sex_column(
                chips.get((group, year, "Jongen"), []), "Jongen", counts.boys
            )
            girls = self._sex_column(
                chips.get((group, year, "Meisje"), []), "Meisje", counts.girls
            )
            sections.append(
                YearSection(
                    year=shift_year(year, self.year_offset),
                    label=year_label(year, self.year_offset),
                    size=boys.new_count + girls.new_count,
                    boys=boys,
                    girls=girls,
                )
            )
        return GroupCard(
            name=group,
            total=comp.boys_total + comp.girls_total,
            boys_total=comp.boys_total,
            girls_total=comp.girls_total,
            year_sections=sections,
        )

    @staticmethod
    def _sex_column(students: list[StudentChip], sex: str, new_count: int) -> SexColumn:
        """One sex column of a section: the newly assigned movers of this sex."""
        return SexColumn(sex=sex, new_count=new_count, students=students)

    def _build_balance_rows(self, group_order: list[str]) -> list[BalanceRow]:
        """The Klassenoverzicht rows: Totaal first, then one row per jaarlaag cohort."""
        comps = self.result.group_composition
        rows = [self._total_balance_row(group_order, comps)]
        years: set[int | None] = set()
        for comp in comps.values():
            years.update(comp.per_year)
        for year in sorted(years, key=year_sort_key):
            rows.append(self._year_balance_row(group_order, comps, year))
        return rows

    def _total_balance_row(self, group_order: list[str], comps: dict) -> BalanceRow:
        """Totaal row: full group sizes including the sitting occupancy."""
        per_group = {
            group: (
                comps[group].boys_total + comps[group].girls_total,
                comps[group].boys_total,
                comps[group].girls_total,
            )
            for group in group_order
        }
        return self._balance_row("Totaal", True, per_group)

    def _year_balance_row(
        self, group_order: list[str], comps: dict, year: int | None
    ) -> BalanceRow:
        """One jaarlaag row: movers only; a group missing this cohort contributes zeroes."""
        per_group = {}
        for group in group_order:
            counts = comps[group].per_year.get(year)
            if counts is None:
                per_group[group] = (0, 0, 0)
            else:
                per_group[group] = (
                    counts.boys + counts.girls,
                    counts.boys,
                    counts.girls,
                )
        return self._balance_row(year_label(year, self.year_offset), False, per_group)

    @staticmethod
    def _balance_row(
        label: str, is_total: bool, per_group: dict[str, tuple[int, int, int]]
    ) -> BalanceRow:
        """Assemble a balance row and derive its size difference and sex imbalance."""
        sizes = [count for count, _, _ in per_group.values()]
        size_diff = max(sizes) - min(sizes) if sizes else 0
        sex_imbalance = max(
            (abs(boys - girls) for _, boys, girls in per_group.values()), default=0
        )
        return BalanceRow(
            label=label,
            is_total=is_total,
            per_group=per_group,
            size_diff=size_diff,
            sex_imbalance=sex_imbalance,
        )
