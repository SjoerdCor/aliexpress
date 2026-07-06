"""Generate test data for the app in the native process-directory format.

Produces the files a process needs to run the solver and navigate the wizard UI:
  relevant_students_and_groups.json — candidates + achterblijvers per doelgroep
  voorkeuren.json     — preferences (StudentEntry path, source="testdata")
  groups.xlsx         — target groups with current boy/girl counts (for the solver)
  not_together.json   — separation rules
  roster.json         — participant list (for the "Wie gaat mee" step)
  input_method.json   — records that the form path was used

Call ``main(n_groups, n_students, n_rules)`` to write a doorzetten scenario to
``testdata/``, or ``main_herindelen(...)`` for a herindelen (redistribute) scenario
(adds ``mode.json``; groups start empty and double as origin groups).

CLI: ``uv run python -m tests.testdata SCHOOL PROCES [herindelen]`` creates a ready-made
process in the running app instance for manual browser testing.
"""

import json
import math
import os
import random
import string
from dataclasses import dataclass

import pandas as pd

from aliexpress import create_app
from aliexpress.data.datareader import matching_key
from aliexpress.data.preferences_form import (
    Preference,
    PreferenceKind,
    StudentEntry,
    build_preference_data,
)
from aliexpress.web.extensions import db
from aliexpress.web.models import Process, School
from aliexpress.web.storage import get_process_path
from tests.testedexmlgeneration import SAMPLE_GROUP_NAMES

random.seed(42)

FOLDER = "testdata"

# Names for the "achterblijvers" (staying students) shown on the Groepen page.
# Kept separate from NativePreferenceGenerator.possible_students to avoid name collisions.
_STAYING_BOYS = [
    "Arjan",
    "Ben",
    "Cas",
    "Dirk",
    "Erik",
    "Frans",
    "Gerben",
    "Henk",
    "Ivo",
    "Jan",
    "Kees",
    "Leo",
    "Mart",
    "Niels",
    "Otto",
    "Pieter",
    "Rick",
    "Stef",
    "Thijs",
    "Ulrich",
    "Victor",
    "Wim",
    "Xander",
    "Yorick",
]
_STAYING_GIRLS = [
    "Alie",
    "Bea",
    "Cora",
    "Demi",
    "Els",
    "Fleur",
    "Griet",
    "Henny",
    "Ineke",
    "Joke",
    "Karen",
    "Lena",
    "Mies",
    "Nel",
    "Olga",
    "Petra",
    "Rianne",
    "Sandra",
    "Truus",
    "Ursula",
    "Vera",
    "Wendy",
    "Xandra",
    "Yvonne",
]


def generate_groups(n_groups=4, sample_group_names=None) -> pd.DataFrame:
    """Create the groups-DataFrame with random boy/girl counts."""
    assert 1 < n_groups <= 10
    if sample_group_names is None:
        sample_group_names = SAMPLE_GROUP_NAMES
    selected_names = sample_group_names[:n_groups]

    rows = {}
    for gr in selected_names:
        n_leerlingen = random.randint(12, 18)
        pct_jongens = 0.3 + 0.4 * random.random()
        n_jongens = int(pct_jongens * n_leerlingen)
        rows[gr] = {"Jongens": n_jongens, "Meisjes": n_leerlingen - n_jongens}

    return pd.DataFrame.from_dict(rows, orient="index").reset_index(names="Groepen")


@dataclass(frozen=True)
class GeneratorScenario:
    """Non-default choices for one ``NativePreferenceGenerator`` scenario."""

    n_groups_from: int = 4
    """Number of origin groups (single letters A, B, …). Ignored when ``groups_from`` is given."""

    groups_from: list[str] | None = None
    """Explicit origin group names. Used for herindelen, where students originate from the
    same groups they are redistributed over."""

    allow_min_satisfaction: bool = True
    """When False, never generate a hard ``MinimaleTevredenheid``. Herindelen redistributes
    every student at once with no fixed achterblijvers, so random per-student minima are much
    more likely to be jointly infeasible than in the doorzetten scenario; a generated test
    scenario must always be solvable."""

    jaargroepen: list[int] | None = None
    """When given, students are spread evenly over these jaargroepen (herindelen's multi-cohort
    scenario). ``None`` (the default) leaves every student without a jaargroep, matching the
    Excel input path's single None-cohort."""


class NativePreferenceGenerator:
    """Generate random StudentEntry preferences and write voorkeuren.json.

    Parallel to the old ``PreferenceExcelGenerator`` but produces the native
    app format (StudentEntry → build_preference_data → voorkeuren.json) instead
    of an Excel file. Same ``possible_students`` list; same ``groups_to`` / ``n_groups_from``
    constructor parameters.
    """

    possible_students = [
        ("Anna", "Meisje"),
        ("Bram", "Jongen"),
        ("Claire", "Meisje"),
        ("Daan", "Jongen"),
        ("Eva", "Meisje"),
        ("Finn", "Jongen"),
        ("Gina", "Meisje"),
        ("Hugo", "Jongen"),
        ("Iris", "Meisje"),
        ("Jesse", "Jongen"),
        ("Kiki", "Meisje"),
        ("Lars", "Jongen"),
        ("Mila", "Meisje"),
        ("Noah", "Jongen"),
        ("Olivia", "Meisje"),
        ("Pim", "Jongen"),
        ("Quinn", "Jongen"),
        ("Rosa", "Meisje"),
        ("Sam", "Jongen"),
        ("Tess", "Meisje"),
        ("Umut", "Jongen"),
        ("Vera", "Meisje"),
        ("Wout", "Jongen"),
        ("Xena", "Meisje"),
        ("Yara", "Meisje"),
        ("Zane", "Jongen"),
        ("Lieke", "Meisje"),
        ("Nina", "Meisje"),
        ("Oscar", "Jongen"),
        ("Paul", "Jongen"),
        ("Rik", "Jongen"),
        ("Sofie", "Meisje"),
        ("Tom", "Jongen"),
        ("Una", "Meisje"),
        ("Valerie", "Meisje"),
        ("Wes", "Jongen"),
        ("Xavi", "Jongen"),
        ("Yentl", "Meisje"),
        ("Zion", "Jongen"),
    ]

    def __init__(
        self, groups_to: list, scenario: GeneratorScenario = GeneratorScenario()
    ):
        """
        Parameters
        ----------
        groups_to : list[str]
            Display names of the destination groups (e.g. ["Blauw", "Geel"]).
        scenario : GeneratorScenario
            Non-default origin-group, minimum-satisfaction and jaargroep choices; see
            ``GeneratorScenario`` for each field.
        """
        self.groups_to = groups_to
        if scenario.groups_from is not None:
            self.groups_from = list(scenario.groups_from)
        else:
            self.groups_from = list(string.ascii_uppercase)[: scenario.n_groups_from]
        self.allow_min_satisfaction = scenario.allow_min_satisfaction
        self.jaargroepen = scenario.jaargroepen

    def generate_minimale_tevredenheid(self) -> float | None:
        """Return None (80 % of the time) or a minimal satisfaction in [0.2, 0.8].

        Always None when ``allow_min_satisfaction`` is False.
        """
        if not self.allow_min_satisfaction or random.random() >= 0.2:
            return None
        return round(random.uniform(0.2, 0.8), 1)

    def _generate_preferences(self, name: str, options: list[str]) -> list[Preference]:
        """Random TOGETHER preferences (0–5) and optionally one APART preference."""
        candidates = [o for o in options if o != name]
        n_together = random.randint(0, min(5, len(candidates)))
        together_targets = random.sample(candidates, k=n_together)
        prefs = [
            Preference(target, float(random.randint(1, 3)), PreferenceKind.TOGETHER)
            for target in together_targets
        ]
        if random.random() < 0.5:
            remaining = [o for o in candidates if o not in together_targets]
            if remaining:
                prefs.append(
                    Preference(
                        random.choice(remaining),
                        float(random.randint(1, 3)),
                        PreferenceKind.APART,
                    )
                )
        return prefs

    def _generate_excluded_groups(self, preferences: list[Preference]) -> list[str]:
        """0–2 groups the student may not be placed in (never all groups)."""
        already_named = {p.target for p in preferences if p.target in self.groups_to}
        possible = [g for g in self.groups_to if g not in already_named]
        max_excl = max(min(2, len(possible) - 1), 0)
        return random.sample(possible, random.randint(0, max_excl))

    def _build_entry(
        self, index: int, name: str, sex: str, options: list[str]
    ) -> StudentEntry:
        """Build one StudentEntry: origin group, preferences, exclusions and jaargroep."""
        prefs = self._generate_preferences(name, options)
        year_group = (
            self.jaargroepen[index % len(self.jaargroepen)]
            if self.jaargroepen
            else None
        )
        return StudentEntry(
            student=name,
            sex=sex,
            origin_group=random.choice(self.groups_from),
            min_satisfaction=self.generate_minimale_tevredenheid(),
            year_group=year_group,
            preferences=prefs,
            excluded_groups=self._generate_excluded_groups(prefs),
        )

    def generate(
        self,
        num_students: int = 35,
        fname: str | None = None,
    ) -> list[StudentEntry]:
        """Build ``num_students`` StudentEntry objects and optionally write voorkeuren.json.

        Parameters
        ----------
        num_students : int
            Number of students to generate (1 – len(possible_students)).
        fname : str or None
            Path for ``voorkeuren.json``; when None the file is not written.

        Returns
        -------
        list[StudentEntry]
            The generated entries (also used for roster.json generation).
        """
        assert 1 <= num_students <= len(self.possible_students)
        selected = self.possible_students[:num_students]
        all_names = [name for name, _ in selected]
        options = all_names + self.groups_to

        entries = [
            self._build_entry(i, name, sex, options)
            for i, (name, sex) in enumerate(selected)
        ]

        if fname is not None:
            all_to_groups = [matching_key(g) for g in self.groups_to]
            preference_data = build_preference_data(entries, all_to_groups)
            _write_voorkeuren_json(fname, preference_data)

        return entries


def generate_not_together(
    leerlingen: list[str], n_groups: int = 4, n_rules: int = 5
) -> list[dict]:
    """Generate random not-together rules as a JSON-serializable list.

    Returns
    -------
    list of {"group": [str, ...], "Max_aantal_samen": int}
        Ready to be written to ``not_together.json`` or passed to the solver.
    """
    rules = []
    for _ in range(n_rules):
        n_children = random.randint(2, min(12, len(leerlingen)))
        group = random.sample(leerlingen, k=n_children)
        rules.append(
            {"group": group, "Max_aantal_samen": math.ceil(n_children / n_groups)}
        )
    return rules


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_groups_to_students(groups_df: pd.DataFrame) -> dict:
    """Build the achterblijver lists per doelgroep from the groups DataFrame.

    Mirrors the format produced by ``candidatedetermination.get_groups_to()``.
    All generated students have ``blijft_in_groep=True`` so their checkboxes are
    pre-ticked on the Groepen page, matching a freshly-uploaded EDEXML.
    """
    boy_names = iter(_STAYING_BOYS * 5)
    girl_names = iter(_STAYING_GIRLS * 5)
    result = {}
    for _, row in groups_df.iterrows():
        students = []
        for _ in range(int(row["Jongens"])):
            students.append(
                {
                    "roepnaam": next(boy_names),
                    "achternaam": "",
                    "geslacht": "Jongen",
                    "jaargroep": 5,
                    "blijft_in_groep": True,
                }
            )
        for _ in range(int(row["Meisjes"])):
            students.append(
                {
                    "roepnaam": next(girl_names),
                    "achternaam": "",
                    "geslacht": "Meisje",
                    "jaargroep": 5,
                    "blijft_in_groep": True,
                }
            )
        result[row["Groepen"]] = students
    return result


def _generate_candidates(entries: list) -> list[dict]:
    """Convert StudentEntry objects to candidate dicts (matching ``get_candidates()`` format)."""
    return [
        {
            "key": matching_key(e.student),
            "roepnaam": e.student,
            "achternaam": "",
            "groepsnaam": e.origin_group,
            "geslacht": e.sex,
            "jaargroep": e.year_group,
        }
        for e in entries
    ]


def _write_voorkeuren_json(path: str, preference_data) -> None:
    """Persist a PreferenceData as voorkeuren.json with source tag "testdata"."""
    payload = json.loads(preference_data.to_json())
    payload["source"] = "testdata"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _build_roster_participants(entries: list[StudentEntry]) -> list[dict]:
    """Convert StudentEntry objects to the roster participant dicts the app expects."""
    return [
        {
            "key": matching_key(e.student),
            "roepnaam": e.student,
            "achternaam": "",
            "geslacht": e.sex,
            "groepsnaam": e.origin_group,
        }
        for e in entries
    ]


def _build_pref_form_state(
    entries: list[StudentEntry], participants: list[dict]
) -> dict:
    """Build preferences_form_state.json content from StudentEntry objects.

    Mirrors the structure that ``_build_form_state`` in wizard.py produces after a
    form POST, so the preferences form pre-fills correctly on the first GET.
    Entries and participants must be in the same order (both come from the same
    generated list).
    """
    state_students = []
    for c, e in zip(participants, entries):
        state_students.append(
            {
                "key": c["key"],
                "roepnaam": c["roepnaam"],
                "achternaam": c["achternaam"],
                "groepsnaam": c["groepsnaam"],
                "geslacht": c["geslacht"],
                "min_satisfaction": e.min_satisfaction,
                "graag_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in e.preferences
                    if p.kind == PreferenceKind.TOGETHER
                ],
                "liever_niet_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in e.preferences
                    if p.kind == PreferenceKind.APART
                ],
                "niet_in": e.excluded_groups,
            }
        )
    return {"students": state_students}


def _write_common_process_files(
    folder: str, entries: list[StudentEntry], n_groups: int, n_rules: int
) -> None:
    """Write the files shared by both modes: not_together, roster, form state, input method."""
    leerlingen = [e.student for e in entries]
    not_together = generate_not_together(leerlingen, n_groups, n_rules)
    with open(os.path.join(folder, "not_together.json"), "w", encoding="utf-8") as fh:
        json.dump(not_together, fh, ensure_ascii=False)

    # roster.json — needed by the preferences form step
    participants = _build_roster_participants(entries)
    with open(os.path.join(folder, "roster.json"), "w", encoding="utf-8") as fh:
        json.dump({"participants": participants}, fh, ensure_ascii=False)

    # preferences_form_state.json — pre-fills the form on first GET
    pref_state = _build_pref_form_state(entries, participants)
    with open(
        os.path.join(folder, "preferences_form_state.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(pref_state, fh, ensure_ascii=False)

    # input_method.json — marks this process as form-based
    with open(os.path.join(folder, "input_method.json"), "w", encoding="utf-8") as fh:
        json.dump({"method": "form"}, fh, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public: write a complete process directory
# ---------------------------------------------------------------------------


def main(n_groups: int = 4, n_students: int = 35, n_rules: int = 5, folder: str = None):
    """Generate a full set of native process files and write them to ``folder``.

    Parameters
    ----------
    n_groups : int
        Number of destination groups (2–10).
    n_students : int
        Number of students to distribute (1–39).
    n_rules : int
        Number of not-together rules to generate.
    folder : str or None
        Output directory; defaults to ``testdata/``.
    """
    if folder is None:
        folder = FOLDER
    os.makedirs(folder, exist_ok=True)

    groups_df = generate_groups(n_groups)
    group_names = groups_df["Groepen"].tolist()

    generator = NativePreferenceGenerator(groups_to=group_names)
    entries = generator.generate(
        num_students=n_students,
        fname=os.path.join(folder, "voorkeuren.json"),
    )

    # groups.xlsx — same format read_groups_excel() expects
    groups_df.to_excel(os.path.join(folder, "groups.xlsx"), index=False)

    # relevant_students_and_groups.json — mirrors what upload_edexml writes;
    # needed so the Groepen wizard step can load and display achterblijvers.
    relevant = {
        "candidates": _generate_candidates(entries),
        "groups_from": generator.groups_from + ["Anders"],
        "groups_to": _generate_groups_to_students(groups_df),
    }
    with open(
        os.path.join(folder, "relevant_students_and_groups.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(relevant, fh, ensure_ascii=False)

    _write_common_process_files(folder, entries, n_groups, n_rules)


def main_herindelen(
    n_groups: int = 3,
    n_students: int = 35,
    n_rules: int = 5,
    folder: str = None,
    jaargroepen: tuple[int, ...] = (6, 7, 8),
):
    """Generate a full set of process files for a herindelen (redistribute) process.

    Differences from ``main()`` (the doorzetten scenario), mirroring what the wizard
    writes for mode "redistribute":
      - ``mode.json`` marks the process as redistribute.
      - Students originate from the destination groups themselves (herkomst = bestemming).
      - Students are spread evenly over ``jaargroepen`` (herindelen's reason to exist:
        redistributing several year groups at once), giving the Klassenoverzicht its
        per-jaarlaag rows instead of one bare "Jaarlaag" row.
      - ``groups_to`` has no achterblijvers: every group starts empty.
      - ``groups.xlsx`` has zero occupancy per group, exactly like
        ``_groups_to_auto_redistribute`` in wizard.py writes it.

    Parameters
    ----------
    n_groups : int
        Number of groups to redistribute over (2–10).
    n_students : int
        Number of students to redistribute (1–39).
    n_rules : int
        Number of not-together rules to generate.
    folder : str or None
        Output directory; defaults to ``testdata/``.
    jaargroepen : tuple[int, ...]
        The jaargroepen students are spread over.
    """
    if folder is None:
        folder = FOLDER
    os.makedirs(folder, exist_ok=True)

    group_names = SAMPLE_GROUP_NAMES[:n_groups]
    generator = NativePreferenceGenerator(
        groups_to=group_names,
        scenario=GeneratorScenario(
            groups_from=group_names,
            allow_min_satisfaction=False,
            jaargroepen=list(jaargroepen),
        ),
    )
    entries = generator.generate(
        num_students=n_students,
        fname=os.path.join(folder, "voorkeuren.json"),
    )

    # mode.json — absent means "forward", so only redistribute processes write it
    with open(os.path.join(folder, "mode.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": "redistribute"}, fh)

    # groups.xlsx — zero occupancy, same format _groups_to_auto_redistribute produces
    distribution = {g: {"Jongens": 0, "Meisjes": 0} for g in group_names}
    pd.DataFrame(distribution).transpose().to_excel(
        os.path.join(folder, "groups.xlsx"), index_label="Groepen"
    )

    # relevant_students_and_groups.json — mirrors handle_edexml_upload_herindelen, plus
    # "jaargroepen" as _select_groups_post records it (settled by the selection, not
    # re-derived from candidates).
    relevant = {
        "candidates": _generate_candidates(entries),
        "groups_from": generator.groups_from + ["Anders"],
        "groups_to": {g: [] for g in group_names},
        "jaargroepen": sorted(jaargroepen),
    }
    with open(
        os.path.join(folder, "relevant_students_and_groups.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(relevant, fh, ensure_ascii=False)

    _write_common_process_files(folder, entries, n_groups, n_rules)


# ---------------------------------------------------------------------------
# Public: create a process in the running app instance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSize:
    """Size parameters shared by ``main()`` and ``main_herindelen()``."""

    n_groups: int = 4
    n_students: int = 35
    n_rules: int = 5


def setup_test_process(
    school_id: str,
    process_name: str,
    size: ScenarioSize = ScenarioSize(),
    mode: str = "forward",
) -> str:
    """Create a DB entry + full process directory for manual browser testing.

    Parameters
    ----------
    school_id : str
        The ``schoolcode`` of an existing school in the database.
    process_name : str
        Name of the process to create (inserted if it does not exist yet).
    size : ScenarioSize
        Passed to ``main()`` / ``main_herindelen()``; see their docstrings.
    mode : str
        "forward" for a doorzetten process, "redistribute" for herindelen.

    Returns
    -------
    str
        Absolute path to the process directory with all files written.

    Raises
    ------
    ValueError
        When ``school_id`` does not exist in the database.
    """
    app = create_app()
    with app.app_context():
        if db.session.get(School, school_id) is None:
            raise ValueError(
                f"School '{school_id}' bestaat niet. "
                "Maak het eerst aan met: uv run flask schools add"
            )
        if Process.by_name(school_id, process_name) is None:
            db.session.add(Process(school_id=school_id, name=process_name))
            db.session.commit()
        folder = get_process_path(school_id, process_name)
    os.makedirs(folder, exist_ok=True)
    generate = main_herindelen if mode == "redistribute" else main
    generate(
        n_groups=size.n_groups,
        n_students=size.n_students,
        n_rules=size.n_rules,
        folder=folder,
    )
    return folder


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def generate_dataframe_function(df, function_name="get_expected_dataframe"):
    """Generate a function that returns a DataFrame with the given data.

    Used to create expected DataFrames for tests.
    """
    data_dict = df.to_dict(orient="list")

    def format_value(v):
        if isinstance(v, float) and math.isnan(v):
            return "np.nan"
        return repr(v)

    formatted_data = "{\n"
    for col, values in data_dict.items():
        formatted_list = ", ".join(format_value(v) for v in values)
        formatted_data += f"    {repr(col)}: [{formatted_list}],\n"
    formatted_data += "}"

    use_index = not isinstance(df.index, pd.RangeIndex)
    if use_index:
        index_vals = [format_value(i) for i in df.index.tolist()]
        index_code = f"    df.index = pd.Index([{', '.join(index_vals)}])\n"
        if any(name is not None for name in df.index.names):
            index_code += f"    df.index.names = {repr(list(df.index.names))}\n"
    else:
        index_code = ""

    if any(name is not None for name in df.columns.names):
        column_code = f"    df.columns.names = {repr(list(df.columns.names))}\n"
    else:
        column_code = ""

    code = f"""\
def {function_name}():
    data = {formatted_data}
    df = pd.DataFrame(data)
{index_code}{column_code}    return df
"""
    return code


def _run_cli(argv: list[str]) -> None:
    """Handle ``python -m tests.testdata SCHOOL PROCES [herindelen]``."""
    if len(argv) >= 3:
        school, process = argv[1], argv[2]
        mode = "redistribute" if "herindelen" in argv[3:] else "forward"
        path = setup_test_process(school, process, mode=mode)
        print(f"Testproces aangemaakt ({mode}): {path}")
    else:
        main(3, 8, 1)


if __name__ == "__main__":
    import sys  # pylint: disable=import-outside-toplevel

    _run_cli(sys.argv)
