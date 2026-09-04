"""Parsing helpers for the wizard's HTML forms.

These functions are pure form-to-dataclass conversions with no Flask route logic,
extracted here so route modules stay thin and the parsing can be unit-tested
without a running Flask app.
"""

import dataclasses
from dataclasses import dataclass
from itertools import zip_longest

from ..errors import ValidationError
from ..solver._balance import BalanceMaxima
from . import datareader
from .preferences_form import Preference, PreferenceKind, StudentEntry


@dataclass
class GroupsToSubmission:
    """Parsed groups-to form.

    ``distribution`` holds the retained boy/girl counts per group (written to
    ``groups.xlsx`` for the solver); ``state`` captures exactly what the teacher did so
    the page can be restored on return (written to ``groups_to_state.json``).
    """

    distribution: dict[str, dict[str, int]]
    state: dict


def _checked_indices(form, groupname: str, n_students: int) -> list[int]:
    """Return the submitted student indices for a group, bounded to the known students.

    Checkbox values are the student's position in the groups-to list (not the gender),
    so the server can both count genders and remember exactly who was ticked.
    """
    indices = []
    for raw in form.getlist(f"group_students[{groupname}]"):
        try:
            index = int(raw)
        except ValueError:
            continue
        if 0 <= index < n_students:
            indices.append(index)
    return indices


def parse_groups_to_form(form, groups_to: dict) -> GroupsToSubmission:
    """Turn the submitted groups-to form into retained counts and a restore state.

    Active groups come from the ``group`` fields. An original group missing from that
    list was switched off; a submitted name that is not an original group is a
    teacher-added empty group. Genders are looked up from ``groups_to`` by the submitted
    student indices, so the counts cannot drift from the source data.

    Every original group's ticks are remembered (their checkboxes submit even when the
    group is switched off), so switching a group back on restores exactly who was ticked.
    Only active groups contribute to ``distribution`` (and thus to ``groups.xlsx``).
    """
    submitted = form.getlist("group")
    # Remember the ticks of every original group, including switched-off ones.
    original_state = {
        name: {"checked_indices": _checked_indices(form, name, len(students))}
        for name, students in groups_to.items()
    }
    distribution: dict[str, dict[str, int]] = {}
    new_groups: list[str] = []

    for name in submitted:
        if name in groups_to:
            students = groups_to[name]
            indices = original_state[name]["checked_indices"]
            distribution[name] = {
                "Jongens": sum(students[i]["geslacht"] == "Jongen" for i in indices),
                "Meisjes": sum(students[i]["geslacht"] == "Meisje" for i in indices),
            }
        else:
            distribution[name] = {"Jongens": 0, "Meisjes": 0}
            new_groups.append(name)

    state = {
        "original_groups": original_state,
        "disabled_groups": [name for name in groups_to if name not in submitted],
        "new_groups": new_groups,
    }
    return GroupsToSubmission(distribution=distribution, state=state)


def parse_preference_list(form, key, soort_field_value) -> list[Preference]:
    """Parse all preferences of one kind for a student from the submitted form."""
    kind = (
        PreferenceKind.APART
        if soort_field_value == "liever_niet_met"
        else PreferenceKind.TOGETHER
    )
    prefix = f"preference_{key}_{soort_field_value}"
    targets = form.getlist(f"{prefix}_target")
    weights = form.getlist(f"{prefix}_gewicht")
    result = []
    for target, weight_raw in zip(targets, weights):
        target = target.strip()
        if not target:
            continue
        try:
            weight = float(weight_raw)
        except ValueError:
            weight = 1.0
        if weight <= 0:
            weight = 1.0
        result.append(Preference(target=target, weight=weight, kind=kind))
    return result


def parse_student_entry(candidate: dict, form) -> StudentEntry:
    """Build a StudentEntry from one candidate dict and the submitted form data.

    Graag-met preferences use ``preference_{key}_graag_met_target[]`` / ``_gewicht[]``.
    Liever-niet-met use ``preference_{key}_liever_niet_met_target[]`` / ``_gewicht[]``.
    Group exclusions use ``nieting_{key}[]``.
    Min. satisfaction uses ``min_sat_{key}``.
    """
    key = candidate["key"]
    name = f"{candidate['roepnaam']} {candidate['achternaam']}"

    preferences = parse_preference_list(form, key, "graag_met") + parse_preference_list(
        form, key, "liever_niet_met"
    )

    excluded = [g.strip() for g in form.getlist(f"nieting_{key}") if g.strip()]

    raw_min_sat = form.get(f"min_sat_{key}", "").strip()
    try:
        min_satisfaction = float(raw_min_sat) / 100.0 if raw_min_sat else None
    except ValueError:
        min_satisfaction = None

    return StudentEntry(
        student=name,
        sex=candidate["geslacht"],
        origin_group=candidate["groepsnaam"],
        min_satisfaction=min_satisfaction,
        year_group=candidate.get("jaargroep"),
        preferences=preferences,
        excluded_groups=excluded,
    )


def build_form_state(entries: list[StudentEntry], participants: list[dict]) -> dict:
    """Serialize submitted preferences to a dict for prefill on next GET.

    The population is already fixed by the roster step (every participant takes part), so
    this only carries each participant's preferences, keyed so the page can restore them.
    """
    entry_by_name = {e.student: e for e in entries}
    state_students = []
    for c in participants:
        name = f"{c['roepnaam']} {c['achternaam']}"
        entry = entry_by_name.get(name)
        state_students.append(
            {
                "key": c["key"],
                "roepnaam": c["roepnaam"],
                "achternaam": c["achternaam"],
                "groepsnaam": c.get("groepsnaam", ""),
                "geslacht": c.get("geslacht", ""),
                "min_satisfaction": entry.min_satisfaction if entry else None,
                "graag_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in (entry.preferences if entry else [])
                    if p.kind == PreferenceKind.TOGETHER
                ],
                "liever_niet_met": [
                    {"target": p.target, "weight": p.weight}
                    for p in (entry.preferences if entry else [])
                    if p.kind == PreferenceKind.APART
                ],
                "niet_in": entry.excluded_groups if entry else [],
            }
        )
    return {"students": state_students}


def reconcile_dangling(
    draft_state, participants, group_labels
) -> list[tuple[str, str]]:
    """Drop draft preferences whose target no longer takes part; return the removed pairs.

    A classmate target is valid only when that leerling is still a participant (the teacher
    may have removed them on the roster step). Group targets stay valid. Mutates
    ``draft_state`` in place and returns one ``(owner, target)`` pair per removed
    preference — the caller formats the Dutch notice.
    """
    valid_keys = {
        datareader.matching_key(f"{p['roepnaam']} {p['achternaam']}")
        for p in participants
    } | {datareader.matching_key(g) for g in group_labels}
    group_keys = {datareader.matching_key(g) for g in group_labels}
    removed = []
    for student in draft_state["students"]:
        owner = f"{student['roepnaam']} {student['achternaam']}".strip()
        owner_key = datareader.matching_key(owner)
        for kind in ("graag_met", "liever_niet_met"):
            kept = []
            for preference in student.get(kind, []):
                target_key = datareader.matching_key(preference["target"])
                if target_key == owner_key and target_key not in group_keys:
                    # Self-targets are invalid input, not a dangling classmate. Drop them
                    # from a draft without claiming that the target left the roster.
                    continue
                if target_key in valid_keys:
                    kept.append(preference)
                else:
                    removed.append((owner, preference["target"]))
            student[kind] = kept
    return removed


def parse_not_together_form(form, n_rules):
    """Parse not-together form fields into rule dicts.

    Raises ``ValidationError`` on malformed input (duplicate student in a rule, a missing
    maximum, or a non-integer maximum); the caller translates it via
    ``to_validation_message`` and flashes it.
    """
    rules = []
    for i in range(n_rules):
        names_raw = form.getlist(f"rule_students[{i}]")
        # Keep the names as entered for display; dedupe on the matching key so the same
        # student picked twice (in any spelling) is caught.
        cleaned = [datareader.display_name(n) for n in names_raw if n.strip()]
        if len({datareader.matching_key(n) for n in cleaned}) != len(cleaned):
            raise ValidationError(
                "duplicate_student_not_together", {"rule_index": i + 1}
            )
        max_together_raw = form.get(f"rule_max[{i}]", "").strip()
        if not max_together_raw:
            raise ValidationError(
                "missing_max_samen_not_together", {"rule_index": i + 1}
            )
        try:
            max_samen = int(max_together_raw)
        except ValueError as exc:
            raise ValidationError(
                "invalid_max_samen_type_not_together", {"rule_index": i + 1}
            ) from exc
        if cleaned:
            rules.append({"group": set(cleaned), "Max_aantal_samen": max_samen})
    return rules


def parse_balance_maxima_form(form) -> BalanceMaxima:
    """Parse balance-maxima form fields into a BalanceMaxima.

    Each of the six families has a checkbox ``maxima_{field}_unlimited`` and a
    number field ``maxima_{field}``. A checked checkbox makes that family
    ``None`` (Onbeperkt), ignoring the number field. Otherwise the number field
    must hold an integer of at least 1.

    Raises ``ValidationError`` on malformed input (missing or non-integer
    number field); the caller translates it via ``to_validation_message`` and
    flashes it.
    """
    values = {}
    for field in dataclasses.fields(BalanceMaxima):
        if form.get(f"maxima_{field.name}_unlimited"):
            values[field.name] = None
            continue
        raw = form.get(f"maxima_{field.name}", "").strip()
        if not raw:
            raise ValidationError("missing_balance_maximum")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValidationError("invalid_balance_maximum") from exc
        if value < 1:
            raise ValidationError("invalid_balance_maximum")
        values[field.name] = value
    return BalanceMaxima(**values)


def build_new_candidates(
    form, groups_from: list, default_jaargroep: int | None = None
) -> list[dict]:
    """Build candidate dicts for incoming students added via the form.

    Expects parallel lists ``new_key[]``, ``new_voornaam[]``, ``new_achternaam[]``,
    ``new_geslacht[]`` and optionally ``new_groep[]``, ``new_jaargroep[]``. Incomplete rows
    are skipped. A row's own ``new_jaargroep[]`` wins; otherwise ``default_jaargroep`` is
    used (the process's single shared jaargroep in doorzetten mode). No ``"jaargroep"`` key
    is added when neither is available, matching the Excel input path's None-cohort.
    """
    fallback = groups_from[0] if groups_from else ""
    candidates = []
    for key, vn, an, geslacht, groep, jaargroep in zip_longest(
        form.getlist("new_key[]"),
        form.getlist("new_voornaam[]"),
        form.getlist("new_achternaam[]"),
        form.getlist("new_geslacht[]"),
        form.getlist("new_groep[]"),
        form.getlist("new_jaargroep[]"),
        fillvalue="",
    ):
        vn, an = vn.strip(), an.strip()
        if vn and an and geslacht and key:
            candidate = {
                "key": key,
                "roepnaam": vn,
                "achternaam": an,
                "geslacht": geslacht,
                "groepsnaam": groep or fallback,
            }
            resolved_jaargroep = (
                int(jaargroep) if jaargroep.strip() else default_jaargroep
            )
            if resolved_jaargroep is not None:
                candidate["jaargroep"] = resolved_jaargroep
            candidates.append(candidate)
    return candidates


def validate_new_students(form, orig_candidates, mode: str) -> None:
    """Validate hand-added new students; raise ValidationError on the first problem.

    The form is best-effort client-side, so the server is the safety net: a row that was
    started but left incomplete, or whose name clashes (compared on matching keys, so
    spelling/case differences still collide) with an existing leerling or another new
    student, is rejected. Entirely empty rows are ignored. In herindelen mode (``mode ==
    "redistribute"``), candidates span several jaargroepen, so a new student must say which
    one explicitly; in doorzetten mode they all share one, so it can be assumed instead
    (see ``build_new_candidates``'s ``default_jaargroep``).
    """
    existing = {
        datareader.matching_key(f"{c['roepnaam']} {c['achternaam']}")
        for c in orig_candidates
    }
    seen = set()
    for vn, an, geslacht, jaargroep in zip_longest(
        form.getlist("new_voornaam[]"),
        form.getlist("new_achternaam[]"),
        form.getlist("new_geslacht[]"),
        form.getlist("new_jaargroep[]"),
        fillvalue="",
    ):
        vn, an = vn.strip(), an.strip()
        if not (vn or an or geslacht):
            continue  # untouched row
        if not (vn and an and geslacht):
            raise ValidationError(code="incomplete_new_student")
        if mode == "redistribute" and not jaargroep.strip():
            raise ValidationError(code="missing_jaargroep_new_student")
        key = datareader.matching_key(f"{vn} {an}")
        if key in existing or key in seen:
            raise ValidationError(
                code="duplicate_new_student", context={"name": f"{vn} {an}"}
            )
        seen.add(key)


def build_participants(form, orig_candidates, groups_from, mode: str) -> list[dict]:
    """Resolve the population: ticked existing candidates plus hand-added new students."""
    checked_keys = set(form.getlist("gaat_over"))
    participants = [c for c in orig_candidates if c["key"] in checked_keys]
    default_jaargroep = (
        orig_candidates[0].get("jaargroep")
        if orig_candidates and mode != "redistribute"
        else None
    )
    participants.extend(build_new_candidates(form, groups_from, default_jaargroep))
    return participants
