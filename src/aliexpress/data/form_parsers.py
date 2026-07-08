"""Parsing helpers for the wizard's HTML forms.

These functions are pure form-to-dataclass conversions with no Flask route logic,
extracted here so route modules stay thin and the parsing can be unit-tested
without a running Flask app.
"""

from dataclasses import dataclass

from ..errors import ValidationError
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
    removed = []
    for student in draft_state["students"]:
        owner = f"{student['roepnaam']} {student['achternaam']}".strip()
        for kind in ("graag_met", "liever_niet_met"):
            kept = []
            for preference in student.get(kind, []):
                if datareader.matching_key(preference["target"]) in valid_keys:
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
