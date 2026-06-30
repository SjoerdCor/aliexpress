"""Parsing helpers for the groups-to HTML form.

These functions are pure form-to-dataclass conversions with no Flask route logic,
extracted here so ``app.py`` stays under the line-count limit and they can be
unit-tested without a running Flask app.
"""

from dataclasses import dataclass


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
