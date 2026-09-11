"""Flask-free presentation transforms for the completed result page."""


def _format_satisfaction(value) -> str | None:
    """Format a stored satisfaction fraction as an exact readable percentage."""
    if value is None:
        return None
    return f"{round(float(value) * 100)}%"


def _preference_result(preference: dict) -> dict:
    """Add display-only wording to one stored preference."""
    result = dict(preference)
    is_positive = result.get("kind") == "graag_met"
    result["wording"] = (
        f"{'Graag' if is_positive else 'Liever niet'} "
        f"{'in' if result.get('target_is_group') else 'bij'} {result.get('target', '')}"
    )
    return result


def _student_result_rows(view: dict) -> list[dict]:
    """Flatten the students in the group cards for the result analysis."""
    students = []
    for card in view.get("groups", []):
        for section in card.get("year_sections", []):
            for sex in ("boys", "girls"):
                for chip in section.get(sex, {}).get("students", []):
                    preferences = [
                        _preference_result(preference)
                        for preference in chip.get("preferences", [])
                    ]
                    # Python's sort is stable, so equal importance keeps input order.
                    preferences.sort(
                        key=lambda preference: (
                            preference.get("kind") != "graag_met",
                            -float(preference.get("weight", 0)),
                            not preference.get("fulfilled"),
                        )
                    )
                    satisfaction = chip.get("satisfaction")
                    students.append(
                        {
                            "name": chip.get("full_name", chip.get("chip_name", "")),
                            "satisfaction": satisfaction,
                            "satisfaction_percent": _format_satisfaction(satisfaction),
                            "has_preferences": bool(preferences),
                            "preferences": preferences,
                        }
                    )
    students.sort(
        key=lambda student: (
            not student["has_preferences"],
            student["satisfaction"] is None,
            -(student["satisfaction"] or 0),
            student["name"].casefold(),
        )
    )
    return students


def _count_tuple(row: dict, group: str) -> tuple[int, int, int]:
    """Read one stored balance cell as learner, boys and girls counts."""
    values = row.get("per_group", {}).get(group, (0, 0, 0))
    return tuple(int(values[index]) if index < len(values) else 0 for index in range(3))


def _student_count_label(value: int) -> str:
    """Format a difference with the correct singular or plural noun."""
    return f"{value} leerling" if value == 1 else f"{value} leerlingen"


def _balance_rows(view: dict) -> list[dict]:
    """Build compact summary cells and detailed cells for each stored balance row."""
    group_order = list(view.get("group_order", []))
    result = []
    for stored_row in view.get("balance_rows", []):
        counts = {group: _count_tuple(stored_row, group) for group in group_order}
        sizes = [cell[0] for cell in counts.values()]
        size_largest = max(sizes, default=0)
        size_smallest = min(sizes, default=0)
        sex_differences = [abs(cell[1] - cell[2]) for cell in counts.values()]
        sex_largest = max(sex_differences, default=0)
        label = (
            "Hele groep" if stored_row.get("is_total") else stored_row.get("label", "")
        )
        result.append(
            {
                "label": label,
                "size_diff": size_largest - size_smallest,
                "sex_imbalance": sex_largest,
                "size_diff_label": _student_count_label(size_largest - size_smallest),
                "sex_imbalance_label": _student_count_label(sex_largest),
                "size_cells": [
                    {
                        "value": counts[group][0],
                        "largest": counts[group][0] == size_largest,
                        "smallest": counts[group][0] == size_smallest,
                    }
                    for group in group_order
                ],
                "sex_cells": [
                    {
                        "boys": counts[group][1],
                        "girls": counts[group][2],
                        "largest": abs(counts[group][1] - counts[group][2])
                        == sex_largest,
                    }
                    for group in group_order
                ],
            }
        )
    return result


def _origin_matrix(view: dict) -> tuple[list[dict], int]:
    """Build a matrix from each new student's current group and origin group."""
    group_order = list(view.get("group_order", []))
    counts = {}
    for card in view.get("groups", []):
        group = card.get("name", "")
        for section in card.get("year_sections", []):
            for sex in ("boys", "girls"):
                for chip in section.get(sex, {}).get("students", []):
                    origin = chip.get("origin_full", "")
                    counts.setdefault(origin, {target: 0 for target in group_order})
                    counts[origin][group] = counts[origin].get(group, 0) + 1
    rows = [
        {
            "origin": origin,
            "cells": [counts[origin].get(group, 0) for group in group_order],
        }
        for origin in sorted(counts, key=str.casefold)
    ]
    max_cell = max((cell for row in rows for cell in row["cells"]), default=0)
    return rows, max_cell


def _occupancy_counts(view: dict) -> tuple[int, int]:
    """Count new students and existing occupancy without changing the stored view."""
    new_count = 0
    existing_count = 0
    for card in view.get("groups", []):
        card_new_count = sum(
            int(section.get("size", 0)) for section in card.get("year_sections", [])
        )
        new_count += card_new_count
        existing_count += max(0, int(card.get("total", 0)) - card_new_count)
    return new_count, existing_count


def build_result_page_view(groepsindeling_view: dict | None) -> dict | None:
    """Build the transient, page-specific view from a stored group-card view."""
    if groepsindeling_view is None:
        return None
    origin_rows, max_clique = _origin_matrix(groepsindeling_view)
    new_count, existing_count = _occupancy_counts(groepsindeling_view)
    return {
        "students": _student_result_rows(groepsindeling_view),
        "balance_rows": _balance_rows(groepsindeling_view),
        "group_order": list(groepsindeling_view.get("group_order", [])),
        "origin_rows": origin_rows,
        "max_clique": max_clique,
        "new_count": new_count,
        "existing_count": existing_count,
    }
