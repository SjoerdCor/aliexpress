"""Shared presentation helpers for the wizard's candidate/leerling listings."""


def sorted_for_display(candidates: list[dict]) -> list[dict]:
    """Order candidates per origin group, alphabetically by roepnaam within each group.

    The "Anders" group (students without a real origin group, e.g. new arrivals) sorts
    last regardless of its name, so it forms the final block on the page.
    """

    def key(candidate: dict):
        group = candidate.get("groepsnaam", "")
        anders_last = group.strip().lower() == "anders"
        return (anders_last, group, candidate.get("roepnaam", ""))

    return sorted(candidates, key=key)
