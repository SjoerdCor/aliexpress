# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

"""Browser tests for the groups-to page JavaScript (enable/disable, adding groups)."""

import json

import pandas as pd


def student(geslacht, roepnaam="Test", *, blijft=True):
    """Build one groups-to student dict as the candidates JSON stores them."""
    return {
        "geslacht": geslacht,
        "roepnaam": roepnaam,
        "achternaam": "Leerling",
        "jaargroep": 4,
        "blijft_in_groep": blijft,
    }


def _state(proc):
    return json.loads((proc / "groups_to_state.json").read_text("utf-8"))


def _groups_xlsx(proc):
    return pd.read_excel(proc / "groups.xlsx", index_col=0)


def test_switched_off_group_keeps_its_ticks(open_groups_to, page):
    """Switching a group off keeps two active groups, drops it from groups.xlsx, but its
    ticks still submit so they are remembered (regression for the disabled-group bug).
    """
    proc = open_groups_to(
        {
            "Klas A": [student("Jongen")],
            "Klas B": [student("Meisje")],  # ticked by default (blijft_in_groep)
            "Klas C": [student("Jongen")],
        }
    )
    page.click('.group-block[data-group="Klas B"] .group-disable')
    page.click("button.next-step")
    page.wait_for_url("**/roster")

    state = _state(proc)
    assert state["disabled_groups"] == ["Klas B"]
    # Its checkbox stayed enabled, so the tick is remembered for when it is switched on.
    assert state["original_groups"]["Klas B"]["checked_indices"] == [0]
    assert "Klas B" not in _groups_xlsx(proc).index


def test_added_empty_group_is_saved(open_groups_to, page):
    """A group added with '+ Nieuwe lege groep' is submitted and stored at 0/0."""
    proc = open_groups_to(
        {"Klas A": [student("Jongen")], "Klas B": [student("Meisje")]}
    )
    page.click('button:has-text("Nieuwe lege groep")')
    page.fill("#new-groups input.group-name-input", "Extra groep")
    page.click("button.next-step")
    page.wait_for_url("**/roster")

    assert _state(proc)["new_groups"] == ["Extra groep"]
    saved = _groups_xlsx(proc)
    assert saved.loc["Extra groep", "Jongens"] == 0
    assert saved.loc["Extra groep", "Meisjes"] == 0
