"""Tests for the /roster wizard step ("Wie gaat mee"): determining the population of
leerlingen that take part in this verdeling, shared by both input routes (ADR 0005).

Only synthetic data is used here, never real student data.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re

import pandas as pd

from tests.helpers import setup_process


class TestRosterPage:
    """Tests for GET/POST /roster."""

    CANDIDATES = [
        {
            "key": "s1",
            "roepnaam": "Anna",
            "achternaam": "Bos",
            "groepsnaam": "Groen",
            "geslacht": "Meisje",
        },
        {
            "key": "s2",
            "roepnaam": "Bram",
            "achternaam": "Dijk",
            "groepsnaam": "Groen",
            "geslacht": "Jongen",
        },
    ]

    def _setup(self, client, tmp_path):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps({"candidates": self.CANDIDATES, "groups_from": ["Groen"]}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        return proc_dir

    def test_get_returns_200_with_candidate_names_and_reassuring_intro(
        self, client, tmp_path
    ):
        """GET /roster shows each candidate's name and an intro making clear the list is
        already loaded, so the teacher need not re-enter anyone."""
        self._setup(client, tmp_path)
        response = client.get("/roster")
        assert response.status_code == 200
        assert b"Anna" in response.data
        assert b"Bram" in response.data
        assert "staan al klaar".encode("utf-8") in response.data

    def test_post_form_choice_writes_roster_method_and_redirects_to_form(
        self, client, tmp_path
    ):
        """POST /roster with the 'form' button writes roster.json with every participant,
        records input_method=form, and redirects to the form preferences page."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster", data={"action": "form", "gaat_over": ["s1", "s2"]}
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        keys = {p["key"] for p in roster["participants"]}
        assert keys == {"s1", "s2"}
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "form"

    def test_post_excel_choice_records_method_and_redirects_to_excel(
        self, client, tmp_path
    ):
        """POST /roster with the 'excel' button records input_method=excel and redirects
        to the Excel preferences page."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster", data={"action": "excel", "gaat_over": ["s1", "s2"]}
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "excel"

    def test_post_unchecked_verlenger_is_excluded(self, client, tmp_path):
        """A verlenger (unticked) is left out of roster.json."""
        proc_dir = self._setup(client, tmp_path)
        client.post("/roster", data={"action": "form", "gaat_over": ["s1"]})
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        keys = {p["key"] for p in roster["participants"]}
        assert keys == {"s1"}

    def test_post_with_new_student_is_included(self, client, tmp_path):
        """A hand-added incoming student is included in roster.json with a new_* key."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/roster",
            data={
                "action": "form",
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Groen",
            },
        )
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        emma = next(
            (p for p in roster["participants"] if p["roepnaam"] == "Emma"), None
        )
        assert emma is not None
        assert emma["key"] == "new_0"
        assert emma["achternaam"] == "Jansen"
        assert emma["geslacht"] == "Meisje"

    def test_post_incomplete_new_student_flashes_and_does_not_save(
        self, client, tmp_path
    ):
        """A started-but-unfinished new student (missing geslacht) is rejected with a
        friendly flash; nothing is persisted and the teacher stays on /roster."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster",
            data={
                "action": "form",
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "",
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")
        assert not (proc_dir / "roster.json").exists()

    def test_get_after_post_restores_verlenger_and_new_student(self, client, tmp_path):
        """GET /roster after a POST reflects the saved roster: a verlenger is unticked and
        a previously added new student is shown again."""
        self._setup(client, tmp_path)
        client.post(
            "/roster",
            data={
                "action": "form",
                "gaat_over": ["s1", "new_0"],  # s2 left behind (verlenger)
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Groen",
            },
        )
        html = client.get("/roster").data.decode("utf-8")
        assert re.search(r'value="s1"\s+checked', html)  # still going
        assert not re.search(r'value="s2"\s+checked', html)  # verlenger, unticked
        assert "Emma" in html
        assert "Jansen" in html

    def test_post_new_student_name_collision_flashes(self, client, tmp_path):
        """A new student whose name clashes with an existing leerling is rejected."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster",
            data={
                "action": "form",
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Anna",
                "new_achternaam[]": "Bos",
                "new_geslacht[]": "Meisje",
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")
        assert not (proc_dir / "roster.json").exists()
