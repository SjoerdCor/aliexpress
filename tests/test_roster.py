"""Tests for the /roster wizard step ("Wie gaat mee"): determining the population of
leerlingen that take part in this verdeling, shared by both input routes (ADR 0005).

Only synthetic data is used here, never real student data.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re

import pandas as pd

from tests.helpers import TWO_STUDENTS_GROEN, setup_process


class TestRosterPage:
    """Tests for GET/POST /roster."""

    CANDIDATES = TWO_STUDENTS_GROEN

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
        assert "Vink aan wie ingedeeld moeten worden".encode("utf-8") in response.data

    def test_post_writes_roster_and_redirects_to_groups_to(self, client, tmp_path):
        """POST /roster writes roster.json with every participant and continues to
        "Groepen naartoe"; the preference method is chosen there now (ADR 0006), so roster
        writes no input_method.json."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post("/roster", data={"gaat_over": ["s1", "s2"]})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/groups_to")
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        keys = {p["key"] for p in roster["participants"]}
        assert keys == {"s1", "s2"}
        assert not (proc_dir / "input_method.json").exists()

    def test_post_unchecked_verlenger_is_excluded(self, client, tmp_path):
        """A verlenger (unticked) is left out of roster.json."""
        proc_dir = self._setup(client, tmp_path)
        client.post("/roster", data={"gaat_over": ["s1"]})
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        keys = {p["key"] for p in roster["participants"]}
        assert keys == {"s1"}

    def test_post_with_new_student_is_included(self, client, tmp_path):
        """A hand-added incoming student is included in roster.json with a new_* key."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/roster",
            data={
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


class TestRosterRedistributeAndForward:
    """Tests for /roster's redistribute_and_forward branch: next step is /select_groups
    (destinations), not /groups_to, and the back button returns to /upload_edexml."""

    CANDIDATES = TWO_STUDENTS_GROEN

    def _setup(self, client, tmp_path):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps(
                {
                    "candidates": self.CANDIDATES,
                    "groups_from": ["Groen"],
                    "groups_to": {},
                }
            ),
            encoding="utf-8",
        )
        return proc_dir

    def test_get_prev_button_points_to_upload_edexml(self, client, tmp_path):
        """GET /roster's back link points to /upload_edexml in redistribute_and_forward
        mode (destinations have not been chosen yet)."""
        self._setup(client, tmp_path)
        response = client.get("/roster")
        assert response.status_code == 200
        assert b'href="/upload_edexml"' in response.data

    def test_post_redirects_to_select_groups(self, client, tmp_path):
        """POST /roster in redistribute_and_forward mode continues to /select_groups, where
        the destination groups are chosen, instead of /groups_to."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post("/roster", data={"gaat_over": ["s1", "s2"]})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/select_groups")
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        assert {p["key"] for p in roster["participants"]} == {"s1", "s2"}


class TestRosterNewStudentJaargroep:
    """A hand-added new student needs a jaargroep too (for the per-year group report).

    In doorzetten mode every candidate already shares one jaargroep (the one chosen on the
    EDEXML upload page), so a new student is assumed to join that same cohort. In herindelen
    mode candidates span several jaargroepen, so the teacher must say which one explicitly.
    """

    CANDIDATES_FORWARD = [
        {
            "key": "s1",
            "roepnaam": "Anna",
            "achternaam": "Bos",
            "groepsnaam": "Groen",
            "geslacht": "Meisje",
            "jaargroep": 5,
        },
    ]

    CANDIDATES_REDISTRIBUTE = [
        {
            "key": "s1",
            "roepnaam": "Anna",
            "achternaam": "Bos",
            "groepsnaam": "Groen",
            "geslacht": "Meisje",
            "jaargroep": 6,
        },
        {
            "key": "s2",
            "roepnaam": "Bram",
            "achternaam": "Dijk",
            "groepsnaam": "Blauw",
            "geslacht": "Jongen",
            "jaargroep": 7,
        },
    ]

    def _setup(self, client, tmp_path, candidates, redistribute_jaargroepen=None):
        """Set up a process; ``redistribute_jaargroepen`` given means herindelen mode."""
        mode = "redistribute" if redistribute_jaargroepen is not None else "forward"
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps(
                {
                    "candidates": candidates,
                    "groups_from": ["Groen"],
                    "jaargroepen": redistribute_jaargroepen or [],
                }
            ),
            encoding="utf-8",
        )
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": mode}), encoding="utf-8"
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        return proc_dir

    def test_forward_mode_new_student_gets_shared_jaargroep(self, client, tmp_path):
        """Doorzetten: a new student with no jaargroep entered gets the process's jaargroep."""
        proc_dir = self._setup(client, tmp_path, self.CANDIDATES_FORWARD)
        client.post(
            "/roster",
            data={
                "gaat_over": ["s1", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Groen",
            },
        )
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        emma = next(p for p in roster["participants"] if p["roepnaam"] == "Emma")
        assert emma["jaargroep"] == 5

    def test_redistribute_mode_new_student_without_jaargroep_flashes(
        self, client, tmp_path
    ):
        """Herindelen: a new student without an explicit jaargroep is rejected."""
        proc_dir = self._setup(
            client,
            tmp_path,
            self.CANDIDATES_REDISTRIBUTE,
            redistribute_jaargroepen=[6, 7],
        )
        response = client.post(
            "/roster",
            data={
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Groen",
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")
        assert not (proc_dir / "roster.json").exists()

    def test_redistribute_mode_new_student_with_jaargroep_is_saved(
        self, client, tmp_path
    ):
        """Herindelen: a new student with an explicit jaargroep is accepted as-entered."""
        proc_dir = self._setup(
            client,
            tmp_path,
            self.CANDIDATES_REDISTRIBUTE,
            redistribute_jaargroepen=[6, 7],
        )
        client.post(
            "/roster",
            data={
                "gaat_over": ["s1", "s2", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Groen",
                "new_jaargroep[]": "7",
            },
        )
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        emma = next(p for p in roster["participants"] if p["roepnaam"] == "Emma")
        assert emma["jaargroep"] == 7

    def test_redistribute_mode_jaargroep_options_reflect_the_group_selection(
        self, client, tmp_path
    ):
        """The dropdown offers the jaargroepen recorded at select_groups time, even when a
        candidate of one of them is no longer present (e.g. unticked/removed since)."""
        candidate_missing_jaargroep_6 = [
            self.CANDIDATES_REDISTRIBUTE[1]
        ]  # only s2, jg 7
        self._setup(
            client,
            tmp_path,
            candidate_missing_jaargroep_6,
            redistribute_jaargroepen=[6, 7],
        )
        html = client.get("/roster").data.decode("utf-8")
        roster_data = json.loads(
            re.search(
                r'<script type="application/json" id="roster-data">\s*(\{.*?\})\s*</script>',
                html,
                re.S,
            ).group(1)
        )
        assert roster_data["jaargroep_options"] == [6, 7]
