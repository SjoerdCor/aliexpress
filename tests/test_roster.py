"""Tests for the /roster wizard step ("Leerlingen controleren"): determining the population of
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
            json.dumps(
                {"candidates": self.CANDIDATES, "groups_from": ["Groen", "Anders"]}
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        return proc_dir

    def test_get_returns_200_with_candidate_names_and_page_navigation(
        self, client, tmp_path
    ):
        """GET /roster shows the roster task and its doorzetten navigation."""
        self._setup(client, tmp_path)
        response = client.get("/roster")
        assert response.status_code == 200
        assert b"Anna" in response.data
        assert b"Bram" in response.data
        assert "Leerlingen controleren".encode("utf-8") in response.data
        assert "Huidige groep: Groen".encode("utf-8") in response.data
        assert "Terug naar leerlinggegevens".encode("utf-8") in response.data
        assert "groepen controleren".encode("utf-8") in response.data
        assert b"roster.css" in response.data

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

    def test_post_with_anders_keeps_the_explicit_current_group(self, client, tmp_path):
        """A student outside the named groups can explicitly choose Anders."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/roster",
            data={
                "gaat_over": ["new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Mila",
                "new_achternaam[]": "Visser",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "Anders",
            },
        )
        roster = json.loads((proc_dir / "roster.json").read_text("utf-8"))
        assert roster["participants"][0]["groepsnaam"] == "Anders"

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
                "new_groep[]": "Groen",
            },
        )
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        assert b"Vul de voornaam, achternaam en het geslacht in." in response.data
        assert b'"roepnaam": "Emma"' in response.data
        assert b'"achternaam": "Jansen"' in response.data

    def test_post_with_no_participants_flashes_and_does_not_save(
        self, client, tmp_path
    ):
        """At least one leerling must remain selected before the roster can continue."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post("/roster", data={"gaat_over": []})
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        assert (
            "Selecteer ten minste één leerling die doorgaat."
            in response.data.decode("utf-8")
        )

    def test_post_with_open_new_student_flashes_and_does_not_save(
        self, client, tmp_path
    ):
        """An added but still open row must be confirmed or removed first."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster",
            data={"gaat_over": ["s1"], "new_key[]": "new_0"},
        )
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        assert (
            "Bevestig de leerling met ‘Leerling aan de lijst toevoegen’ of verwijder de invoer."
            in response.data.decode("utf-8")
        )

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
                "new_groep[]": "Groen",
            },
        )
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        html = response.data.decode("utf-8")
        assert "Er staat al een leerling met de naam ‘Anna Bos’ in de lijst." in html
        assert "Pas de naam aan" not in html

    def test_post_requires_an_explicit_current_group_and_preserves_values(
        self, client, tmp_path
    ):
        """A new student cannot use the old first-group fallback on a rejected POST."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/roster",
            data={
                "gaat_over": ["s1", "new_0"],
                "new_key[]": "new_0",
                "new_voornaam[]": "Emma",
                "new_achternaam[]": "Jansen",
                "new_geslacht[]": "Meisje",
                "new_groep[]": "",
            },
        )
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        html = response.data.decode("utf-8")
        assert "Kies de huidige groep, of kies ‘Anders’." in html
        assert re.search(r'value="s1"\s+checked', html)
        assert not re.search(r'value="s2"\s+checked', html)
        assert '"roepnaam": "Emma"' in html
        assert '"groepsnaam": ""' in html


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
                    "groups_from": ["Groen", "Anders"],
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

    def test_get_shows_forward_navigation_labels(self, client, tmp_path):
        """The third flow keeps its upload back route and new-groups next route."""
        self._setup(client, tmp_path)
        html = client.get("/roster").data.decode("utf-8")
        assert "Leerlingen controleren" in html
        assert "Terug naar leerlinggegevens" in html
        assert "Verder naar nieuwe groepen" in html


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
                    "groups_from": ["Groen", "Anders"],
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
        assert response.status_code == 200
        assert not (proc_dir / "roster.json").exists()
        assert "Kies ook de huidige jaarlaag." in response.data.decode("utf-8")

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
