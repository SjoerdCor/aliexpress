"""Tests for the GET/POST /preferences_form route (web-form preference input path).

Split out of test_wizard.py: these route tests are a cohesive, growing group around the
preferences_form redesign. Only synthetic data is used here, never real student data.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json

import pandas as pd

from tests.helpers import setup_process


class TestPreferencesForm:
    """Tests for GET/POST /preferences_form."""

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

    def _setup(self, client, tmp_path, participants=None):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps({"candidates": self.CANDIDATES, "groups_from": ["Groen"]}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        # The roster step has run: the population is settled before preferences are entered.
        (proc_dir / "roster.json").write_text(
            json.dumps(
                {
                    "participants": (
                        participants if participants is not None else self.CANDIDATES
                    )
                }
            ),
            encoding="utf-8",
        )
        return proc_dir

    def test_get_returns_200_with_student_names_and_target_groups(
        self, client, tmp_path
    ):
        """GET /preferences_form shows each candidate's name and the target group names."""
        self._setup(client, tmp_path)
        response = client.get("/preferences_form")
        assert response.status_code == 200
        assert b"Anna" in response.data
        assert b"Bram" in response.data
        assert b"Klas A" in response.data

    def test_post_writes_voorkeuren_json_for_all_participants_and_redirects(
        self, client, tmp_path
    ):
        """POST /preferences_form writes voorkeuren.json for every roster participant and
        redirects to not_together (the population is fixed by the roster step)."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post("/preferences_form", data={})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/not_together")
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        assert payload["source"] == "form"
        display_names = set(payload["student_display"].values())
        assert "Anna Bos" in display_names
        assert "Bram Dijk" in display_names

    def test_get_redirects_to_roster_when_no_roster_yet(self, client, tmp_path):
        """Without a settled roster the page sends the teacher to 'Wie gaat mee' first."""
        proc_dir = self._setup(client, tmp_path)
        (proc_dir / "roster.json").unlink()
        response = client.get("/preferences_form")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/roster")

    def test_post_with_wish_appears_in_preference_frame(self, client, tmp_path):
        """POST with a 'Graag met' wish is stored in the preference frame."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={
                "preference_s1_graag_met_target": ["Bram Dijk"],
                "preference_s1_graag_met_gewicht": ["1"],
            },
        )
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        records = payload["preferences"]["records"]
        assert any(
            r["Leerling"] == "annabos"
            and r["TypeWens"] == "Graag met"
            and r["Waarde"] == "bramdijk"
            for r in records
        )

    def test_autosave_writes_draft_only(self, client, tmp_path):
        """A POST with action='autosave' saves the draft state, not voorkeuren.json,
        and returns 204 (background save, no navigation)."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/preferences_form",
            data={
                "action": "autosave",
                "preference_s1_graag_met_target": ["Bram Dijk"],
                "preference_s1_graag_met_gewicht": ["1"],
            },
        )
        assert response.status_code == 204
        assert (proc_dir / "preferences_form_state.json").exists()
        assert not (proc_dir / "voorkeuren.json").exists()

    def test_get_candidates_sorted_by_group(self, client, tmp_path):
        """GET /preferences_form returns candidates sorted by groepsnaam."""
        proc_dir = setup_process(client, tmp_path)
        candidates = [
            {
                "key": "s1",
                "roepnaam": "Zes",
                "achternaam": "Z",
                "groepsnaam": "Zulu",
                "geslacht": "Jongen",
            },
            {
                "key": "s2",
                "roepnaam": "Alfa",
                "achternaam": "A",
                "groepsnaam": "Alpha",
                "geslacht": "Meisje",
            },
        ]
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": candidates}), encoding="utf-8"
        )
        pd.DataFrame(
            {"Jongens": [1, 1], "Meisjes": [0, 1]},
            index=pd.Index(["Klas A", "Klas B"], name="Groepen"),
        ).to_excel(proc_dir / "groups.xlsx")
        html = client.get("/preferences_form").data.decode("utf-8")
        pos_alfa = html.find("Alfa")
        pos_zes = html.find("Zes")
        assert pos_alfa < pos_zes, "Alpha moet vóór Zulu staan"

    def test_post_min_satisfaction_stored_in_students_info(self, client, tmp_path):
        """POST /preferences_form with min_satisfaction percentage is stored as 0-1 decimal."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={"min_sat_s1": "50"},
        )
        payload = json.loads((proc_dir / "voorkeuren.json").read_text("utf-8"))
        info = payload["students_info"]
        anna_key = next(
            k for k, v in payload["student_display"].items() if v == "Anna Bos"
        )
        assert abs(info[anna_key]["MinimaleTevredenheid"] - 0.5) < 0.001

    def test_get_candidates_sorted_within_group_and_anders_last(self, client, tmp_path):
        """Within a group candidates are alphabetical by roepnaam; the 'Anders' group of
        students without an origin group is shown last, regardless of its name."""
        proc_dir = setup_process(client, tmp_path)
        candidates = [
            {
                "key": "s1",
                "roepnaam": "Tom",
                "achternaam": "T",
                "groepsnaam": "Beren",
                "geslacht": "Jongen",
            },
            {
                "key": "s2",
                "roepnaam": "Anne",
                "achternaam": "A",
                "groepsnaam": "Beren",
                "geslacht": "Meisje",
            },
            {
                "key": "s3",
                "roepnaam": "Bo",
                "achternaam": "B",
                "groepsnaam": "Anders",
                "geslacht": "Jongen",
            },
        ]
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": candidates}), encoding="utf-8"
        )
        pd.DataFrame(
            {"Jongens": [1], "Meisjes": [1]}, index=pd.Index(["Klas A"], name="Groepen")
        ).to_excel(proc_dir / "groups.xlsx")
        html = client.get("/preferences_form").data.decode("utf-8")
        assert html.find("Anne") < html.find("Tom") < html.find("Bo")

    def test_post_with_unknown_target_flashes_and_preserves_draft(
        self, client, tmp_path
    ):
        """An invalid preference (unknown target) must not 500: it flashes a friendly
        error, redirects back to the form, preserves the draft, and persists nothing."""
        proc_dir = self._setup(client, tmp_path)
        response = client.post(
            "/preferences_form",
            data={
                "preference_s1_graag_met_target": ["Spook Persoon"],
                "preference_s1_graag_met_gewicht": ["1"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")
        # Invalid input is not persisted as the canonical preferences...
        assert not (proc_dir / "voorkeuren.json").exists()
        # ...but the teacher's work is kept as a draft so nothing is lost.
        assert (proc_dir / "preferences_form_state.json").exists()
        # The flash survives the redirect and is shown on the form.
        html = client.get("/preferences_form").data.decode("utf-8")
        assert "flash" in html.lower()

    def test_get_after_post_prefills_wishes_from_state(self, client, tmp_path):
        """GET /preferences_form after a POST prefills the previously saved wish."""
        proc_dir = self._setup(client, tmp_path)
        client.post(
            "/preferences_form",
            data={
                "preference_s1_graag_met_target": ["Bram Dijk"],
                "preference_s1_graag_met_gewicht": ["2"],
            },
        )
        assert (proc_dir / "preferences_form_state.json").exists()
        html = client.get("/preferences_form").data.decode("utf-8")
        assert "Bram Dijk" in html

    def test_dangling_preference_dropped_with_notice_when_target_left_roster(
        self, client, tmp_path
    ):
        """If a preference points to a leerling who was later removed from the roster, the
        GET drops it and shows a friendly notice about the removal."""
        proc_dir = self._setup(client, tmp_path)
        # Anna (s1) prefers Bram (s2); save that as a draft.
        client.post(
            "/preferences_form",
            data={
                "action": "autosave",
                "preference_s1_graag_met_target": ["Bram Dijk"],
                "preference_s1_graag_met_gewicht": ["1"],
            },
        )
        # Now Bram is no longer a participant (removed on the roster step).
        (proc_dir / "roster.json").write_text(
            json.dumps({"participants": [self.CANDIDATES[0]]}), encoding="utf-8"
        )
        html = client.get("/preferences_form").data.decode("utf-8")
        assert "Bram Dijk" in html  # named in the removal notice
        assert "is verwijderd" in html
