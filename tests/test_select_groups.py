"""Tests for GET/POST /select_groups (herindelen's group-selection step).

Split out of test_wizard.py, which was bumping into pylint's module-length limit.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
from unittest.mock import MagicMock

import pandas as pd

import aliexpress.web.routes.wizard as wizard_module
from tests.helpers import flashes, setup_process


def _make_edexml_reader(df):
    """Return an EdexReader replacement whose get_full_df() returns df."""
    instance = MagicMock()
    instance.get_full_df.return_value = df
    return MagicMock(return_value=instance)


_SELECT_GROUPS_FAKE_DF = pd.DataFrame(
    {
        "key": ["k1", "k2", "k3"],
        "roepnaam": ["Anna", "Ben", "Carl"],
        "achternaam": ["A", "B", "C"],
        "groepsnaam": ["3A", "3B", "3A"],
        "geslacht": ["Meisje", "Jongen", "Jongen"],
        "jaargroep": [3, 3, 3],
    }
)


class TestSelectGroups:
    """Tests for GET/POST /select_groups."""

    def _write_fake_edex(self, proc_dir):
        (proc_dir / "edex.xml").write_bytes(b"fake")

    def test_get_shows_groups_from_edexml(self, client, tmp_path, monkeypatch):
        """GET /select_groups shows checkboxes for each group found in the EDEXML."""
        proc_dir = setup_process(client, tmp_path)
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.get("/select_groups")
        assert resp.status_code == 200
        assert b"3A" in resp.data
        assert b"3B" in resp.data

    def test_get_without_edex_redirects_to_upload(self, client, tmp_path):
        """GET /select_groups without an uploaded EDEXML redirects back to upload."""
        setup_process(client, tmp_path)
        resp = client.get("/select_groups")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")

    def test_post_fewer_than_two_groups_flashes_error(
        self, client, tmp_path, monkeypatch
    ):
        """POST /select_groups with only one group selected flashes an error."""
        proc_dir = setup_process(client, tmp_path)
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.post("/select_groups", data={"groups": ["3A"]})
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/select_groups")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_post_valid_selection_saves_json_and_redirects_to_roster(
        self, client, tmp_path, monkeypatch
    ):
        """POST /select_groups with valid groups saves candidates JSON and goes to roster."""
        proc_dir = setup_process(client, tmp_path)
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.post("/select_groups", data={"groups": ["3A", "3B"]})
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert len(saved["candidates"]) == 3
        assert set(saved["groups_to"].keys()) == {"3A", "3B"}
        assert saved["groups_to"]["3A"] == []
        assert saved["groups_to"]["3B"] == []
        assert saved["jaargroepen"] == [3]

    def test_get_redistribute_and_forward_shows_adapted_heading_and_back_to_roster(
        self, client, tmp_path, monkeypatch
    ):
        """GET /select_groups in redistribute_and_forward mode shows the destination-groups
        heading and a back button to /roster (not /upload_edexml)."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.get("/select_groups")
        assert resp.status_code == 200
        assert "komen deze jaargroepen volgend jaar".encode() in resp.data
        assert b'href="/roster"' in resp.data

    def test_post_redistribute_and_forward_sets_groups_to_and_redirects_to_groups_to(
        self, client, tmp_path, monkeypatch
    ):
        """POST /select_groups in redistribute_and_forward mode keeps the candidates and
        groups_from already settled at upload time, only sets groups_to to the chosen
        destinations, and redirects to /groups_to."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        candidates = [
            {
                "key": "k1",
                "roepnaam": "Anna",
                "achternaam": "A",
                "groepsnaam": "3A",
                "geslacht": "Meisje",
                "jaargroep": 3,
            }
        ]
        (proc_dir / "relevant_students_and_groups.json").write_text(
            json.dumps(
                {
                    "candidates": candidates,
                    "groups_from": ["3A", "3B", "Anders"],
                    "groups_to": {},
                    "jaargroepen": [3],
                }
            ),
            encoding="utf-8",
        )
        resp = client.post("/select_groups", data={"groups": ["4A", "4B"]})
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/groups_to")
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert saved["candidates"] == candidates
        assert saved["groups_from"] == ["3A", "3B", "Anders"]
        assert saved["groups_to"] == {"4A": [], "4B": []}
        assert saved["jaargroepen"] == [3]

    def test_get_redistribute_and_forward_marks_step_3_active_in_stepper(
        self, client, tmp_path, monkeypatch
    ):
        """GET /select_groups in redistribute_and_forward mode is reached after "Wie gaat
        mee" (step 2), so the stepper must mark step 3 ("Groepen naartoe") as active, not
        step 1 ("Schoolinformatie")."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.get("/select_groups")
        html = resp.data.decode("utf-8")
        assert re.search(r"step active\">\s*<span>Groepen naartoe<", html)
        assert re.search(r"step done\">\s*<span>Schoolinformatie<", html)

    def test_get_redistribute_marks_step_1_active_in_stepper(
        self, client, tmp_path, monkeypatch
    ):
        """Regression: GET /select_groups in plain redistribute mode still marks step 1
        ("Schoolinformatie") as active, unchanged from before."""
        proc_dir = setup_process(client, tmp_path)
        self._write_fake_edex(proc_dir)
        monkeypatch.setattr(
            wizard_module.datareader,
            "EdexReader",
            _make_edexml_reader(_SELECT_GROUPS_FAKE_DF),
        )
        resp = client.get("/select_groups")
        html = resp.data.decode("utf-8")
        assert re.search(r"step active\">\s*<span>Schoolinformatie<", html)

    def test_post_persists_every_jaargroep_in_a_combination_class(
        self, client, tmp_path, monkeypatch
    ):
        """A selected combination class ("6/7A") spans two jaargroepen; both are recorded
        for the roster page's new-student dropdown, not just one per selected group."""
        proc_dir = setup_process(client, tmp_path)
        self._write_fake_edex(proc_dir)
        df = pd.DataFrame(
            {
                "key": ["k1", "k2", "k3"],
                "roepnaam": ["Anna", "Ben", "Carl"],
                "achternaam": ["A", "B", "C"],
                "groepsnaam": ["6/7A", "6/7A", "8B"],
                "geslacht": ["Meisje", "Jongen", "Jongen"],
                "jaargroep": [6, 7, 8],
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(df)
        )
        resp = client.post("/select_groups", data={"groups": ["6/7A", "8B"]})
        assert resp.status_code == 302
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert saved["jaargroepen"] == [6, 7, 8]
