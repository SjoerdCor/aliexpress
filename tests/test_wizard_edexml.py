"""Tests for the /upload_edexml wizard route."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd

import aliexpress.web.routes.wizard as wizard_module
from tests.helpers import flashes, setup_process


class TestUploadEdexmlErrors:  # pylint: disable=too-few-public-methods  # one test
    """Tests for friendly EDEXML upload errors."""

    def test_garbage_edexml_redirects_with_error_flash(self, client, tmp_path):
        """Uploading a garbage EDEXML file flashes an error and redirects back."""
        setup_process(client, tmp_path)

        response = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"garbage"), "edex.xml"),
                "jaargroep": "4",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _msg in flashes(client))


def _make_edexml_reader(df):
    """Return an EdexReader replacement whose get_full_df() returns df."""
    instance = MagicMock()
    instance.get_full_df.return_value = df
    return MagicMock(return_value=instance)


class TestUploadEdexmlMode:
    """Tests for mode-branching in the upload_edexml route."""

    def test_get_forward_mode_shows_jaargroep(self, client, tmp_path):
        """GET /upload_edexml in forward mode shows the jaargroep selector."""
        setup_process(client, tmp_path)
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b"jaargroep" in resp.data

    def test_get_redistribute_mode_hides_jaargroep(self, client, tmp_path):
        """GET /upload_edexml in redistribute mode hides the jaargroep selector."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b"jaargroep" not in resp.data

    def test_post_redistribute_valid_edexml_redirects_to_select_groups(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute mode redirects to /select_groups."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame({"groepsnaam": ["3A", "3B"], "jaargroep": [3, 3]})
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/select_groups")

    def test_get_redistribute_and_forward_mode_shows_jaargroep_checkboxes(
        self, client, tmp_path
    ):
        """GET /upload_edexml in redistribute_and_forward mode shows jaargroep checkboxes
        (not the forward-mode dropdown) and the shared "Selecteer leerlingen en groepen"
        button text."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        resp = client.get("/upload_edexml")
        assert resp.status_code == 200
        assert b'name="jaargroep"' not in resp.data
        assert resp.data.count(b'name="jaargroepen"') == 8
        assert "Selecteer leerlingen en groepen".encode() in resp.data

    def test_post_redistribute_and_forward_valid_selection_saves_json_and_redirects_to_roster(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode, with jaargroepen ticked,
        determines candidates school-wide for those jaargroepen, leaves groups_to empty
        (destinations are picked later at /select_groups) and redirects to /roster."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A", "6A", "6B", "7A", "8A"],
                "roepnaam": ["Anna", "Ben", "Carl", "Dana", "Eva"],
                "achternaam": ["A", "B", "C", "D", "E"],
                "key": ["k1", "k2", "k3", "k4", "k5"],
                "geslacht": ["Meisje", "Jongen", "Jongen", "Meisje", "Meisje"],
                "jaargroep": pd.array([5, 6, 6, 7, 8], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["5", "6", "7"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert {c["roepnaam"] for c in saved["candidates"]} == {
            "Anna",
            "Ben",
            "Carl",
            "Dana",
        }
        assert saved["groups_to"] == {}
        assert saved["jaargroepen"] == [5, 6, 7]

    def test_post_redistribute_and_forward_without_jaargroep_flashes(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode without ticking any
        jaargroep flashes a warning and redirects back to /upload_edexml."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A"],
                "roepnaam": ["Anna"],
                "achternaam": ["A"],
                "key": ["k1"],
                "geslacht": ["Meisje"],
                "jaargroep": pd.array([5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))
        assert not (proc_dir / "relevant_students_and_groups.json").exists()

    def test_post_redistribute_and_forward_no_candidates_flashes(
        self, client, tmp_path, monkeypatch
    ):
        """POST /upload_edexml in redistribute_and_forward mode with jaargroepen that match
        no students flashes a warning and redirects back to /upload_edexml."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A"],
                "roepnaam": ["Anna"],
                "achternaam": ["A"],
                "key": ["k1"],
                "geslacht": ["Meisje"],
                "jaargroep": pd.array([5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["8"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))
        assert not (proc_dir / "relevant_students_and_groups.json").exists()

    def test_post_redistribute_and_forward_missing_jaargroep_flashes_info(
        self, client, tmp_path, monkeypatch
    ):
        """A student without a jaargroep in the uploaded file is reported (info flash) as
        not taking part, while the rest of the upload succeeds normally."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["5A", "5A"],
                "roepnaam": ["Anna", "Ben"],
                "achternaam": ["A", "B"],
                "key": ["k1", "k2"],
                "geslacht": ["Meisje", "Jongen"],
                "jaargroep": pd.array([5, None], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["5"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        assert any(
            cat == "info" and "geen jaargroep" in msg for cat, msg in flashes(client)
        )

    def test_post_redistribute_garbage_edexml_flashes_error(self, client, tmp_path):
        """POST /upload_edexml in redistribute mode with a garbage file flashes an error."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute"}), encoding="utf-8"
        )
        resp = client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"garbage"), "edex.xml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/upload_edexml")
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_post_redistribute_and_forward_reupload_clears_stale_wizard_state(
        self, client, tmp_path, monkeypatch
    ):
        """Re-uploading EDEXML in redistribute_and_forward mode (e.g. after going back and
        picking different jaargroepen) must wipe stale downstream artifacts from the
        earlier upload, so the roster/preferences pages don't merge old and new
        populations."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text(
            json.dumps({"mode": "redistribute_and_forward"}), encoding="utf-8"
        )
        # Simulate leftover state from an earlier upload of jaargroepen 5-6-7.
        (proc_dir / "roster.json").write_text("{}", encoding="utf-8")
        (proc_dir / "voorkeuren.json").write_text("{}", encoding="utf-8")
        (proc_dir / "groups.xlsx").write_text("stale", encoding="utf-8")

        fake_df = pd.DataFrame(
            {
                "groepsnaam": ["3A", "4A", "5A"],
                "roepnaam": ["Anna", "Ben", "Carl"],
                "achternaam": ["A", "B", "C"],
                "key": ["k1", "k2", "k3"],
                "geslacht": ["Meisje", "Jongen", "Jongen"],
                "jaargroep": pd.array([3, 4, 5], dtype="Int64"),
            }
        )
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(b"anything"), "edex.xml"),
                "jaargroepen": ["3", "4", "5"],
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/roster")
        assert not (proc_dir / "roster.json").exists()
        assert not (proc_dir / "voorkeuren.json").exists()
        assert not (proc_dir / "groups.xlsx").exists()
        saved = json.loads(
            (proc_dir / "relevant_students_and_groups.json").read_text("utf-8")
        )
        assert {c["roepnaam"] for c in saved["candidates"]} == {"Anna", "Ben", "Carl"}

    def test_post_upload_edexml_removes_stale_downstream_files(
        self, client, tmp_path, monkeypatch
    ):
        """A general check that any new EDEXML upload (forward mode here) removes stale
        wizard artifacts derived from a previous upload."""
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "roster.json").write_text("{}", encoding="utf-8")
        (proc_dir / "voorkeuren.json").write_text("{}", encoding="utf-8")
        (proc_dir / "groups.xlsx").write_text("stale", encoding="utf-8")

        fake_df = pd.DataFrame({"groepsnaam": ["3A", "3B"], "jaargroep": [3, 3]})
        monkeypatch.setattr(
            wizard_module.datareader, "EdexReader", _make_edexml_reader(fake_df)
        )
        monkeypatch.setattr(
            wizard_module.candidatedetermination,
            "handle_edexml_upload",
            lambda df, jaargroep: ([], {}, {}),
        )
        client.post(
            "/upload_edexml",
            data={"edexml": (BytesIO(b"anything"), "edex.xml"), "jaargroep": "4"},
            content_type="multipart/form-data",
        )

        assert not (proc_dir / "roster.json").exists()
        assert not (proc_dir / "voorkeuren.json").exists()
        assert not (proc_dir / "groups.xlsx").exists()
