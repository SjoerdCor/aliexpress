"""Tests for the /groups_to wizard route."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
# pylint: disable=duplicate-code  # route setup repeats intentional integration scenarios

import json
import re

import pandas as pd

from tests.helpers import flashes, make_students, setup_process, write_groups_to_json


class TestGroupsToPage:
    """Tests for GET/POST /groups_to."""

    def test_get_renders_groups_from_json(self, client, tmp_path):
        """GET /groups_to reads the candidates JSON and renders group names in the page."""
        proc_dir = setup_process(client, tmp_path)
        # groups_to is a dict {groupname: [students]}; the template calls .items()
        write_groups_to_json(proc_dir, {"Klas A": [], "Klas B": []})
        response = client.get("/groups_to")
        assert response.status_code == 200
        assert b"Klas A" in response.data

    def test_post_too_few_groups_flashes_error(self, client, tmp_path):
        """POST /groups_to with fewer than 2 groups flashes an error and redirects back."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(proc_dir, {"Klas A": make_students("Jongen")})
        response = client.post(
            "/groups_to",
            data={"group": ["Klas A"], "group_students[Klas A]": ["0"]},
        )
        assert response.status_code == 302
        assert any(cat == "error" for cat, _ in flashes(client))

    def test_post_duplicate_group_names_flashes_error(self, client, tmp_path):
        """POST /groups_to with duplicate group names flashes an error and redirects back."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(proc_dir, {"Klas A": make_students("Jongen")})
        response = client.post(
            "/groups_to",
            data={"group": ["Klas A", "Klas A"]},
        )
        assert response.status_code == 302
        messages = flashes(client)
        assert any(cat == "error" for cat, _ in messages)
        assert any("Klas A" in msg for _, msg in messages)

    def test_post_form_choice_records_method_and_redirects_to_form(
        self, client, tmp_path
    ):
        """POST /groups_to with the 'form' button saves groups.xlsx, records
        input_method=form, and continues to the form preferences page (ADR 0006)."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
        )
        response = client.post(
            "/groups_to",
            data={
                "action": "form",
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_form")
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "form"

    def test_post_excel_choice_records_method_and_redirects_to_excel(
        self, client, tmp_path
    ):
        """POST /groups_to with the 'excel' button records input_method=excel and continues
        to the Excel preferences page (ADR 0006)."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
        )
        response = client.post(
            "/groups_to",
            data={
                "action": "excel",
                "group": ["Klas A", "Klas B"],
                "group_students[Klas A]": ["0", "1"],
                "group_students[Klas B]": ["0"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "excel"

    def test_post_empty_group_is_kept_with_zero_counts(self, client, tmp_path):
        """A group submitted via 'group' but without retained students is kept at 0/0."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir, {"Klas A": make_students("Jongen", "Meisje", "Meisje")}
        )
        response = client.post(
            "/groups_to",
            data={
                "group": ["Klas A", "Nieuwe groep 1"],
                "group_students[Klas A]": ["0", "1", "2"],
            },
        )
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/preferences_excel")
        saved = pd.read_excel(proc_dir / "groups.xlsx", index_col=0)
        assert saved.loc["Klas A", "Jongens"] == 1
        assert saved.loc["Klas A", "Meisjes"] == 2
        assert saved.loc["Nieuwe groep 1", "Jongens"] == 0
        assert saved.loc["Nieuwe groep 1", "Meisjes"] == 0

    def test_post_persists_restore_state(self, client, tmp_path):
        """POST writes groups_to_state.json capturing ticks, disabled and new groups."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
        )
        client.post(
            "/groups_to",
            data={
                # Klas B switched off (absent from 'group'); a new empty group added.
                "group": ["Klas A", "Nieuwe groep 1"],
                "group_students[Klas A]": ["1"],
            },
        )
        state = json.loads((proc_dir / "groups_to_state.json").read_text("utf-8"))
        assert state["original_groups"]["Klas A"]["checked_indices"] == [1]
        assert state["disabled_groups"] == ["Klas B"]
        assert state["new_groups"] == ["Nieuwe groep 1"]

    def test_get_restores_state_into_the_form(self, client, tmp_path):
        """GET after a save pre-ticks the right boxes and marks a disabled group."""
        proc_dir = setup_process(client, tmp_path)
        write_groups_to_json(
            proc_dir,
            {
                "Klas A": make_students("Jongen", "Meisje"),
                "Klas B": make_students("Jongen"),
            },
        )
        (proc_dir / "groups_to_state.json").write_text(
            json.dumps(
                {
                    "original_groups": {"Klas A": {"checked_indices": [1]}},
                    "disabled_groups": ["Klas B"],
                    "new_groups": ["Nieuwe groep 1"],
                }
            ),
            encoding="utf-8",
        )
        html = client.get("/groups_to").data.decode("utf-8")
        checkboxes = re.findall(r'<input type="checkbox".*?>', html, re.DOTALL)
        ticked = [c for c in checkboxes if "checked" in c]
        # Exactly one box is ticked: the second student of Klas A (index 1).
        assert len(ticked) == 1
        assert 'value="1"' in ticked[0]
        assert "group-disabled" in html  # Klas B comes in switched off
        assert "Nieuwe groep 1" in html  # restored new group is rendered


class TestGroupsToRedistribute:
    """Tests for /groups_to in redistribute mode (auto-passthrough, no page shown)."""

    def _setup(self, client, tmp_path, groups=("3A", "3B")):
        proc_dir = setup_process(client, tmp_path)
        (proc_dir / "mode.json").write_text('{"mode":"redistribute"}', encoding="utf-8")
        write_groups_to_json(proc_dir, {g: [] for g in groups})
        return proc_dir

    def test_get_auto_redirects_and_writes_groups_xlsx(self, client, tmp_path):
        """GET /groups_to in redistribute mode auto-writes groups.xlsx and redirects."""
        proc_dir = self._setup(client, tmp_path)
        resp = client.get("/groups_to")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/preferences_form")
        df = pd.read_excel(proc_dir / "groups.xlsx", index_col=0)
        assert (df.loc[["3A", "3B"]] == 0).all().all()
        method = json.loads((proc_dir / "input_method.json").read_text("utf-8"))
        assert method["method"] == "form"

    def test_post_also_auto_redirects_to_preferences_form(self, client, tmp_path):
        """POST /groups_to in redistribute mode is the same: transparent passthrough."""
        self._setup(client, tmp_path)
        resp = client.post("/groups_to")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/preferences_form")
