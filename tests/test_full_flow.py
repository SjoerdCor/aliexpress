"""Full-flow integration test: chained handoff through the real wizard routes (form path).

WHY THIS TEST EXISTS
--------------------
Every other test in the suite drives wizard steps in isolation with hand-built session /
file state.  Nothing was testing the CHAIN: that the output (files + session written by
step N) is actually what step N+1 reads.  That is exactly where handoff / wiring bugs hide
— a renamed session key, a file written under the wrong name, a redirect pointing at the
wrong URL.

This test walks the complete form path end-to-end using the real Flask test client:

    /processes/create
    → upload_edexml          (writes relevant_students_and_groups.json)
    → /roster                (writes roster.json)
    → /groups_to             (writes groups.xlsx + input_method.json)
    → /preferences_form      (writes voorkeuren.json + preferences_form_state.json)
    → /not_together          (writes not_together.json, redirects to /processing — the
                               idle panel with the "Start verdeling" button)
    → /start_distribution    (POST from the idle panel; redirects to /processing —
                               solver NOT awaited)

After each transition the test asserts:
1. The expected HTTP redirect / status code.
2. The file(s) the next step depends on now exist in the process directory.

After /not_together the process directory must be in a complete, ready-to-solve state:
    voorkeuren.json, groups.xlsx, not_together.json, roster.json, input_method.json.

The solver is not run (start_distribution is driven, but background threads are replaced
by no-ops so the test is fast and deterministic — we only want to know the kickoff is
accepted, not that the LP terminates).
"""

import dataclasses
import json
import pathlib
import xml.etree.ElementTree as ET
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd

from aliexpress.data.preferences_form import StudentEntry, build_preference_data
from aliexpress.solver._balance import BalanceMaxima
from tests.helpers import SCHOOL_ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Groups spanning jaargroeps 4 and 5.  EdexReader merges students on groepscode
# (the group "key" attribute), so every group key must appear in both the
# <groepen> section and referenced by <leerling><groep key="...">.
#
# get_candidates(df, jaargroep=4)  → students at jaargroep 4 (will be redistributed)
# get_groups_to(df,   jaargroep=4) → groups that have students at jaargroep 5
#
# Group naming format: "4-5 Naam (Leerkracht)" — the reader uses this as the
# display name.  A group spans jaargroeps 4 and 5, so it appears at both levels.
_EDEXML_GROUPS = [
    # Groups with jaargroep-4 students (current year, the ones being redistributed)
    {"key": "GRP_A_lj_4", "naam": "4-5 Alpacas (Juf Nora)", "jaargroep": "4"},
    {"key": "GRP_B_lj_4", "naam": "4-5 Beren (Meester Tim)", "jaargroep": "4"},
    # Same groups at jaargroep 5 — this is what get_groups_to looks for
    {"key": "GRP_A_lj_5", "naam": "4-5 Alpacas (Juf Nora)", "jaargroep": "5"},
    {"key": "GRP_B_lj_5", "naam": "4-5 Beren (Meester Tim)", "jaargroep": "5"},
    # Two fresh jaargroep-5-only groups: the actual TARGET groups the teacher will
    # choose to distribute students into via the /groups_to page.
    {"key": "GRP_C_lj_5", "naam": "5-6 Ceders (Juf Mia)", "jaargroep": "5"},
    {"key": "GRP_D_lj_5", "naam": "5-6 Dolfijnen (Meester Jo)", "jaargroep": "5"},
]


# Twelve students total.  geslacht 1=Jongen, 2=Meisje.
# jaargroep-4 students are the candidates.
# jaargroep-5 students give get_groups_to its non-empty set so groups_to != {}.
def _student(fields):
    """Unpack a compact (key, roepnaam, achternaam, geslacht, jaargroep, groep) tuple."""
    key, roepnaam, achternaam, geslacht, jaargroep, groep = fields
    return {
        "key": key,
        "roepnaam": roepnaam,
        "achternaam": achternaam,
        "geslacht": geslacht,
        "jaargroep": jaargroep,
        "groep": groep,
    }


_EDEXML_STUDENTS = [
    # jaargroep 4 → candidates (the students being redistributed)
    _student(("s001", "Anna", "Berg", "2", "4", "GRP_A_lj_4")),
    _student(("s002", "Bram", "Dijk", "1", "4", "GRP_A_lj_4")),
    _student(("s003", "Clara", "Groot", "2", "4", "GRP_A_lj_4")),
    _student(("s004", "Daan", "Hoek", "1", "4", "GRP_A_lj_4")),
    _student(("s005", "Emma", "Jansen", "2", "4", "GRP_B_lj_4")),
    _student(("s006", "Finn", "Kuiper", "1", "4", "GRP_B_lj_4")),
    _student(("s007", "Gina", "Laan", "2", "4", "GRP_B_lj_4")),
    _student(("s008", "Hugo", "Mulder", "1", "4", "GRP_B_lj_4")),
    # jaargroep 5 → current occupants of the target groups; makes get_groups_to
    # return a non-empty dict so groups_to != {} after the EDEXML upload.
    _student(("s009", "Iris", "Naald", "2", "5", "GRP_A_lj_5")),
    _student(("s010", "Jesse", "Otter", "1", "5", "GRP_B_lj_5")),
    _student(("s011", "Kim", "Prins", "2", "5", "GRP_C_lj_5")),
    _student(("s012", "Lars", "Roos", "1", "5", "GRP_D_lj_5")),
]


def _build_minimal_edexml() -> bytes:
    """Build a minimal but valid EDEXML blob that EdexReader can parse.

    Contains two origin groups (Alpacas, Beren) each with jaargroep-4 students
    that will become candidates when jaargroep=4 is selected.  Also includes
    jaargroep-3 students (verlengers) so the reader's ``blijft_in_groep`` logic
    works correctly.
    """
    root = ET.Element("EDEX")

    groepen_el = ET.SubElement(root, "groepen")
    for g in _EDEXML_GROUPS:
        groep_el = ET.SubElement(groepen_el, "groep", key=g["key"])
        ET.SubElement(groep_el, "naam").text = g["naam"]
        ET.SubElement(groep_el, "jaargroep").text = g["jaargroep"]

    leerlingen_el = ET.SubElement(root, "leerlingen")
    for s in _EDEXML_STUDENTS:
        # The EdexReader reads the student key from the <leerling key="..."> attribute.
        ll = ET.SubElement(leerlingen_el, "leerling", key=s["key"])
        ET.SubElement(ll, "roepnaam").text = s["roepnaam"]
        ET.SubElement(ll, "achternaam").text = s["achternaam"]
        ET.SubElement(ll, "geslacht").text = s["geslacht"]
        ET.SubElement(ll, "jaargroep").text = s["jaargroep"]
        ET.SubElement(ll, "groep", key=s["groep"])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _write_voorkeuren_payload(proc_dir: pathlib.Path, participants: list[dict]) -> None:
    """Write a valid voorkeuren.json for *participants* with two target groups.

    Used to seed the preferences_form POST so the form step can produce a valid
    voorkeuren.json without needing the browser-driven chip UI.
    """
    entries = [
        StudentEntry(
            student=f"{p['roepnaam']} {p['achternaam']}".strip(),
            sex=p["geslacht"],
            origin_group=p.get("groepsnaam", "A"),
            min_satisfaction=None,
            preferences=[],
            excluded_groups=[],
        )
        for p in participants
    ]
    all_to_groups = ["klas a", "klas b"]
    preference_data = build_preference_data(entries, all_to_groups)
    payload = json.loads(preference_data.to_json())
    payload["source"] = "form"
    (proc_dir / "voorkeuren.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _proc_dir(tmp_path: pathlib.Path, process_name: str) -> pathlib.Path:
    """Return the process directory path (mirrors storage.get_process_path)."""
    return tmp_path / SCHOOL_ID / process_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The chain test
# ---------------------------------------------------------------------------


class TestFullWizardFormFlow:  # pylint: disable=too-few-public-methods  # one test; many private _step_* helpers
    """One chained test through every wizard step (form path).

    Each helper method drives one step and asserts the concrete handoff:
    the redirect target AND the files / session the next step depends on.
    """

    PROCESS_NAME = "volledig-test-proces"
    JAARGROEP = 4

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _step_create_process(self, client, tmp_path):
        """POST /processes/create → session['process_id'] set, dir exists."""
        resp = client.post(
            "/processes/create",
            data={"process_name": self.PROCESS_NAME},
            follow_redirects=False,
        )
        # Should redirect to /upload_edexml (the first wizard step)
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /processes/create, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/upload_edexml"
        ), f"Expected redirect to /upload_edexml, got {resp.headers['Location']}"
        pdir = _proc_dir(tmp_path, self.PROCESS_NAME)
        assert pdir.exists(), "Process directory was not created"
        # Session must carry the process id so subsequent routes work
        with client.session_transaction() as sess:
            assert (
                sess.get("process_id") == self.PROCESS_NAME
            ), "Session key 'process_id' not set after process creation"
        return pdir

    def _step_upload_edexml(self, client, pdir):
        """POST /upload_edexml with a valid EDEXML → relevant_students_and_groups.json."""
        edexml_bytes = _build_minimal_edexml()
        resp = client.post(
            "/upload_edexml",
            data={
                "edexml": (BytesIO(edexml_bytes), "edex.xml"),
                "jaargroep": str(self.JAARGROEP),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /upload_edexml, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/roster"
        ), f"Expected redirect to /roster, got {resp.headers['Location']}"
        candidates_file = pdir / "relevant_students_and_groups.json"
        assert (
            candidates_file.exists()
        ), "relevant_students_and_groups.json not written after EDEXML upload"
        data = json.loads(candidates_file.read_text("utf-8"))
        assert data.get("candidates"), "Candidates list is empty after EDEXML upload"
        assert data.get("groups_to"), "groups_to dict is empty after EDEXML upload"
        return data

    def _step_roster(self, client, pdir, candidates):
        """POST /roster ticking all candidates → roster.json."""
        keys = [c["key"] for c in candidates]
        resp = client.post(
            "/roster",
            data={"gaat_over": keys},
            follow_redirects=False,
        )
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /roster, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/groups_to"
        ), f"Expected redirect to /groups_to, got {resp.headers['Location']}"
        roster_file = pdir / "roster.json"
        assert roster_file.exists(), "roster.json not written after /roster POST"
        roster = json.loads(roster_file.read_text("utf-8"))
        assert roster.get("participants"), "Participants list is empty in roster.json"
        # Note: input_method.json must NOT exist yet (it is written by /groups_to, ADR 0006)
        assert not (pdir / "input_method.json").exists(), (
            "input_method.json written too early "
            "(should be written by /groups_to, not /roster)"
        )

    def _step_groups_to(self, client, pdir, groups_to: dict):
        """POST /groups_to choosing the form path → groups.xlsx + input_method.json."""
        # We need at least 2 groups; use up to the first 2 from groups_to keys
        group_names = list(groups_to.keys())[:2]
        form_data = {"action": "form", "group": group_names}
        # Each group needs to contribute current-student counts; the students lists in
        # groups_to are the existing-group members.  Send all indices so every student
        # is counted — the form uses student-index checkboxes.
        for gname in group_names:
            students = groups_to.get(gname, [])
            indices = [str(i) for i in range(len(students))]
            for idx in indices:
                form_data.setdefault(f"group_students[{gname}]", []).append(idx)
        resp = client.post(
            "/groups_to",
            data=form_data,
            follow_redirects=False,
        )
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /groups_to, got {resp.status_code}"
        assert resp.headers["Location"].endswith("/preferences_form"), (
            f"Expected redirect to /preferences_form after 'form' action, "
            f"got {resp.headers['Location']}"
        )
        assert (
            pdir / "groups.xlsx"
        ).exists(), "groups.xlsx not written after /groups_to POST"
        assert (
            pdir / "input_method.json"
        ).exists(), "input_method.json not written after /groups_to POST"
        method = json.loads((pdir / "input_method.json").read_text("utf-8"))
        assert (
            method.get("method") == "form"
        ), f"input_method.json records wrong method: {method}"

    def _step_preferences_form(self, client, pdir):
        """POST /preferences_form (empty — no wishes) → voorkeuren.json."""
        # The route reads participants from roster.json and builds preference_data from
        # the submitted form.  An empty POST (no wishes) is valid; every student gets
        # an empty entry.  This is the minimal form that always succeeds.
        resp = client.post(
            "/preferences_form",
            data={},
            follow_redirects=False,
        )
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /preferences_form, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/not_together"
        ), f"Expected redirect to /not_together, got {resp.headers['Location']}"
        voorkeuren_file = pdir / "voorkeuren.json"
        assert (
            voorkeuren_file.exists()
        ), "voorkeuren.json not written after /preferences_form POST"
        payload = json.loads(voorkeuren_file.read_text("utf-8"))
        assert (
            payload.get("source") == "form"
        ), f"voorkeuren.json source tag should be 'form', got {payload.get('source')}"
        assert payload.get(
            "student_display"
        ), "voorkeuren.json student_display is empty — no students were serialised"

    def _step_not_together(self, client, pdir):
        """POST /not_together with zero rules → not_together.json."""
        resp = client.post(
            "/not_together",
            data={"n_rules": "0"},
            follow_redirects=False,
        )
        assert (
            resp.status_code == 302
        ), f"Expected 302 from /not_together, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/processing"
        ), f"Expected redirect to /processing, got {resp.headers['Location']}"
        nt_file = pdir / "not_together.json"
        assert (
            nt_file.exists()
        ), "not_together.json not written after /not_together POST"
        rules = json.loads(nt_file.read_text("utf-8"))
        assert isinstance(rules, list), "not_together.json should be a list"

    def _assert_ready_to_solve(self, pdir):
        """Assert the process directory holds a complete, consistent input set.

        These are the exact files run_solve_thread (in web/tasks.py) reads before
        calling the solver. A missing file here means start_distribution would
        immediately error out in production even though the wizard appeared to
        complete successfully.
        """
        required = {
            "voorkeuren.json": "preferences read by the solver thread",
            "groups.xlsx": "target groups (read by run_solve_thread via process_files.load_groups)",
            "not_together.json": "separation rules (loaded by start_distribution route)",
            "roster.json": "settled participant list (read by preferences_form on resume)",
            "input_method.json": "input method tag (controls back-link in not_together)",
        }
        for fname, role in required.items():
            assert (pdir / fname).exists(), (
                f"Required file '{fname}' missing from process directory. "
                f"Role: {role}"
            )

        # Cross-check: the students in voorkeuren.json must be a non-empty set
        voorkeuren = json.loads((pdir / "voorkeuren.json").read_text("utf-8"))
        assert voorkeuren.get(
            "student_display"
        ), "voorkeuren.json student_display is empty — solver has no students to distribute"

        # Cross-check: groups.xlsx must have at least 2 rows (solver requires >= 2 groups)
        groups_df = pd.read_excel(pdir / "groups.xlsx", index_col=0)
        assert (
            len(groups_df) >= 2
        ), f"groups.xlsx has only {len(groups_df)} group(s) — solver needs at least 2"

    def _step_start_distribution(self, client):
        """POST /start_distribution (the idle panel's "Start verdeling" button), with
        solver thread replaced by a no-op.

        Every balance-maxima family is submitted as Onbeperkt so the form parses without
        needing real numbers. We replace Thread so the solver does not actually run —
        the test only cares that the kickoff route accepts the request
        (i.e. all input files were found) and enters explicit processing watch mode.
        """
        noop_thread = MagicMock()
        noop_thread.start.return_value = None
        maxima_form = {
            f"maxima_{field.name}_unlimited": "on"
            for field in dataclasses.fields(BalanceMaxima)
        }

        with patch("aliexpress.web.routes.wizard.Thread", return_value=noop_thread):
            resp = client.post(
                "/start_distribution", data=maxima_form, follow_redirects=False
            )

        assert (
            resp.status_code == 302
        ), f"Expected 302 from /start_distribution, got {resp.status_code}"
        assert resp.headers["Location"].endswith(
            "/processing?watch=1"
        ), f"Expected redirect to processing watch mode, got {resp.headers['Location']}"

    # ------------------------------------------------------------------
    # The test
    # ------------------------------------------------------------------

    def test_form_path_handoff_chain(self, client, tmp_path):
        """Walk every wizard step end-to-end; assert each handoff in sequence.

        The solver is not run.  The test stops after start_distribution returns
        its redirect — the back-half (poll /status → /result → /download) is
        already covered by tests/browser/test_distribution_browser.py.
        """

        # Step 1: create process
        pdir = self._step_create_process(client, tmp_path)

        # Step 2: upload EDEXML → relevant_students_and_groups.json
        edexml_data = self._step_upload_edexml(client, pdir)
        candidates = edexml_data["candidates"]
        groups_to = edexml_data["groups_to"]

        # Step 3: roster → roster.json
        self._step_roster(client, pdir, candidates)

        # Step 4: groups_to → groups.xlsx + input_method.json
        self._step_groups_to(client, pdir, groups_to)

        # Step 5: preferences_form → voorkeuren.json
        self._step_preferences_form(client, pdir)

        # Step 6: not_together → not_together.json
        self._step_not_together(client, pdir)

        # Invariant: full ready-to-solve input set present
        self._assert_ready_to_solve(pdir)

        # Step 7: start_distribution (solver not run; just assert kickoff accepted)
        self._step_start_distribution(client)
