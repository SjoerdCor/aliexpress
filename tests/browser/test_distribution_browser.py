"""End-to-end browser test for the processing -> result -> download flow.

Drives the real app and the real solver on the small synthetic dataset: it starts the
distribution, lets the processing page poll /status until done, and checks that the result
tables render and the workbook downloads. This is the automated end-to-end check for Fase 1.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
from playwright.sync_api import expect

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.process_files import save_voorkeuren
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE
from tests.helpers import make_interim_view

_INTEGRATION = Path(__file__).parents[1] / "integration"


def _make_process(live_server, tmp_path, page, name="browserrun"):
    """Create a process with ready-to-solve input files and select it in the browser."""
    proc = tmp_path / TEST_SCHOOLCODE / name
    proc.mkdir(parents=True, exist_ok=True)
    shutil.copy(_INTEGRATION / "groepen_small.xlsx", proc / "groups.xlsx")
    # The solver reads voorkeuren.json (the canonical format both input paths write);
    # build it from the synthetic workbook exactly like the Excel upload route does.
    groups_to, _ = datareader.read_groups_excel(str(proc / "groups.xlsx"))
    processor = datareader.VoorkeurenProcessor(_INTEGRATION / "voorkeuren_small.xlsx")
    processor.process(all_to_groups=list(groups_to.keys()))
    with app.app_context():
        save_voorkeuren(
            TEST_SCHOOLCODE, name, processor.to_preference_data(), source="excel"
        )
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name=name))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/{name}")  # sets the session process
    return proc


@pytest.mark.usefixtures("login")
def test_processing_to_result_to_download(live_server, tmp_path, page):
    """Starting a distribution lands on the result page and the workbook downloads."""
    proc = _make_process(live_server, tmp_path, page)

    page.goto(f"{live_server}/start_distribution")
    # The processing page polls /status and redirects here once the solve is done.
    page.wait_for_url("**/result", timeout=60000)

    # The three analysis tables are rendered as tabs.
    assert page.locator(".tab").count() == 3

    # The artifacts were written to the process dir before "done".
    assert (proc / "results.xlsx").exists()
    assert (proc / "result_tables.json").exists()
    assert (proc / "groepsindeling_view.json").exists()

    with page.expect_download() as download_info:
        page.click("text=Download groepsindeling")
    assert download_info.value.suggested_filename == "results.xlsx"


@pytest.mark.usefixtures("login")
def test_processing_shows_input_overview(live_server, tmp_path, page):
    """The processing page renders the input overview from the /status payload.

    The real solver on the small dataset finishes in about a second, so racing the DOM
    against the redirect is flaky (and the small fixture has no Jaarlaag to show anyway).
    Instead we stub /status with a fixed running payload so the JS rendering — including
    the jaarlagen line and the per-source-group counts — is asserted deterministically.
    The end-to-end path (real solver -> progress.json -> input_summary) is covered by
    ``test_processing_stepper_completes``.
    """
    _make_process(live_server, tmp_path, page, name="overviewrun")

    fake_status = {
        "status_studentdistribution": "running",
        "steps": {"floor": "busy", "balance": "pending", "satisfaction": "pending"},
        "stage_seconds": [],
        "input_summary": {
            "n_students": 87,
            "n_boys": 44,
            "n_girls": 43,
            "source_groups": {"Klas A": 22, "Klas B": 21, "Klas C": 22, "Klas D": 22},
            "n_target_groups": 4,
            "years": [6, 7],
        },
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    overview = page.locator("#input-overview")
    expect(overview).to_have_class("input-overview input-overview--visible")
    text = overview.inner_text()
    assert "87 leerlingen (44 jongens, 43 meisjes)" in text
    assert "jaarlagen 6 en 7" in text
    # Origin groups are listed with their counts; the target side drops "nieuwe".
    assert "Klas A (22)" in text
    assert "→ 4 groepen" in text
    assert "nieuwe" not in text


@pytest.mark.usefixtures("login")
def test_processing_shows_sociogram_card_and_no_logs(live_server, tmp_path, page):
    """The sociogram card appears once /status reports sociogram_ready; no raw log block.

    Stubs /status like ``test_processing_shows_input_overview`` does: the real
    sociogram thread's readiness races the redirect on the small dataset, so the
    deterministic assertion comes from a fixed stubbed payload rather than the real run.
    """
    _make_process(live_server, tmp_path, page, name="sociogramrun")

    fake_status = {
        "status_studentdistribution": "running",
        "steps": {"floor": "busy", "balance": "pending", "satisfaction": "pending"},
        "sociogram_ready": False,
    }
    # Route handler reads the mutable dict each request, so flipping sociogram_ready
    # below is picked up by the next 1 s poll without re-registering the route.
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    # No raw log stream anymore.
    assert page.locator("#logs").count() == 0

    card = page.locator("#sociogram-card")
    expect(card).to_have_class("sociogram-card")

    fake_status["sociogram_ready"] = True
    page.wait_for_timeout(1200)  # let the next 1 s poll pick up the updated stub

    expect(card).to_have_class("sociogram-card sociogram-card--visible")
    link = card.locator("a")
    expect(link).to_have_attribute("href", "/sociogram")
    expect(link).to_have_attribute("target", "_blank")


@pytest.mark.usefixtures("login")
def test_processing_shows_plateaus_and_tiebreak(live_server, tmp_path, page):
    """The satisfaction step lists each completed plateau and the tie-break line.

    Stubs /status like test_processing_shows_input_overview does: the real solve on
    the small dataset finishes too fast to reliably observe an intermediate plateau
    list, so this asserts the JS rendering deterministically from a fixed payload.
    The end-to-end path (real solver -> progress.json -> plateaus) is covered by
    test_stages_fire_in_order_with_nonnegative_durations at the integration level.
    """
    _make_process(live_server, tmp_path, page, name="plateaurun")

    fake_status = {
        "status_studentdistribution": "running",
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "plateaus": [
            {"min_pct": 62, "n_can_improve": 34},
            {"min_pct": 78, "n_can_improve": 5},
        ],
        "tiebreak_busy": True,
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    lines = page.locator("#plateaus li")
    expect(lines).to_have_count(2)
    expect(lines.nth(0)).to_have_text(
        "Minst tevreden leerling: nu 62% — 34 leerlingen kunnen nog omhoog"
    )
    expect(lines.nth(1)).to_have_text(
        "Minst tevreden leerling: nu 78% — 5 leerlingen kunnen nog omhoog"
    )
    expect(page.locator("#tiebreak-line")).to_be_visible()


@pytest.mark.usefixtures("login")
def test_processing_stepper_completes(live_server, tmp_path, page):
    """The processing page shows the three-step stepper and it all ends up 'done'."""
    proc = _make_process(live_server, tmp_path, page, name="stepperrun")

    page.goto(f"{live_server}/start_distribution")
    # The stepper renders all three steps immediately (they only ever refine in
    # place, never appear/disappear — see the "rustregels" in the plan).
    assert page.locator(".solve-step").count() == 3

    page.wait_for_url("**/result", timeout=60000)

    # progress.json was written throughout the solve and ended with every step done.
    with open(proc / "progress.json", encoding="utf-8") as fh:
        progress = json.load(fh)
    assert progress["steps"] == {
        "floor": "done",
        "balance": "done",
        "satisfaction": "done",
    }
    # ...and the real solver emitted the input overview (small fixture: 5 students,
    # 2 boys, 3 girls, origin groups A/D/B, 2 target groups, no Jaarlaag).
    assert progress["input_summary"] == {
        "n_students": 5,
        "n_boys": 2,
        "n_girls": 3,
        "source_groups": {"A": 3, "D": 1, "B": 1},
        "n_target_groups": 2,
        "years": [],
    }


@pytest.mark.usefixtures("login")
def test_result_group_cards_and_popover(live_server, tmp_path, page):
    """The structured group-card view renders: cards, click-popover, legend, overview."""
    _make_process(live_server, tmp_path, page, name="cardsrun")

    page.goto(f"{live_server}/start_distribution")
    page.wait_for_url("**/result", timeout=60000)

    # Group cards render with at least one chip.
    assert page.locator(".gi-card").count() >= 1
    first_chip = page.locator(".gi-chip").first
    first_chip.wait_for()

    # Popover opens on click and is visible.
    first_chip.click()
    pop = first_chip.locator(".gi-pop")
    assert pop.is_visible()

    # ...and closes on Escape.
    page.keyboard.press("Escape")
    assert not pop.is_visible()

    # ...opens again and closes on an outside click.
    first_chip.click()
    assert pop.is_visible()
    page.locator("h1").click()
    assert not pop.is_visible()

    # The legend is a collapsible <details> that starts open.
    legend = page.locator("details.gi-legend-details")
    assert legend.count() == 1
    assert legend.evaluate("el => el.open") is True

    # The klassenoverzicht is present with both balance columns.
    assert page.locator(".gi-baltable").count() == 1
    assert page.locator(".gi-baltable th", has_text="Grootteverschil").count() == 1
    assert page.locator(".gi-baltable th", has_text="Onbalans").count() == 1


@pytest.mark.usefixtures("login")
def test_processing_shows_interim_result(live_server, tmp_path, page):
    """The processing page fetches and renders /interim_result once /status reports a
    fresh interim_result_updated_at.

    Stubs only /status (like test_processing_shows_input_overview): the real
    /interim_result route is exercised for real, reading a real interim_result.json
    written directly into the process dir, so this covers the real route + real
    partial rendering the group cards. The end-to-end path (real solver ->
    ProgressWriter.interim_result_view -> interim_result.json) is covered at the unit
    level by test_progress_writer.py.
    """
    proc = _make_process(live_server, tmp_path, page, name="interimrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    cards = page.locator("#interim-result .gi-card")
    expect(cards).to_have_count(1)
    caption = page.locator("#interim-result .interim-result-caption")
    expect(caption).to_have_text("voorlopig — dit kan nog veranderen (ook verbeteren)")
