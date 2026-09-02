"""End-to-end browser test for the processing -> result -> download flow.

Drives the real app and the real solver on the small synthetic dataset: it starts the
distribution, lets the processing page poll /status until done, and checks that the result
tables render and the workbook downloads. This is the automated end-to-end check for Fase 1.
"""

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from playwright.sync_api import expect

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process, Run
from aliexpress.web.process_files import save_voorkeuren
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE
from tests.helpers import make_interim_view

_INTEGRATION = Path(__file__).parents[1] / "integration"


def _make_process(live_server, tmp_path, page, name="browserrun", running=True):
    """Create a process with ready-to-solve input files and select it in the browser.

    ``running=True`` (the default) also creates a Run row with status "running", so a
    plain ``page.goto(".../processing")`` lands straight on the live-progress view — most
    tests here stub ``/status`` themselves and only care about that view's JS. Pass
    ``running=False`` for the handful of tests that drive a real solve: they need the
    processing page's idle panel (no Run yet) so they can submit its "Start verdeling"
    form themselves.
    """
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
        proc_row = Process(school_id=TEST_SCHOOLCODE, name=name)
        flask_db.session.add(proc_row)
        flask_db.session.flush()
        if running:
            flask_db.session.add(Run(process_id=proc_row.id, status="running"))
        flask_db.session.commit()
    page.goto(f"{live_server}/processes/select/{name}")  # sets the session process
    return proc


def _start_distribution_from_idle_panel(live_server, page):
    """Navigate to the idle processing panel and click "Start verdeling".

    The idle panel prefills every balance-maxima field with the data-driven defaults, so
    submitting it unmodified reproduces the previous "start immediately" behaviour for
    tests that only care about the solve itself, not the balance-limits UI.
    """
    page.goto(f"{live_server}/processing")
    page.click('button:has-text("Start verdeling")')


@pytest.mark.usefixtures("login")
def test_balance_limits_can_be_changed_unlimited_and_submitted(
    live_server, tmp_path, page
):
    """The real processing form submits an edited cap and an unlimited cap.

    The small real instance keeps this focused browser acceptance test quick while
    still exercising the rendered form, its JavaScript, the POST route, and the
    persisted values consumed by the solve thread.
    """
    proc = _make_process(
        live_server, tmp_path, page, name="balance-limits", running=False
    )

    page.goto(f"{live_server}/processing")
    page.locator("details.instructions-box > summary").click()

    page.locator('input[name="maxima_max_clique"]').fill("7")
    unlimited_number = page.locator('input[name="maxima_max_clique_sex"]')
    unlimited_number.fill("6")
    unlimited = page.locator('input[name="maxima_max_clique_sex_unlimited"]')
    unlimited.check()

    expect(unlimited_number).to_be_disabled()
    expect(unlimited_number).to_have_value("")
    expect(unlimited_number).to_have_attribute("placeholder", "Geen maximum")
    unlimited.uncheck()
    expect(unlimited_number).to_be_enabled()
    expect(unlimited_number).to_have_value("6")
    unlimited.check()
    expect(unlimited_number).to_have_value("")
    assert page.locator('[title="Placeholder: uitleg volgt."]').count() == 0

    page.click('button:has-text("Start verdeling")')
    page.wait_for_url("**/result", timeout=60000)

    saved = json.loads((proc / "balance_limits.json").read_text("utf-8"))
    assert saved["max_clique"] == 7
    assert saved["max_clique_sex"] is None


@pytest.mark.usefixtures("login")
def test_balance_limit_without_number_stays_on_form(live_server, tmp_path, page):
    """A missing active cap is blocked by native form validation without a reload."""
    _make_process(live_server, tmp_path, page, name="balance-validation", running=False)

    page.goto(f"{live_server}/processing")
    page.locator("details.instructions-box > summary").click()
    number = page.locator('input[name="maxima_max_clique"]')
    number.fill("")
    assert number.evaluate("element => element.validity.valueMissing") is True

    page.click('button:has-text("Start verdeling")')
    page.wait_for_timeout(250)

    assert page.url == f"{live_server}/processing"
    expect(number).to_be_visible()


@pytest.mark.usefixtures("login")
def test_processing_idle_links_back_to_not_together(live_server, tmp_path, page):
    """The idle processing page offers the previous wizard step."""
    _make_process(live_server, tmp_path, page, name="balance-navigation", running=False)

    page.goto(f"{live_server}/processing")
    back = page.locator("a.previous-step")
    expect(back).to_have_attribute("href", "/not_together")
    expect(back).to_contain_text("niet samen")

    back.click()
    page.wait_for_url(f"{live_server}/not_together")


@pytest.mark.usefixtures("login")
def test_processing_to_result_to_download(live_server, tmp_path, page):
    """Starting a distribution lands on the result page and the workbook downloads."""
    proc = _make_process(live_server, tmp_path, page, running=False)

    _start_distribution_from_idle_panel(live_server, page)
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
def test_completed_distribution_can_be_adjusted_and_run_again(
    live_server, tmp_path, page
):
    """A completed process reopens its saved limits and follows a second real run."""
    proc = _make_process(
        live_server, tmp_path, page, name="rerun-completed", running=False
    )

    _start_distribution_from_idle_panel(live_server, page)
    page.wait_for_url("**/result", timeout=60000)

    page.get_by_role("link", name="← Nog niet helemaal... opnieuw invoeren").click()
    page.wait_for_url(f"{live_server}/processing")

    download_link = page.get_by_role("link", name="Download huidige groepsindeling")
    expect(download_link).to_have_attribute("href", "/download")
    notice = page.locator(".recalculation-note")
    expect(notice).to_have_css("background-color", "rgb(247, 247, 247)")
    assert "button" not in (download_link.get_attribute("class") or "").split()
    with page.expect_download() as download_info:
        download_link.click()
    assert download_info.value.suggested_filename == "results.xlsx"
    assert page.url == f"{live_server}/processing"

    details = page.locator("details.instructions-box")
    assert details.evaluate("element => element.open") is False
    expect(page.get_by_role("button", name="Start nieuwe indeling")).to_be_visible()

    details.locator("summary").click()
    assert details.evaluate("element => element.open") is True
    clique_limit = page.locator('input[name="maxima_max_clique"]')
    saved_limit = int(clique_limit.input_value())
    loosened_limit = saved_limit + 1
    clique_limit.fill(str(loosened_limit))

    page.get_by_role("button", name="Start nieuwe indeling").click()
    page.wait_for_url("**/result", timeout=60000)

    saved = json.loads((proc / "balance_limits.json").read_text("utf-8"))
    assert saved["max_clique"] == loosened_limit


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
def test_processing_pending_status_shows_spinner(live_server, tmp_path, page):
    """The initial pending status is treated as active by the processing poll."""
    _make_process(live_server, tmp_path, page, name="pendingrun")

    page.route(
        "**/status",
        lambda route: route.fulfill(
            json={
                "status_studentdistribution": "pending",
                "steps": {
                    "floor": "pending",
                    "balance": "pending",
                    "satisfaction": "pending",
                },
            }
        ),
    )
    page.goto(f"{live_server}/processing")

    expect(page.locator(".loading-spinner")).to_be_visible()
    assert page.locator("a.previous-step").count() == 0


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
    expect(card).to_be_hidden()

    fake_status["sociogram_ready"] = True
    page.wait_for_timeout(1200)  # let the next 1 s poll pick up the updated stub

    expect(card).to_be_visible()
    link = card.locator("a")
    expect(link).to_have_attribute("href", "/sociogram")
    expect(link).to_have_attribute("target", "_blank")


@pytest.mark.usefixtures("login")
def test_processing_wait_section_groups_sociogram_and_interim(
    live_server, tmp_path, page
):
    """The sociogram button and interim result live in one "Terwijl je wacht" section.

    The section stays hidden (via the HTML `hidden` attribute) until the first wait
    activity is ready, so the heading never appears above an empty block. Stubs
    /status like ``test_processing_shows_sociogram_card_and_no_logs`` does.
    """
    _make_process(live_server, tmp_path, page, name="waitsectionrun")

    fake_status = {
        "status_studentdistribution": "running",
        "steps": {"floor": "busy", "balance": "pending", "satisfaction": "pending"},
        "sociogram_ready": False,
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    section = page.locator("#wait-activities")
    expect(section).to_be_hidden()
    assert page.locator("#wait-activities #sociogram-card").count() == 1
    assert page.locator("#wait-activities #interim-result").count() == 1

    # The spinner is the closing element of the progress block, so it must precede
    # the wait section in document order (DOCUMENT_POSITION_FOLLOWING = 4).
    position = page.evaluate(
        """() => document.querySelector('.loading-spinner')
            .compareDocumentPosition(document.getElementById('wait-activities'))"""
    )
    assert position & 4

    fake_status["sociogram_ready"] = True
    page.wait_for_timeout(1200)  # let the next 1 s poll pick up the updated stub

    expect(section).not_to_be_hidden()
    heading = page.locator(".wait-activities-heading")
    expect(heading).to_be_visible()
    expect(heading).to_have_text("Terwijl je wacht")


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

    # started_at 60s in the past clears the 45s reveal threshold, so the plateau
    # list is expected to render (see the gating in templates/processing.html).
    started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    fake_status = {
        "status_studentdistribution": "running",
        "started_at": started_at,
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
def test_processing_hides_plateaus_before_reveal_threshold(live_server, tmp_path, page):
    """Plateau updates stay hidden while the run is still young (< 45s elapsed).

    Same stub shape as test_processing_shows_plateaus_and_tiebreak, but started_at
    is "now" so the 45s reveal threshold has not been crossed yet: a short run
    should not flash the plateau list in its last seconds before redirect.
    """
    _make_process(live_server, tmp_path, page, name="plateaugatedrun")

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
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
    page.wait_for_timeout(1200)  # let a poll happen so hiding is a real assertion

    expect(page.locator("#plateaus li")).to_have_count(0)


@pytest.mark.usefixtures("login")
def test_processing_shows_plateaus_after_reveal_threshold(live_server, tmp_path, page):
    """Plateau updates render once elapsed time crosses the 45s reveal threshold."""
    _make_process(live_server, tmp_path, page, name="plateaurevealedrun")

    started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    fake_status = {
        "status_studentdistribution": "running",
        "started_at": started_at,
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


@pytest.mark.usefixtures("login")
def test_processing_hides_tiebreak_before_reveal_threshold(live_server, tmp_path, page):
    """The tie-break line stays hidden while the run is still young (< 45s elapsed)."""
    _make_process(live_server, tmp_path, page, name="tiebreakgatedrun")

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "tiebreak_busy": True,
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")
    page.wait_for_timeout(1200)  # let a poll happen so hiding is a real assertion

    expect(page.locator("#tiebreak-line")).to_be_hidden()


@pytest.mark.usefixtures("login")
def test_processing_shows_tiebreak_after_reveal_threshold(live_server, tmp_path, page):
    """The tie-break line renders once elapsed time crosses the 45s reveal threshold."""
    _make_process(live_server, tmp_path, page, name="tiebreakrevealedrun")

    started_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    fake_status = {
        "status_studentdistribution": "running",
        "started_at": started_at,
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "tiebreak_busy": True,
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    expect(page.locator("#tiebreak-line")).to_be_visible()


@pytest.mark.usefixtures("login")
def test_processing_shows_static_estimate_line_and_no_elapsed_clock(
    live_server, tmp_path, page
):
    """The always-visible static estimate line replaces a ticking elapsed-time clock.

    The page deliberately does not show elapsed time anywhere (a ticking clock draws
    attention to the wait) — only the calm, static estimate text. Here started_at="now"
    so no elapsed-time reveal could kick in either, isolating the static text.
    """
    _make_process(live_server, tmp_path, page, name="etalinerun")

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "stage_seconds": [],
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    expect(page.locator("#eta-line")).to_contain_text(
        "dit duurt meestal minder dan een minuut, soms enkele minuten"
    )
    # No ticking elapsed-time clock anywhere on the page: no element carries an
    # id/class suggestive of a timer/elapsed-seconds display. This is a robust proxy
    # for "we render no elapsed-time counter" without depending on exact wording.
    expect(page.locator("#elapsed, .elapsed, #timer, .timer")).to_have_count(0)


@pytest.mark.usefixtures("login")
def test_processing_shows_dynamic_estimate_text(live_server, tmp_path, page):
    """The estimate line switches to the dynamic ETA text once /status reports one.

    started_at="now" so the elapsed-time reveal cannot explain the change; the text
    change must come from data.estimate alone (see updateEstimate in processing.html).
    """
    _make_process(live_server, tmp_path, page, name="etadynamicrun")

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "estimate": {
            "phase": "c",
            "seconds": 120,
            "text": "naar verwachting nog ~2 minuten (ruwe schatting)",
        },
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    expect(page.locator("#eta-line")).to_have_text(
        "naar verwachting nog ~2 minuten (ruwe schatting)"
    )


@pytest.mark.usefixtures("login")
def test_processing_reveals_early_when_estimate_predicts_a_long_run(
    live_server, tmp_path, page
):
    """A high estimate reveals the rich components even before 45s have elapsed.

    started_at="now" (elapsed < 45s) but estimate.seconds=120 (> 45): the reveal is
    driven by the estimate, not just elapsed time (see revealed() in processing.html).
    """
    proc = _make_process(live_server, tmp_path, page, name="etaearlyrevealrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "plateaus": [{"min_pct": 62, "n_can_improve": 34}],
        "tiebreak_busy": True,
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
        "estimate": {
            "phase": "c",
            "seconds": 120,
            "text": "naar verwachting nog ~2 minuten (ruwe schatting)",
        },
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    expect(page.locator("#plateaus li")).to_have_count(1)
    expect(page.locator("#tiebreak-line")).to_be_visible()
    expect(page.locator("#interim-result .gi-card")).to_have_count(1)


@pytest.mark.usefixtures("login")
def test_processing_stays_gated_when_estimate_predicts_a_short_run(
    live_server, tmp_path, page
):
    """A low estimate does not trigger the early reveal (mirror of the case above).

    started_at="now" (elapsed < 45s) and estimate.seconds=20 (< 45): the rich
    components stay hidden, same as with no estimate at all.
    """
    _make_process(live_server, tmp_path, page, name="etanorevealrun")

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "done", "satisfaction": "busy"},
        "stage_seconds": [],
        "plateaus": [{"min_pct": 62, "n_can_improve": 34}],
        "tiebreak_busy": True,
        "estimate": {
            "phase": "b",
            "seconds": 20,
            "text": "naar verwachting nog ~20 seconden (ruwe schatting)",
        },
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")
    page.wait_for_timeout(1200)  # let a poll happen so hiding is a real assertion

    expect(page.locator("#plateaus li")).to_have_count(0)
    expect(page.locator("#tiebreak-line")).to_be_hidden()


@pytest.mark.usefixtures("login")
def test_processing_stepper_completes(live_server, tmp_path, page):
    """The processing page shows the three-step stepper and it all ends up 'done'."""
    proc = _make_process(live_server, tmp_path, page, name="stepperrun", running=False)

    _start_distribution_from_idle_panel(live_server, page)
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
    _make_process(live_server, tmp_path, page, name="cardsrun", running=False)

    _start_distribution_from_idle_panel(live_server, page)
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
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    cards = page.locator("#interim-result .gi-card")
    expect(cards).to_have_count(1)
    expect(page.locator(".interim-summary-title")).to_have_text("Voorlopige indeling")
    expect(page.locator(".interim-summary-subtext")).to_have_text(
        "Wordt nog verbeterd…"
    )


@pytest.mark.usefixtures("login")
def test_processing_hides_interim_result_before_reveal_threshold(
    live_server, tmp_path, page
):
    """The interim result stays hidden while the run is still young (< 45s elapsed).

    Same stub shape as test_processing_shows_interim_result, but started_at is
    "now" so the 45s reveal threshold has not been crossed yet.
    """
    proc = _make_process(live_server, tmp_path, page, name="interimgatedrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")
    page.wait_for_timeout(1200)  # let a poll happen so hiding is a real assertion

    expect(page.locator("#interim-result .gi-card")).to_have_count(0)


@pytest.mark.usefixtures("login")
def test_processing_shows_interim_result_after_reveal_threshold(
    live_server, tmp_path, page
):
    """The interim result renders once elapsed time crosses the 45s reveal threshold."""
    proc = _make_process(live_server, tmp_path, page, name="interimrevealedrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    cards = page.locator("#interim-result .gi-card")
    expect(cards).to_have_count(1)


@pytest.mark.usefixtures("login")
def test_processing_interim_result_chip_popover(live_server, tmp_path, page):
    """The shared gi-popover.js is loaded on the processing page too, so a chip in the
    dynamically-injected interim result is click-toggleable just like on /result."""
    proc = _make_process(live_server, tmp_path, page, name="interimpopoverrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    summary = page.locator("#interim-details > summary")
    summary.wait_for()  # waits for the poll that unhides #interim-details
    summary.click()  # open the collapsed <details> before the chip can be visible
    first_chip = page.locator("#interim-result .gi-chip").first
    first_chip.wait_for()

    first_chip.click()
    pop = first_chip.locator(".gi-pop")
    assert pop.is_visible()

    page.keyboard.press("Escape")
    assert not pop.is_visible()


@pytest.mark.usefixtures("login")
def test_processing_interim_result_collapsed_by_default_and_stays_open(
    live_server, tmp_path, page
):
    """The <details> starts closed and stays open across interim-result updates.

    A closed-by-default disclosure keeps the tentative distribution from competing
    with the solve-stepper for attention; once the user opens it, a later poll that
    replaces #interim-result's inner HTML must not reset the open state.
    """
    proc = _make_process(live_server, tmp_path, page, name="interimcollapsedrun")
    view = make_interim_view()
    (proc / "interim_result.json").write_text(
        json.dumps(asdict(view)), encoding="utf-8"
    )

    fake_status = {
        "status_studentdistribution": "running",
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
        "steps": {"floor": "done", "balance": "busy", "satisfaction": "pending"},
        "interim_result_updated_at": "2026-07-11T12:00:00+00:00",
    }
    page.route("**/status", lambda route: route.fulfill(json=fake_status))
    page.goto(f"{live_server}/processing")

    details = page.locator("#interim-details")
    first_card = page.locator("#interim-result .gi-card").first
    first_card.wait_for(state="attached")  # rendered but collapsed, so not visible
    expect(first_card).not_to_be_visible()
    assert details.evaluate("el => el.open") is False

    details.locator("> summary").click()
    expect(first_card).to_be_visible()
    assert details.evaluate("el => el.open") is True

    fake_status["interim_result_updated_at"] = "2026-07-11T12:00:05+00:00"
    page.wait_for_timeout(1200)  # let the next poll pick up the updated stub

    assert details.evaluate("el => el.open") is True
    expect(first_card).to_be_visible()
