"""End-to-end browser test for the processing -> result -> download flow.

Drives the real app and the real solver on the small synthetic dataset: it starts the
distribution, lets the processing page poll /status until done, and checks that the result
tables render and the workbook downloads. This is the automated end-to-end check for Fase 1.
"""

import shutil
from pathlib import Path

import pytest

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.process_files import save_voorkeuren
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE

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
