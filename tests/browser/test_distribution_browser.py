"""End-to-end browser test for the processing -> result -> download flow.

Drives the real app and the real solver on the small synthetic dataset: it starts the
distribution, lets the processing page poll /status until done, and checks that the result
tables render and the workbook downloads. This is the automated end-to-end check for Fase 1.
"""

import shutil
from pathlib import Path

import pytest

_INTEGRATION = Path(__file__).parents[1] / "integration"


def _make_process(live_server, tmp_path, page, name="browserrun"):
    """Create a process with ready-to-solve input files and select it in the browser."""
    proc = tmp_path / name
    proc.mkdir(exist_ok=True)
    shutil.copy(_INTEGRATION / "voorkeuren_small.xlsx", proc / "preferences.xlsx")
    shutil.copy(_INTEGRATION / "groepen_small.xlsx", proc / "groups.xlsx")
    page.goto(f"{live_server}/processes/select/{name}")  # sets the session process
    return proc


@pytest.mark.usefixtures("login")
def test_processing_to_result_to_download(live_server, tmp_path, page):
    """Starting a distribution lands on the result page and the workbook downloads."""
    proc = _make_process(live_server, tmp_path, page)

    page.goto(f"{live_server}/start_distribution")
    # The processing page polls /status and redirects here once the solve is done.
    page.wait_for_url("**/result", timeout=60000)

    # All five result tables are rendered as tabs.
    assert page.locator(".tab").count() == 5
    assert page.locator(".tab", has_text="Groepsindeling").is_visible()

    # The artifacts were written to the process dir before "done".
    assert (proc / "results.xlsx").exists()
    assert (proc / "result_tables.json").exists()

    with page.expect_download() as download_info:
        page.click("text=Download groepsindeling")
    assert download_info.value.suggested_filename == "results.xlsx"
