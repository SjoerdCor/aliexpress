"""Browser acceptance tests for the Cytoscape sociogram."""

import json

import pytest

from aliexpress.data import datareader
from aliexpress.web.extensions import db as flask_db
from aliexpress.web.models import Process
from aliexpress.web.process_files import save_voorkeuren
from app import app
from tests.browser.conftest import TEST_SCHOOLCODE


def _make_sociogram_process(
    live_server, tmp_path, page, name, preferences_path="testdata/voorkeuren_klein.xlsx"
):
    """Create and open a process backed by a real preference workbook."""
    proc = tmp_path / TEST_SCHOOLCODE / name
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    processor = datareader.VoorkeurenProcessor(preferences_path)
    processor.process(["blauw", "groen", "geel", "oranje"])
    with app.app_context():
        save_voorkeuren(
            TEST_SCHOOLCODE,
            name,
            processor.to_preference_data(),
            source="excel",
        )
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name=name))
        flask_db.session.commit()

    page.goto(f"{live_server}/processes/select/{name}")
    page.goto(f"{live_server}/sociogram")
    page.wait_for_function("() => window.sociogramReady === true")
    page.locator("#sociogram").scroll_into_view_if_needed()


def _make_small_sociogram_process(live_server, tmp_path, page, name):
    """Create and open a process backed by the small real preference workbook."""
    _make_sociogram_process(live_server, tmp_path, page, name)


def _assert_nodes_inside_canvas(page, snapshot):
    """Assert that rendered node boxes fit inside the Cytoscape canvas."""
    canvas_size = page.evaluate(
        """() => ({
            width: document.querySelector('#sociogram').clientWidth,
            height: document.querySelector('#sociogram').clientHeight
        })"""
    )
    outside = [
        node["id"]
        for node in snapshot["nodes"]
        if not (
            node["bounding_box"]["x1"] >= 0
            and node["bounding_box"]["y1"] >= 0
            and node["bounding_box"]["x2"] <= canvas_size["width"]
            and node["bounding_box"]["y2"] <= canvas_size["height"]
        )
    ]
    assert not outside, {
        "outside": outside,
        "zoom": snapshot["zoom"],
        "canvas": canvas_size,
    }


@pytest.mark.usefixtures("login")
def test_sociogram_renders_real_nodes_and_directed_preferences(
    live_server, tmp_path, page
):
    """A real process produces Cytoscape nodes and preference endpoints in the browser."""
    _make_small_sociogram_process(live_server, tmp_path, page, "sociogramrun")
    snapshot = page.evaluate("window.sociogramSnapshot()")

    assert len(snapshot["nodes"]) == 4
    assert len(snapshot["preferences"]) == 1
    assert page.locator("#sociogram canvas").count() >= 1
    legend = page.locator(".sociogram-legend")
    assert "Positief: doorgetrokken" in legend.inner_text()
    assert "Negatief: rood en gestreept" in legend.inner_text()
    assert "Lijndikte = absoluut gewicht" in legend.inner_text()
    instructions = page.locator(".instructions-box")
    assert "Elke cirkel is een leerling." in instructions.inner_text()
    assert "node" not in instructions.inner_text().lower()


@pytest.mark.usefixtures("login")
def test_sociogram_renders_reference_workbook(live_server, tmp_path, page):
    """The reference workbook renders its full social structure in the browser."""
    _make_sociogram_process(
        live_server, tmp_path, page, "reference-run", "testdata/voorkeuren.xlsx"
    )
    snapshot = page.evaluate("window.sociogramSnapshot()")

    assert len(snapshot["nodes"]) == 43
    assert len(snapshot["preferences"]) == 102
    _assert_nodes_inside_canvas(page, snapshot)
    assert page.locator("#sociogram canvas").count() >= 1


@pytest.mark.usefixtures("login")
def test_sociogram_limits_zoom_and_restores_overview(live_server, tmp_path, page):
    """Zoom stays bounded and the reset control returns to the initial view."""
    _make_small_sociogram_process(live_server, tmp_path, page, "zoomrun")
    graph = page.locator("#sociogram")
    graph_box = graph.bounding_box()
    initial = page.evaluate("window.sociogramSnapshot()")

    page.mouse.move(
        graph_box["x"] + graph_box["width"] / 2,
        graph_box["y"] + graph_box["height"] / 2,
    )
    page.mouse.wheel(0, 100000)
    page.wait_for_timeout(100)
    zoomed_in = page.evaluate("window.sociogramSnapshot()")
    assert zoomed_in["zoom"] <= 2.5

    page.mouse.wheel(0, -100000)
    page.wait_for_timeout(100)
    zoomed_out = page.evaluate("window.sociogramSnapshot()")
    assert zoomed_out["zoom"] >= 0.35

    page.get_by_role("button", name="Herstel overzicht").click()
    restored = page.evaluate("window.sociogramSnapshot()")
    assert restored["zoom"] == pytest.approx(initial["zoom"], abs=0.05)


@pytest.mark.usefixtures("login")
def test_sociogram_keeps_nodes_inside_canvas(live_server, tmp_path, page):
    """After layout, every rendered node stays within the visible canvas."""
    _make_small_sociogram_process(live_server, tmp_path, page, "boundsrun")
    snapshot = page.evaluate("window.sociogramSnapshot()")
    _assert_nodes_inside_canvas(page, snapshot)


@pytest.mark.usefixtures("login")
def test_sociogram_focuses_students_and_preferences_and_clears_on_background(
    live_server, tmp_path, page
):
    """A teacher can inspect one learner or arrow and return to the full view."""
    _make_small_sociogram_process(live_server, tmp_path, page, "focusrun")
    snapshot = page.evaluate("window.sociogramSnapshot()")
    graph_box = page.locator("#sociogram").bounding_box()

    maya = next(node for node in snapshot["nodes"] if node["id"] == "maya")
    page.mouse.click(
        graph_box["x"] + maya["center"]["x"],
        graph_box["y"] + maya["center"]["y"],
    )
    assert page.locator("#sociogram-detail-title").inner_text() == "Leerling: Maya"
    assert page.locator("#sociogram-detail-directions").inner_text() == (
        "1 inkomend / 0 uitgaand"
    )
    popup = page.locator("#sociogram-popup")
    assert popup.is_visible()
    stage_box = page.locator(".sociogram-stage").bounding_box()
    popup_box = popup.bounding_box()
    assert popup_box["x"] >= stage_box["x"]
    assert popup_box["y"] >= stage_box["y"]
    assert popup_box["x"] + popup_box["width"] <= stage_box["x"] + stage_box["width"]
    assert popup_box["y"] + popup_box["height"] <= stage_box["y"] + stage_box["height"]
    assert page.locator("#sociogram-detail-content").is_visible()

    graph_box = page.locator("#sociogram").bounding_box()
    preference = snapshot["preferences"][0]
    midpoint = {
        axis: (preference["source"][axis] + preference["target"][axis]) / 2
        for axis in ("x", "y")
    }
    page.mouse.click(
        graph_box["x"] + midpoint["x"],
        graph_box["y"] + midpoint["y"],
    )
    assert page.locator("#sociogram-detail-title").inner_text() == "Geselecteerde pijl"
    assert page.locator("#sociogram-detail-kind").inner_text() == "Negatieve voorkeur"
    assert page.locator("#sociogram-detail-source").inner_text() == "Harrie"
    assert page.locator("#sociogram-detail-target").inner_text() == "Maya"
    assert page.locator("#sociogram-detail-weight").inner_text() == "-1"

    graph_box = page.locator("#sociogram").bounding_box()
    background = {
        "x": graph_box["x"] + graph_box["width"] - 100,
        "y": graph_box["y"] + 20,
    }
    page.mouse.click(background["x"], background["y"])
    assert page.locator("#sociogram-detail-title").inner_text() == "Details"
    assert not popup.is_visible()


@pytest.mark.usefixtures("login")
def test_sociogram_closes_popup_with_escape(live_server, tmp_path, page):
    """Escape closes the focused sociogram popup."""
    _make_small_sociogram_process(live_server, tmp_path, page, "escaperun")
    snapshot = page.evaluate("window.sociogramSnapshot()")
    graph_box = page.locator("#sociogram").bounding_box()
    maya = next(node for node in snapshot["nodes"] if node["id"] == "maya")

    page.mouse.click(
        graph_box["x"] + maya["center"]["x"],
        graph_box["y"] + maya["center"]["y"],
    )
    popup = page.locator("#sociogram-popup")
    assert popup.is_visible()

    page.keyboard.press("Escape")
    assert not popup.is_visible()
