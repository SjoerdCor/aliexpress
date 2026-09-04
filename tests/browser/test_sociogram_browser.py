"""Browser acceptance tests for the Cytoscape sociogram."""

import json
import math
import statistics
from urllib.parse import urlsplit

import pandas as pd
import pytest

from aliexpress.data import datareader
from aliexpress.data.preferences_data import PreferenceData
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
    preference_data = processor.to_preference_data()
    with app.app_context():
        save_voorkeuren(
            TEST_SCHOOLCODE,
            name,
            preference_data,
            source="excel",
        )
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name=name))
        flask_db.session.commit()

    page.goto(f"{live_server}/processes/select/{name}")
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{live_server}/sociogram")
    page.wait_for_function("() => window.sociogramReady === true")
    page.locator("#sociogram").scroll_into_view_if_needed()
    return preference_data


def _make_small_sociogram_process(live_server, tmp_path, page, name):
    """Create and open a process backed by the small real preference workbook."""
    _make_sociogram_process(live_server, tmp_path, page, name)


def _preference_data_from_edges(edges, student_count=30):
    """Build canonical data for a focused browser layout regression."""
    students = [f"s{index}" for index in range(student_count)]
    records = []
    index = []
    counters = {}
    for source, target, weight in edges:
        kind = "Graag met" if weight > 0 else "Liever niet met"
        counters[(source, kind)] = counters.get((source, kind), 0) + 1
        index.append((source, kind, float(counters[(source, kind)])))
        records.append({"Waarde": target, "Gewicht": float(weight)})
    preferences = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(index, names=["Leerling", "TypeWens", "Nr"]),
        columns=["Waarde", "Gewicht"],
    )
    return PreferenceData(
        preferences=preferences,
        students_info={student: {} for student in students},
        student_display={student: f"Leerling {student[1:]}" for student in students},
        stamgroep_display={},
        input_sheet=pd.DataFrame(),
    )


def _make_star_layout_process(live_server, tmp_path, page, name):
    """Open 30 students where one has one positive and one negative preference."""
    proc = tmp_path / TEST_SCHOOLCODE / name
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    pd.DataFrame(
        {"Jongens": [0], "Meisjes": [0]},
        index=pd.Index(["Klas A"], name="Groepen"),
    ).to_excel(proc / "groups.xlsx")
    preference_data = _preference_data_from_edges([("s0", "s1", 1), ("s0", "s2", -1)])
    with app.app_context():
        save_voorkeuren(TEST_SCHOOLCODE, name, preference_data, source="form")
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name=name))
        flask_db.session.commit()

    page.goto(f"{live_server}/processes/select/{name}")
    page.goto(f"{live_server}/sociogram")
    page.wait_for_function("() => window.sociogramReady === true")
    return preference_data


def _large_preference_data(student_count=150, preference_count=1000):
    """Build canonical data for the agreed sociogram scale smoke test."""
    students = [f"s{index}" for index in range(student_count)]
    edges = []
    for preference_number in range(preference_count):
        source_index, offset = divmod(preference_number, student_count - 1)
        source = students[source_index % student_count]
        target = students[(source_index + offset + 1) % student_count]
        weight = -1.0 if preference_number % 17 == 0 else 1.0
        edges.append((source, target, weight))
    return _preference_data_from_edges(edges, student_count)


def _make_data_sociogram_process(live_server, tmp_path, page, name, preference_data):
    """Persist canonical data and open its sociogram in the browser."""
    proc = tmp_path / TEST_SCHOOLCODE / name
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "relevant_students_and_groups.json").write_text(
        json.dumps({"candidates": [], "groups_from": []}), encoding="utf-8"
    )
    with app.app_context():
        save_voorkeuren(TEST_SCHOOLCODE, name, preference_data, source="form")
        flask_db.session.add(Process(school_id=TEST_SCHOOLCODE, name=name))
        flask_db.session.commit()

    page.goto(f"{live_server}/processes/select/{name}")
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{live_server}/sociogram")
    return preference_data


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


def _student_preference_records(preference_data):
    """Return visible student preferences in the same order as the browser snapshot."""
    students = set(preference_data.students_info)
    return [
        (index[0], row["Waarde"], float(row["Gewicht"]))
        for index, row in preference_data.preferences.iterrows()
        if index[0] in students and row["Waarde"] in students
    ]


def _layout_pair_categories(preference_data):
    """Derive the documented layout categories from canonical preference records."""
    students = list(preference_data.students_info)
    order = {student: position for position, student in enumerate(students)}
    pairs = {}
    for source, target, weight in _student_preference_records(preference_data):
        pair = tuple(sorted((source, target), key=order.__getitem__))
        pairs.setdefault(pair, []).append(weight)

    categories = {}
    for pair, weights in pairs.items():
        if any(weight < 0 for weight in weights):
            category = "negative"
        elif len(weights) == 2:
            category = "mutual_positive"
        else:
            category = "positive"
        categories[pair] = category
    return categories


def _properly_crosses(first, second):
    """Return whether two center-to-center segments have an interior crossing."""

    def cross(point_a, point_b, point_c):
        return (point_b["x"] - point_a["x"]) * (point_c["y"] - point_a["y"]) - (
            point_b["y"] - point_a["y"]
        ) * (point_c["x"] - point_a["x"])

    first_source, first_target = first
    second_source, second_target = second
    first_orientation = (
        cross(first_source, first_target, second_source),
        cross(first_source, first_target, second_target),
    )
    second_orientation = (
        cross(second_source, second_target, first_source),
        cross(second_source, second_target, first_target),
    )
    return (
        first_orientation[0] * first_orientation[1] < 0
        and second_orientation[0] * second_orientation[1] < 0
    )


def _overlapping_nodes(snapshot):
    """Return pairs whose rendered node bounding boxes overlap."""
    overlaps = []
    for index, first in enumerate(snapshot["nodes"]):
        for second in snapshot["nodes"][index + 1 :]:
            first_box = first["bounding_box"]
            second_box = second["bounding_box"]
            if max(first_box["x1"], second_box["x1"]) < min(
                first_box["x2"], second_box["x2"]
            ) and max(first_box["y1"], second_box["y1"]) < min(
                first_box["y2"], second_box["y2"]
            ):
                overlaps.append((first["id"], second["id"]))
    return overlaps


def _count_crossings(records, positions):
    """Count proper crossings between visible preference center-line proxies."""
    crossings = 0
    for index, first_record in enumerate(records):
        first_pair = first_record[:2]
        first_segment = (positions[first_pair[0]], positions[first_pair[1]])
        for second_index in range(index + 1, len(records)):
            second_record = records[second_index]
            second_pair = second_record[:2]
            if set(first_pair) & set(second_pair):
                continue
            second_segment = (positions[second_pair[0]], positions[second_pair[1]])
            crossings += _properly_crosses(first_segment, second_segment)
    return crossings


def _category_distances(categories, positions):
    """Return rendered center distances grouped by the documented relation category."""
    distances = {
        category: [] for category in ("mutual_positive", "positive", "negative")
    }
    for pair, category in categories.items():
        distances[category].append(
            math.hypot(
                positions[pair[1]]["x"] - positions[pair[0]]["x"],
                positions[pair[1]]["y"] - positions[pair[0]]["y"],
            )
        )
    return distances


def _reference_layout_metrics(preference_data, snapshot):
    """Measure the observable reference-layout geometry used by slice 5."""
    nodes = {node["id"]: node for node in snapshot["nodes"]}
    records = _student_preference_records(preference_data)
    categories = _layout_pair_categories(preference_data)
    positions = {student: nodes[student]["center"] for student in nodes}
    distances = _category_distances(categories, positions)

    return {
        "overlaps": _overlapping_nodes(snapshot),
        "crossings": _count_crossings(records, positions),
        "category_counts": {
            category: len(values) for category, values in distances.items()
        },
        "median_distances": {
            category: statistics.median(values)
            for category, values in distances.items()
        },
        "negative_longer_than_positive_median": sum(
            distance
            > statistics.median(distances["mutual_positive"] + distances["positive"])
            for distance in distances["negative"]
        ),
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
    assert page.locator(".stepper").count() == 0
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
    preference_data = _make_sociogram_process(
        live_server, tmp_path, page, "reference-run", "testdata/voorkeuren.xlsx"
    )
    snapshot = page.evaluate("window.sociogramSnapshot()")

    assert len(snapshot["nodes"]) == 43
    assert len(snapshot["preferences"]) == 102
    _assert_nodes_inside_canvas(page, snapshot)
    assert page.locator("#sociogram canvas").count() >= 1

    metrics = _reference_layout_metrics(preference_data, snapshot)
    assert metrics["category_counts"] == {
        "mutual_positive": 26,
        "positive": 41,
        "negative": 6,
    }
    assert not metrics["overlaps"], metrics
    assert metrics["crossings"] < 30, metrics
    median_distances = metrics["median_distances"]
    assert (
        median_distances["mutual_positive"]
        < median_distances["positive"]
        < median_distances["negative"]
    ), metrics
    assert metrics["negative_longer_than_positive_median"] >= 4, metrics


@pytest.mark.usefixtures("login")
def test_negative_preference_is_not_closer_than_positive_preference(
    live_server, tmp_path, page
):
    """A 30-student star keeps a negative relation farther away than a positive one."""
    preference_data = _make_star_layout_process(
        live_server, tmp_path, page, "relation-distance-run"
    )
    snapshot = page.evaluate("window.sociogramSnapshot()")

    distances = []
    for edge, geometry in zip(
        preference_data.preferences.itertuples(index=True), snapshot["preferences"]
    ):
        source = geometry["source"]
        target = geometry["target"]
        distances.append(
            (
                edge.Gewicht,
                math.hypot(target["x"] - source["x"], target["y"] - source["y"]),
            )
        )

    positive_distance = next(distance for weight, distance in distances if weight > 0)
    negative_distance = next(distance for weight, distance in distances if weight < 0)
    assert negative_distance > positive_distance


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

    page.get_by_role("button", name="Toon volledig overzicht").click()
    restored = page.evaluate("window.sociogramSnapshot()")
    assert restored["zoom"] == pytest.approx(initial["zoom"], abs=0.05)


@pytest.mark.usefixtures("login")
def test_sociogram_keeps_nodes_inside_canvas(live_server, tmp_path, page):
    """After layout, every rendered node stays within the visible canvas."""
    _make_small_sociogram_process(live_server, tmp_path, page, "boundsrun")
    snapshot = page.evaluate("window.sociogramSnapshot()")
    _assert_nodes_inside_canvas(page, snapshot)


@pytest.mark.usefixtures("login")
def test_sociogram_uses_only_same_origin_requests(live_server, tmp_path, page):
    """The sociogram page loads its code and assets without contacting another origin."""
    _make_small_sociogram_process(live_server, tmp_path, page, "offline-run")
    requests = []
    page.on("request", lambda request: requests.append(request.url))

    page.reload()
    page.wait_for_function("() => window.sociogramReady === true")

    origins = {f"{urlsplit(url).scheme}://{urlsplit(url).netloc}" for url in requests}
    assert origins == {live_server}


@pytest.mark.usefixtures("login")
def test_sociogram_reaches_ready_at_150_learners_and_1000_preferences(
    live_server, tmp_path, page
):
    """The Cytoscape view reaches ready for the agreed large input size."""
    preference_data = _large_preference_data()
    _make_data_sociogram_process(
        live_server, tmp_path, page, "large-run", preference_data
    )
    page.wait_for_function(
        "() => window.sociogramReady === true",
        timeout=60_000,
    )
    snapshot = page.evaluate("window.sociogramSnapshot()")

    assert len(snapshot["nodes"]) == 150
    assert len(snapshot["preferences"]) == 1000


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
