"""Create a Sociogram, displaying relations and popularity"""

# pylint: disable=too-many-locals,use-dict-literal  # file is pending a full rewrite

import base64
import io
import math
from dataclasses import dataclass

import matplotlib
import networkx as nx
import plotly.graph_objects as go

matplotlib.use("Agg")  # headless backend — must precede pyplot import
# pylint: disable=wrong-import-position  # matplotlib.use() must come first
from matplotlib import pyplot as plt

from .data import datareader


@dataclass(frozen=True)
class SociogramNode:
    """A student node for the browser sociogram."""

    id: str
    label: str
    received_preference_score: float
    size: float


@dataclass(frozen=True)
class PreferenceEdge:
    """One original, directed student-to-student preference."""

    id: str
    source: str
    target: str
    weight: float
    kind: str
    line_width: float = 2.0


@dataclass(frozen=True)
class LayoutRelation:
    """One undirected, derived relation used only to position two learners."""

    source: str
    target: str
    kind: str
    strength: float
    ideal_distance: float


@dataclass(frozen=True)
class SociogramView:
    """Flat, JSON-serialisable data needed to render the browser sociogram."""

    nodes: list[SociogramNode]
    preferences: list[PreferenceEdge]
    layout_relations: list[LayoutRelation]


def build_sociogram_view(preference_data) -> SociogramView:
    """Build student nodes and visible preference arrows from canonical preference data.

    Destination-group preferences are intentionally left out: only preferences whose
    target is another known student belong in the sociogram. Every retained input row
    becomes its own directed edge, including a negative preference.
    """
    student_keys = list(preference_data.students_info)
    student_set = set(student_keys)
    student_display = preference_data.student_display
    unique_name = preference_data.unique_name

    incoming_scores = {student: 0.0 for student in student_keys}
    edges = []
    pair_preferences = {}
    for position, (index, row) in enumerate(preference_data.preferences.iterrows()):
        source = index[0]
        target = row["Waarde"]
        if source not in student_set or target not in student_set:
            continue
        weight = float(row["Gewicht"])
        incoming_scores[target] += _clip_received_preference(weight)
        edges.append(
            PreferenceEdge(
                id=f"preference-{position}",
                source=source,
                target=target,
                weight=weight,
                kind="negative" if weight < 0 else "positive",
                line_width=_edge_line_width(weight),
            )
        )
        if source != target:
            pair = _ordered_pair(source, target, student_keys)
            pair_preferences.setdefault(pair, []).append((source, target, weight))

    nodes = [
        SociogramNode(
            id=student,
            label=unique_name.get(student, student_display.get(student, student)),
            received_preference_score=score,
            size=_node_size(score),
        )
        for student, score in incoming_scores.items()
    ]
    layout_relations = [
        _build_layout_relation(source, target, preferences)
        for (source, target), preferences in pair_preferences.items()
    ]
    return SociogramView(
        nodes=nodes, preferences=edges, layout_relations=layout_relations
    )


def _ordered_pair(source: str, target: str, student_keys: list[str]) -> tuple[str, str]:
    """Return a stable pair orientation based on the canonical learner order."""
    order = {student: index for index, student in enumerate(student_keys)}
    if order[source] <= order[target]:
        return source, target
    return target, source


def _build_layout_relation(source, target, preferences) -> LayoutRelation:
    """Combine the at-most-two directed preferences into one relation.

    The preference validator guarantees that a learner can mention another learner at
    most once, so the pair contains at most one weight per direction.
    """
    weights_by_source = {
        preference_source: weight for preference_source, _, weight in preferences
    }
    negative_weights = [
        abs(weight) for weight in weights_by_source.values() if weight < 0
    ]

    if negative_weights:
        kind = "negative"
        strength = max(negative_weights)
    elif source in weights_by_source and target in weights_by_source:
        kind = "mutual_positive"
        strength = (weights_by_source[source] + weights_by_source[target]) / 2
    else:
        kind = "positive"
        strength = next(iter(weights_by_source.values()))

    return LayoutRelation(
        source=source,
        target=target,
        kind=kind,
        strength=float(strength),
        ideal_distance=_ideal_distance(kind, strength),
    )


def _ideal_distance(kind: str, strength: float) -> float:
    """Map relation strength into a non-overlapping, bounded distance band."""
    bounded_strength = _bounded_strength(strength)
    bands = {
        "mutual_positive": (60.0, 90.0),
        "positive": (110.0, 145.0),
        "negative": (220.0, 290.0),
    }
    closest, furthest = bands[kind]
    if kind == "negative":
        return closest + (furthest - closest) * bounded_strength
    return furthest - (furthest - closest) * bounded_strength


def _bounded_strength(strength: float) -> float:
    """Compress non-negative strength to [0, 1): x/(x+1) grows but approaches 1."""
    return strength / (strength + 1.0)


def _node_size(received_preference_score: float) -> float:
    """Map a received-preference score to a restrained Cytoscape node diameter."""
    bounded_score = max(-2.0, min(10.0, received_preference_score))
    return 36.0 + 28.0 * (bounded_score + 2.0) / 12.0


def _edge_line_width(weight: float) -> float:
    """Map absolute preference strength to a bounded, non-linear line width."""
    return 2.0 + 6.0 * _bounded_strength(abs(weight))


def _clip_received_preference(weight: float) -> float:
    """Limit one contribution to the received score without changing the edge weight."""
    return max(-2.0, min(2.0, weight))


# pylint: enable=wrong-import-position


class SociogramMaker:
    """
    Create a Sociogram, displaying relations and popularity, based on the preferences-excel

    Parameters
    ----------
    fname: str
        The location of the file of the preference-excel
    groups: list
        the groups to which the students can be sent (used for validating the excel)
    """

    def __init__(self, fname, groups):
        self.fname = fname
        processor = datareader.VoorkeurenProcessor(fname)
        self.preferences = processor.process(groups)
        self.students_info = processor.get_students_meta_info()

    @classmethod
    def from_preference_data(cls, preference_data):
        """Build a SociogramMaker from a PreferenceData object (no Excel file needed).

        ``preference_data.preferences`` and ``preference_data.students_info`` are the
        same fields the constructor derives via VoorkeurenProcessor, so the sociogram
        graph is identical regardless of which input path produced the PreferenceData.
        """
        obj = cls.__new__(cls)
        obj.fname = None
        obj.preferences = preference_data.preferences
        obj.students_info = preference_data.students_info
        return obj

    @staticmethod
    def min_max_scaler(
        this_value, min_desired, max_desired, min_possible, max_possible
    ):
        """Scale value from range [min_possible, max_possible] to range [min_desired, max_desired]

        Useful for popularity -> node size, or calculating edge widths on weight, etc.
        """
        scale = max_desired - min_desired
        return min_desired + scale * (this_value - min_possible) / (
            max_possible - min_possible
        )

    def calculate_node_size(self, g):
        """Calculate the node size of nodes of g based on their popularity"""
        min_node_size = 25
        max_node_size = 375

        node_sizes = []
        for _, data in g.nodes(data=True):
            node_sizes.append(
                self.min_max_scaler(
                    data.get("popularity"),
                    min_node_size,
                    max_node_size,
                    -2,
                    10,
                )
            )

        return node_sizes

    def plot_sociogram(self):
        """Returns a matplolib Figure of the sociogram"""
        g = nx.MultiDiGraph()

        popularity = (
            self.preferences.groupby("Waarde")["Gewicht"]
            .apply(lambda s: s.clip(-2, 2).sum())
            .reindex(self.students_info.keys())
            .fillna(0)
        )
        for student in self.students_info:
            g.add_node(student, popularity=popularity[student])

        sociogram_preferences = (
            self.preferences.loc[
                lambda df: df["Waarde"].isin(self.students_info.keys()),
                ["Waarde", "Gewicht"],
            ]
            .reset_index("Leerling")
            .reset_index(drop=True)
        )
        for _, row in sociogram_preferences.iterrows():
            g.add_edge(row["Leerling"], row["Waarde"], weight=row["Gewicht"])

        node_sizes = self.calculate_node_size(g)

        fig, ax = plt.subplots(figsize=(6, 6))
        pos = nx.spring_layout(g, k=1, seed=42)
        nx.draw_networkx_nodes(g, pos, node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)

        self.draw_edges(g, ax, pos)

        plt.axis("off")
        return fig, g, pos

    def draw_edges(self, g, ax, pos):
        """Draw edges on graph g on ax using positions given"""
        positive_edges = [
            (u, v, k)
            for u, v, k, d in g.edges(keys=True, data=True)
            if d["weight"] >= 0
        ]
        negative_edges = [
            (u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d["weight"] < 0
        ]

        nx.draw_networkx_edges(
            g, pos, edgelist=positive_edges, edge_color="black", ax=ax, arrows=True
        )
        nx.draw_networkx_edges(
            g, pos, edgelist=negative_edges, edge_color="red", style="dashed", ax=ax
        )

    def get_as_b64_bytes(self):
        """Get sociogram as base-64 string

        This is useful for showing in HTML
        """
        fig = self.plot_sociogram()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


def networkx_to_plotly(g, pos):
    """Convert a NetworkX graph to a Plotly figure

    Parameters
    ----------
    g : networkx.Graph
        The graph to convert
    pos : dict
        A dictionary mapping nodes to their (x, y) positions
    """
    edge_traces = []
    seen_pairs = set()

    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 1)
        is_bidirectional = (v, u) in g.edges()

        width = abs(weight)
        color = "red" if weight < 0 else "#888"
        text = f"{u} → {v}<br>Gewicht: {weight:.2f}"

        # Arrowhead parameters
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx**2 + dy**2)
        offset_scale = 0.01 if is_bidirectional else 0
        ox = -dy / length * offset_scale
        oy = dx / length * offset_scale

        ux, uy = dx / length, dy / length

        # Shorten line so it doesn't overlap node marker
        shrink = 0.02
        x0s = x0 + ox + shrink * ux
        y0s = y0 + oy + shrink * uy
        x1s = x1 + ox - shrink * ux
        y1s = y1 + oy - shrink * uy

        edge_traces.append(
            go.Scatter(
                x=[x0s, x1s],
                y=[y0s, y1s],
                mode="lines+markers",
                line={"width": width, "color": color},
                marker={
                    "symbol": "triangle-up",
                    "size": 6,
                    "color": color,
                    "angleref": "previous",
                },
                hoverinfo="text",
                text=text,
                showlegend=False,
            )
        )

        seen_pairs.add((u, v))

    node_x = []
    node_y = []
    labels = []
    node_sizes = []

    def calc_node_size(popularity, default_size=10):
        if popularity > 0:
            return default_size + 1.5 * popularity
        return max(default_size + 3 * popularity, 1)

    for node, data in g.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))
        node_sizes.append(calc_node_size(data["popularity"]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hoverinfo="text",
        marker={"size": node_sizes, "color": "skyblue", "line_width": 2},
    )
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            width=600,
            height=600,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        ),
    )

    return fig
