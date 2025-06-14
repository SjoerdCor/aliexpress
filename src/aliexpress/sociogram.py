import base64
import io
import math

import networkx as nx
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from . import datareader


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

    @staticmethod
    def min_max_scaler(
        this_value, min_desired, max_desired, min_possible, max_possible
    ):
        """Scale this value, which is in range [min_possible, max_possible] to range [min_desired, max_desired]

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
        for child, data in g.nodes(data=True):
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
                line=dict(width=width, color=color),
                marker=dict(
                    symbol="triangle-up",
                    size=6,
                    color=color,
                    angleref="previous",
                ),
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
        marker=dict(size=node_sizes, color="skyblue", line_width=2),
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
