import base64
import io

import networkx as nx
from matplotlib import pyplot as plt

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
        popularity = (
            self.preferences.groupby("Waarde")["Gewicht"]
            .apply(lambda s: s.clip(-2, 2).sum())
            .reindex(self.students_info.keys())
            .fillna(0)
        )

        min_node_size = 25
        max_node_size = 375
        node_sizes = [
            self.min_max_scaler(
                popularity[child],
                min_node_size,
                max_node_size,
                popularity.min(),
                popularity.max(),
            )
            for child in g.nodes()
        ]

        return node_sizes

    def plot_sociogram(self):
        """Returns a matplolib Figure of the sociogram"""
        g = nx.MultiDiGraph()
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
        return fig

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
