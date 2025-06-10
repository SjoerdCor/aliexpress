import networkx as nx
from matplotlib import pyplot as plt


def plot_sociogram(preferences, students_info):
    g = nx.MultiDiGraph()
    popularity = (
        preferences.groupby("Waarde")["Gewicht"]
        .apply(lambda s: s.clip(-2, 2).sum())
        .reindex(students_info.keys())
        .fillna(0)
        .sort_values(ascending=False)
    )

    def get_node_size(child_popularity, min_popularity=-5, max_popularity=10):
        min_size = 25
        max_size = 375

        return min_size + (child_popularity - min_popularity) / (
            max_popularity - min_popularity
        ) * (max_size - min_size)

    node_sizes = [
        get_node_size(popularity[child], popularity.min(), popularity.max())
        for child in g.nodes()
    ]
    positiveprefs = (
        preferences.loc[
            lambda df: df["Waarde"].isin(students_info.keys()), ["Waarde", "Gewicht"]
        ]
        .reset_index("Leerling")
        .reset_index(drop=True)
    )

    for _, row in positiveprefs.iterrows():
        g.add_edge(row["Leerling"], row["Waarde"], weight=row["Gewicht"])

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    positive_edges = [
        (u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d["weight"] >= 0
    ]
    negative_edges = [
        (u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d["weight"] < 0
    ]

    pos = nx.spring_layout(g, k=1, seed=42)
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        g, pos, edgelist=positive_edges, edge_color="black", ax=ax, arrows=True
    )
    nx.draw_networkx_edges(
        g, pos, edgelist=negative_edges, edge_color="red", style="dashed", ax=ax
    )

    plt.axis("off")

    return ax
