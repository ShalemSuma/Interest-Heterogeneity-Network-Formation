"""
Network Visualisation

Visualise the YouTube subscription network with community colouring,
subscriber-count sizing, and top-node labels.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx

from ..utils.config import get_data_path, get_results_path


def visualize_network_from_processed_data(processed_data_path=None, output_path=None, dpi=300):
    """
    Create a publication-quality network visualisation.
    """
    processed_data_path = processed_data_path or str(get_data_path("raw") / "processed_network_data.json")
    output_path = output_path or str(get_results_path("figures") / "youtube_network.png")

    with open(processed_data_path) as f:
        data = json.load(f)

    G = nx.Graph()
    node_ids = data["node_ids"]
    for i, nid in enumerate(node_ids):
        G.add_node(nid, community=data["node_communities"][i],
                    size=data["node_sizes"][i], label=data["node_labels"][i])

    edge_x, edge_y = data["edge_x"], data["edge_y"]
    for i in range(0, len(edge_x), 3):
        if i + 1 < len(edge_x) and edge_x[i] is not None and edge_x[i + 1] is not None:
            src = tgt = None
            for j, nid in enumerate(node_ids):
                if data["node_x"][j] == edge_x[i] and data["node_y"][j] == edge_y[i]:
                    src = nid
                if data["node_x"][j] == edge_x[i + 1] and data["node_y"][j] == edge_y[i + 1]:
                    tgt = nid
                if src and tgt:
                    break
            if src and tgt:
                G.add_edge(src, tgt)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Layout
    init_pos = {node_ids[i]: (data["node_x"][i], data["node_y"][i]) for i in range(len(node_ids))}
    pos = nx.spring_layout(G, pos=init_pos, k=0.35, iterations=120, scale=40, seed=42) if G.number_of_edges() else init_pos

    palette = ["#FF5252", "#7A4EB5", "#00CCFF", "#76FF03", "#FF9800",
               "#E040FB", "#1DE9B6", "#FFD600", "#448AFF", "#FF6D00", "#43A047"]
    communities = sorted({G.nodes[n]["community"] for n in G.nodes()})
    comm_colors = {c: palette[i % len(palette)] for i, c in enumerate(communities)}

    # Size scaling
    sizes_raw = [G.nodes[n]["size"] for n in G.nodes()]
    log_min = np.log1p(min(sizes_raw)) if min(sizes_raw) > 0 else 0
    log_max = np.log1p(max(sizes_raw)) if max(sizes_raw) > 0 else 1
    log_range = log_max - log_min

    display_sizes = {}
    for n in G.nodes():
        s = G.nodes[n]["size"]
        if log_range > 0 and s > 0:
            factor = ((np.log1p(s) - log_min) / log_range) ** 3
            display_sizes[n] = 10 + factor * 4990
        else:
            display_sizes[n] = 10

    # Draw
    fig = plt.figure(figsize=(30, 30), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        avg = (display_sizes[u] + display_sizes[v]) / (2 * 5000)
        plt.plot([x1, x2], [y1, y2], color="gray",
                 alpha=min(0.35, avg * 0.6 + 0.1), linewidth=0.8 + avg * 2, zorder=1)

    for comm in communities:
        nodes = [n for n in G.nodes() if G.nodes[n]["community"] == comm]
        if not nodes:
            continue
        plt.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
                    s=[display_sizes[n] for n in nodes],
                    c=comm_colors[comm], edgecolors="black", linewidths=0.8, alpha=0.9, zorder=2)

    threshold = np.percentile(sizes_raw, 95)
    for n in G.nodes():
        if G.nodes[n]["size"] >= threshold:
            x, y = pos[n]
            label = G.nodes[n]["label"][:12] + "..." if len(G.nodes[n]["label"]) > 15 else G.nodes[n]["label"]
            fs = min(8 + (display_sizes[n] / 5000) * 14, np.sqrt(display_sizes[n]) / 3.5)
            txt = plt.text(x, y, label, fontsize=fs, ha="center", va="center",
                           fontweight="bold", zorder=3)
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor="white", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")
