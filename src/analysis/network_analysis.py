"""
Network Analysis

Centrality calculations, core-node detection, and consensus analysis
on the YouTube subscription network.
"""

import json
import os

import networkx as nx

from ..utils.config import get_data_path


def _build_graph_from_processed(processed_data_path):
    """Create a NetworkX graph from the processed JSON data."""
    with open(processed_data_path) as f:
        data = json.load(f)

    G = nx.Graph()
    node_ids = data["node_ids"]
    for i, nid in enumerate(node_ids):
        G.add_node(nid,
                    community=data["node_communities"][i],
                    size=data["node_sizes"][i],
                    label=data["node_labels"][i])

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

    return G


def calculate_core_nodes(G, method="degree", top_percent=5):
    """
    Return the top *top_percent* % nodes by the chosen centrality method.

    Parameters
    ----------
    G : networkx.Graph
    method : str
        One of 'degree', 'betweenness', 'eigenvector', 'pagerank', 'kcore', 'weighted'.
    top_percent : float

    Returns
    -------
    list of (node_id, label, score)
    """
    if method == "degree":
        cent = nx.degree_centrality(G)
    elif method == "betweenness":
        cent = nx.betweenness_centrality(G)
    elif method == "eigenvector":
        try:
            cent = nx.eigenvector_centrality_numpy(G)
        except Exception:
            cent = nx.eigenvector_centrality(G, max_iter=1000)
    elif method == "pagerank":
        cent = nx.pagerank(G)
    elif method == "kcore":
        core_nums = nx.core_number(G)
        max_core = max(core_nums.values())
        k_core_set = {n for n, c in core_nums.items() if c == max_core}
        cent = {n: (1.0 if n in k_core_set else 0.0) for n in G.nodes()}
    elif method == "weighted":
        cent = {n: G.degree(n) * G.nodes[n]["size"] for n in G.nodes()}
    else:
        raise ValueError(f"Unknown method: {method}")

    num = max(1, int(len(G.nodes()) * top_percent / 100))
    top = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:num]
    return [(nid, G.nodes[nid]["label"], score) for nid, score in top]


def identify_core_nodes(processed_data_path=None, methods=None, top_percent=5):
    """
    Run multiple centrality methods and return results.

    Returns
    -------
    dict mapping method_name -> list of (node_id, label, score)
    """
    processed_data_path = processed_data_path or str(
        get_data_path("raw") / "processed_network_data.json"
    )
    if methods is None:
        methods = ["degree", "betweenness", "eigenvector", "pagerank", "kcore", "weighted"]

    G = _build_graph_from_processed(processed_data_path)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results = {}
    for m in methods:
        try:
            results[m] = calculate_core_nodes(G, m, top_percent)
        except Exception as e:
            print(f"Error with {m}: {e}")
            results[m] = []

    return results
