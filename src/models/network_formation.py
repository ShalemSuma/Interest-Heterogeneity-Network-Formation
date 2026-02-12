"""
Noncooperative Network Formation Model (Bala & Goyal, 2000)

Agents strategically form directed links to access information from others.
Link formation is costly, information decays over distance, and some agents
are designated as high-value "influencers".  The simulation tests whether
the resulting network follows the "Law of the Few" pattern.
"""

import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class NoncooperativeNetworkFormation:
    """Bala & Goyal (2000) network formation with heterogeneous node values."""

    def __init__(
        self,
        n_nodes=100,
        decay_parameter=0.9,
        cost_parameter=0.1,
        heterogeneity_factor=3.0,
        strategic_update_prob=0.8,
        noise_level=0.05,
        influencer_ratio=0.1,
        seed=None,
    ):
        self.n_nodes = n_nodes
        self.decay_parameter = decay_parameter
        self.cost_parameter = cost_parameter
        self.heterogeneity_factor = heterogeneity_factor
        self.strategic_update_prob = strategic_update_prob
        self.noise_level = noise_level
        self.influencer_ratio = influencer_ratio

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.G = nx.DiGraph()
        self.timestep = 0
        self.history = []
        self.convergence_metrics = {
            "star_nodes_fraction": [],
            "clustering_coefficient": [],
            "average_path_length": [],
            "network_density": [],
            "gini_coefficient": [],
            "top_20_percent_share": [],
        }

        self.initialize_network()
        self.snapshot_network_state()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize_network(self):
        influencer_count = int(self.n_nodes * self.influencer_ratio)
        influencer_set = set(random.sample(range(self.n_nodes), influencer_count))

        for i in range(self.n_nodes):
            is_inf = i in influencer_set
            self.G.add_node(i, is_influencer=is_inf,
                            value=self.heterogeneity_factor if is_inf else 1.0)

        for i in range(self.n_nodes):
            for j in influencer_set:
                if i != j and random.random() < 0.2:
                    self.G.add_edge(i, j)
            non_inf = set(range(self.n_nodes)) - influencer_set - {i}
            for j in random.sample(list(non_inf), min(2, len(non_inf))):
                if random.random() < 0.05:
                    self.G.add_edge(i, j)

    # ------------------------------------------------------------------
    # Utility calculations
    # ------------------------------------------------------------------
    def calculate_utility(self, node, potential_links=None):
        temp_G = self.G.copy()
        if potential_links:
            for t in potential_links:
                if node != t and not temp_G.has_edge(node, t):
                    temp_G.add_edge(node, t)

        total_benefit = 0
        try:
            distances = nx.single_source_shortest_path_length(temp_G, node)
            for target, dist in distances.items():
                if target != node:
                    total_benefit += self.G.nodes[target]["value"] * (
                        self.decay_parameter ** (dist - 1)
                    )
        except nx.NetworkXError:
            pass

        total_cost = self.cost_parameter * temp_G.out_degree(node)
        utility = total_benefit - total_cost
        if self.noise_level > 0:
            utility += np.random.normal(0, self.noise_level)
        return utility

    def find_best_link_change(self, node):
        current = self.calculate_utility(node)
        best_action, best_delta = None, 0

        for t in range(self.n_nodes):
            if t != node and not self.G.has_edge(node, t):
                delta = self.calculate_utility(node, potential_links={t}) - current
                if delta > best_delta:
                    best_delta = delta
                    best_action = ("add", t)

        for t in list(self.G.successors(node)):
            self.G.remove_edge(node, t)
            delta = self.calculate_utility(node) - current
            self.G.add_edge(node, t)
            if delta > best_delta:
                best_delta = delta
                best_action = ("remove", t)

        return best_action, best_delta

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def run_step(self):
        self.timestep += 1
        nodes = random.sample(range(self.n_nodes), max(1, int(0.1 * self.n_nodes)))

        for node in nodes:
            if random.random() < self.strategic_update_prob:
                action, delta = self.find_best_link_change(node)
                if action and delta > 0:
                    kind, target = action
                    if kind == "add":
                        self.G.add_edge(node, target)
                    else:
                        self.G.remove_edge(node, target)
            else:
                if random.random() < 0.5:
                    candidates = [
                        i for i in range(self.n_nodes)
                        if i != node and not self.G.has_edge(node, i)
                    ]
                    if candidates:
                        self.G.add_edge(node, random.choice(candidates))
                else:
                    targets = list(self.G.successors(node))
                    if targets:
                        self.G.remove_edge(node, random.choice(targets))

        self.snapshot_network_state()

    def snapshot_network_state(self):
        G_und = self.G.to_undirected()
        in_degrees = [d for _, d in self.G.in_degree()]
        out_degrees = [d for _, d in self.G.out_degree()]

        density = nx.density(self.G) if self.G.number_of_nodes() else 0
        try:
            clustering = nx.average_clustering(G_und)
        except Exception:
            clustering = 0
        try:
            wcc = max(nx.weakly_connected_components(self.G), key=len)
            sg = self.G.subgraph(wcc)
            avg_path = nx.average_shortest_path_length(sg) if len(sg) > 1 else 0
        except Exception:
            avg_path = 0

        sorted_in = sorted(self.G.in_degree(), key=lambda x: x[1], reverse=True)
        node_ids, degrees = (zip(*sorted_in) if sorted_in else ([], []))

        # BUG FIX: convert to numpy arrays before arithmetic
        if len(degrees) > 1:
            sorted_deg = np.array(sorted(degrees), dtype=float)
            cum = np.cumsum(sorted_deg)
            cum_norm = cum / cum[-1]
            total = sorted_deg.sum()
            gini = 1 - 2 * np.sum(cum_norm - sorted_deg / (2 * total))
        else:
            gini = 0

        top_20_count = max(1, int(len(node_ids) * 0.2))
        top_20_links = sum(degrees[:top_20_count]) if degrees else 0
        total_links = sum(degrees) if degrees else 0
        top_20_share = (top_20_links / total_links * 100) if total_links else 0

        self.convergence_metrics["star_nodes_fraction"].append(
            sum(1 for n in self.G.nodes() if self.G.nodes[n]["value"] > 1) / self.n_nodes
        )
        self.convergence_metrics["clustering_coefficient"].append(clustering)
        self.convergence_metrics["average_path_length"].append(avg_path)
        self.convergence_metrics["network_density"].append(density)
        self.convergence_metrics["gini_coefficient"].append(gini)
        self.convergence_metrics["top_20_percent_share"].append(top_20_share)

        # Influencer stats
        inf_in, reg_in = [], []
        for n in self.G.nodes():
            (inf_in if self.G.nodes[n].get("is_influencer") else reg_in).append(
                self.G.in_degree(n)
            )

        self.history.append({
            "timestep": self.timestep,
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "avg_in_degree": np.mean(in_degrees) if in_degrees else 0,
            "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "density": density,
            "clustering_coefficient": clustering,
            "average_path_length": avg_path,
            "gini_coefficient": gini,
            "top_20_percent_share": top_20_share,
            "follows_law_of_few": top_20_share >= 80 and gini >= 0.6,
            "graph": self.G.copy(),
            "in_degree_distribution": Counter(in_degrees),
            "out_degree_distribution": Counter(out_degrees),
            "influencer_data": {
                "avg_influencer_in_degree": np.mean(inf_in) if inf_in else 0,
                "avg_regular_in_degree": np.mean(reg_in) if reg_in else 0,
            },
        })

    def run_simulation(self, n_steps=50):
        for _ in range(n_steps):
            self.run_step()
        return pd.DataFrame([
            {k: v for k, v in h.items()
             if k not in ("graph", "in_degree_distribution",
                          "out_degree_distribution", "influencer_data")}
            for h in self.history
        ])

    def has_converged(self, window_size=10, threshold=0.02):
        if len(self.history) < window_size * 2:
            return False, {"reason": "Not enough history"}
        results, is_converged = {}, True
        for name, values in self.convergence_metrics.items():
            if len(values) < window_size * 2:
                continue
            recent = values[-window_size:]
            avg = np.mean(recent)
            cv = np.std(recent) / avg if avg else 0
            results[f"{name}_cv"] = cv
            ok = cv < threshold
            results[f"{name}_converged"] = ok
            is_converged = is_converged and ok
        return is_converged, results

    # ------------------------------------------------------------------
    # Law of the Few analysis
    # ------------------------------------------------------------------
    def analyze_law_of_few(self, timestep=None):
        G = self.history[timestep]["graph"] if timestep is not None and 0 <= timestep < len(self.history) else self.G
        in_degrees = [d for _, d in G.in_degree()]
        if not in_degrees:
            return {"status": "Empty graph", "follows_law_of_few": False}

        sorted_nodes = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
        node_ids, degrees = zip(*sorted_nodes)

        top_20_count = max(1, int(len(node_ids) * 0.2))
        top_20_links = sum(degrees[:top_20_count])
        total_links = sum(degrees)
        top_20_share = (top_20_links / total_links * 100) if total_links else 0

        sorted_deg = np.array(sorted(in_degrees), dtype=float)
        cum = np.cumsum(sorted_deg)
        cum_norm = cum / cum[-1] if cum[-1] > 0 else cum
        total = sorted_deg.sum()
        gini = 1 - 2 * np.sum(cum_norm - sorted_deg / (2 * total)) if total > 0 else 0

        return {
            "top_20_percent_share": top_20_share,
            "gini_coefficient": gini,
            "avg_in_degree": np.mean(in_degrees),
            "max_in_degree": max(in_degrees),
            "follows_law_of_few": top_20_share >= 80 and gini >= 0.6,
            "top_nodes": list(node_ids[:10]),
        }

    def run_comparative_analysis(self, parameter_variations, n_steps=50, seeds=None):
        """Run multiple simulations varying parameters."""
        if seeds is None:
            seeds = [random.randint(0, 10000) for _ in range(3)]

        results = []
        param_names = list(parameter_variations.keys())
        param_values = list(parameter_variations.values())

        for seed in seeds:
            for values in itertools.product(*param_values):
                params = dict(zip(param_names, values))
                sim = NoncooperativeNetworkFormation(
                    n_nodes=self.n_nodes,
                    decay_parameter=params.get("decay_parameter", self.decay_parameter),
                    cost_parameter=params.get("cost_parameter", self.cost_parameter),
                    heterogeneity_factor=params.get("heterogeneity_factor", self.heterogeneity_factor),
                    strategic_update_prob=params.get("strategic_update_prob", self.strategic_update_prob),
                    noise_level=params.get("noise_level", self.noise_level),
                    influencer_ratio=params.get("influencer_ratio", self.influencer_ratio),
                    seed=seed,
                )
                sim.run_simulation(n_steps=n_steps)
                analysis = sim.analyze_law_of_few()
                row = {"seed": seed, **params, **{
                    k: v for k, v in analysis.items() if k != "top_nodes"
                }}
                results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_network(self, timestep=None, layout="spring", color_by="in_degree", size_by="in_degree"):
        plt.figure(figsize=(12, 10))
        G = self.history[timestep]["graph"] if (timestep is not None and 0 <= timestep < len(self.history)) else self.G
        if not G.number_of_nodes():
            plt.title("Empty Graph"); plt.axis("off"); return plt.gcf()

        G_viz = G.to_undirected()
        pos = nx.spring_layout(G_viz, k=0.3, iterations=50) if layout == "spring" else nx.circular_layout(G_viz)

        if color_by == "in_degree":
            node_colors = [G.in_degree(n) for n in G_viz.nodes()]
        elif color_by == "influencer":
            node_colors = ["red" if G.nodes[n].get("is_influencer") else "skyblue" for n in G_viz.nodes()]
        else:
            node_colors = "skyblue"

        if size_by == "in_degree":
            node_sizes = [100 + 20 * G.in_degree(n) for n in G_viz.nodes()]
        else:
            node_sizes = 100

        nodes = nx.draw_networkx_nodes(
            G_viz, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8,
            cmap=plt.cm.viridis if isinstance(node_colors, list) and not isinstance(node_colors[0], str) else None,
        )
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.5, alpha=0.6, arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.1")

        if isinstance(node_colors, list) and node_colors and not isinstance(node_colors[0], str):
            plt.colorbar(nodes)
        plt.title(f"Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        plt.axis("off"); plt.tight_layout()
        return plt.gcf()

    def plot_network_evolution(self, metrics=None):
        if not self.history:
            return None
        df = pd.DataFrame([
            {k: v for k, v in h.items()
             if k not in ("graph", "in_degree_distribution", "out_degree_distribution", "influencer_data")}
            for h in self.history
        ])
        if metrics is None:
            metrics = ["avg_in_degree", "gini_coefficient", "top_20_percent_share", "clustering_coefficient"]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for i, m in enumerate(metrics):
            if m in df.columns:
                axes[i].plot(df["timestep"], df[m], "o-")
                axes[i].set_ylabel(m); axes[i].grid(alpha=0.3)
                if m == "top_20_percent_share":
                    axes[i].axhline(80, color="r", ls="--", alpha=0.5, label="80% threshold")
                    axes[i].legend()
                elif m == "gini_coefficient":
                    axes[i].axhline(0.6, color="r", ls="--", alpha=0.5, label="0.6 threshold")
                    axes[i].legend()
        plt.tight_layout()
        return fig

    def export_results(self, filename_prefix="network_formation"):
        df = pd.DataFrame([
            {k: v for k, v in h.items()
             if k not in ("graph", "in_degree_distribution", "out_degree_distribution", "influencer_data")}
            for h in self.history
        ])
        df.to_csv(f"{filename_prefix}_metrics.csv", index=False)

        final_G = self.G.copy()
        for n, attrs in final_G.nodes(data=True):
            for k, v in attrs.items():
                if isinstance(v, bool):
                    final_G.nodes[n][k] = str(v)
        nx.write_graphml(final_G, f"{filename_prefix}_final_network.graphml")
