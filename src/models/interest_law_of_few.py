"""
Interest-Based Law of the Few Model

Extension of Galeotti & Goyal (2010) that introduces multidimensional interest
vectors.  Nodes form connections based on both information acquisition and
interest similarity, leading to the emergence of *multiple* interest-specific
hubs rather than a single core-periphery structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from tqdm import tqdm
import random
from scipy.spatial.distance import cosine


class InterestBasedLawOfFewModel:
    """
    Network formation model with heterogeneous interest vectors.
    """

    def __init__(self, n=10, num_interests=5, c=1.0, k=0.5,
                 y_hat=1.0, interest_weight=0.5, seed=None):
        """
        Parameters
        ----------
        n : int
            Number of players.
        num_interests : int
            Dimensionality of interest vectors.
        c : float
            Information acquisition cost.
        k : float
            Base link-formation cost.
        y_hat : float
            Optimal isolated-player information level.
        interest_weight : float
            Weight (0-1) given to interest similarity in link cost.
        seed : int, optional
            Random seed.
        """
        self.n = n
        self.num_interests = num_interests
        self.c = c
        self.k = k
        self.y_hat = y_hat
        self.interest_weight = interest_weight

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate and normalise interest vectors
        self.interests = np.random.normal(0, 1, (self.n, self.num_interests))
        for i in range(self.n):
            norm = np.linalg.norm(self.interests[i])
            if norm > 0:
                self.interests[i] /= norm

        # Cosine-similarity matrix
        self.similarity_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.similarity_matrix[i, j] = 1 - cosine(self.interests[i], self.interests[j])

        # Initialise information levels
        self.x = np.zeros(self.n)
        num_initial = min(int(n * 0.2), 10)
        for i in np.random.choice(n, num_initial, replace=False):
            self.x[i] = np.random.uniform(0.3, 0.7)

        # Initialise adjacency matrix
        self.g = np.zeros((self.n, self.n))
        link_density = 0.05
        for i in range(self.n):
            probs = link_density * (1 + self.similarity_matrix[i]) / 2
            for j in range(self.n):
                if i != j and np.random.random() < probs[j]:
                    self.g[i, j] = 1

    # ------------------------------------------------------------------
    # Core model methods
    # ------------------------------------------------------------------
    def information_utility(self, y):
        return np.log(1 + y)

    def get_accessible_information(self, player):
        g_closure = np.maximum(self.g, self.g.T)
        return self.x[player] + sum(
            self.x[j] for j in range(self.n) if g_closure[player, j] == 1
        )

    def adjusted_link_cost(self, player, target):
        sim = self.similarity_matrix[player, target]
        return max(0.1, self.k * (1 - self.interest_weight * (sim + 1) / 2))

    def calculate_payoff(self, player):
        info = self.get_accessible_information(player)
        link_cost = sum(
            self.adjusted_link_cost(player, j)
            for j in range(self.n) if self.g[player, j] == 1
        )
        return self.information_utility(info) - self.c * self.x[player] - link_cost

    def best_response_x(self, player):
        g_closure = np.maximum(self.g, self.g.T)
        neighbor_info = sum(self.x[j] for j in range(self.n) if g_closure[player, j] == 1)
        return max(0, min(self.y_hat - neighbor_info, self.y_hat))

    def best_response_links(self, player):
        current_links = self.g[player].copy()
        original_x = self.x[player]

        # Baseline: no links
        self.g[player] = np.zeros(self.n)
        self.x[player] = self.best_response_x(player)
        best_payoff = self.calculate_payoff(player)
        best_links = np.zeros(self.n)

        # Try each individual link
        for j in range(self.n):
            if j == player or self.x[j] <= 0:
                continue
            temp = np.zeros(self.n)
            temp[j] = 1
            self.g[player] = temp
            self.x[player] = self.best_response_x(player)
            payoff = self.calculate_payoff(player)
            if payoff > best_payoff:
                best_payoff = payoff
                best_links = temp.copy()

        # Try top-k most similar
        top_similar = np.argsort(-self.similarity_matrix[player])[:5]
        for kk in range(1, 6):
            temp = np.zeros(self.n)
            for j in top_similar[:kk]:
                if j != player:
                    temp[j] = 1
            self.g[player] = temp
            self.x[player] = self.best_response_x(player)
            payoff = self.calculate_payoff(player)
            if payoff > best_payoff:
                best_payoff = payoff
                best_links = temp.copy()

        # Try top information providers
        top_info = np.argsort(-self.x)[:10]
        for kk in range(1, 4):
            for combo in self._get_combinations(top_info, kk):
                temp = np.zeros(self.n)
                for j in combo:
                    if j != player:
                        temp[j] = 1
                self.g[player] = temp
                self.x[player] = self.best_response_x(player)
                payoff = self.calculate_payoff(player)
                if payoff > best_payoff:
                    best_payoff = payoff
                    best_links = temp.copy()

        self.x[player] = original_x
        self.g[player] = current_links
        return best_links

    def _get_combinations(self, items, k):
        if k > len(items):
            return []
        if k == 1:
            return [[item] for item in items]
        result = []
        for i in range(len(items)):
            for combo in self._get_combinations(items[i + 1:], k - 1):
                result.append([items[i]] + combo)
        return result[:20]

    def update_player(self, player):
        old_x = self.x[player]
        old_links = self.g[player].copy()
        self.g[player] = self.best_response_links(player)
        self.x[player] = self.best_response_x(player)
        return abs(old_x - self.x[player]) > 1e-3 or not np.array_equal(old_links, self.g[player])

    def simulate_step(self):
        players = list(range(self.n))
        random.shuffle(players)
        return sum(1 for p in players if self.update_player(p))

    def simulate(self, max_steps=1000, convergence_threshold=0, check_window=5):
        """
        Run the simulation.

        Returns
        -------
        history : dict
        converged : bool
        steps_to_convergence : int
        """
        history = {
            "x": [self.x.copy()],
            "g": [self.g.copy()],
            "hub_count": [],
            "avg_degree": [],
            "total_info": [],
            "changes": [],
            "interest_clusters": [],
            "homophily": [],
        }

        converged = False
        steps_to_convergence = max_steps
        change_history = []

        for step in tqdm(range(max_steps)):
            changes = self.simulate_step()
            history["changes"].append(changes)
            change_history.append(changes)

            if step % 10 == 0 or step == max_steps - 1:
                history["x"].append(self.x.copy())
                history["g"].append(self.g.copy())

            hub_threshold = 0.3 * self.y_hat
            incoming = np.sum(self.g, axis=0)
            hub_count = sum(
                1 for i in range(self.n)
                if self.x[i] > hub_threshold and incoming[i] > 2
            )
            history["hub_count"].append(hub_count)
            history["avg_degree"].append(
                np.mean(np.sum(np.maximum(self.g, self.g.T), axis=1))
            )
            history["total_info"].append(np.sum(self.x))

            if step % 20 == 0:
                history["interest_clusters"].append(self._identify_interest_clusters())
                history["homophily"].append(self._calculate_homophily())

            if step >= 100 and convergence_threshold > 0:
                if len(change_history) > check_window:
                    change_history = change_history[-check_window:]
                    if max(change_history) <= convergence_threshold:
                        converged = True
                        steps_to_convergence = step + 1
                        break

        if not converged or len(history["x"]) == 1 or step % 10 != 0:
            history["x"].append(self.x.copy())
            history["g"].append(self.g.copy())

        return history, converged, steps_to_convergence

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def _identify_interest_clusters(self, min_similarity=0.7):
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.similarity_matrix[i, j] > min_similarity:
                    G.add_edge(i, j)
        return list(nx.connected_components(G))

    def _calculate_homophily(self):
        total_sim, count = 0.0, 0
        for i in range(self.n):
            for j in range(self.n):
                if self.g[i, j] == 1:
                    total_sim += self.similarity_matrix[i, j]
                    count += 1
        return total_sim / count if count else 0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_network(self, ax=None, title=None, color_by_interest=True, interest_dim=0):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        G = nx.DiGraph()
        for i in range(self.n):
            G.add_node(i, info=self.x[i])
        for i in range(self.n):
            for j in range(self.n):
                if self.g[i, j] > 0:
                    G.add_edge(i, j)

        hub_threshold = 0.3 * self.y_hat
        incoming = np.sum(self.g, axis=0)
        hub_nodes = [i for i in range(self.n) if self.x[i] > hub_threshold and incoming[i] > 2]

        if hub_nodes:
            pos = {}
            for idx, node in enumerate(hub_nodes):
                angle = 2 * np.pi * idx / len(hub_nodes)
                pos[node] = np.array([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
            pos = nx.spring_layout(G, k=0.3, pos=pos, fixed=hub_nodes, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42, k=0.3)

        max_info = max(1e-6, self.y_hat)
        node_sizes = [200 + 1800 * (self.x[i] / max_info) for i in range(self.n)]

        if color_by_interest:
            vals = self.interests[:, interest_dim]
            lo, hi = vals.min(), vals.max()
            norm_vals = (vals - lo) / (hi - lo) if hi > lo else np.full(self.n, 0.5)
            cmap = cm.viridis
            node_colors = [cmap(v) for v in norm_vals]
            sm_obj = plt.cm.ScalarMappable(cmap=cmap)
            sm_obj.set_array([])
            plt.colorbar(sm_obj, ax=ax, label=f"Interest dimension {interest_dim}")
        else:
            node_colors = []
            for i in range(self.n):
                if self.x[i] >= hub_threshold and i in hub_nodes:
                    node_colors.append("gold")
                elif self.x[i] >= 0.1 * self.y_hat:
                    node_colors.append("limegreen")
                else:
                    node_colors.append("purple")

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax, arrowsize=15, connectionstyle="arc3,rad=0.1")
        labels = {i: str(i) for i in hub_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", ax=ax)

        if title:
            ax.set_title(title)
        ax.axis("off")
        return ax

    def plot_interest_distribution(self, top_hubs=5):
        hub_threshold = 0.3 * self.y_hat
        incoming = np.sum(self.g, axis=0)
        scores = [
            (i, self.x[i] * incoming[i])
            for i in range(self.n)
            if self.x[i] > hub_threshold
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [s[0] for s in scores[:top_hubs]]

        if not top_indices:
            print("No significant hubs found.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.8 / len(top_indices)
        for i, hub in enumerate(top_indices):
            positions = np.arange(self.num_interests) + i * width
            ax.bar(positions, self.interests[hub], width, label=f"Hub {hub}")

        ax.set_xlabel("Interest Dimension")
        ax.set_ylabel("Interest Value")
        ax.set_title("Interest Profiles of Top Hubs")
        ax.set_xticks(np.arange(self.num_interests) + width * (len(top_indices) - 1) / 2)
        ax.set_xticklabels([f"Topic {i}" for i in range(self.num_interests)])
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_metrics(self, history):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        axes[0].plot(history["hub_count"])
        axes[0].set_title("Hub Count")
        axes[0].grid(True)

        axes[1].plot(history["avg_degree"])
        axes[1].set_title("Average Degree")
        axes[1].grid(True)

        axes[2].plot(history["total_info"])
        axes[2].set_title("Total Information")
        axes[2].grid(True)

        axes[3].plot(history["changes"])
        axes[3].set_title("Strategy Changes per Step")
        axes[3].grid(True)

        if history["interest_clusters"]:
            counts = [len(c) for c in history["interest_clusters"]]
            x_pts = list(range(0, len(history["changes"]), 20))[: len(counts)]
            if x_pts:
                axes[4].plot(x_pts, counts)
                axes[4].set_title("Interest Clusters")
                axes[4].grid(True)

        if history["homophily"]:
            x_pts = list(range(0, len(history["changes"]), 20))[: len(history["homophily"])]
            if x_pts:
                axes[5].plot(x_pts, history["homophily"])
                axes[5].set_title("Network Homophily")
                axes[5].grid(True)

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------
def run_interest_simulation(
    n=100, num_interests=5, c=1.0, k=0.5, y_hat=1.0,
    interest_weight=0.5, max_steps=300, convergence_threshold=0,
    use_seed=None,
):
    """
    Run a simulation and return (model, history).
    """
    model = InterestBasedLawOfFewModel(
        n=n, num_interests=num_interests, c=c, k=k,
        y_hat=y_hat, interest_weight=interest_weight, seed=use_seed,
    )
    history, _, _ = model.simulate(
        max_steps=max_steps, convergence_threshold=convergence_threshold
    )
    return model, history
