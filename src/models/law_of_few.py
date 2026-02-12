"""
Law of the Few Model (Galeotti & Goyal, 2010)

Implements the theoretical model where individuals decide between personally
acquiring information and forming strategic connections, leading to the
emergence of core-periphery network structures with information specialization.

Key insights:
1. In equilibrium, each player either acquires information personally or
   forms links with others who do, creating "hubs" and "spokes".
2. Aggregate information in equilibrium equals the optimal level for an
   isolated player.
3. The number of hubs is bounded and becomes a tiny fraction of the
   population as n grows.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import random


class LawOfFewModel:
    """
    Implementation of the Law of the Few model from Galeotti and Goyal (2010).
    """

    def __init__(self, n=10, c=1.0, k=0.5, y_hat=1.0, seed=None):
        """
        Parameters
        ----------
        n : int
            Number of players.
        c : float
            Cost of acquiring one unit of information.
        k : float
            Cost of forming a link with another player.
        y_hat : float
            Optimal information level for an isolated player
            (the amount that maximizes f(y) - cy).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.n = n
        self.c = c
        self.k = k
        self.y_hat = y_hat

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize information acquisition
        self.x = np.zeros(self.n)
        num_initial_info = min(int(n * 0.2), 10)
        initial_info_players = np.random.choice(n, num_initial_info, replace=False)
        for i in initial_info_players:
            self.x[i] = np.random.uniform(0.3, 0.7)

        # Initialize the network (adjacency matrix)
        self.g = np.zeros((self.n, self.n))
        link_density = 0.05
        for i in range(self.n):
            for j in range(self.n):
                if i != j and np.random.random() < link_density:
                    self.g[i, j] = 1

    def information_utility(self, y):
        """Utility function f(y) for information, concave in y."""
        return np.log(1 + y)

    def get_accessible_information(self, player):
        """Total information accessible by *player* (own + neighbours')."""
        g_closure = np.maximum(self.g, self.g.T)
        neighbor_info = sum(self.x[j] for j in range(self.n) if g_closure[player, j] == 1)
        return self.x[player] + neighbor_info

    def calculate_payoff(self, player):
        """Net payoff for a given player."""
        accessible_info = self.get_accessible_information(player)
        return (
            self.information_utility(accessible_info)
            - self.c * self.x[player]
            - self.k * np.sum(self.g[player])
        )

    def best_response_x(self, player):
        """Best-response information acquisition for *player*."""
        g_closure = np.maximum(self.g, self.g.T)
        neighbor_info = sum(self.x[j] for j in range(self.n) if g_closure[player, j] == 1)
        return max(0, min(self.y_hat - neighbor_info, self.y_hat))

    def best_response_links(self, player):
        """Best-response linking strategy for *player*."""
        current_links = self.g[player].copy()
        best_links = np.zeros(self.n)

        temp_links = np.zeros(self.n)
        self.g[player] = temp_links
        best_x_no_links = self.best_response_x(player)

        original_x = self.x[player]
        self.x[player] = best_x_no_links
        best_payoff = self.calculate_payoff(player)

        # Try individual links
        for j in range(self.n):
            if j != player and self.x[j] > 0:
                temp_links = np.zeros(self.n)
                temp_links[j] = 1
                self.g[player] = temp_links
                self.x[player] = self.best_response_x(player)
                payoff = self.calculate_payoff(player)
                if payoff > best_payoff:
                    best_payoff = payoff
                    best_links = temp_links.copy()

        # Try linking to top information providers
        sorted_info = np.argsort(-self.x)
        top_info = [i for i in sorted_info if i != player and self.x[i] > 0][:5]
        if top_info:
            temp_links = np.zeros(self.n)
            for j in top_info:
                temp_links[j] = 1
            self.g[player] = temp_links
            self.x[player] = self.best_response_x(player)
            payoff = self.calculate_payoff(player)
            if payoff > best_payoff:
                best_payoff = payoff
                best_links = temp_links.copy()

        self.x[player] = original_x
        self.g[player] = current_links
        return best_links

    def update_player(self, player):
        """Update strategy for a player. Returns True if strategy changed."""
        old_x = self.x[player]
        old_links = self.g[player].copy()

        self.g[player] = self.best_response_links(player)
        self.x[player] = self.best_response_x(player)

        return abs(old_x - self.x[player]) > 1e-3 or not np.array_equal(old_links, self.g[player])

    def simulate_step(self):
        """Single step: update all players in random order."""
        players = list(range(self.n))
        random.shuffle(players)
        return sum(1 for p in players if self.update_player(p))

    def simulate(self, max_steps=1000, convergence_threshold=0, check_window=5):
        """
        Run the simulation until convergence or *max_steps*.

        Returns
        -------
        history : dict
        converged : bool
        steps_to_convergence : int
        """
        history = {
            "x": [self.x.copy()],
            "g": [self.g.copy()],
            "core_size": [],
            "avg_degree": [],
            "total_info": [],
            "changes": [],
        }

        converged = False
        steps_to_convergence = max_steps
        change_history = []

        for step in tqdm(range(max_steps), desc="Simulating network evolution"):
            changes = self.simulate_step()
            history["changes"].append(changes)
            change_history.append(changes)

            if step % 10 == 0 or step == max_steps - 1:
                history["x"].append(self.x.copy())
                history["g"].append(self.g.copy())

            core_size = sum(1 for xi in self.x if xi > 0.1 * self.y_hat)
            history["core_size"].append(core_size)
            history["avg_degree"].append(
                np.mean(np.sum(np.maximum(self.g, self.g.T), axis=1))
            )
            history["total_info"].append(np.sum(self.x))

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
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_network(self, ax=None, title=None, use_circular_layout=False):
        """Visualise the current network."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        G = nx.DiGraph()
        for i in range(self.n):
            G.add_node(i, info=self.x[i])
        for i in range(self.n):
            for j in range(self.n):
                if self.g[i, j] > 0:
                    G.add_edge(i, j)

        hub_threshold = 0.5 * self.y_hat
        medium_threshold = 0.1 * self.y_hat
        hub_nodes = [n for n, d in G.nodes(data=True) if d["info"] >= hub_threshold]

        if use_circular_layout and hub_nodes:
            pos = {}
            if len(hub_nodes) == 1:
                pos[hub_nodes[0]] = np.array([0.5, 0.5])
            else:
                for i, node in enumerate(hub_nodes):
                    angle = 2 * np.pi * i / len(hub_nodes)
                    pos[node] = np.array([0.5 + 0.1 * np.cos(angle), 0.5 + 0.1 * np.sin(angle)])
            non_hub = [n for n in G.nodes() if n not in hub_nodes]
            for i, node in enumerate(non_hub):
                angle = 2 * np.pi * i / len(non_hub)
                pos[node] = np.array([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        else:
            pos = nx.spring_layout(G, seed=42, k=0.2)

        max_info = max(1e-6, self.y_hat)
        node_sizes = [300 + 2700 * (d["info"] / max_info) for _, d in G.nodes(data=True)]

        node_colors = []
        for _, d in G.nodes(data=True):
            if d["info"] >= hub_threshold:
                node_colors.append("gold")
            elif d["info"] >= medium_threshold:
                node_colors.append("limegreen")
            else:
                node_colors.append("purple")

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax, arrowsize=15, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, ax=ax)

        if title:
            ax.set_title(title)
        ax.axis("off")
        return ax

    def plot_metrics(self, history):
        """Plot simulation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        axes[0].plot(history["core_size"], lw=2.5, color="darkblue")
        axes[0].set_title("Core Size")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Players")
        axes[0].grid(alpha=0.3)

        axes[1].plot(history["avg_degree"], lw=2.5, color="darkgreen")
        axes[1].set_title("Average Degree")
        axes[1].set_xlabel("Step")
        axes[1].grid(alpha=0.3)

        axes[2].plot(history["total_info"], lw=2.5, color="darkred")
        axes[2].set_title("Total Information")
        axes[2].set_xlabel("Step")
        axes[2].axhline(y=self.y_hat, color="r", ls="--", alpha=0.7)
        axes[2].grid(alpha=0.3)

        axes[3].plot(history["changes"], lw=2.5, color="darkorange")
        axes[3].set_title("Strategy Changes per Step")
        axes[3].set_xlabel("Step")
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def run_simulation(
    n=200, c=1.0, k=0.5, y_hat=1.0,
    max_steps=300, convergence_threshold=3,
    seed=42, save_plots=False, output_dir=".",
):
    """
    Run a complete simulation and display results.
    """
    if k >= c * y_hat:
        print(f"Warning: k ({k}) >= c*y_hat ({c * y_hat}), expecting no links in equilibrium")

    print(f"Starting simulation with {n} players (c={c}, k={k}, y_hat={y_hat})...")
    model = LawOfFewModel(n=n, c=c, k=k, y_hat=y_hat, seed=seed)

    initial_x = model.x.copy()
    initial_g = model.g.copy()

    history, converged, steps = model.simulate(
        max_steps=max_steps, convergence_threshold=convergence_threshold
    )

    if converged:
        print(f"\nConverged after {steps} steps!")
    else:
        print(f"\nDid not converge within {max_steps} steps.")

    # Visualise before / after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    model.x, model.g = initial_x, initial_g
    model.plot_network(ax=ax1, title="Initial Network", use_circular_layout=False)
    model.x, model.g = history["x"][-1], history["g"][-1]
    model.plot_network(ax=ax2, title="Final Network", use_circular_layout=True)
    plt.tight_layout()

    if save_plots:
        import os
        plt.savefig(os.path.join(output_dir, "law_of_few_networks.png"), dpi=300)

    model.plot_metrics(history)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "law_of_few_metrics.png"), dpi=300)

    # Print summary
    final_x = history["x"][-1]
    final_g = history["g"][-1]
    threshold = 0.1 * model.y_hat
    core_players = [i for i, xi in enumerate(final_x) if xi > threshold]

    print(f"\nCore size: {len(core_players)}/{n} = {len(core_players) / n:.2%}")
    print(f"Total information: {np.sum(final_x):.4f} (target: {model.y_hat})")

    return model, history
