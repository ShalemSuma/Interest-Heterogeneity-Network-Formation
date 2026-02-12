"""
Hub Prediction Model

Uses Random Forest and Gradient Boosting classifiers to predict:
1. Which nodes become hubs in the interest-based network.
2. When multiple hubs emerge (as opposed to a single hub).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from .interest_law_of_few import InterestBasedLawOfFewModel, run_interest_simulation


class HubPredictionModel:
    """Predict hub formation using node- and network-level features."""

    def __init__(self, n=100, num_interests=5):
        self.n = n
        self.num_interests = num_interests
        self.node_model = None
        self.multi_hub_model = None
        self.node_scaler = StandardScaler()
        self.network_scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_node_features(self, model, history, node):
        initial_g = history["g"][0]
        initial_x = history["x"][0]

        G = nx.DiGraph()
        G.add_nodes_from(range(model.n))
        for i in range(model.n):
            for j in range(model.n):
                if initial_g[i, j] > 0:
                    G.add_edge(i, j)

        try:
            betweenness = nx.betweenness_centrality(G)[node]
        except Exception:
            betweenness = 0
        try:
            closeness = nx.closeness_centrality(G)[node]
        except Exception:
            closeness = 0

        iv = model.interests[node]
        sims = [model.similarity_matrix[node, j] for j in range(model.n) if j != node]
        info_holders = [j for j in range(model.n) if initial_x[j] > 0.1]
        sim_to_info = (
            np.mean([model.similarity_matrix[node, j] for j in info_holders if j != node])
            if info_holders else 0
        )
        try:
            clustering = nx.clustering(nx.Graph(G), node)
        except Exception:
            clustering = 0

        neighbors = [j for j in range(model.n) if initial_g[node, j] > 0 or initial_g[j, node] > 0]
        neighbor_div = np.mean([np.var(model.interests[j]) for j in neighbors]) if neighbors else 0

        return {
            "initial_info": initial_x[node],
            "initial_out_degree": np.sum(initial_g[node]),
            "initial_in_degree": np.sum(initial_g[:, node]),
            "initial_degree": np.sum(initial_g[node]) + np.sum(initial_g[:, node]),
            "betweenness": betweenness,
            "closeness": closeness,
            "interest_norm": np.linalg.norm(iv),
            "interest_variance": np.var(iv),
            "interest_range": np.ptp(iv),
            "interest_extremity": np.mean(np.abs(iv - np.mean(model.interests, axis=0))),
            "avg_similarity": np.mean(sims),
            "max_similarity": np.max(sims),
            "min_similarity": np.min(sims),
            "similarity_variance": np.var(sims),
            "similarity_to_info_holders": sim_to_info,
            "clustering": clustering,
            "neighbor_interest_diversity": neighbor_div,
            "interest_polarization": np.linalg.norm(iv - np.mean(model.interests, axis=0)),
        }

    def extract_network_features(self, model, history):
        g0 = history["g"][0]
        x0 = history["x"][0]

        G = nx.DiGraph()
        G.add_nodes_from(range(model.n))
        for i in range(model.n):
            for j in range(model.n):
                if g0[i, j] > 0:
                    G.add_edge(i, j)

        dc = list(nx.degree_centrality(G).values())
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(nx.Graph(G))
            modularity = nx.algorithms.community.modularity(nx.Graph(G), comms)
        except Exception:
            modularity = 0

        sorted_info = np.sort(x0)
        cum = np.cumsum(sorted_info)
        total = sorted_info.sum()
        gini_info = 1 - 2 * np.sum(cum) / (model.n * total) if total > 0 else 0

        return {
            "initial_density": np.sum(g0) / (model.n * (model.n - 1)),
            "initial_avg_degree": np.mean(np.sum(np.maximum(g0, g0.T), axis=1)),
            "initial_total_info": np.sum(x0),
            "degree_centralization": np.max(dc) - np.mean(dc),
            "modularity": modularity,
            "interest_diversity": np.mean([np.var(model.interests[i]) for i in range(model.n)]),
            "avg_pairwise_similarity": np.mean(model.similarity_matrix),
            "interest_clusters": len(model._identify_interest_clusters()),
            "info_variance": np.var(x0),
            "info_holders": int(np.sum(x0 > 0.1)),
            "gini_info": gini_info,
            "initial_homophily": model._calculate_homophily(),
            "cost_ratio": model.k / model.c,
            "interest_weight": model.interest_weight,
        }

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------
    def run_single_simulation(self, sim_id, params=None):
        if params is None:
            params = {}
        model, history = run_interest_simulation(
            n=params.get("n", 100),
            num_interests=params.get("num_interests", 5),
            c=params.get("c", 1.0),
            k=params.get("k", 0.5),
            y_hat=params.get("y_hat", 1.0),
            interest_weight=params.get("interest_weight", 0.5),
            max_steps=params.get("max_steps", 200),
            use_seed=sim_id,
        )

        final_x = history["x"][-1]
        final_g = history["g"][-1]
        hub_threshold = 0.3 * model.y_hat
        incoming = np.sum(final_g, axis=0)
        hubs = [i for i in range(model.n) if final_x[i] > hub_threshold and incoming[i] > 2]

        node_data = []
        for node in range(model.n):
            feats = self.extract_node_features(model, history, node)
            feats["simulation_id"] = sim_id
            feats["node_id"] = node
            feats["became_hub"] = 1 if node in hubs else 0
            node_data.append(feats)

        net_feats = self.extract_network_features(model, history)
        net_feats["simulation_id"] = sim_id
        net_feats["multi_hub"] = 1 if len(hubs) > 1 else 0
        net_feats["hub_count"] = len(hubs)

        return node_data, net_feats

    def generate_training_data(self, num_simulations=300, vary_params=True, n_jobs=4):
        print(f"Generating training data from {num_simulations} simulations...")
        param_sets = []
        for sid in range(num_simulations):
            params = (
                {"k": np.random.uniform(0.3, 0.7), "interest_weight": np.random.uniform(0.2, 0.8)}
                if vary_params and sid > 0 else None
            )
            param_sets.append((sid, params))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_single_simulation)(sid, p) for sid, p in tqdm(param_sets)
        )

        node_data, network_data = [], []
        for nd, nf in results:
            node_data.extend(nd)
            network_data.append(nf)

        node_df = pd.DataFrame(node_data)
        network_df = pd.DataFrame(network_data)
        print(f"Hub rate: {node_df['became_hub'].mean():.2%}, Multi-hub rate: {network_df['multi_hub'].mean():.2%}")
        return node_df, network_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_node_model(self, node_df):
        X = node_df.drop(["simulation_id", "node_id", "became_hub"], axis=1)
        y = node_df["became_hub"]
        X_scaled = self.node_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        self.node_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.node_model.fit(X_train, y_train)

        acc = self.node_model.score(X_test, y_test)
        print(f"Hub prediction test accuracy: {acc:.4f}")
        print(classification_report(y_test, self.node_model.predict(X_test)))

        importance = pd.DataFrame({"Feature": X.columns, "Importance": self.node_model.feature_importances_})
        return acc, importance.sort_values("Importance", ascending=False)

    def train_multi_hub_model(self, network_df):
        X = network_df.drop(["simulation_id", "multi_hub", "hub_count"], axis=1)
        y = network_df["multi_hub"]
        X_scaled = self.network_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        self.multi_hub_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.multi_hub_model.fit(X_train, y_train)

        acc = self.multi_hub_model.score(X_test, y_test)
        print(f"Multi-hub prediction test accuracy: {acc:.4f}")
        print(classification_report(y_test, self.multi_hub_model.predict(X_test)))

        importance = pd.DataFrame({"Feature": X.columns, "Importance": self.multi_hub_model.feature_importances_})
        return acc, importance.sort_values("Importance", ascending=False)

    # ------------------------------------------------------------------
    # Prediction & visualisation
    # ------------------------------------------------------------------
    def predict_hub_formation(self, model, history):
        feats = [self.extract_node_features(model, history, n) for n in range(model.n)]
        X = pd.DataFrame(feats)
        return self.node_model.predict_proba(self.node_scaler.transform(X))[:, 1]

    def analyze_feature_importance(self, node_importance, network_importance):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        sns.barplot(x="Importance", y="Feature", data=node_importance.head(10), ax=axes[0])
        axes[0].set_title("Top Features for Hub Formation")
        sns.barplot(x="Importance", y="Feature", data=network_importance.head(10), ax=axes[1])
        axes[1].set_title("Top Features for Multi-Hub Formation")
        plt.tight_layout()
        return fig


def run_hub_prediction_demo(num_simulations=300, n_jobs=4):
    """End-to-end demonstration of the hub prediction pipeline."""
    hub_model = HubPredictionModel(n=100, num_interests=5)
    node_df, network_df = hub_model.generate_training_data(num_simulations=num_simulations, n_jobs=n_jobs)
    node_acc, node_imp = hub_model.train_node_model(node_df)
    net_acc, net_imp = hub_model.train_multi_hub_model(network_df)
    hub_model.analyze_feature_importance(node_imp, net_imp)
    print(f"\nNode accuracy: {node_acc:.4f}, Network accuracy: {net_acc:.4f}")
    return hub_model, node_df, network_df
