"""
Statistical Analysis

Logistic regression pipelines for:
1. Simulated data — predicting hub formation from interest features.
2. Real (YouTube) data — predicting core-node status.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import statsmodels.api as sm


# ------------------------------------------------------------------
# Simulated data analysis
# ------------------------------------------------------------------
def generate_simulation_data(n_nodes=100, n_simulations=20):
    """
    Generate synthetic data mimicking interest-based network simulations.

    Returns
    -------
    node_df : DataFrame  (per-node features + is_hub label)
    network_df : DataFrame  (per-network features + multi_hub label)
    """
    np.random.seed(42)
    node_data, network_data = [], []

    for sim_id in range(n_simulations):
        k = np.random.uniform(0.3, 0.7)
        iw = np.random.uniform(0.2, 0.8)
        hub_thr = 0.92 + 0.04 * np.random.normal()

        for nid in range(n_nodes):
            base = np.random.normal()
            im = 0.3 + 0.2 * np.random.normal() - 0.15 * base
            ig = 0.4 + 0.15 * np.random.normal() - 0.1 * base
            ik = 2 + np.random.normal() - 0.2 * base
            dn = 0.5 + 0.2 * np.random.normal() + 0.2 * base
            cn = 0.6 + 0.15 * np.random.normal() + 0.1 * base
            ip = 0.3 + 0.1 * np.random.normal() + 0.15 * base

            hub_score = (-1.8 * im - 1.4 * ig - 0.8 * ik
                         + 2.2 * dn + 1.9 * cn + 1.1 * ip - 0.5
                         + np.random.normal(0, 1.5))
            is_hub = int(1 / (1 + np.exp(-hub_score)) > hub_thr)

            node_data.append({
                "simulation_id": sim_id, "node_id": nid,
                "interest_mean": im, "interest_gini": ig,
                "interest_kurtosis": ik, "distance_to_network": dn,
                "cosine_to_network": cn, "interest_polarization": ip,
                "interest_entropy": 1.5 + 0.3 * np.random.normal() - 0.1 * ig,
                "interest_std": 0.2 + 0.05 * np.random.normal() + 0.1 * ig,
                "interest_max": im + (0.2 + 0.05 * np.random.normal() + 0.1 * ig) * np.random.normal(1.5, 0.3),
                "interest_skewness": np.random.normal(),
                "top3_concentration": 0.6 + 0.1 * np.random.normal() + 0.2 * ig,
                "dominant_interests": int(1 + np.random.poisson(2) * (1 - ig)),
                "degree": np.random.poisson(3),
                "is_hub": is_hub,
            })

        sim_nodes = node_data[-n_nodes:]
        hc = sum(n["is_hub"] for n in sim_nodes)
        mh_score = 2.4 * iw - 2.0 * k + np.random.normal()
        multi_hub = int(hc >= 2 and 1 / (1 + np.exp(-mh_score)) > 0.5)

        ns = {"simulation_id": sim_id, "multi_hub": multi_hub, "hub_count": hc, "k": k, "interest_weight": iw}
        for stat in ["interest_mean", "interest_gini", "interest_kurtosis", "interest_entropy", "interest_polarization"]:
            ns[f"avg_{stat}"] = np.mean([n[stat] for n in sim_nodes])
            ns[f"std_{stat}"] = np.std([n[stat] for n in sim_nodes])
        network_data.append(ns)

    return pd.DataFrame(node_data), pd.DataFrame(network_data)


def analyze_hub_formation(node_df, network_df):
    """
    Logistic regression analysis for hub and multi-hub formation.
    """
    # --- Single hub ---
    features = [
        "interest_mean", "interest_gini", "interest_kurtosis", "interest_entropy",
        "cosine_to_network", "distance_to_network", "interest_polarization",
        "top3_concentration", "interest_std", "interest_max",
    ]
    X = node_df[features]
    y = node_df["is_hub"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(Xtr, ytr)
    auc = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    print(f"Hub AUC: {auc:.4f}")
    print(classification_report(yte, lr.predict(Xte)))

    fi = pd.DataFrame({"Feature": features, "Importance": np.abs(lr.coef_[0]),
                        "Coefficient": lr.coef_[0]}).sort_values("Importance", ascending=False)

    # Statsmodels
    top = fi.head(7)["Feature"].tolist()
    Xsm = sm.add_constant(node_df[top])
    sm_model = sm.Logit(node_df["is_hub"], Xsm).fit(disp=0)
    print(sm_model.summary())

    single_results = {"feature_importance": fi, "sm_model": sm_model, "auc": auc}

    # --- Multi-hub ---
    mf = ["k", "interest_weight", "avg_interest_entropy", "avg_interest_mean",
          "avg_interest_gini", "avg_interest_kurtosis", "std_interest_mean",
          "std_interest_gini", "std_interest_kurtosis", "avg_interest_polarization"]
    Xn = network_df[mf]
    yn = network_df["multi_hub"]

    if yn.nunique() < 2 or yn.value_counts().min() < 3:
        print("Not enough multi-hub examples.")
        return single_results, None

    lr2 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr2.fit(StandardScaler().fit_transform(Xn), yn)
    fi2 = pd.DataFrame({"Feature": mf, "Importance": np.abs(lr2.coef_[0]),
                         "Coefficient": lr2.coef_[0]}).sort_values("Importance", ascending=False)

    return single_results, {"feature_importance": fi2}


# ------------------------------------------------------------------
# Real (YouTube) data analysis
# ------------------------------------------------------------------
def run_youtube_regression(df, features=None, target="core_nodes"):
    """
    Run logistic regression on real YouTube channel data.

    Parameters
    ----------
    df : DataFrame with feature columns and a binary target column.
    features : list of column names (default: interest + channel features).
    target : str

    Returns
    -------
    dict with auc, classification report, feature importance DataFrame.
    """
    if features is None:
        features = [
            "interest_mean", "interest_gini", "interest_kurtosis",
            "channel_age_days", "interest_std", "content_amount", "interest_entropy",
        ]

    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    report = classification_report(yte, model.predict(Xte))
    fi = pd.DataFrame({"Feature": features, "Importance": np.abs(model.coef_[0])}).sort_values("Importance", ascending=False)

    # Statsmodels summary
    Xsm = sm.add_constant(df[features])
    sm_model = sm.Logit(df[target], Xsm).fit(disp=0)

    return {"auc": auc, "report": report, "feature_importance": fi, "sm_model": sm_model}
