"""
Comparison Plots

Side-by-side comparisons of real YouTube data vs simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_real_vs_simulation(real_df, sim_df, features=None, output_path=None):
    """
    Compare distributions of key features between real and simulated data.

    Parameters
    ----------
    real_df : DataFrame with columns including *features* and a binary
              'core_nodes' (or 'is_hub') column.
    sim_df : DataFrame with columns including *features* and 'is_hub'.
    features : list of column names to compare.
    output_path : optional path to save the figure.
    """
    if features is None:
        features = ["interest_mean", "interest_gini", "interest_kurtosis",
                     "interest_entropy", "interest_std"]

    # Align column names
    real = real_df.copy()
    sim = sim_df.copy()
    if "core_nodes" in real.columns:
        real = real.rename(columns={"core_nodes": "is_hub"})

    n_feat = len(features)
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        if feat in real.columns and feat in sim.columns:
            ax.hist(real[feat].dropna(), bins=30, alpha=0.5, label="Real", density=True)
            ax.hist(sim[feat].dropna(), bins=30, alpha=0.5, label="Simulated", density=True)
            ax.set_title(feat)
            ax.legend()
        else:
            ax.set_title(f"{feat}\n(missing)")

    plt.suptitle("Real vs Simulated Feature Distributions", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")

    return fig


def plot_hub_characteristic_comparison(real_df, sim_df, hub_col="is_hub",
                                        features=None, output_path=None):
    """
    Compare hub vs non-hub feature distributions for real and sim data.
    """
    if features is None:
        features = ["interest_mean", "interest_gini", "interest_kurtosis"]

    real = real_df.copy()
    sim = sim_df.copy()
    if "core_nodes" in real.columns:
        real = real.rename(columns={"core_nodes": hub_col})

    fig, axes = plt.subplots(2, len(features), figsize=(5 * len(features), 8))

    for j, feat in enumerate(features):
        for i, (df, label) in enumerate([(real, "Real"), (sim, "Simulated")]):
            ax = axes[i, j] if len(features) > 1 else axes[i]
            if feat in df.columns and hub_col in df.columns:
                for hub_val, colour in [(0, "steelblue"), (1, "tomato")]:
                    sub = df[df[hub_col] == hub_val][feat].dropna()
                    ax.hist(sub, bins=20, alpha=0.6, color=colour,
                            label="Hub" if hub_val else "Non-hub", density=True)
                ax.set_title(f"{label} — {feat}")
                ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
