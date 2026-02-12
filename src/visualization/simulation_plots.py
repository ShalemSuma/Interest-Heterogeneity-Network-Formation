"""
Simulation Plots

Visualise the Bala & Goyal network formation simulation:
time series, comparative parameter sweeps, heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_network_evolution(history_df, metrics=None):
    """
    Plot time-series evolution of key metrics.

    Parameters
    ----------
    history_df : DataFrame  (output of NoncooperativeNetworkFormation.run_simulation)
    metrics : list of column names
    """
    if metrics is None:
        metrics = ["gini_coefficient", "top_20_percent_share",
                    "density", "clustering_coefficient",
                    "average_path_length", "max_in_degree"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for i, m in enumerate(metrics):
        ax = axes.flat[i]
        if m in history_df.columns:
            ax.plot(history_df["timestep"], history_df[m], "o-", lw=2)
            ax.set_title(f"Evolution of {m}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(m)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparative_results(results_df, x_param, target="top_20_percent_share", group_by=None):
    """
    Visualise results of a comparative parameter sweep.
    """
    fig = plt.figure(figsize=(12, 6))

    if group_by:
        for grp in results_df[group_by].unique():
            sub = results_df[results_df[group_by] == grp]
            xs = sorted(sub[x_param].unique())
            ys = [sub[sub[x_param] == x][target].mean() for x in xs]
            es = [sub[sub[x_param] == x][target].std() for x in xs]
            plt.errorbar(xs, ys, yerr=es, marker="o", label=f"{group_by}={grp}")
        plt.legend()
    else:
        xs = sorted(results_df[x_param].unique())
        ys = [results_df[results_df[x_param] == x][target].mean() for x in xs]
        es = [results_df[results_df[x_param] == x][target].std() for x in xs]
        plt.errorbar(xs, ys, yerr=es, marker="o")

    plt.xlabel(x_param)
    plt.ylabel(target)
    plt.title(f"Effect of {x_param} on {target}")
    plt.grid(alpha=0.3)
    return fig


def plot_parameter_heatmap(results_df, row_param, col_param, value, ax=None):
    """Draw a heat-map of *value* indexed by two parameters."""
    pivot = results_df.pivot_table(index=row_param, columns=col_param, values=value, aggfunc="mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{value} by {row_param} & {col_param}")
    return ax
