"""
pareto.py
---------
Computes Pareto-optimal models across multiple objectives and generates the
four required plots.

A model M is Pareto-optimal if no other model is *better or equal in every
metric and strictly better in at least one*. We minimize {latency, memory,
energy} and maximize {accuracy}.

Generates:
    outputs/plots/latency_vs_accuracy.png
    outputs/plots/memory_vs_accuracy.png
    outputs/plots/pareto_frontier.png
    outputs/plots/energy_vs_accuracy.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
    """
    costs: (N, K) — K cost dimensions, all to be MINIMIZED.
    Returns boolean mask of Pareto-optimal points.
    """
    n = costs.shape[0]
    is_opt = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_opt[i]:
            continue
        # i is dominated if there exists j with all costs[j] <= costs[i] and any <
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        dominated[i] = False
        if dominated.any():
            is_opt[i] = False
    return is_opt


def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Pareto cost matrix on (latency, memory, energy, -accuracy)
    so all dimensions are minimized.
    """
    df = df.copy()
    # Replace NaN accuracy with 0 so it acts as worst-case
    acc = df["accuracy"].fillna(0.0).to_numpy()
    costs = np.column_stack([
        df["latency_ms"].to_numpy(),
        df["memory_mb"].to_numpy(),
        df["energy_mj"].to_numpy(),
        -acc,  # negate because we want to MAXIMIZE accuracy
    ])
    df["pareto_optimal"] = is_pareto_optimal(costs)
    return df


def _scatter_with_labels(df, x_col, y_col, x_label, y_label, title, fname):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d62728" if p else "#1f77b4" for p in df["pareto_optimal"]]
    ax.scatter(df[x_col], df[y_col], c=colors, s=110, edgecolors="black", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row[x_col], row[y_col]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d62728", label="Pareto-optimal"),
        Patch(color="#1f77b4", label="Dominated"),
    ], loc="best")

    plt.tight_layout()
    out = PLOTS_DIR / fname
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[pareto] saved {out}")


def make_all_plots(df: pd.DataFrame) -> None:
    df = compute_pareto(df)

    _scatter_with_labels(
        df, "latency_ms", "accuracy",
        "Latency (ms, lower is better)", "Top-1 Accuracy (higher is better)",
        "Latency vs Accuracy (Edge CXR)",
        "latency_vs_accuracy.png",
    )
    _scatter_with_labels(
        df, "memory_mb", "accuracy",
        "Memory (MB, lower is better)", "Top-1 Accuracy",
        "Memory vs Accuracy (Edge CXR)",
        "memory_vs_accuracy.png",
    )
    _scatter_with_labels(
        df, "energy_mj", "accuracy",
        "Energy per inference (mJ, lower is better)", "Top-1 Accuracy",
        "Energy vs Accuracy (Edge CXR)",
        "energy_vs_accuracy.png",
    )

    # Pareto frontier plot: latency vs energy, color by accuracy, marker by Pareto
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(
        df["latency_ms"], df["energy_mj"],
        c=df["accuracy"].fillna(0.0), cmap="viridis",
        s=[200 if p else 110 for p in df["pareto_optimal"]],
        edgecolors=["red" if p else "black" for p in df["pareto_optimal"]],
        linewidths=2, zorder=3,
    )
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["latency_ms"], row["energy_mj"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Accuracy")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Energy per inference (mJ)")
    ax.set_title("Pareto Frontier — Latency × Energy × Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "pareto_frontier.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[pareto] saved {out}")

    # Save augmented CSV with pareto column
    df.to_csv(RESULTS_CSV, index=False)


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Run `python -m src.benchmark` first.")
    df = pd.read_csv(RESULTS_CSV)
    make_all_plots(df)
    # Re-read after make_all_plots writes pareto_optimal back
    df = pd.read_csv(RESULTS_CSV)
    if "pareto_optimal" in df.columns:
        pareto_models = df[df["pareto_optimal"] == True]["model"].tolist()
    else:
        pareto_models = []
    print(f"[pareto] Pareto-optimal models: {pareto_models}")


if __name__ == "__main__":
    main()
