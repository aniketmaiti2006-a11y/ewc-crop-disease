"""Generate paper-quality figures from checkpoints/results.json.

Outputs (saved into ./figures/):
  - loss_curves.{pdf,png}      — Total loss / CE loss / EWC penalty per task
  - accuracy_curves.{pdf,png}  — Train vs Val accuracy per task
  - forgetting.{pdf,png}       — R_tt vs R_Tt bar chart with BWT annotation
  - combined.{pdf,png}         — 2x3 grid: rows = metrics, cols = tasks

All figures are saved as vector PDF (Type-42 fonts) and 300 DPI PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style — paper-friendly defaults (matches IEEE / ACM / Springer guidelines).
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "pdf.fonttype": 42,        # TrueType, not Type-3
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Season → colour mapping, consistent across all figures.
SEASON_COLOURS = {
    "Spring": "#2ca02c",  # green
    "Summer": "#ff7f0e",  # orange
    "Autumn": "#d62728",  # red
}
SEASONS = ["Spring", "Summer", "Autumn"]


# ---------------------------------------------------------------------------
# Load results.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "checkpoints" / "results.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path = RESULTS_PATH) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as both vector PDF and 300 DPI PNG."""
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf_path.relative_to(HERE)}")
    print(f"  wrote {png_path.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Figure 1 — loss decomposition (Total / CE / EWC), one panel per task.
# ---------------------------------------------------------------------------
def plot_loss_curves(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False)

    epoch_logs = results["epoch_logs"]
    for ax, season in zip(axes, SEASONS):
        logs = epoch_logs[season]
        epochs = [e["epoch"] for e in logs]
        total = [e["loss"] for e in logs]
        ce = [e["ce"] for e in logs]
        ewc = [e["ewc"] for e in logs]

        c = SEASON_COLOURS[season]
        ax.plot(epochs, total, marker="o", lw=1.8, color=c, label="Total loss")
        ax.plot(epochs, ce, marker="s", lw=1.4, color=c, ls="--", alpha=0.75, label="CE loss")
        ax.plot(epochs, ewc, marker="^", lw=1.4, color="#444444", ls=":", label="EWC penalty")

        ax.set_title(season, color=c, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_xlim(0.8, max(epochs) + 0.2)

    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", frameon=True)

    fig.suptitle("Training loss decomposition across seasonal tasks", y=1.04)
    _save(fig, "loss_curves")


# ---------------------------------------------------------------------------
# Figure 2 — accuracy curves (Train vs Val), one panel per task.
# ---------------------------------------------------------------------------
def plot_accuracy_curves(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)

    epoch_logs = results["epoch_logs"]
    for ax, season in zip(axes, SEASONS):
        logs = epoch_logs[season]
        epochs = [e["epoch"] for e in logs]
        train_acc = [e["train_acc"] for e in logs]
        val_acc = [e["val_acc"] for e in logs]

        c = SEASON_COLOURS[season]
        ax.plot(epochs, train_acc, marker="o", lw=1.8, color=c, label="Train")
        ax.plot(epochs, val_acc, marker="o", lw=1.8, color="#1f77b4", label="Validation")

        # Mark the final val accuracy as the task's R_tt.
        ax.scatter([epochs[-1]], [val_acc[-1]], s=70, facecolor="none",
                   edgecolor="#1f77b4", lw=1.5, zorder=5)

        ax.set_title(season, color=c, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_xlim(0.8, max(epochs) + 0.2)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right", frameon=True)

    fig.suptitle("Train vs. validation accuracy across seasonal tasks", y=1.04)
    _save(fig, "accuracy_curves")


# ---------------------------------------------------------------------------
# Figure 3 — forgetting summary: R_tt vs R_Tt per task, with BWT annotation.
# ---------------------------------------------------------------------------
def plot_forgetting(results: dict) -> None:
    r_tt = results["R_tt"]
    r_Tt = results["R_Tt"]
    bwt = results["BWT"]

    x = np.arange(len(SEASONS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    rtt_bars = ax.bar(x - width / 2, [r_tt[s] for s in SEASONS], width,
                      color="#9ecae1", edgecolor="#1f77b4",
                      label=r"$R_{t,t}$ (just after training)")
    rTt_bars = ax.bar(x + width / 2, [r_Tt[s] for s in SEASONS], width,
                      color="#fdae6b", edgecolor="#d95f02",
                      label=r"$R_{T,t}$ (after all tasks)")

    # Numeric labels above each bar.
    for bars in (rtt_bars, rTt_bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Δ arrow between paired bars.
    for i, season in enumerate(SEASONS):
        delta = r_Tt[season] - r_tt[season]
        y_top = max(r_tt[season], r_Tt[season]) + 6
        ax.annotate(
            f"Δ={delta:+.2f}",
            xy=(i, y_top),
            ha="center",
            fontsize=8,
            color=("#d62728" if delta < 0 else "#2ca02c"),
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(max(r_tt.values()), max(r_Tt.values())) * 1.25)
    ax.legend(loc="upper right", frameon=True)

    ax.set_title(
        f"Forgetting across tasks   |   BWT = {bwt:+.4f}",
        fontweight="bold",
    )

    _save(fig, "forgetting")


# ---------------------------------------------------------------------------
# Figure 4 — headline "result card" for the README hero spot.
# ---------------------------------------------------------------------------
def plot_result_card(results: dict) -> None:
    """Single hero image: BWT headline + per-season forgetting bars.

    Sized for a GitHub README (~1000x520 px at 100 DPI). Two panels:
      • Left  — the BWT score with a short verdict line.
      • Right — the R_tt / R_Tt per-season bars (the same data as `forgetting`
                but compressed to one strip so it pairs with the headline).
    """
    fig = plt.figure(figsize=(10.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.18)

    # ----- Left panel: BWT headline -----
    ax_head = fig.add_subplot(gs[0, 0])
    ax_head.axis("off")

    bwt = results["BWT"]
    passes = bwt >= -2.9  # catastrophic-forgetting threshold from README.
    verdict_colour = "#2ca02c" if passes else "#d62728"
    verdict_text = "PASSED" if passes else "FAILED"
    bwt_colour = "#2ca02c" if bwt >= 0 else "#d62728"

    ax_head.text(
        0.5, 0.84,
        "Backward Transfer",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#222222",
    )
    ax_head.text(
        0.5, 0.55,
        f"{bwt:+.2f} %",
        ha="center", va="center",
        fontsize=56, fontweight="bold", color=bwt_colour,
    )
    ax_head.text(
        0.5, 0.27,
        "Catastrophic-forgetting criterion",
        ha="center", va="center", fontsize=10, color="#555555",
    )
    ax_head.text(
        0.5, 0.13,
        f"{verdict_text}   (threshold ≥ −2.9 %)",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color=verdict_colour,
    )

    # Decorative underline below the headline.
    ax_head.plot([0.18, 0.82], [0.66, 0.66],
                 color="#cccccc", lw=1.2, transform=ax_head.transAxes)

    # ----- Right panel: per-season forgetting -----
    ax = fig.add_subplot(gs[0, 1])
    r_tt = results["R_tt"]
    r_Tt = results["R_Tt"]

    x = np.arange(len(SEASONS))
    width = 0.36

    rtt_bars = ax.bar(x - width / 2, [r_tt[s] for s in SEASONS], width,
                      color="#9ecae1", edgecolor="#1f77b4",
                      label=r"$R_{t,t}$ (just after training)")
    rTt_bars = ax.bar(x + width / 2, [r_Tt[s] for s in SEASONS], width,
                      color="#fdae6b", edgecolor="#d95f02",
                      label=r"$R_{T,t}$ (after all tasks)")

    for bars in (rtt_bars, rTt_bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    for i, season in enumerate(SEASONS):
        delta = r_Tt[season] - r_tt[season]
        y_top = max(r_tt[season], r_Tt[season]) + 6
        ax.annotate(
            f"Δ={delta:+.2f}",
            xy=(i, y_top),
            ha="center", fontsize=8,
            color=("#d62728" if delta < 0 else "#2ca02c"),
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, max(max(r_tt.values()), max(r_Tt.values())) * 1.30)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title("Per-season forgetting", fontsize=11, fontweight="bold")

    fig.suptitle(
        "EWC continual learning  —  seasonal crop-disease tasks",
        y=1.00, fontsize=13, fontweight="bold",
    )
    _save(fig, "result")


# ---------------------------------------------------------------------------
# Figure 5 — combined 2x3 grid for the paper's main results section.
# ---------------------------------------------------------------------------
def plot_combined(results: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4), sharex="col")

    epoch_logs = results["epoch_logs"]

    # Row 1 — loss decomposition (Total + CE).
    for col, season in enumerate(SEASONS):
        ax = axes[0, col]
        logs = epoch_logs[season]
        epochs = [e["epoch"] for e in logs]
        c = SEASON_COLOURS[season]
        ax.plot(epochs, [e["loss"] for e in logs], marker="o", lw=1.6, color=c, label="Total")
        ax.plot(epochs, [e["ce"] for e in logs], marker="s", lw=1.4, ls="--",
                color=c, alpha=0.7, label="CE")
        ax.set_title(season, color=c, fontweight="bold")
        ax.set_xticks(epochs)
        ax.set_xlim(0.8, max(epochs) + 0.2)
        ax.set_ylabel("Loss")
        if col == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    # Row 2 — accuracy (Train vs Val).
    for col, season in enumerate(SEASONS):
        ax = axes[1, col]
        logs = epoch_logs[season]
        epochs = [e["epoch"] for e in logs]
        c = SEASON_COLOURS[season]
        ax.plot(epochs, [e["train_acc"] for e in logs], marker="o", lw=1.6, color=c, label="Train")
        ax.plot(epochs, [e["val_acc"] for e in logs], marker="o", lw=1.6,
                color="#1f77b4", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_xlim(0.8, max(epochs) + 0.2)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        if col == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(
        f"EWC continual-learning results   |   "
        f"$\\lambda_{{EWC}}$ = {results['lambda_ewc']},   "
        f"epochs/task = {results['epochs']},   "
        f"BWT = {results['BWT']:+.4f}",
        y=1.00,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "combined")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    results = load_results()
    print(f"Loaded results from {RESULTS_PATH.relative_to(HERE)}")
    print(f"BWT = {results['BWT']:+.4f}")
    print(f"Writing figures to {FIG_DIR.relative_to(HERE)}/")
    plot_loss_curves(results)
    plot_accuracy_curves(results)
    plot_forgetting(results)
    plot_combined(results)
    plot_result_card(results)
    print("Done.")


if __name__ == "__main__":
    main()