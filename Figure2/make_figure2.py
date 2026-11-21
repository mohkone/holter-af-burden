"""Render the first 24 paired reviews of patient 102 from the archived endpoints."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.draws)
    rows = data.loc[(data["patient"] == "patient_102") & data["draw"].between(1, 24)]
    panels = [("fixed_radius", "A", "Fixed radius"), ("scaled_radius", "B", "Adaptive radius")]
    by_method = {key: group.sort_values("draw") for key, group in rows.groupby("method")}
    assert set(by_method) == {key for key, _, _ in panels}
    assert rows["truth_pp"].nunique() == 1
    truth = rows["truth_pp"].iloc[0]
    for group in by_method.values():
        np.testing.assert_array_equal(group["draw"], np.arange(1, 25))
        np.testing.assert_array_equal(group["covered"], (group["lower_pp"] <= truth) & (truth <= group["upper_pp"]))
    np.testing.assert_allclose(by_method["fixed_radius"]["center_pp"], by_method["scaled_radius"]["center_pp"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.04), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.093, right=0.988, top=0.805, bottom=0.285, wspace=0.24)
    blue, rust, ink = "#205A70", "#A24621", "#1B1E20"
    bottom = int(np.floor((rows["lower_pp"].min() - 1) / 5) * 5)
    top = int(np.ceil((rows["upper_pp"].max() + 1) / 5) * 5)
    for ax, (key, letter, heading) in zip(axes, panels):
        ax.text(0.0, 1.16, letter, transform=ax.transAxes, ha="left", va="center",
                weight="bold", fontsize=15, color="white", clip_on=False,
                bbox=dict(facecolor=ink, edgecolor="none", boxstyle="square,pad=0.17"))
        ax.text(0.122, 1.16, heading, transform=ax.transAxes, ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=ink, clip_on=False)
        for row in by_method[key].itertuples():
            color, marker = (blue, "o") if row.covered else (rust, "x")
            ax.plot((row.draw, row.draw), (row.lower_pp, row.upper_pp), c=color, lw=1.18)
            ax.scatter(row.draw, row.center_pp, c=color, s=12.5, marker=marker, linewidths=1.0, zorder=4)
        ax.axhline(truth, color=ink, ls=(0, (3, 2)), lw=1.15, zorder=3)
        ax.set(xlim=(0.25, 24.75), ylim=(bottom, top), xticks=[1, 6, 12, 18, 24],
               yticks=list(range(((bottom + 9) // 10) * 10, top + 1, 10)), xlabel="Simulated review")
        ax.spines[["left", "bottom"]].set_color("#91979A")
        ax.tick_params(length=3, color="#91979A", labelleft=True)
    axes[0].set_ylabel("First-day AF burden (%)")
    keys = [Line2D([], [], marker="o", color=blue, label="Contains reference", markersize=4),
            Line2D([], [], marker="x", color=rust, label="Misses reference", markersize=4),
            Line2D([], [], color=ink, ls="--", label=f"Annotated burden ({truth:.2f}%)")]
    fig.legend(handles=keys, loc="lower center", frameon=False, ncol=3, fontsize=9,
               bbox_to_anchor=(0.52, -0.004), columnspacing=1.5)
    args.out.mkdir(parents=True, exist_ok=True)
    basename = args.out / "Figure2_paired_interval_examples"
    fig.savefig(basename.with_suffix(".pdf"), metadata={"Title": "Paired AF burden intervals for patient 102"})
    fig.savefig(basename.with_suffix(".png"), dpi=450, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
