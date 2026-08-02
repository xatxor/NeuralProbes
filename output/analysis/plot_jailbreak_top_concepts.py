"""Presentation version of the layer-25 top-concept comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/pdf/boundary_steering_paper_v2/data/viewer"
OUT = Path(__file__).resolve().parent / "jailbreak_top_concepts_presentation_large_text.png"

index = json.loads((DATA / "index.json").read_text())
pairs = {row["pair"]: row["concept"] for row in index["pairs"]}
aggregate = json.loads((DATA / "aggregate-response-L25-zscore.json").read_text())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), sharex=True)
panels = (("baseline", "Baseline", "#20805c"), ("gcg", "GCG", "#c84343"))

for ax, (condition, title, color) in zip(axes, panels):
    rows = aggregate[f"{condition}:25:all"]["top"][:8][::-1]
    labels = [pairs[row["pair"]] for row in rows]
    values = [row["z_score"] for row in rows]
    ax.barh(range(len(rows)), values, color=color, alpha=.82)
    ax.set_yticks(range(len(rows)), labels, fontsize=13)
    ax.set_title(title, fontsize=19)
    ax.set_xlim(0, 0.9)
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean standardized z-score", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)

fig.suptitle("Experiment-level most activated concepts at layer 25", fontsize=21)
fig.tight_layout()
fig.savefig(OUT, dpi=240, facecolor="white")
print(OUT)
