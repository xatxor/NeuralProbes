"""L25 positive steering sweep from the latest viewer summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


alphas = [0.0, 0.1, 0.25, 0.5]
attack_success = [0.61, 0.41, 0.22, 0.06]
out = Path(__file__).with_name("gcg_attack_success_under_steering_heatmap.png")

fig, ax = plt.subplots(figsize=(8.4, 2.5))
image = ax.imshow(np.asarray([attack_success]), cmap="RdYlGn_r", vmin=0, vmax=.61, aspect="auto")
ax.set(
    xlabel="Positive steering strength α",
    ylabel="Layer",
)
ax.set_xticks(range(len(alphas)), ["0", "0.1", "0.25", "0.5"])
ax.set_yticks([0], ["L25"])
for col, rate in enumerate(attack_success):
    ax.text(col, 0, f"{rate:.0%}", ha="center", va="center", fontsize=13,
            color="white" if rate >= .5 else "black")
colorbar = fig.colorbar(image, ax=ax, pad=.02)
colorbar.set_label("Attack success rate")
fig.suptitle("GCG attack success under steering", fontsize=16, y=.99)
fig.text(.5, .81, "Layer 25 · refusing unethical orders", ha="center", color="#4b5563", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .70))
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(out)
