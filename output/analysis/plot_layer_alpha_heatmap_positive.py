"""Positive-alpha subset of the report's layer-by-steering ASR heatmap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


layers = ["L11", "L14", "L18", "L22", "L25"]
alphas = ["0", "0.1", "0.25", "0.5"]
# Original report heatmap values, retaining only non-negative steering strengths.
asr = np.array([
    [.610, .493, .333, .142],
    [.610, .483, .360, .200],
    [.610, .385, .222, .110],
    [.610, .358, .202, .118],
    [.610, .405, .197, .062],
])
out = Path(__file__).with_name("layer_alpha_heatmap_positive.png")

fig, ax = plt.subplots(figsize=(10, 4.4))
image = ax.imshow(asr, cmap="RdYlGn_r", vmin=0, vmax=.61, aspect="auto")
ax.set(title="GCG attack success under steering", xlabel="Steering strength α", ylabel="Layer")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(17)
ax.yaxis.label.set_fontsize(17)
ax.set_xticks(range(len(alphas)), alphas, fontsize=16)
ax.set_yticks(range(len(layers)), layers, fontsize=16)
for row in range(asr.shape[0]):
    for col in range(asr.shape[1]):
        value = asr[row, col]
        ax.text(col, row, f"{value:.1%}", ha="center", va="center", fontsize=16,
                color="white" if value >= .45 else "black")
colorbar = fig.colorbar(image, ax=ax, pad=.02)
colorbar.set_label("Mean ASR", fontsize=17)
colorbar.ax.tick_params(labelsize=15)
fig.tight_layout()
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(out)
