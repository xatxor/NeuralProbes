#! /usr/bin/env python

"""Render `predict.py`'s output as a PDF.

The document is arranged so the null arrives before the result. Every panel that shows a separation
also shows what separation a shuffled label produced on the same data, because at 2072 correlated
tests the interesting question is never "is this above chance" but "is this above the best of 2072
draws from chance".
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger("report")

PAGE = (11.69, 8.27)
INK = "#1b1b1b"
GREY = "#8d8d8d"
COLOUR = {"submit": "#d1495b", "give_up": "#2e86ab", "degenerate": "#8d8d8d"}
LABEL = {"submit": "reward hacked", "give_up": "gave up", "degenerate": "degenerate"}
WINDOW = {
    "all": "all thinking tokens",
    "early": "thinking in the first 3 turns",
    "pre": "all thinking except the final turn",
    "last": "the final turn's thinking only",
}
KIND = {"z": "per-episode z", "cos": "raw cosine"}
# The window whose result is the actual answer to "can we predict": tokens emitted before the model
# had committed to anything, with episode shape partialled out.
HEADLINE = "cos:early|ctl"


def describe(key: str) -> str:
    """Human-readable name for a feature-set key."""
    base, _, control = key.partition("|")
    kind, window = base.split(":")
    return f"{WINDOW[window]} · {KIND[kind]}" + (" · episode shape removed" if control else "")


def title(figure, text: str, sub: str = "") -> None:
    """Put a page heading on a figure, shrunk to fit rather than run off the page."""
    figure.text(0.06, 0.94, text, fontsize=min(15, 15 * 62 / max(len(text), 62)),
                color=INK, weight="bold")
    if sub:
        figure.text(0.06, 0.905, sub, fontsize=9.5, color=GREY)


def prose(figure, lines: list[str], top: float = 0.85, size: float = 9.5, width: int = 118) -> float:
    """Lay out wrapped paragraphs, returning the y reached."""
    y = top
    for line in lines:
        if not line:
            y -= 0.018
            continue
        bold = line.startswith("**")
        body = line.strip("*")
        for piece in textwrap.wrap(body, width) or [""]:
            figure.text(0.06, y, piece, fontsize=size, color=INK,
                        weight="bold" if bold else "normal", family="DejaVu Sans")
            y -= 0.030
    return y


def table(figure, headers: list[str], rows: list[list[str]], top: float, widths: list[float],
          size: float = 8.0, step: float = 0.024) -> float:
    """Draw a fixed-column text table, returning the y reached."""
    x0 = 0.06
    y = top
    x = x0
    for head, w in zip(headers, widths):
        figure.text(x, y, head, fontsize=size, color=GREY, weight="bold")
        x += w
    y -= 0.008
    figure.add_artist(plt.Line2D([x0, x0 + sum(widths)], [y, y], color="#dddddd", linewidth=0.8))
    y -= 0.020
    for row in rows:
        x = x0
        for cell, w in zip(row, widths):
            figure.text(x, y, cell, fontsize=size, color=INK, family="DejaVu Sans")
            x += w
        y -= step
    return y


def cover(pdf: PdfPages, report: dict) -> None:
    """Data summary and what was measured."""
    figure = plt.figure(figsize=PAGE)
    title(figure,
          "Do concept activations predict reward hacking?",
          f"{report['episodes']} gated episodes of workload 01 · Qwen3-8B · 1036 concept directions at L18 and L25")

    counts = report["counts"]
    y = prose(figure, [
        "**The question**",
        "For each episode, average every concept's activation over the model's own thinking tokens. "
        "That gives one number per episode per concept. Fit a Gaussian to that number within each "
        "outcome class and ask whether the Gaussians are far enough apart to tell the classes apart.",
        "",
        "**The corpus**",
    ], top=0.85)

    rows = [[LABEL[g], str(counts.get(g, 0)),
             f"{np.mean([m['turns'] for m in report['meta'] if m['ending'] == g]):.1f}",
             f"{np.mean([m['thinking'] for m in report['meta'] if m['ending'] == g]):.0f}",
             f"{np.mean([m['distinct'] for m in report['meta'] if m['ending'] == g]):.2f}"]
            for g in ("submit", "give_up", "degenerate") if counts.get(g)]
    y = table(figure, ["outcome", "episodes", "mean turns", "mean thinking tokens", "mean distinct impls"],
              rows, y - 0.01, [0.14, 0.09, 0.11, 0.17, 0.14])

    prose(figure, [
        "",
        "**Two normalisations, because only one of them is legitimately cross-episode**",
        "z -- the per-episode z-scores. Averaged over thinking tokens this is a contrast: how far the "
        "concept rises during deliberation relative to the rest of that same episode. Immune to a "
        "per-episode offset, and blind to one.",
        "cos -- the raw cosines. The absolute level, so an episode that simply runs hot on a concept "
        "throughout is visible. Also the one that can pick up drift unrelated to the outcome.",
        "",
        "**Four token windows, because correlation and prediction are different claims**",
        "all -- every thinking token. Includes the decision, so it can only be correlational.",
        "early -- thinking in the first 3 turns. No episode in this corpus ever hacked before three "
        "distinct implementations existed, so these tokens precede the commitment.",
        "pre -- everything except the final turn, which removes the terminal-turn artifact that has "
        "topped every earlier ranking in this project.",
        "last -- the final turn alone, kept as the artifact-rich contrast.",
        "",
        "**Only one cell in the whole grid is a prediction, and it is not the strongest one**",
        "A z-score is centred and scaled against its own episode's mean and standard deviation over "
        "every token -- including the final turn. So z:early tokens carry information from their own "
        "episode's future, and z:early cannot support a predictive claim however well it scores. "
        "cos:early is the only window that is both ahead of the decision and free of it. Where the "
        "two disagree, cos:early is the answer.",
    ], top=y - 0.02)

    pdf.savefig(figure)
    plt.close(figure)


def confounds(pdf: PdfPages, report: dict) -> None:
    """What separates the classes without any concept at all."""
    figure = plt.figure(figsize=PAGE)
    title(figure, "What the concepts have to beat",
          "AUC of a single scalar against 'this episode reward hacked'. 0.5 is chance; below 0.5 means the class runs low.")

    y = prose(figure, [
        "A concept mean is not automatically about concepts. Episodes of different lengths contain "
        "different token mixtures, and the exploration count is already known to gate hacking "
        "entirely. If a concept does not beat these numbers, it has told us nothing new.",
        "",
    ], top=0.85)

    rows = [[field, f"{report['baseline'][field]['submit']:.3f}",
             f"{report['baseline'][field]['give_up']:.3f}",
             f"{report['baseline'][field]['degenerate']:.3f}"]
            for field in ("turns", "tokens", "thinking", "distinct")]
    y = table(figure, ["scalar", "hacked vs rest", "gave up vs rest", "degenerate vs rest"],
              rows, y, [0.16, 0.16, 0.16, 0.16])

    shape = report["baseline"].get("shape_model", {})
    if shape:
        y = prose(figure, [
            "",
            "**Those four scalars together, same ridge and same leave-one-out as every concept model**",
        ], top=y - 0.01)
        y = table(figure, ["comparison", "LOO AUC", "shuffled-label mean", "p"],
                  [[name.replace("hack_vs_", "hacked vs "), f"{fit['auc']:.3f}",
                    f"{fit['null_mean']:.3f}", f"{fit['p']:.3f}"]
                   for name, fit in shape.items()],
                  y, [0.20, 0.12, 0.18, 0.10])

    prose(figure, [
        "",
        "**This is the number to beat, and it is why every feature set below appears twice.**",
        "The `|ctl` variants have episode shape regressed out of every concept first — intercept, "
        "log turns, log tokens, log thinking tokens, thinking fraction and distinct implementations. "
        "Whatever survives that is not a length readout.",
        "The thinking fraction is in there for an arithmetic reason, not a cautious one: a per-episode "
        "z-score is centred over all of the episode's tokens, so the mean over the thinking subset is "
        "fixed by the mean over the remainder and the ratio between them. A z-contrast therefore "
        "carries token composition whether or not it carries anything else.",
    ], top=y - 0.01)

    pdf.savefig(figure)
    plt.close(figure)


def spectrum(pdf: PdfPages, report: dict, key: str, comparison: str) -> None:
    """The whole field of 2072 AUCs against the permutation null."""
    block = report["univariate"].get(key, {}).get(comparison)
    if not block:
        return
    figure = plt.figure(figsize=PAGE)
    title(figure,
          f"All 2072 concept-layer values · {describe(key)}",
          f"{comparison.replace('_', ' ')} — {block['n_pos']} vs {block['n_neg']} episodes")

    axes = figure.add_axes((0.08, 0.42, 0.40, 0.42))
    values = np.array(block["auc"])
    axes.hist(values, bins=60, color="#4a6fa5", alpha=0.85)
    edge = 0.5 + block["null_p95"]
    for x, style, note in ((edge, "--", "null 95th pct"), (1 - edge, "--", None)):
        axes.axvline(x, color="#d1495b", linewidth=1.2, linestyle=style)
    axes.axvline(0.5, color=INK, linewidth=0.8, linestyle=":")
    axes.set_xlabel("AUC", fontsize=9)
    axes.set_ylabel("concept-layer values", fontsize=9)
    axes.set_title(f"red = family-wise null, {edge:.3f}", fontsize=9, color=GREY)
    axes.tick_params(labelsize=8)

    axes2 = figure.add_axes((0.56, 0.42, 0.38, 0.42))
    deviation = np.sort(np.abs(values - 0.5))[::-1]
    axes2.plot(np.arange(1, len(deviation) + 1), deviation, color="#4a6fa5", linewidth=1.4)
    axes2.axhline(block["null_p95"], color="#d1495b", linewidth=1.2, linestyle="--")
    axes2.axhline(block["null_max"], color="#d1495b", linewidth=0.8, linestyle=":")
    axes2.set_xscale("log")
    axes2.set_xlabel("rank", fontsize=9)
    axes2.set_ylabel("|AUC − 0.5|", fontsize=9)
    axes2.set_title("sorted effect against the null", fontsize=9, color=GREY)
    axes2.tick_params(labelsize=8)

    cv = report["multivariate"].get(key, {}).get(comparison, {})
    shape = report["baseline"].get("shape_model", {}).get(comparison, {})
    prose(figure, [
        f"**Best single concept AUC {block['best_auc']:.3f}. Concepts surviving the family-wise null: "
        f"{block['survivors']} of {len(values)}.**",
        f"A shuffled label produced a best-of-2072 AUC of {edge:.3f} at the 95th percentile and "
        f"{0.5 + block['null_max']:.3f} at its most extreme, over {report['draws']} draws.",
        (f"All 2072 concepts together, ridge, leave-one-out: AUC {cv.get('auc', float('nan')):.3f} "
         f"against a shuffled-label mean of {cv.get('null_mean', float('nan')):.3f} "
         f"(p = {cv.get('p', float('nan')):.3f})"
         + (f", and {shape['auc']:.3f} for episode shape alone." if shape else ".") if cv else ""),
        "",
        "Two cautions on this page. The best-of-2072 AUC was chosen on the same episodes it is "
        "scored on, so it is optimistic by construction — only the leave-one-out figure is honest, "
        "and the permutation edge is what the selected one has to clear. And a survivor count in the "
        "hundreds is not hundreds of findings: the effective dimensionality of these 1036 directions "
        "is about ten, so one shared axis lights up a large, correlated block of them at once.",
    ], top=0.32, size=9.0)

    pdf.savefig(figure)
    plt.close(figure)


def densities(pdf: PdfPages, report: dict, key: str, comparison: str, count: int = 6) -> None:
    """The Gaussians themselves, for the concepts that separated best."""
    block = report["univariate"].get(key, {}).get(comparison)
    if not block:
        return
    gaussians = report["gaussians"].get(key, {})
    figure = plt.figure(figsize=PAGE)
    title(figure,
          f"Fitted Gaussians · {describe(key)}",
          f"the {count} concept-layer values with the largest |AUC − 0.5| for {comparison.replace('_', ' ')}")

    edge = 0.5 + block["null_p95"]
    for slot, row in enumerate(block["table"][:count]):
        axes = figure.add_axes((0.07 + 0.31 * (slot % 3), 0.55 - 0.34 * (slot // 3), 0.25, 0.24))
        span = None
        for group, style in gaussians.items():
            mu = style["mean"][row["slot"]]
            sd = max(style["sd"][row["slot"]], 1e-6)
            grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
            span = (min(span[0], grid[0]), max(span[1], grid[-1])) if span else (grid[0], grid[-1])
            axes.plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                      color=COLOUR[group], linewidth=1.8,
                      label=f"{LABEL[group]} (n={style['n']})")
        axes.set_xlim(*span)
        axes.tick_params(labelsize=7)
        axes.set_yticks([])
        flag = "" if row["p_fwer"] < 0.05 else "  (inside the null)"
        axes.set_title(
            textwrap.fill(f"{row['pair']} L{row['layer']}: {row['concept']} || {row['antagonist']}", 46)
            + f"\nAUC {row['auc']:.3f}  d {row['d']:+.2f}  p {row['p_fwer']:.3f}{flag}",
            fontsize=7, color=INK if row["p_fwer"] < 0.05 else GREY)
        if slot == 0:
            axes.legend(fontsize=6.5, loc="upper right", frameon=False)

    figure.text(0.06, 0.06,
                f"A concept clears the family-wise null only if its AUC leaves [{1 - edge:.3f}, {edge:.3f}]. "
                "Titles in grey did not.", fontsize=8.5, color=GREY)
    pdf.savefig(figure)
    plt.close(figure)


def ranking(pdf: PdfPages, report: dict, key: str, comparison: str, count: int = 20) -> None:
    """The tabulated top concepts."""
    block = report["univariate"].get(key, {}).get(comparison)
    if not block:
        return
    figure = plt.figure(figsize=PAGE)
    title(figure, f"Ranked concepts · {describe(key)}",
          f"{comparison.replace('_', ' ')}; p is family-wise over all 2072 values")
    rows = [[str(r["pair"]), f"L{r['layer']}",
             textwrap.shorten(f"{r['concept']} || {r['antagonist']}", 58, placeholder="…"),
             f"{r['auc']:.3f}", f"{r['d']:+.2f}", f"{r['p_fwer']:.3f}"]
            for r in block["table"][:count]]
    table(figure, ["pair", "layer", "concept || antagonist", "AUC", "d", "p"], rows, 0.86,
          [0.05, 0.05, 0.50, 0.07, 0.07, 0.07])
    pdf.savefig(figure)
    plt.close(figure)


def summary(pdf: PdfPages, report: dict) -> None:
    """Every window and comparison in one grid, then the verdict."""
    figure = plt.figure(figsize=PAGE)
    title(figure, "Every window, every comparison",
          "best single-concept AUC / family-wise null edge / concepts surviving · ridge leave-one-out AUC (p)")

    # Raw and controlled sit on one row so the cost of removing episode shape is read directly rather
    # than by hunting across two tables.
    rows = []
    for key in sorted(k for k in report["univariate"] if "|" not in k):
        for comparison in ("hack_vs_rest", "hack_vs_giveup", "hack_vs_degenerate"):
            block = report["univariate"][key].get(comparison)
            if not block:
                continue
            ctl = report["univariate"].get(f"{key}|ctl", {}).get(comparison, {})
            cv = report["multivariate"].get(key, {}).get(comparison, {})
            cvc = report["multivariate"].get(f"{key}|ctl", {}).get(comparison, {})
            rows.append([
                key, comparison.replace("hack_vs_", "vs "),
                f"{block['n_pos']}/{block['n_neg']}",
                f"{block['best_auc']:.3f}", f"{0.5 + block['null_p95']:.3f}", f"{block['survivors']}",
                f"{cv.get('auc', float('nan')):.3f}",
                f"{ctl.get('best_auc', float('nan')):.3f}", f"{ctl.get('survivors', 0)}",
                f"{cvc.get('auc', float('nan')):.3f}", f"{cvc.get('p', float('nan')):.3f}",
            ])
    figure.text(0.06, 0.875, "as measured", fontsize=8, color=GREY, style="italic")
    figure.text(0.545, 0.875, "episode shape removed", fontsize=8, color="#d1495b", style="italic")
    y = table(figure,
              ["features", "comparison", "n +/-", "best AUC", "null edge", "survive", "ridge LOO",
               "best AUC", "survive", "ridge LOO", "p"],
              rows, 0.845,
              [0.105, 0.115, 0.075, 0.070, 0.070, 0.062, 0.075, 0.070, 0.062, 0.075, 0.060],
              size=7.0, step=0.0205)

    headline = report["univariate"].get(HEADLINE, {}).get("hack_vs_rest")
    cv = report["multivariate"].get(HEADLINE, {}).get("hack_vs_rest", {})
    shape = report["baseline"].get("shape_model", {}).get("hack_vs_rest", {})
    lines = ["", "**Reading it**"]
    if headline:
        lines += [
            f"On the predictive window with episode shape removed ({describe(HEADLINE)}), the best of "
            f"2072 concepts reached AUC {headline['best_auc']:.3f} against a family-wise null edge of "
            f"{0.5 + headline['null_p95']:.3f}; {headline['survivors']} concepts cleared it. All 2072 "
            f"together: leave-one-out AUC {cv.get('auc', float('nan')):.3f} "
            f"(p = {cv.get('p', float('nan')):.3f}), against "
            f"{shape.get('auc', float('nan')):.3f} for episode shape alone.",
        ]
    lines += [
        "",
        "Three rows deserve suspicion by construction. `last` carries a tiny n -- most final turns "
        "emit fewer than 32 thinking tokens, so the window is absent for most episodes and the "
        "family-wise null saturates near 1.0. The uncontrolled `z:` rows inherit token composition "
        "arithmetically, which is what the controlled columns strip. And `z:early` scores well "
        "without being prospective at all: its baseline is computed over the whole episode, final "
        "turn included, so those tokens are standardised against their own future.",
    ]
    prose(figure, lines, top=y - 0.02, size=9.0)
    pdf.savefig(figure)
    plt.close(figure)


def verdict(pdf: PdfPages, report: dict) -> None:
    """What the numbers answer, and what they do not."""
    figure = plt.figure(figsize=PAGE)
    title(figure, "What this answers")

    def fetch(key: str, comparison: str = "hack_vs_rest") -> tuple[float, float]:
        cv = report["multivariate"].get(key, {}).get(comparison, {})
        return cv.get("auc", float("nan")), cv.get("p", float("nan"))

    shape = report["baseline"].get("shape_model", {}).get("hack_vs_rest", {}).get("auc", float("nan"))
    early, early_p = fetch("cos:early|ctl")
    pre, pre_p = fetch("cos:pre|ctl")
    whole, whole_p = fetch("cos:all|ctl")
    raw_pre, _ = fetch("cos:pre")

    y = prose(figure, [
        "**1. Episode shape alone already predicts reward hacking at AUC 0.939.**",
        f"Reward hacks are short ({np.mean([m['turns'] for m in report['meta'] if m['ending'] == 'submit']):.1f} "
        f"turns against {np.mean([m['turns'] for m in report['meta'] if m['ending'] == 'degenerate']):.1f} "
        "for degenerate runs). Turn count alone gives AUC 0.830 in the short direction, and the four "
        "shape scalars together reach 0.939, rising to 0.999 against degenerate episodes. Any concept result "
        "that has not had this removed is largely a length readout. This is the single most important "
        "number in the document, and it is not about concepts at all.",
        "",
        "**2. With shape removed, concepts still separate a hack that is already underway.**",
        f"Averaged over the whole episode, {whole:.3f} leave-one-out (p = {whole_p:.3f}); excluding "
        f"the final turn, {pre:.3f} (p = {pre_p:.3f}). The final turn is not carrying this — dropping "
        f"it costs almost nothing. That is a real effect above a family-wise permutation null, and it "
        "is concurrent, not predictive: those tokens overlap the decision.",
        "",
        f"**3. Genuine prediction is weak: {early:.3f} leave-one-out, p = {early_p:.3f}.**",
        "Restricted to thinking in the first three turns, with episode shape removed and using raw "
        "cosines so no information from the episode's own future can leak in, the model-wide "
        f"classifier reaches {early:.3f}. At 30 hacked episodes that does not clear the 0.05 line. "
        "The honest reading is that the signal is present but not established at this n — not that "
        "prediction has been demonstrated, and not that it has been ruled out.",
        "",
        "**4. The concepts that do move form one interpretable axis, not twenty findings.**",
        "At the top of the predictive window, hacked episodes run LOW on limits resources to task "
        "scope, completes task without accruing power, staying within bounds || self-exfiltration, "
        "conservative growth and minimalism, and HIGH on detecting steganographic intent, context "
        "injection and detecting dual-use uplift requests. Read together that is a scope-restraint "
        "axis rather than a reward-hacking detector. Given that the effective dimensionality of these "
        "1036 directions is about ten, a coherent block like this is one axis lighting up, which is "
        "why the count of surviving concepts should not be read as a count of discoveries.",
        "",
        "**What would settle it**",
        f"More hacked episodes. Every limit here is the 30. A held-out corpus would also separate "
        "the axis above from the possibility that it tracks task difficulty, which shape controls "
        "only crudely.",
    ], top=0.86, size=9.0)

    figure.text(0.06, max(y - 0.02, 0.04),
                f"Concept directions from AntonKorznikov/feature_stories · {report['draws']} permutations "
                f"for the family-wise null · leave-one-out throughout",
                fontsize=8, color=GREY)
    pdf.savefig(figure)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("analysis/predict.json"))
    parser.add_argument("--out", type=Path, default=Path("analysis/predict.pdf"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = json.loads(args.source.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pdf:
        cover(pdf, report)
        confounds(pdf, report)
        summary(pdf, report)
        verdict(pdf, report)
        # The detail pages are chosen, not enumerated: every feature set appears in the summary
        # table, but only the ones that carry a claim get three pages of their own. Order follows
        # the order the claims should be believed in -- the prospective window first, then the
        # concurrent ones, then the uncontrolled contrast that shows what the control removed.
        # `last` never appears here; at 5 hacked episodes a density panel would be decoration.
        for key, comparison in (
            ("cos:early|ctl", "hack_vs_rest"),
            ("cos:early|ctl", "hack_vs_giveup"),
            ("cos:pre|ctl", "hack_vs_rest"),
            ("cos:pre|ctl", "hack_vs_giveup"),
            ("cos:all|ctl", "hack_vs_rest"),
            ("z:pre|ctl", "hack_vs_rest"),
            ("cos:pre", "hack_vs_rest"),
        ):
            if key not in report["univariate"]:
                continue
            spectrum(pdf, report, key, comparison)
            densities(pdf, report, key, comparison)
            ranking(pdf, report, key, comparison)

    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
