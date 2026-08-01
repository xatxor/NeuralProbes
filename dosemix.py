#! /usr/bin/env python

"""Is there a responding subpopulation, without thresholding?

Counting concepts above `|z| >= 4` throws the distribution away and makes the answer depend on a
threshold nobody chose on principle. This asks the same question from the whole distribution.

Write `f_c` for the density of `|z|` over the 1036 concepts and `f_r` for the density over the 512
random directions. Three quantities follow, in increasing order of usefulness.

`f_c - f_r`   where concepts are over-represented. Integrating its positive part gives the **excess
              mass**: the fraction of concepts not accounted for by the null, and hence an estimate
              of how many respond that never mentions a threshold.

`f_c / f_r`   enrichment as a function of `|z|`. A pure null sits at 1 everywhere. A mixture of null
              plus a responding subpopulation rises in the tail, and *where* it lifts off is where
              the subpopulation starts.

`d/d|z| log(f_c/f_r)`  the slope of that. Positive and sustained means the tail keeps getting richer
              in real responders; flat means the tail is just the null's own tail. This is the
              derivative worth taking -- of the ratio, not of either density alone, because the
              derivative of `f_c` by itself reports the shape of the null it is dominated by.

The Kolmogorov--Smirnov statistic is reported alongside as a distribution-free check that the two
samples differ at all.

Densities are Gaussian KDEs on `log10|z|` rather than `|z|`, because `|z|` is bounded below by zero
and strongly right-skewed; a symmetric kernel on the raw scale leaks mass through the boundary and
manufactures a spurious peak near zero.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

log = logging.getLogger("dosemix")

LAYERS = [11, 14, 18, 22, 25]


def kde(sample: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Gaussian kernel density estimate.

    :param sample: observations, one dimension.
    :param grid: points to evaluate on.
    :param bandwidth: kernel width; Silverman's rule when omitted.

    :return: density on `grid`.
    """
    n = len(sample)
    if bandwidth is None:
        spread = min(sample.std(ddof=1), (np.percentile(sample, 75) - np.percentile(sample, 25)) / 1.349)
        bandwidth = 0.9 * spread * n ** (-0.2)
    z = (grid[:, None] - sample[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))


def kolmogorov(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov--Smirnov statistic and its asymptotic p-value.

    :param a: first sample.
    :param b: second sample.

    :return: the statistic and the p-value.
    """
    pooled = np.sort(np.concatenate([a, b]))
    gap = np.abs(np.searchsorted(np.sort(a), pooled, "right") / len(a)
                 - np.searchsorted(np.sort(b), pooled, "right") / len(b))
    statistic = float(gap.max())
    scale = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (scale + 0.12 + 0.11 / scale) * statistic
    terms = np.arange(1, 101)
    return statistic, float(2 * np.sum((-1) ** (terms - 1) * np.exp(-2 * terms ** 2 * lam ** 2)))


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs = {label: np.load(path / "dose-derived.npz", allow_pickle=False)
            for label, path in zip(args.labels, args.runs)}
    concepts = args.concepts
    slot = LAYERS.index(args.layer)
    grid = np.linspace(np.log10(args.floor), np.log10(args.ceiling), 400)

    report: dict[str, dict[str, float]] = {}
    with PdfPages(args.out) as pdf:
        fig, axes = plt.subplots(2, len(runs), figsize=(6.5 * len(runs), 7.0), squeeze=False)
        for column, (label, blob) in enumerate(runs.items()):
            z = blob[f"{args.stat}.tylenol"][slot]
            real = np.clip(np.abs(z[:concepts]), args.floor, None)
            rand = np.clip(np.abs(z[concepts:]), args.floor, None)
            fc, fr = kde(np.log10(real), grid), kde(np.log10(rand), grid)

            step = grid[1] - grid[0]
            excess = float(np.clip(fc - fr, 0, None).sum() * step)
            ratio = fc / np.maximum(fr, 1e-9)
            slope = np.gradient(np.log(np.maximum(ratio, 1e-9)), step)
            statistic, pvalue = kolmogorov(np.log10(real), np.log10(rand))
            # Where the tail stops looking like the null: the smallest |z| beyond which the ratio
            # never falls back below `lift`.
            above = ratio >= args.lift
            liftoff = float(10 ** grid[np.argmax(np.cumprod(above[::-1])[::-1] > 0)]) if above.any() else float("nan")

            report[label] = {
                "excess_mass": excess, "implied_responders": excess * concepts,
                "ks": statistic, "ks_p": pvalue, "liftoff_z": liftoff,
                "median_ratio": float(np.median(real) / np.median(rand)),
                "max_log_ratio_slope": float(np.nanmax(slope)),
            }

            ax = axes[0, column]
            ax.plot(10 ** grid, fc, color="C3", lw=1.4, label=f"{concepts} concepts")
            ax.plot(10 ** grid, fr, color="0.5", lw=1.4, label="512 random")
            ax.fill_between(10 ** grid, fr, fc, where=fc > fr, color="C3", alpha=0.2,
                            label=f"excess mass {excess:.3f}")
            ax.set_xscale("log")
            ax.set_xlabel(r"$|z|$")
            ax.set_ylabel("density on $\\log_{10}|z|$")
            ax.set_title(f"{label}: densities", fontsize=9)
            ax.legend(fontsize=7)

            ax = axes[1, column]
            ax.plot(10 ** grid, ratio, color="C0", lw=1.5)
            ax.axhline(1, color="k", lw=0.7, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$|z|$")
            ax.set_ylabel(r"$f_{\rm concepts}/f_{\rm random}$")
            twin = ax.twinx()
            twin.plot(10 ** grid, slope, color="C2", lw=1.0, alpha=0.75)
            twin.axhline(0, color="C2", lw=0.5, ls=":")
            twin.set_ylabel(r"$d\log(\rm ratio)/d\log_{10}|z|$", color="C2", fontsize=8)
            twin.tick_params(labelsize=6, colors="C2")
            ax.set_title(f"{label}: enrichment and its slope; KS={statistic:.3f} "
                         f"(p={pvalue:.1e})", fontsize=9)
        fig.suptitle(f"block {args.layer}, {args.stat}, Tylenol ladder", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    print(f"\n{'':<24}" + "".join(f"{k:>14}" for k in report))
    print("-" * (24 + 14 * len(report)))
    for field in ("median_ratio", "excess_mass", "implied_responders", "liftoff_z", "ks", "ks_p"):
        print(f"{field:<24}" + "".join(f"{report[k][field]:>14.4g}" for k in report))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--stat", default="zf", choices=["zf", "zr", "zp"])
    parser.add_argument("--layer", type=int, default=25, choices=LAYERS)
    parser.add_argument("--concepts", type=int, default=1036)
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--ceiling", type=float, default=60.0)
    parser.add_argument("--lift", type=float, default=1.5, help="ratio that counts as lift-off")
    parser.add_argument("--out", type=Path, default=Path(".bak/dose-lda/dose-mixture.pdf"))
    main(parser.parse_args())
