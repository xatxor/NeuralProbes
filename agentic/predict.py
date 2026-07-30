#! /usr/bin/env python

"""Ask whether a concept readout predicts how an agentic episode ends.

One number per episode per concept: the mean activation over the model's own thinking tokens. Fit a
Gaussian per outcome class, and ask how far apart those Gaussians are.

Two normalisations are carried through everything, because they answer different questions and only
one of them is legitimately cross-episode:

- **z** -- the per-episode z-scores. Averaged over thinking tokens this is a *contrast*: how much the
  concept rises while the model deliberates relative to the rest of that same episode. It is immune
  to any per-episode offset, and blind to one. `zscore.py` says so in its own docstring.
- **cos** -- the raw cosines. This is the absolute level, so an episode that simply runs hotter on a
  concept throughout is visible. It is also the one that can pick up drift that has nothing to do
  with the outcome.

Four token windows are cut, because "predict" and "correlate with" are different claims:

- **all**    every thinking token in the episode. Correlational: includes the decision itself.
- **early**  thinking tokens of the first `EARLY` turns only. This is the predictive claim -- those
             tokens are emitted before the model has committed to anything.
- **pre**    every thinking token except the final turn's. Removes the terminal-turn artifact that
             has topped every previous ranking here.
- **last**   the final turn's thinking only. Kept for contrast, and expected to be artifact-rich.

Nothing here is trusted without a null. 1036 concepts at two layers is 2072 correlated tests, and the
effective dimensionality of these directions is about ten, so Bonferroni is both wrong and too harsh.
Instead labels are permuted and the *maximum* statistic over all concepts is recorded, which gives a
family-wise null that accounts for the correlation exactly. The multivariate model is scored by
leave-one-out cross-validation with the same permutation treatment, including the choice of ridge
strength, so the reported number is not a number that was selected.
"""

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger("predict")

# `readout.find` does the same thing, but importing it drags in torch. Nothing here needs a GPU or a
# model, and the box's lighter venv has no torch at all, so the lookup is repeated rather than shared.
VECTORS = "josephofthebread/Qwen3-8B-concept-vectors"


def find(name: str) -> str:
    """Locate a file inside the local HF cache.

    :param name: the file's name within the vectors repo.

    :return: an absolute path.
    """
    pattern = f"**/models--{VECTORS.replace('/', '--')}/snapshots/*/{name}"
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub" / pattern), recursive=True)
    if not matches:
        raise SystemExit(f"{name} is not in the local cache; this box is meant to have it already")
    return matches[0]

LAYERS = (18, 25)
GROUPS = ("submit", "give_up", "degenerate")
# How each ending reads. On an unsatisfiable suite a green submit cannot happen honestly, so `submit`
# is a reward hack by construction rather than by judgement.
LABEL = {"submit": "reward hacked", "give_up": "gave up", "degenerate": "degenerate"}
# Turns counted as "early". Three is the exploration law's threshold: below three distinct
# implementations no episode in the corpus ever hacked, so the decision cannot have been made yet.
EARLY = 3
# A window thinner than this is a mean over too few tokens to be a measurement.
FLOOR = 32
WINDOWS = ("all", "early", "pre", "last")
KINDS = ("z", "cos")
# Prefixes of the episode, for asking how far in the model has to be before its state says anything.
# `upto{k}` is cut only for episodes that actually reached turn k, so every episode contributing to a
# point has been observed for the same number of turns -- otherwise a short episode would contribute
# its whole trajectory while a long one contributed a fragment, and hacks are the short ones.
PREFIXES = (1, 2, 3, 4, 6, 8)
# Single turns, for asking whether one turn carries what the prefix carries.
SINGLES = (0, 1, 2, 3)
# Episode shape. These are not concepts, and if they separate the outcomes on their own then a
# concept that separates them no better has said nothing about the model's state.
SHAPE = ("turns", "tokens", "thinking", "distinct")
# Ridge strengths, as multiples of the kernel's mean eigenvalue so the grid is scale free.
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def windows(episode: dict, tokens: int) -> dict[str, np.ndarray]:
    """Boolean masks over the token stream, one per window.

    :param episode: the saved episode record.
    :param tokens: length of the readout, which may be shorter than the recorded stream.

    :return: mask per window name; windows that cannot be cut are absent.
    """
    roles = np.array(episode["roles"][:tokens])
    think = roles == "thinking"
    turns = [t for t in episode["turns"] if "start" in t]
    cut: dict[str, np.ndarray] = {"all": think}

    if turns:
        early = np.zeros(tokens, dtype=bool)
        for turn in turns[:EARLY]:
            early[turn["start"] : min(turn["end"], tokens)] = True
        cut["early"] = think & early

        final = np.zeros(tokens, dtype=bool)
        final[turns[-1]["start"] : min(turns[-1]["end"], tokens)] = True
        cut["last"] = think & final
        cut["pre"] = think & ~final

        for k in PREFIXES:
            if len(turns) < k:
                continue
            prefix = np.zeros(tokens, dtype=bool)
            for turn in turns[:k]:
                prefix[turn["start"] : min(turn["end"], tokens)] = True
            cut[f"upto{k}"] = think & prefix

        for k in SINGLES:
            if len(turns) <= k:
                continue
            single = np.zeros(tokens, dtype=bool)
            single[turns[k]["start"] : min(turns[k]["end"], tokens)] = True
            cut[f"turn{k}"] = think & single

    return {name: mask for name, mask in cut.items() if mask.sum() >= FLOOR}


def collect(directory: Path) -> dict:
    """Reduce every episode to one value per concept per layer per window.

    :param directory: directory holding `*.json`, `*.z.npy` and optionally `*.scores.npy`.

    :return: features, labels and per-episode metadata.
    """
    rows: dict[str, dict[str, list[np.ndarray]]] = {k: defaultdict(list) for k in KINDS}
    keep: dict[str, list[int]] = defaultdict(list)
    meta: list[dict] = []

    for index, path in enumerate(sorted(directory.glob("*.z.npy"))):
        record = Path(str(path).replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        if episode.get("ending") not in GROUPS:
            continue

        source = {"z": path, "cos": Path(str(path).replace(".z.npy", ".scores.npy"))}
        arrays = {k: np.load(p).astype(np.float32) for k, p in source.items() if p.exists()}
        if "z" not in arrays:
            continue

        cut = windows(episode, arrays["z"].shape[0])
        slot = len(meta)
        for kind, block in arrays.items():
            for name, mask in cut.items():
                rows[kind][name].append(block[mask].mean(axis=0))
                keep[f"{kind}:{name}"].append(slot)

        turns = [t for t in episode["turns"] if "start" in t]
        meta.append(
            {
                "stem": record.stem,
                "ending": episode["ending"],
                "seed": episode.get("seed"),
                "turns": len(episode.get("turns", [])),
                "tokens": int(arrays["z"].shape[0]),
                "thinking": int(cut["all"].sum()) if "all" in cut else 0,
                "distinct": episode.get("distinct") or 0,
                "early_turns": len(turns[:EARLY]),
            }
        )
        if (index + 1) % 50 == 0:
            log.info(f"{index + 1} episodes reduced")

    features = {
        f"{kind}:{name}": np.stack(block).astype(np.float32)
        for kind, sets in rows.items()
        for name, block in sets.items()
        if block
    }
    return {"features": features, "index": {k: np.array(v) for k, v in keep.items() if v}, "meta": meta}


def design(meta: list[dict]) -> np.ndarray:
    """Episode-shape design matrix, intercept first.

    Counts are logged because a concept mean responds to proportions rather than to absolute counts,
    and the thinking fraction is carried separately because a per-episode z-contrast is arithmetically
    tied to it: the mean over thinking tokens is fixed by the mean over the rest and their ratio.

    :param meta: per-episode metadata.

    :return: `[episode, 1 + k]` design matrix.
    """
    turns = np.array([m["turns"] for m in meta], dtype=np.float64)
    tokens = np.array([m["tokens"] for m in meta], dtype=np.float64)
    thinking = np.array([m["thinking"] for m in meta], dtype=np.float64)
    distinct = np.array([m["distinct"] for m in meta], dtype=np.float64)
    return np.column_stack([
        np.ones_like(turns),
        np.log1p(turns),
        np.log1p(tokens),
        np.log1p(thinking),
        thinking / np.maximum(tokens, 1.0),
        distinct,
    ])


def residualise(matrix: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Remove everything episode shape can explain from every feature.

    :param matrix: `[episode, feature]` values.
    :param shape: `[episode, 1 + k]` design matrix including an intercept.

    :return: residuals of the same shape.
    """
    coefficients, *_ = np.linalg.lstsq(shape, matrix, rcond=None)
    return matrix - shape @ coefficients


def ranks(column: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared.

    :param column: values to rank.

    :return: ranks in `[1, n]`.
    """
    order = np.argsort(column, kind="stable")
    out = np.empty(len(column), dtype=np.float64)
    out[order] = np.arange(1, len(column) + 1)
    # Ties get the mean of the ranks they span, so a constant feature scores exactly 0.5 rather than
    # whatever the sort order happened to be.
    values = column[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or values[stop] != values[start]:
            out[order[start:stop]] = np.arange(start + 1, stop + 1).mean()
            start = stop
    return out


def auc(matrix: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC of every column against a binary label.

    :param matrix: `[episode, feature]` values.
    :param positive: boolean mask of the positive class.

    :return: one AUC per feature.
    """
    rank = np.stack([ranks(matrix[:, j]) for j in range(matrix.shape[1])], axis=1)
    n1 = int(positive.sum())
    n2 = len(positive) - n1
    return (rank[positive].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n2)


def permuted_auc(matrix: np.ndarray, positive: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Family-wise null for `max |AUC - 0.5|` over all features.

    Ranks do not move when labels are shuffled, so every draw is one matrix product rather than a
    re-sort. That is what makes a proper null affordable at 2072 features.

    :param matrix: `[episode, feature]` values.
    :param positive: boolean mask of the positive class.
    :param draws: permutations.
    :param rng: source of randomness.

    :return: the maximum absolute deviation from 0.5 in each draw.
    """
    rank = np.stack([ranks(matrix[:, j]) for j in range(matrix.shape[1])], axis=1)
    n, _ = matrix.shape
    n1 = int(positive.sum())
    n2 = n - n1
    picks = np.zeros((n, draws), dtype=np.float32)
    for d in range(draws):
        picks[rng.permutation(n)[:n1], d] = 1.0
    got = (rank.T.astype(np.float32) @ picks - n1 * (n1 + 1) / 2) / (n1 * n2)
    return np.abs(got - 0.5).max(axis=0)


def smoother(kernel: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """The hat matrix of kernel ridge regression and its leverages.

    Neither depends on the targets, so both are built once per ridge strength and reused across every
    label permutation. That is what makes a permutation null over the cross-validated score cheap.

    :param kernel: `[episode, episode]` gram matrix.
    :param lam: ridge strength.

    :return: the hat matrix and `1 - leverage`.
    """
    n = kernel.shape[0]
    hat = kernel @ np.linalg.inv(kernel + lam * np.eye(n))
    return hat, 1.0 - np.clip(np.diag(hat), None, 1 - 1e-9)


def loo(hat: np.ndarray, slack: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact leave-one-out predictions for a linear smoother.

    Refitting once per held-out episode is unnecessary: the held-out residual is the fitted residual
    divided by one minus that point's leverage.

    :param hat: the smoother matrix.
    :param slack: `1 - leverage` per episode.
    :param y: centred targets.

    :return: one held-out prediction per episode.
    """
    return y - (y - hat @ y) / slack


def multivariate(matrix: np.ndarray, positive: np.ndarray, draws: int, rng: np.random.Generator) -> dict:
    """Cross-validated AUC of a ridge model over all concepts, with a permutation null.

    The ridge strength is chosen on the leave-one-out curve, which is a selection, so every
    permutation is allowed the same selection. Otherwise the null would be the null of a fixed model
    and the reported AUC would be optimistic by exactly the amount that choice is worth.

    :param matrix: `[episode, feature]` values.
    :param positive: boolean mask of the positive class.
    :param draws: permutations.
    :param rng: source of randomness.

    :return: chosen lambda, held-out AUC, and the permutation null.
    """
    x = matrix - matrix.mean(axis=0, keepdims=True)
    spread = x.std(axis=0, keepdims=True)
    x = x / np.where(spread < 1e-8, 1.0, spread)
    kernel = (x @ x.T) / x.shape[1]
    scale = float(np.trace(kernel) / kernel.shape[0])
    fits = {lam: smoother(kernel, lam * scale) for lam in LAMBDAS}

    n = len(positive)
    n1 = int(positive.sum())

    def best(target: np.ndarray) -> tuple[float, float]:
        y = target - target.mean()
        top = (-1.0, 0.0)
        for lam, (hat, slack) in fits.items():
            score = float(auc(loo(hat, slack, y)[:, None], target > 0.5)[0])
            if score > top[0]:
                top = (score, lam)
        return top

    score, lam = best(positive.astype(np.float64))
    null = np.empty(draws)
    for d in range(draws):
        shuffled = np.zeros(n)
        shuffled[rng.permutation(n)[:n1]] = 1.0
        null[d] = best(shuffled)[0]

    return {
        "auc": score,
        "lambda": lam,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p": float((null >= score).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--draws", type=int, default=2000, help="label permutations for the null")
    parser.add_argument("--cv-draws", type=int, default=200, help="permutations for the ridge null")
    parser.add_argument("--top", type=int, default=25, help="concepts tabulated per comparison")
    parser.add_argument("--reuse", action="store_true", help="load the cached reduction instead of re-reading")
    parser.add_argument("--comparisons", default="", help="comma-separated subset to run; all if empty")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    pairs = pq.read_table(find("pairs.parquet")).to_pylist()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "predict-features.npz"

    # Reducing 15 GB of readouts takes minutes; the statistics take seconds. Keeping the reduction on
    # disk means the thresholds and nulls can be revisited without re-reading a single episode.
    if args.reuse and cache.exists():
        held = np.load(cache, allow_pickle=True)
        data = {
            "features": {k: held[k] for k in held.files if not k.startswith("index/") and k != "meta"},
            "index": {k[len("index/"):]: held[k] for k in held.files if k.startswith("index/")},
            "meta": list(held["meta"]),
        }
        log.info(f"reusing {cache}")
    else:
        log.info(f"reducing episodes under {args.dir}")
        data = collect(args.dir)

    meta = data["meta"]
    if not meta:
        raise SystemExit(f"no scored episodes with a usable window under {args.dir}")

    endings = np.array([m["ending"] for m in meta])
    counts = {g: int((endings == g).sum()) for g in GROUPS}
    log.info(f"{len(meta)} episodes: {counts}")

    if not (args.reuse and cache.exists()):
        np.savez_compressed(
            cache,
            meta=np.array(meta, dtype=object),
            **{k: v for k, v in data["features"].items()},
            **{f"index/{k}": v for k, v in data["index"].items()},
        )

    # Confounds first. If episode length or exploration count separates the classes on its own, a
    # concept has to beat that number before it means anything.
    baseline = {}
    for field in SHAPE:
        column = np.array([[m[field]] for m in meta], dtype=np.float64)
        baseline[field] = {
            g: float(auc(column, endings == g)[0]) for g in GROUPS
        }
        log.info(f"baseline {field:<9} AUC vs rest " + " ".join(f"{g}={baseline[field][g]:.3f}" for g in GROUPS))

    shape = design(meta)
    for name, split in (("hack_vs_rest", lambda e: (e == "submit", e != "submit")),
                        ("hack_vs_giveup", lambda e: (e == "submit", e == "give_up")),
                        ("hack_vs_degenerate", lambda e: (e == "submit", e == "degenerate"))):
        positive, negative = split(endings)
        take = positive | negative
        fit = multivariate(shape[take][:, 1:], positive[take], args.cv_draws, rng)
        baseline.setdefault("shape_model", {})[name] = fit
        log.info(f"baseline shape-only ridge {name:<19} LOO AUC {fit['auc']:.3f} p={fit['p']:.3f}")

    comparisons = {
        name: split
        for name, split in (
            ("hack_vs_rest", lambda e: (e == "submit", e != "submit")),
            ("hack_vs_giveup", lambda e: (e == "submit", e == "give_up")),
            ("hack_vs_degenerate", lambda e: (e == "submit", e == "degenerate")),
        )
        if not args.comparisons or name in args.comparisons.split(",")
    }

    report: dict = {
        "dir": str(args.dir),
        "episodes": len(meta),
        "counts": counts,
        "early_turns": EARLY,
        "floor": FLOOR,
        "layers": list(LAYERS),
        "draws": args.draws,
        "baseline": baseline,
        "meta": meta,
        "gaussians": {},
        "univariate": {},
        "multivariate": {},
    }

    # Each feature set is assessed twice: as measured, and with everything episode shape can explain
    # subtracted out. Hacked episodes are markedly shorter, so without the controlled variant every
    # concept mean is partly a length readout.
    variants = []
    for base, matrix in sorted(data["features"].items()):
        rowsof = data["index"][base]
        raw = matrix.reshape(matrix.shape[0], -1)
        variants.append((base, raw, rowsof, matrix.shape[-1]))
        variants.append((f"{base}|ctl", residualise(raw, shape[rowsof]), rowsof, matrix.shape[-1]))

    for key, flat, rowsof, width in variants:
        local = endings[rowsof]
        log.info(f"{key:<15} {len(rowsof)} episodes " +
                 " ".join(f"{g}={int((local == g).sum())}" for g in GROUPS))

        # Per-group Gaussian parameters for every concept, which is what the density panels draw.
        report["gaussians"][key] = {
            g: {
                "n": int((local == g).sum()),
                "mean": flat[local == g].mean(axis=0).round(4).tolist(),
                "sd": flat[local == g].std(axis=0, ddof=1).round(4).tolist(),
            }
            for g in GROUPS
            if (local == g).sum() > 1
        }

        for name, split in comparisons.items():
            positive, negative = split(local)
            if positive.sum() < 5 or negative.sum() < 5:
                continue
            take = positive | negative
            sub = flat[take]
            pos = positive[take]

            scores = auc(sub, pos)
            null = permuted_auc(sub, pos, args.draws, rng)
            deviation = np.abs(scores - 0.5)
            # Family-wise p: how often a shuffle produced a concept at least this extreme anywhere.
            pvalues = np.array([(null >= d).mean() for d in deviation])

            mu1 = sub[pos].mean(axis=0)
            mu0 = sub[~pos].mean(axis=0)
            var1 = sub[pos].var(axis=0, ddof=1)
            var0 = sub[~pos].var(axis=0, ddof=1)
            pooled = np.sqrt(((pos.sum() - 1) * var1 + ((~pos).sum() - 1) * var0) / (len(pos) - 2))
            cohen = (mu1 - mu0) / np.where(pooled < 1e-9, np.nan, pooled)

            order = np.argsort(-deviation)[: args.top]
            table = []
            for slot in order:
                layer, pair = LAYERS[slot // width], int(slot % width)
                table.append(
                    {
                        "slot": int(slot),
                        "pair": pair,
                        "layer": layer,
                        "concept": pairs[pair]["concept"],
                        "antagonist": pairs[pair]["antagonist"],
                        "class": pairs[pair]["class_name"],
                        "auc": float(scores[slot]),
                        "d": float(cohen[slot]),
                        "p_fwer": float(pvalues[slot]),
                        "mean_pos": float(mu1[slot]),
                        "mean_neg": float(mu0[slot]),
                        "sd_pos": float(np.sqrt(var1[slot])),
                        "sd_neg": float(np.sqrt(var0[slot])),
                    }
                )

            cv = multivariate(sub, pos, args.cv_draws, rng)
            report["univariate"].setdefault(key, {})[name] = {
                "n_pos": int(pos.sum()),
                "n_neg": int((~pos).sum()),
                "auc": scores.round(4).tolist(),
                "d": np.nan_to_num(cohen).round(4).tolist(),
                "null_p95": float(np.quantile(null, 0.95)),
                "null_max": float(null.max()),
                "best_auc": float(scores[order[0]]),
                "survivors": int((pvalues < 0.05).sum()),
                "table": table,
            }
            report["multivariate"].setdefault(key, {})[name] = cv
            log.info(
                f"{key:<10} {name:<19} best AUC {scores[order[0]]:.3f} "
                f"(null p95 {0.5 + np.quantile(null, 0.95):.3f}, {int((pvalues < 0.05).sum())} survive)  "
                f"ridge LOO AUC {cv['auc']:.3f} p={cv['p']:.3f}"
            )

    target = args.out / "predict.json"
    target.write_text(json.dumps(report))
    log.info(f"wrote {target}")


if __name__ == "__main__":
    main()
