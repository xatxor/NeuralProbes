#! /usr/bin/env python

"""Extract a steering vector for the reward-hacking decision by preference optimisation.

The object is a single vector `v` added to the residual stream at one layer, with the model frozen --
BiPO's parameterisation. Training it with a DPO objective on pairs of trajectories is the plan, but
the objective does not have to be run end to end to find where it converges.

Write the DPO loss over `v` for a pair sharing a prefix `x`:

    h(v) = [log pi_v(y_w|x) - log pi_0(y_w|x)] - [log pi_v(y_l|x) - log pi_0(y_l|x)],
    L(v) = -log sigmoid(beta * h(v)).

Because h(0) = 0, a Taylor expansion at the origin gives h(v) = <v, dg> + O(||v||^2) with

    dg = grad_v log pi_v(y_w|x)|_0  -  grad_v log pi_v(y_l|x)|_0,

so the loss is logistic regression on the features `dg`, with no bias term and every label positive.
The consequence that matters operationally: `g` is **one backward pass per trajectory**, and every
pair is then a subtraction. 196 backward passes yield all 30 x 166 pairs rather than 4980 training
steps, and the direction that logistic regression converges to is the direction DPO converges to --
gradient descent on a separable logistic loss converges to the L2 max-margin separator.

What that buys and what it does not. It buys the correct object cheaply, inside the linear regime the
theory describes. It does not buy identification: the max-margin separator of these features is *a*
direction that flips the outcome at every pair, not necessarily *the* reward-hacking direction. Any
cheaper axis that also separates -- trajectory length, tool-call syntax, the `give_up` token itself --
can be the thing that gets found. That is a property of the data, not of the optimiser, and it is why
this file computes both a length-normalised and a raw variant, and why a paired-at-the-branch-point
dataset is worth building before believing any of it.
"""

# Annotations are strings, so `torch.Tensor` in a signature does not require torch at import time --
# which is what lets `fit` run in a venv that has no torch at all.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# torch is imported inside `features` rather than here. Only that stage needs a GPU; `fit` is
# arithmetic on saved gradients, and keeping it torch-free lets it run in the lighter venv on a box
# whose GPUs are busy with someone else's job.
log = logging.getLogger("bipo")

# Roles the model itself produced. Tool observations are inputs, not emissions, so they are excluded
# from the log-probability: pi(y|x) must not be credited for text the environment wrote.
GENERATED = ("thinking", "tool_call", "answer")
GROUPS = {"submit": "hack", "give_up": "giveup", "degenerate": "degenerate"}
# Chunk for the language-model head. A 151936-wide logits tensor over a full trajectory is several
# gigabytes for a quantity we reduce to one scalar per position.
CHUNK = 512


def inject(model: Any, layer: int, vector: torch.Tensor) -> Any:
    """Add a vector to the residual stream at the output of one block.

    The published vectors are named one-indexed, so layer 18 is `model.layers[17]`, matching
    `readout.py`. The hook is registered permanently and reads `vector` by closure, so the same
    handle serves both the gradient pass (vector of zeros, requires_grad) and later steering.

    :param model: the loaded causal LM.
    :param layer: one-indexed block whose output is displaced.
    :param vector: the displacement, broadcast over every position.

    :return: the hook handle.
    """

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vector.to(output[0].dtype),) + output[1:]
        return output + vector.to(output.dtype)

    return model.model.layers[layer - 1].register_forward_hook(hook)


def emitted(roles: list[str], length: int) -> torch.Tensor:
    """Mask of positions whose token the model produced.

    :param roles: per-token role labels.
    :param length: token count to consider.

    :return: boolean mask over positions.
    """
    return torch.tensor([r in GENERATED for r in roles[:length]], dtype=torch.bool)


def logprob(model: Any, ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Total log-probability the model assigns to its own emissions.

    :param model: the loaded causal LM.
    :param ids: `[1, tokens]` token ids.
    :param keep: `[tokens]` mask of model-emitted positions.

    :return: a scalar with a gradient path back to any injected vector.
    """
    hidden = model.model(input_ids=ids).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    # Position t predicts token t+1, so a kept position at t+1 is scored from hidden state t.
    for start in range(0, hidden.shape[0] - 1, CHUNK):
        stop = min(start + CHUNK, hidden.shape[0] - 1)
        wanted = keep[start + 1 : stop + 1]
        if not wanted.any():
            continue
        logits = model.lm_head(hidden[start:stop]).float()
        chosen = ids[0, start + 1 : stop + 1]
        step = torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None])[:, 0]
        total = total + step[wanted].sum()
    return total


def features(args: argparse.Namespace) -> None:
    """Compute one gradient feature per trajectory and save them.

    :param args: parsed arguments.
    """
    import torch

    from model import load

    model, _ = load(device=args.device, dtype=args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    width = model.config.hidden_size
    vector = torch.zeros(width, device=args.device, dtype=torch.float32, requires_grad=True)
    inject(model, args.layer, vector)

    # No sharding, deliberately. One forward and one backward per trajectory is a couple of seconds on
    # an A100, so all 288 finish inside twenty minutes on a single GPU -- less than the staggered
    # weight download that ten containers would need before any of them started.
    rows, meta = [], []
    # `._*` are macOS AppleDouble sidecars. A tar built on the Mac carries one per file that has an
    # xattr, they are binary, and they match `*.json` -- which is exactly how this job died once.
    paths = [p for p in sorted(args.dir.glob("*.json")) if not p.name.startswith("._")]
    for count, path in enumerate(paths, start=1):
        episode = json.loads(path.read_text())
        if episode.get("ending") not in GROUPS:
            continue
        ids = episode["ids"][: args.max_tokens]
        keep = emitted(episode["roles"], len(ids))
        if not keep.any():
            continue

        tensor = torch.tensor([ids], device=args.device)
        if vector.grad is not None:
            vector.grad = None
        logprob(model, tensor, keep.to(args.device)).backward()

        rows.append(vector.grad.detach().float().cpu().numpy().copy())
        meta.append({
            "stem": path.stem,
            "ending": episode["ending"],
            "group": GROUPS[episode["ending"]],
            "seed": episode.get("seed"),
            "turns": len(episode.get("turns", [])),
            "tokens": len(ids),
            "emitted": int(keep.sum()),
        })
        if count % 20 == 0:
            log.info(f"{count}/{len(paths)} trajectories, {len(rows)} kept")

    args.out.mkdir(parents=True, exist_ok=True)
    # Static name: `${SHARD}` is not substituted inside a DataSphere `outputs:` block, and a missing
    # declared output aborts the whole upload after the job has already exited 0.
    target = args.out / "grads.npz"
    np.savez_compressed(target, g=np.stack(rows).astype(np.float32), meta=np.array(meta, dtype=object),
                        layer=np.array(args.layer))
    counts: dict[str, int] = {}
    for row in meta:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    log.info(f"wrote {target}: {len(rows)} trajectories {counts}")


def margin(deltas: np.ndarray, steps: int, rate: float, radius: float) -> np.ndarray:
    """Fit the separating direction by gradient descent on the linearised DPO loss.

    Plain gradient descent rather than a solver, because the point is to converge where DPO converges;
    the norm ball keeps the iterate inside the region where the linearisation that produced these
    features is still a fair description of the model.

    :param deltas: `[pair, hidden]` gradient differences, all labelled positive.
    :param steps: iterations.
    :param rate: step size.
    :param radius: norm constraint.

    :return: the fitted direction, unnormalised.
    """
    scale = float(np.linalg.norm(deltas, axis=1).mean()) or 1.0
    x = deltas / scale
    v = np.zeros(x.shape[1], dtype=np.float64)
    for step in range(steps):
        # d/dv of mean log(1 + exp(-<v, x>)) is -mean sigma(-<v,x>) x. Written as a matrix-vector
        # product: the obvious `(weight[:, None] * x).mean(axis=0)` allocates a full copy of x on
        # every one of the thousands of iterations, which is the whole running time.
        weight = 1.0 / (1.0 + np.exp(np.clip(x @ v, -60, 60)))
        v += rate * (x.T @ weight) / len(weight)
        norm = np.linalg.norm(v)
        if norm > radius:
            v *= radius / norm
    return v


def reduce(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite gradients in an orthonormal basis of their own row space.

    Every pair difference is a difference of two rows, so it lies in that row space -- which has rank
    at most the number of trajectories, 288, against 4096 ambient dimensions. Fitting there is exact,
    not an approximation, and shrinks the optimisation by more than an order of magnitude.

    :param block: `[trajectory, hidden]` gradients.

    :return: coordinates in the basis, and the basis itself as `[rank, hidden]`.
    """
    _, singular, right = np.linalg.svd(block, full_matrices=False)
    basis = right[singular > singular[0] * 1e-10]
    return block @ basis.T, basis


def fit(args: argparse.Namespace) -> None:
    """Form pairs, fit the direction, and score it out of sample.

    :param args: parsed arguments.
    """
    held = np.load(args.grads or (args.out / "grads.npz"), allow_pickle=True)
    g = held["g"].astype(np.float64)
    meta = list(held["meta"])
    emitted_counts = np.array([m["emitted"] for m in meta], dtype=np.float64)
    group = np.array([m["group"] for m in meta])

    report: dict = {"layer": args.layer, "counts": {k: int((group == k).sum()) for k in set(group)}}
    rng = np.random.default_rng(args.seed)

    for scaling in ("sum", "mean"):
        # `sum` is the DPO objective's own quantity, sequence log-probability. Its gradient grows with
        # trajectory length, and hacks are the short trajectories, so length enters the features
        # directly. `mean` divides it out. Disagreement between the two IS the length confound.
        ambient = g if scaling == "sum" else g / emitted_counts[:, None]
        # Fit in the row space and map the answer back, rather than optimising in 4096 dimensions
        # where all but ~288 of them are exactly zero for every pair.
        block, basis = reduce(ambient)

        wins = np.flatnonzero(group == "giveup")
        losses = np.flatnonzero(group == "hack")
        rng.shuffle(wins)
        rng.shuffle(losses)
        # Split by trajectory, never by pair: the same trajectory appearing in both folds would let
        # the fit memorise its gradient and score itself.
        cut_w, cut_l = int(0.8 * len(wins)), int(0.8 * len(losses))
        folds = {
            "train": (wins[:cut_w], losses[:cut_l]),
            "test": (wins[cut_w:], losses[cut_l:]),
        }

        pairs = {
            name: np.array([(w, l) for w in ws for l in ls])
            for name, (ws, ls) in folds.items()
        }
        deltas = {name: block[p[:, 0]] - block[p[:, 1]] for name, p in pairs.items()}

        reduced = margin(deltas["train"], args.steps, args.rate, args.radius)
        scores = {name: float((d @ reduced > 0).mean()) for name, d in deltas.items()}
        v = basis.T @ reduced

        # A direction is only interesting if it beats the cheapest thing that also separates these
        # trajectories. Length is that thing, so it is fitted the same way and reported alongside.
        length = np.log(emitted_counts)[:, None]
        lv = margin(length[pairs["train"][:, 0]] - length[pairs["train"][:, 1]], args.steps, args.rate, args.radius)
        naive = float(((length[pairs["test"][:, 0]] - length[pairs["test"][:, 1]]) @ lv > 0).mean())

        unit = v / (np.linalg.norm(v) or 1.0)
        report[scaling] = {
            "train_pairs": int(len(pairs["train"])),
            "test_pairs": int(len(pairs["test"])),
            "train_accuracy": scores["train"],
            "test_accuracy": scores["test"],
            "length_only_test_accuracy": naive,
            "norm": float(np.linalg.norm(v)),
        }
        np.save(args.out / f"vector-L{args.layer}-{scaling}.npy", unit.astype(np.float32))
        log.info(
            f"{scaling:<5} pairs {len(pairs['train'])}/{len(pairs['test'])}  "
            f"train {scores['train']:.3f}  held-out {scores['test']:.3f}  "
            f"(length alone {naive:.3f})"
        )

    # The adversarial control, in the same space rather than as a scalar: the direction in gradient
    # space that best predicts how long a trajectory is. Corollary 2 says the max-margin solution can
    # be any separator, and length is the cheapest one available here -- so the cosine between the
    # fitted direction and this one is the number that says whether that is what happened.
    centred = g - g.mean(axis=0, keepdims=True)
    target = np.log(emitted_counts) - np.log(emitted_counts).mean()
    length_direction, *_ = np.linalg.lstsq(centred, target, rcond=None)
    length_direction /= np.linalg.norm(length_direction) or 1.0
    np.save(args.out / f"vector-L{args.layer}-length.npy", length_direction.astype(np.float32))

    a = np.load(args.out / f"vector-L{args.layer}-sum.npy").astype(np.float64)
    b = np.load(args.out / f"vector-L{args.layer}-mean.npy").astype(np.float64)
    report["cosine_sum_vs_mean"] = float(a @ b)
    report["cosine_sum_vs_length"] = float(a @ length_direction)
    report["cosine_mean_vs_length"] = float(b @ length_direction)
    log.info(f"cosine sum vs mean {report['cosine_sum_vs_mean']:+.3f}; "
             f"vs the length direction: sum {report['cosine_sum_vs_length']:+.3f}, "
             f"mean {report['cosine_mean_vs_length']:+.3f}")

    (args.out / f"bipo-L{args.layer}.json").write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.out / f'bipo-L{args.layer}.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["features", "fit"])
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("bipo"))
    parser.add_argument("--grads", type=Path, default=None, help="gradient features; defaults to <out>/grads.npz")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"features": features, "fit": fit}[args.stage](args)


if __name__ == "__main__":
    main()
