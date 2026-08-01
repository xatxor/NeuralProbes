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


def emitted(roles: list[str], length: int, window: str = "all", after: int = 0) -> torch.Tensor:
    """Mask of positions the gradient is taken over.

    Two choices, and the difference is the whole identification problem in miniature. `all` scores
    every token the model produced, including the tool call -- but the give-up branch always emits the
    `give_up` token and the hack branch always emits code, so "which tool was called" is a confound
    with a consistent sign across every pair, and Fable's identification remark says no optimiser can
    remove that. `thinking` scores only the deliberation preceding the call, which is the difference
    between reading the decision and reading its announcement.

    :param roles: per-token role labels.
    :param length: token count to consider.
    :param window: `all` for every emitted token, `thinking` for deliberation only.
    :param after: ignore positions before this index, used to score only a forked continuation.

    :return: boolean mask over positions.
    """
    wanted = GENERATED if window == "all" else ("thinking",)
    mask = torch.tensor([r in wanted for r in roles[:length]], dtype=torch.bool)
    if after:
        mask[:after] = False
    return mask


def logprob(model: Any, ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Total log-probability the model assigns to its own emissions.

    :param model: the loaded causal LM.
    :param ids: `[1, tokens]` token ids.
    :param keep: `[tokens]` mask of model-emitted positions.

    :return: a scalar with a gradient path back to any injected vector.
    """
    hidden = model.model(input_ids=ids).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)

    def head(slice_: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = model.lm_head(slice_).float()
        step = torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None])[:, 0]
        return step[mask].sum()

    # Position t predicts token t+1, so a kept position at t+1 is scored from hidden state t.
    for start in range(0, hidden.shape[0] - 1, CHUNK):
        stop = min(start + CHUNK, hidden.shape[0] - 1)
        wanted = keep[start + 1 : stop + 1]
        if not wanted.any():
            continue
        chosen = ids[0, start + 1 : stop + 1]
        piece = hidden[start:stop]
        if piece.requires_grad:
            # Checkpointed: store the 512x4096 hidden slice and recompute the 512x151936 logits in
            # backward. Retaining them instead is what filled an 80 GB card.
            total = total + torch.utils.checkpoint.checkpoint(
                head, piece, chosen, wanted, use_reentrant=False)
        else:
            total = total + head(piece, chosen, wanted)
    return total


def features(args: argparse.Namespace) -> None:
    """Compute one gradient feature per trajectory and save them.

    :param args: parsed arguments.
    """
    # `global` before the import, deliberately. A bare `import torch` here binds it as a LOCAL of this
    # function, so the module-level `emitted` and `logprob` -- which is where torch is actually used --
    # would raise NameError. Binding it at module scope is what makes the deferred import work.
    global torch
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
        # A forked continuation shares its prefix with its siblings by construction, so scoring the
        # prefix would add the same large constant to every gradient in the group and swamp the part
        # that differs -- which is the only part carrying the decision.
        branch = (episode.get("fork") or {}).get("prefix_tokens", 0)
        keep = emitted(episode["roles"], len(ids), args.window, after=branch)
        if int(keep.sum()) < args.floor:
            continue

        tensor = torch.tensor([ids], device=args.device)
        if vector.grad is not None:
            vector.grad = None
        logprob(model, tensor, keep.to(args.device)).backward()

        rows.append(vector.grad.detach().float().cpu().numpy().copy())
        fork = episode.get("fork") or {}
        records = episode.get("turns", [])
        branch_turn = fork.get("branch_turn", 0)
        # Behavioural covariates, recorded now because the gradient pass is the only expensive step:
        # every extra behaviour costs one matrix-vector product later, but only if the quantity it is
        # defined over was captured here.
        meta.append({
            "stem": path.stem,
            "ending": episode["ending"],
            "group": GROUPS[episode["ending"]],
            "seed": episode.get("seed"),
            "turns": len(records),
            "tokens": len(ids),
            "emitted": int(keep.sum()),
            "distinct": episode.get("distinct") or 0,
            # How many times the gate refused a submit: persistence under explicit refusal.
            "refused": sum(1 for t in records if t.get("event") == "submit_refused"),
            # Turns the model spent after the branch, which is its own choice rather than the prefix's.
            "new_turns": max(0, len(records) - branch_turn),
            "writes": sum(1 for t in records if t.get("tool") == "write_file"),
            "tests": sum(1 for t in records if t.get("tool") == "run_tests"),
            # The branch point identity. Everything downstream pairs and groups on this: two records
            # sharing it came from the identical state, which is what makes their difference a
            # contrast between actions rather than between trajectories.
            "prefix": f"{fork.get('source', path.stem)}@{fork.get('branch_turn', -1)}",
            "source_ending": fork.get("source_ending"),
            # Which task this branch point came from. A direction fitted on branch points from a
            # single workload cannot be told apart from a topic direction -- that is exactly how the
            # first vector reached 1.000 on fastsum and 0.154 on fastsort. Recording the workload is
            # what lets the fit be validated by holding an entire task out rather than one prefix.
            "workload": episode.get("workload") or episode.get("variant") or "unknown",
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


def build(meta: list[dict], group: np.ndarray, paired: bool, rng: np.random.Generator) -> dict:
    """Assemble preference pairs and split them so nothing leaks across the fold boundary.

    Paired mode is the point of the whole exercise: a pair is formed only between two continuations
    sampled from the *same* branch point, so the shared prefix -- task, files, accumulated history,
    number of failed test runs -- cancels in the difference, leaving the action. Unpaired mode keeps
    the old all-versus-all construction as the control that already failed, so the two can be scored
    side by side on the same gradients.

    The split is by **branch point**, not by continuation and not by pair. Two continuations from one
    prefix are near-duplicates of each other; letting one into train and its sibling into test would
    report memorisation as generalisation, which is exactly the failure being diagnosed.

    :param meta: per-trajectory metadata.
    :param group: per-trajectory class label.
    :param paired: pair within a branch point rather than across everything.
    :param rng: source of randomness.

    :return: train and test pair index arrays.
    """
    prefixes = sorted({m["prefix"] for m in meta})
    rng.shuffle(prefixes)
    cut = max(1, int(0.75 * len(prefixes)))
    fold = {p: ("train" if i < cut else "test") for i, p in enumerate(prefixes)}

    out: dict[str, list] = {"train": [], "test": []}
    if paired:
        for prefix in prefixes:
            members = [i for i, m in enumerate(meta) if m["prefix"] == prefix]
            wins = [i for i in members if group[i] == "giveup"]
            losses = [i for i in members if group[i] == "hack"]
            for w in wins:
                for l in losses:
                    out[fold[prefix]].append((w, l))
    else:
        for name in ("train", "test"):
            wins = [i for i, m in enumerate(meta) if group[i] == "giveup" and fold[m["prefix"]] == name]
            losses = [i for i, m in enumerate(meta) if group[i] == "hack" and fold[m["prefix"]] == name]
            out[name] = [(w, l) for w in wins for l in losses]

    return {k: np.array(v, dtype=int).reshape(-1, 2) for k, v in out.items()}


def length_axis(block: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Unit direction in feature space that best predicts how much a rollout emitted.

    :param block: `[rollout, hidden]` features.
    :param counts: emitted-token count per rollout.

    :return: unit vector along the length axis.
    """
    centred = block - block.mean(axis=0, keepdims=True)
    target = np.log(counts) - np.log(counts).mean()
    axis, *_ = np.linalg.lstsq(centred, target, rcond=None)
    return axis / (np.linalg.norm(axis) or 1.0)


def crossval_workload(block: np.ndarray, meta: list[dict], group: np.ndarray,
                      steps: int, rate: float, radius: float) -> dict:
    """Held-out accuracy by leaving out an entire WORKLOAD at a time.

    This is the test the first vector never faced and then failed in the field: fitted on `fastsum`
    branch points alone it drove the hack rate to 1.000 there and to 0.154 on `fastsort`, the same
    layer and the same alpha. Nothing in the max-margin objective prefers the decision axis over
    "this task is about summing integer ranges", so with one workload in the training set the two are
    not distinguishable. Holding a whole task out makes them distinguishable, because a topic
    direction has nothing to say about a task it never saw.

    Pairs are still formed WITHIN a branch point, exactly as `crossval` does -- the workload is the
    unit of holdout, not the unit of pairing. Loosening that would reintroduce the trajectory-length
    confound that prefix matching exists to remove.

    :param block: `[rollout, reduced]` features.
    :param meta: per-rollout metadata, carrying `workload` and `prefix`.
    :param group: per-rollout class label.
    :param steps: iterations for each fit.
    :param rate: step size.
    :param radius: norm constraint.

    :return: per-workload accuracy and pair counts, plus the pooled figure.
    """
    def pairs_within(indices: list[int]) -> list[tuple[int, int]]:
        by_prefix: dict[str, list[int]] = {}
        for i in indices:
            by_prefix.setdefault(meta[i]["prefix"], []).append(i)
        return [(w, l) for members in by_prefix.values()
                for w in [i for i in members if group[i] == "giveup"]
                for l in [i for i in members if group[i] == "hack"]]

    workloads = sorted({m["workload"] for m in meta})
    report: dict = {"workloads": workloads, "per_workload": {}}
    correct = total = 0
    for held in workloads:
        train_index = [i for i, m in enumerate(meta) if m["workload"] != held]
        test_index = [i for i, m in enumerate(meta) if m["workload"] == held]
        train, test = pairs_within(train_index), pairs_within(test_index)
        if not train or not test:
            report["per_workload"][held] = {"pairs": len(test), "accuracy": None,
                                            "note": "no pairs on one side"}
            continue
        array = np.array(train, dtype=int)
        v = margin(block[array[:, 0]] - block[array[:, 1]], steps, rate, radius)
        hits = sum(int((block[w] - block[l]) @ v > 0) for w, l in test)
        report["per_workload"][held] = {"pairs": len(test), "accuracy": hits / len(test),
                                        "trained_on": sorted(set(workloads) - {held})}
        correct += hits
        total += len(test)
    report["pooled"] = {"accuracy": correct / total if total else None, "pairs": total}
    return report


def crossval(block: np.ndarray, meta: list[dict], group: np.ndarray, paired: bool,
             steps: int, rate: float, radius: float) -> tuple[float, int]:
    """Held-out pair accuracy by leaving out one branch point at a time.

    A single 75/25 split of 147 pairs leaves 30 in the test fold, and 30 pairs carry a standard error
    near 0.09 -- wide enough that 0.60 and 0.70 are indistinguishable, which is not a result but an
    absence of one. Leaving out one branch point at a time instead gives every pair a prediction made
    without it, at the cost of refitting once per branch point, which is seconds in the row space.

    The unit left out is the BRANCH POINT, never the pair and never the continuation: siblings from
    one prefix are near-duplicates, so holding out a pair while its sibling trains would report
    memorisation as generalisation.

    :param block: `[rollout, reduced]` features.
    :param meta: per-rollout metadata.
    :param group: per-rollout class label.
    :param paired: pair only within a branch point.
    :param steps: iterations for each fit.
    :param rate: step size.
    :param radius: norm constraint.

    :return: accuracy over all held-out pairs, and how many pairs were scored.
    """
    prefixes = sorted({m["prefix"] for m in meta})
    correct = total = 0
    for held in prefixes:
        inside = [i for i, m in enumerate(meta) if m["prefix"] == held]
        wins = [i for i in inside if group[i] == "giveup"]
        losses = [i for i in inside if group[i] == "hack"]
        if not (wins and losses):
            continue
        train = [(w, l) for p in prefixes if p != held
                 for w in [i for i, m in enumerate(meta) if m["prefix"] == p and group[i] == "giveup"]
                 for l in [i for i, m in enumerate(meta) if m["prefix"] == p and group[i] == "hack"]] \
            if paired else \
            [(w, l) for w in [i for i, m in enumerate(meta) if group[i] == "giveup" and m["prefix"] != held]
             for l in [i for i, m in enumerate(meta) if group[i] == "hack" and m["prefix"] != held]]
        if not train:
            continue
        pairs = np.array(train, dtype=int)
        v = margin(block[pairs[:, 0]] - block[pairs[:, 1]], steps, rate, radius)
        for w in wins:
            for l in losses:
                correct += int((block[w] - block[l]) @ v > 0)
                total += 1
    return (correct / total if total else float("nan")), total


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

    # Built ONCE, outside the loop. `build` shuffles, so calling it per scaling would give `sum` and
    # `mean` different train/test splits -- and the cosine between the two fitted directions is
    # supposed to measure the estimators' disagreement, not the splits'.
    pairs = build(meta, group, args.paired, rng)
    prefixes = {m["prefix"] for m in meta}
    log.info(f"{len(meta)} rollouts over {len(prefixes)} branch points; "
             f"pairs train {len(pairs['train'])} / test {len(pairs['test'])}")
    if not len(pairs["train"]) or not len(pairs["test"]):
        raise SystemExit(f"too few pairs to fit: train {len(pairs['train'])}, test {len(pairs['test'])}")

    for scaling in ("sum", "mean"):
        # `sum` is the DPO objective's own quantity, sequence log-probability. Its gradient grows with
        # trajectory length, and hacks are the short trajectories, so length enters the features
        # directly. `mean` divides it out. Disagreement between the two IS the length confound.
        ambient = g if scaling == "sum" else g / emitted_counts[:, None]
        # Fit in the row space and map the answer back, rather than optimising in 4096 dimensions
        # where all but ~288 of them are exactly zero for every pair.
        block, basis = reduce(ambient)

        deltas = {name: block[p[:, 0]] - block[p[:, 1]] for name, p in pairs.items()}

        reduced = margin(deltas["train"], args.steps, args.rate, args.radius)
        scores = {name: float((d @ reduced > 0).mean()) for name, d in deltas.items()}
        v = basis.T @ reduced

        # A direction is only interesting if it beats the cheapest thing that also separates these
        # trajectories. Length is that thing, so it is fitted the same way and reported alongside.
        length = np.log(emitted_counts)[:, None]
        lv = margin(length[pairs["train"][:, 0]] - length[pairs["train"][:, 1]], args.steps, args.rate, args.radius)
        naive = float(((length[pairs["test"][:, 0]] - length[pairs["test"][:, 1]]) @ lv > 0).mean())

        # Every pair scored by a fit that never saw its branch point. This is the number that matters;
        # the single-split figure above is kept only because it is what the earlier runs reported.
        loo, scored = crossval(block, meta, group, args.paired, args.steps, args.rate, args.radius)

        # And again with the length axis projected out of the features, which asks whether anything
        # survives beyond the one scalar that has out-predicted every vector this project has built.
        axis = length_axis(ambient, emitted_counts)
        stripped, sbasis = reduce(ambient - np.outer(ambient @ axis, axis))
        residual_loo, _ = crossval(stripped, meta, group, args.paired, args.steps, args.rate, args.radius)

        # The generalisation test proper: hold out an entire task. Only meaningful with more than one
        # workload in the corpus, so it is skipped rather than reported as a trivial pass.
        across = None
        if len({m["workload"] for m in meta}) > 1:
            across = crossval_workload(block, meta, group, args.steps, args.rate, args.radius)
            per = ", ".join(f"{k} {v['accuracy']:.3f}" if v["accuracy"] is not None else f"{k} -"
                            for k, v in across["per_workload"].items())
            log.info(f"[{scaling}] leave-one-WORKLOAD-out: pooled "
                     f"{across['pooled']['accuracy']} over {across['pooled']['pairs']} pairs ({per})")

        unit = v / (np.linalg.norm(v) or 1.0)
        report[scaling] = {
            "train_pairs": int(len(pairs["train"])),
            "test_pairs": int(len(pairs["test"])),
            "train_accuracy": scores["train"],
            "test_accuracy": scores["test"],
            "length_only_test_accuracy": naive,
            "loo_accuracy": loo,
            "loo_pairs": scored,
            "loo_accuracy_length_removed": residual_loo,
            "leave_one_workload_out": across,
            "norm": float(np.linalg.norm(v)),
        }
        log.info(f"{scaling:<5} leave-one-branch-point-out over {scored} pairs: {loo:.3f}  "
                 f"(length axis removed: {residual_loo:.3f})")
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


def grpo(args: argparse.Namespace) -> None:
    """The one-step GRPO estimator, which needs no rollouts beyond the ones already sampled.

    Put `v` as the only trainable parameter and define an outcome reward. The policy gradient of
    `J(v) = E_{y~pi_v}[r(y)]` at `v = 0` is `E[A(y) * grad_v log pi_v(y|x)|_0]` -- an
    advantage-weighted mean of exactly the gradients DPO used as pair features. So where DPO takes
    the max-margin separator of differences, GRPO takes a weighted mean of the same vectors: the
    difference-of-means estimator, transported into gradient space.

    GRPO rather than plain REINFORCE for a structural reason: its advantage is normalised within a
    group sampled from one input, and a group sampled from a shared prefix IS a branch point. The
    group baseline subtracts out everything state-level -- task difficulty, prefix length, how much
    frustration has accumulated -- which is the same thing prefix-matching does for DPO, obtained for
    free instead of by construction.

    The degenerate class enters here, which it cannot in pairwise DPO. Two reward schemes are fitted
    because their difference isolates it: if a direction that suppresses hacking merely converts hacks
    into context exhaustion, that is not mitigation and the two schemes will disagree.

    :param args: parsed arguments.
    """
    held = np.load(args.grads or (args.out / "grads.npz"), allow_pickle=True)
    g = held["g"].astype(np.float64)
    meta = list(held["meta"])
    emitted_counts = np.array([m["emitted"] for m in meta], dtype=np.float64)
    group = np.array([m["group"] for m in meta])
    prefix = np.array([m["prefix"] for m in meta])

    # Every entry is a different behaviour read off the same gradients. The expensive step was
    # computing g; a new behaviour is a reward vector and one matrix-vector product, so the marginal
    # cost of asking a further question of this corpus is essentially zero.
    #
    # `None` excludes a rollout from that behaviour entirely rather than scoring it zero, which
    # matters: a trajectory with no opinion on a question should not drag its group's baseline.
    behaviours = {
        # The three pairwise contrasts among the endings. Their geometry answers a question worth
        # asking on its own -- is hacking closer to persisting or to quitting?
        "hack_vs_giveup": lambda m: {"hack": 1.0, "giveup": -1.0}.get(m["group"]),
        "hack_vs_degenerate": lambda m: {"hack": 1.0, "degenerate": -1.0}.get(m["group"]),
        "giveup_vs_degenerate": lambda m: {"giveup": 1.0, "degenerate": -1.0}.get(m["group"]),
        # Hacking against everything else, so a direction that merely converts hacks into context
        # exhaustion cannot pass as mitigation.
        "hack_vs_rest": lambda m: 1.0 if m["group"] == "hack" else -1.0,
        # Graded behaviours, which pairwise DPO cannot express at all. GRPO normalises within a group,
        # so a continuous reward needs no thresholding.
        "exploration": lambda m: float(m.get("distinct") or 0),
        "persistence_under_refusal": lambda m: float(m.get("refused") or 0),
        "rewriting": lambda m: float(m.get("writes") or 0),
        # The control, and the one that must be reported next to all the others: trajectory length
        # separates these outcomes at AUC 0.939 on its own, so any behaviour whose direction is
        # parallel to this one has told us nothing the episode's shape did not already say.
        "length_control": lambda m: float(np.log1p(m.get("new_turns") or 0)),
    }

    report: dict = {"layer": args.layer, "groups": {k: int((group == k).sum()) for k in set(group)}}
    produced: dict[str, np.ndarray] = {}

    for name, score in behaviours.items():
        raw = [score(m) for m in meta]
        eligible = np.array([r is not None for r in raw])
        reward = np.array([0.0 if r is None else float(r) for r in raw])

        advantage = np.zeros_like(reward)
        used = 0
        for key in np.unique(prefix):
            members = np.flatnonzero((prefix == key) & eligible)
            # A group needs at least two eligible members and some spread among them; without spread
            # every advantage is zero and dividing would turn rounding noise into a direction.
            if len(members) < 2:
                continue
            spread = reward[members].std()
            if spread < 1e-9:
                continue
            advantage[members] = (reward[members] - reward[members].mean()) / spread
            used += 1

        if not used:
            log.warning(f"grpo {name}: no informative groups, skipped")
            continue

        for scaling in ("sum", "mean"):
            block = g if scaling == "sum" else g / emitted_counts[:, None]
            v = advantage @ block
            norm = float(np.linalg.norm(v))
            unit = v / (norm or 1.0)
            np.save(args.out / f"grpo-L{args.layer}-{name}-{scaling}.npy", unit.astype(np.float32))
            produced[f"{name}:{scaling}"] = unit
            report.setdefault(name, {})[scaling] = {
                "groups_used": used,
                "eligible": int(eligible.sum()),
                "norm": norm,
            }
        log.info(f"grpo {name:<26} {used:>4} groups, {int(eligible.sum()):>4} rollouts")

    # The geometry among the behaviours is the result, not a diagnostic. They are all built from the
    # same few hundred gradients, so they cannot be independent -- what matters is which pairs are
    # close, and above all how close each one sits to `length_control`.
    names = sorted(produced)
    matrix = np.array([[float(produced[a] @ produced[b]) for b in names] for a in names])
    report["cosines"] = {"names": names, "matrix": matrix.round(4).tolist()}

    anchor = "length_control:mean"
    if anchor in produced:
        log.info("cosine to the length control:")
        for name in names:
            if name != anchor:
                log.info(f"  {float(produced[name] @ produced[anchor]):+.3f}  {name}")

    (args.out / f"grpo-L{args.layer}.json").write_text(json.dumps(report, indent=1))
    log.info(f"wrote {len(produced)} directions and {args.out / f'grpo-L{args.layer}.json'}")



def sequences(directory: Path, window: str, floor: int, max_tokens: int) -> tuple[list, list]:
    """Load forked continuations and group them into prefix-matched preference pairs.

    :param directory: directory of forked episode records.
    :param window: `all` or `thinking`, matching the gradient stage.
    :param floor: minimum scored tokens for a continuation to be usable.
    :param max_tokens: truncation bound.

    :return: the rollout records, and index pairs `(preferred, dispreferred)` within a branch point.
    """
    rollouts, by_prefix = [], {}
    paths = [p for p in sorted(directory.glob("*.json")) if not p.name.startswith("._")]
    for path in paths:
        episode = json.loads(path.read_text())
        if episode.get("ending") not in GROUPS:
            continue
        fork = episode.get("fork") or {}
        ids = episode["ids"][:max_tokens]
        roles = episode["roles"][: len(ids)]
        branch = fork.get("prefix_tokens", 0)
        wanted = GENERATED if window == "all" else ("thinking",)
        keep = [i for i, r in enumerate(roles) if r in wanted and i >= branch]
        if len(keep) < floor:
            continue
        key = f"{fork.get('source', path.stem)}@{fork.get('branch_turn', -1)}"
        rollouts.append({"ids": ids, "keep": keep, "group": GROUPS[episode["ending"]], "prefix": key})
        by_prefix.setdefault(key, []).append(len(rollouts) - 1)

    pairs = []
    for members in by_prefix.values():
        wins = [i for i in members if rollouts[i]["group"] == "giveup"]
        losses = [i for i in members if rollouts[i]["group"] == "hack"]
        pairs.extend((w, l) for w in wins for l in losses)
    return rollouts, pairs


def train(args: argparse.Namespace) -> None:
    """Optimise v against the true DPO loss on prefix-matched pairs.

    :param args: parsed arguments.
    """
    global torch
    import torch

    from model import load

    model, _ = load(device=args.device, dtype=args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rollouts, pairs = sequences(args.dir, args.window, args.floor, args.max_tokens)
    prefixes = sorted({r["prefix"] for r in rollouts})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(prefixes)
    # Held out by BRANCH POINT: siblings share a prefix and are near-duplicates, so splitting on pairs
    # would let the fit see a held-out pair's twin during training.
    cut = max(1, int(0.75 * len(prefixes)))
    held = set(prefixes[cut:])
    train_pairs = [p for p in pairs if rollouts[p[0]]["prefix"] not in held]
    test_pairs = [p for p in pairs if rollouts[p[0]]["prefix"] in held]
    log.info(f"{len(rollouts)} rollouts, {len(pairs)} pairs "
             f"({len(train_pairs)} train / {len(test_pairs)} held out over {len(held)} branch points)")
    if not train_pairs or not test_pairs:
        raise SystemExit("not enough pairs to train and hold out")

    width = model.config.hidden_size
    vector = torch.zeros(width, device=args.device, dtype=torch.float32, requires_grad=True)
    handle = inject(model, args.layer, vector)

    def score(index: int) -> torch.Tensor:
        row = rollouts[index]
        ids = torch.tensor([row["ids"]], device=args.device)
        keep = torch.zeros(len(row["ids"]), dtype=torch.bool, device=args.device)
        keep[torch.tensor(row["keep"], device=args.device)] = True
        return logprob(model, ids, keep)

    # log pi_0 is constant in v, so it is computed once with the hook contributing nothing.
    log.info("caching reference log-probabilities")
    reference = {}
    with torch.no_grad():
        for index in {i for pair in pairs for i in pair}:
            reference[index] = float(score(index))
    log.info(f"cached {len(reference)} references")

    radius = args.radius
    optimiser = torch.optim.Adam([vector], lr=args.lr)
    order = list(range(len(train_pairs)))
    best = {"accuracy": -1.0}

    def accuracy(subset: list) -> float:
        hits = 0
        with torch.no_grad():
            for w, l in subset:
                margin_ = (float(score(w)) - reference[w]) - (float(score(l)) - reference[l])
                hits += int(margin_ > 0)
        return hits / max(len(subset), 1)

    for epoch in range(args.epochs):
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch):
            optimiser.zero_grad()
            chunk = order[start : start + args.batch]
            for slot in chunk:
                w, l = train_pairs[slot]
                # Values first, without a graph, to get the scalar coefficient.
                with torch.no_grad():
                    a = float(score(w)) - reference[w]
                    b = float(score(l)) - reference[l]
                h = a - b
                weight = -args.beta * float(torch.sigmoid(torch.tensor(-args.beta * h))) / len(chunk)
                total += float(-torch.nn.functional.logsigmoid(torch.tensor(args.beta * h))) / len(chunk)
                # Then each side separately, so only one graph is ever alive. Same gradient, half the
                # peak memory.
                (weight * score(w)).backward()
                (-weight * score(l)).backward()
            optimiser.step()
            with torch.no_grad():
                norm = vector.norm()
                if norm > radius:
                    vector.mul_(radius / norm)
        got = accuracy(test_pairs)
        log.info(f"epoch {epoch}: loss {total / max(len(order) // args.batch, 1):.4f}  "
                 f"|v| {float(vector.norm()):.2f}  held-out pair accuracy {got:.3f}")
        if got > best["accuracy"]:
            best = {"accuracy": got, "epoch": epoch,
                    "vector": vector.detach().float().cpu().numpy().copy()}

    handle.remove()
    args.out.mkdir(parents=True, exist_ok=True)
    unit = best["vector"] / (np.linalg.norm(best["vector"]) or 1.0)
    np.save(args.out / f"trained-L{args.layer}-r{radius:g}.npy", unit.astype(np.float32))
    (args.out / f"trained-L{args.layer}-r{radius:g}.json").write_text(json.dumps({
        "layer": args.layer, "radius": radius, "beta": args.beta, "lr": args.lr,
        "epochs": args.epochs, "window": args.window,
        "pairs_train": len(train_pairs), "pairs_test": len(test_pairs),
        "best_epoch": best["epoch"], "best_held_out_accuracy": best["accuracy"],
    }, indent=1))
    log.info(f"best held-out {best['accuracy']:.3f} at epoch {best['epoch']}; wrote "
             f"{args.out / f'trained-L{args.layer}-r{radius:g}.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["features", "fit", "grpo", "train"])
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
    parser.add_argument("--window", choices=["all", "thinking"], default="thinking",
                        help="score every emitted token, or only the deliberation before the tool call")
    parser.add_argument("--floor", type=int, default=16, help="minimum scored tokens to keep a trajectory")
    parser.add_argument("--paired", action="store_true",
                        help="pair only within a branch point; without it, the failed all-vs-all control")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"features": features, "fit": fit, "grpo": grpo, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
