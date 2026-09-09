"""Small dependency-free checks for the steering experiment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import steer
from steer import ALPHAS, CONCEPTS, DEFAULT_CONCEPT_PAIRS, DEFAULT_LAYERS, Steerer, condition_specs, task_key, worker_command
from summarize import plot_results, summarize


def main() -> None:
    conditions = condition_specs(list(DEFAULT_CONCEPT_PAIRS), list(DEFAULT_LAYERS), list(ALPHAS))
    assert len(conditions) == 171
    assert {len((conditions * 30)[worker::10]) for worker in range(10)} == {513}
    assert sum(row["alpha"] == 0 for row in conditions) == 1
    # Baselines lead, so a run cut short still has the reference every delta needs.
    assert conditions[0]["alpha"] == 0.0 and conditions[1]["alpha"] != 0.0
    # Default order finishes one concept's whole alpha curve before the next concept.
    steered_conditions = [row for row in conditions if row["alpha"] != 0.0]
    leading = [row for row in steered_conditions if row["pair"] == steered_conditions[0]["pair"]]
    assert steered_conditions[: len(leading)] == leading
    assert {row["alpha"] for row in leading} == set(ALPHAS)

    # The other order covers every concept at one strength first.
    by_alpha = condition_specs(list(DEFAULT_CONCEPT_PAIRS), list(DEFAULT_LAYERS), list(ALPHAS), 1, "alpha")
    steered_by_alpha = [row for row in by_alpha if row["alpha"] != 0.0]
    first = [row for row in steered_by_alpha if row["alpha"] == steered_by_alpha[0]["alpha"]]
    assert steered_by_alpha[: len(first)] == first
    assert {row["pair"] for row in first} == set(DEFAULT_CONCEPT_PAIRS)
    assert sorted(map(repr, by_alpha)) == sorted(map(repr, conditions))
    assert all(
        row["alpha"] == 0 or (row["pair"] is not None and row["layer"] is not None)
        for row in conditions
    )

    layers = [torch.nn.Identity() for _ in range(25)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    steerer = Steerer(model, {(367, 11): torch.tensor([2.0, 0.0])})
    values = torch.zeros(1, 3, 2)
    with steerer.apply(367, 11, 0.1):
        steered = layers[10](values)
        untouched = layers[9](values)
    assert torch.equal(steered[0, :2], values[0, :2])
    assert torch.allclose(steered[0, -1], torch.tensor([0.2, 0.0]))
    assert torch.equal(untouched, values)
    assert torch.equal(layers[10](values), values)

    task = {"benchmark": "aime_2024", "id": "0", "pair": 367, "layer": 11, "alpha": -0.05}
    assert task_key(task) == "aime_2024:0:pair-367:L11:a-0.05"
    assert task_key({**task, "pair": None, "layer": None, "alpha": 0.0}) == "aime_2024:0:baseline"
    assert task_key({**task, "pair": None, "layer": None, "alpha": 0.0, "baseline_repeat": 1}) == "aime_2024:0:baseline:repeat-1"
    args = SimpleNamespace(
        benchmark="math_500", num_workers=4, concept_pairs=list(DEFAULT_CONCEPT_PAIRS),
        layers=list(DEFAULT_LAYERS), alphas=list(ALPHAS), baseline_repeats=1, limit=1,
        vector_dir=Path("/fake/vector/dir"), disable_loop_detection=False, loop_ngram_size=4,
        loop_window_tokens=1024, loop_unique_ratio_threshold=0.2, loop_check_every=64,
        loop_consecutive_windows=3, loop_min_new_tokens=2048, loop_extra_tokens=512,
        max_new_tokens=16384, order="concept", batch_size=8,
        results_dir=Path("/fake/results"),
    )
    command = worker_command(args, 0)
    assert "--order" in command and "concept" in command
    assert "--batch-size" in command
    assert "--results-dir" in command
    assert "--alphas=-3.0,-2.5,-2.0,-1.5,-1.0,1.0,1.5,2.0,2.5,3.0" in command
    assert "--loop-window-tokens" in command and "--disable-loop-detection" not in command
    assert "--max-new-tokens" in command
    assert steer.loop_options(args)["unique_ratio_threshold"] == 0.2
    assert steer.loop_options(SimpleNamespace(**{**vars(args), "disable_loop_detection": True})) is None

    base = {
        "key": "aime_2024:0:baseline", "model": steer.MODEL_ID, "dtype": "float16",
        "steering_version": steer.STEERING_VERSION, "vector_capture_key": "key-a",
        "prompt_sha256": steer.prompt_hash({"prompt": "p"}),
    }
    baseline_task = {"benchmark": "aime_2024", "id": "0", "pair": None, "layer": None, "alpha": 0.0, "prompt": "p"}
    # A run that stopped on its own inside the new budget survives a budget change.
    kept = {**base, "generated_token_count": 4000, "hit_context_limit": False, "hit_token_budget": False}
    assert steer.compatible(kept, baseline_task, "key-a", 16384)
    # One that ran to the context wall has to be regenerated under the budget.
    truncated = {**base, "generated_token_count": 40823, "hit_context_limit": True, "hit_token_budget": False}
    assert not steer.compatible(truncated, baseline_task, "key-a", 16384)
    assert not steer.compatible(kept, baseline_task, "other-key", 16384)

    # Batches never mix steering conditions, and every task lands in exactly one.
    mixed = [
        {"pair": 657, "layer": 18, "alpha": 1.0, "id": str(i)} for i in range(10)
    ] + [
        {"pair": 657, "layer": 18, "alpha": -1.0, "id": str(i)} for i in range(3)
    ] + [
        {"pair": None, "layer": None, "alpha": 0.0, "id": str(i)} for i in range(2)
    ]
    batches = steer.condition_batches(mixed, 4)
    assert [len(batch) for batch in batches] == [4, 4, 2, 3, 2]
    for batch in batches:
        cells = {(task["pair"], task["layer"], task["alpha"]) for task in batch}
        assert len(cells) == 1
    assert sum(len(batch) for batch in batches) == len(mixed)

    # A row is cut at its own EOS, so padding from longer rows never leaks into it.
    assert steer.split_continuation([5, 6, 99, 0, 0], {99}) == [5, 6, 99]
    assert steer.split_continuation([5, 6, 7], {99}) == [5, 6, 7]

    # Loop detection is per row: a repeating row must not condemn its neighbours, and a
    # row that already emitted EOS must not trip on the padding that follows it.
    options = {
        "ngram_size": 2, "window_tokens": 32, "unique_ratio_threshold": 0.3,
        "check_every": 8, "consecutive_windows": 2, "min_new_tokens": 32, "extra_tokens": 0,
    }
    prompt = [1, 2, 3, 4]
    detector = steer.BatchedLoopDetector(prompt_tokens=len(prompt), batch_size=3, eos={99}, options=options)
    stopped = torch.zeros(3, dtype=torch.bool)
    for length in range(1, 140):
        rows = [
            ([7, 8] * length)[:length],  # two tokens forever: no diversity
            list(range(100, 100 + length)),  # never repeats
            ([5, 99] + [0] * length)[:length],  # emits EOS, then gets padded
        ]
        input_ids = torch.tensor([prompt + row for row in rows])
        stopped |= detector(input_ids, None).cpu()
    assert stopped.tolist() == [True, False, False]
    assert detector.metadata(0, 140)["detected"]
    assert not detector.metadata(1, 140)["detected"]
    # The padded row was retired at its EOS rather than being read as a loop.
    assert detector.finished[2] and not detector.metadata(2, 140)["detected"]

    rows = pd.DataFrame(
        [
            {"benchmark": "aime_2024", "id": "0", "concept_pair": None, "concept": None, "layer": None, "alpha": 0.0, "baseline_repeat": None, "correct": False, "reasoning_token_count": 100, "generation_seconds": 10, "hit_context_limit": False},
            {"benchmark": "aime_2024", "id": "0", "concept_pair": None, "concept": None, "layer": None, "alpha": 0.0, "baseline_repeat": 1, "correct": True, "reasoning_token_count": 120, "generation_seconds": 10, "hit_context_limit": False},
            {"benchmark": "aime_2024", "id": "0", "concept_pair": 367, "concept": CONCEPTS[367], "layer": 11, "alpha": 0.05, "baseline_repeat": None, "correct": True, "reasoning_token_count": 120, "generation_seconds": 12, "hit_context_limit": False},
        ]
    )
    _, effects = summarize(rows)
    assert effects.iloc[0].delta_reasoning_tokens == 0
    assert effects.iloc[0].delta_accuracy_pp == 0
    original_results = sys.modules["summarize"].RESULTS
    with tempfile.TemporaryDirectory() as directory:
        sys.modules["summarize"].RESULTS = Path(directory)
        assert len(plot_results(rows)) == 2
    sys.modules["summarize"].RESULTS = original_results
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        old_results = steer.RESULTS
        steer.RESULTS = root
        (root / "steering.jsonl").write_text('{"key": "previous"}\n')
        (root / "steering.worker-00-of-01.jsonl").write_text("")
        steer.merge_shards(
            SimpleNamespace(num_workers=1, worker_index=None, results_dir=root), [], "test-capture-key"
        )
        assert (root / "steering.jsonl").read_text() == '{"key": "previous"}\n'
        steer.RESULTS = old_results
    print("steering checks passed")


if __name__ == "__main__":
    main()
