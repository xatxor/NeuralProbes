"""Small dependency-free checks for the steering experiment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steer import ALPHAS, CONCEPTS, LAYERS, Steerer, condition_specs, task_key, worker_command
from summarize import plot_results, summarize


def main() -> None:
    conditions = condition_specs(list(CONCEPTS), list(LAYERS), list(ALPHAS))
    assert len(conditions) == 301
    assert sum(row["alpha"] == 0 for row in conditions) == 1
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
    args = SimpleNamespace(
        benchmark="math_500", num_workers=4, concept_pairs=list(CONCEPTS),
        layers=list(LAYERS), alphas=list(ALPHAS), limit=1,
    )
    assert "--alphas=-0.1,-0.05,0.0,0.05,0.1" in worker_command(args, 0)

    rows = pd.DataFrame(
        [
            {"benchmark": "aime_2024", "id": "0", "concept_pair": None, "concept": None, "layer": None, "alpha": 0.0, "correct": False, "reasoning_token_count": 100, "generation_seconds": 10, "hit_context_limit": False},
            {"benchmark": "aime_2024", "id": "0", "concept_pair": 367, "concept": CONCEPTS[367], "layer": 11, "alpha": 0.05, "correct": True, "reasoning_token_count": 120, "generation_seconds": 12, "hit_context_limit": False},
        ]
    )
    _, effects = summarize(rows)
    assert effects.iloc[0].delta_reasoning_tokens == 20
    assert effects.iloc[0].delta_accuracy_pp == 100
    original_results = sys.modules["summarize"].RESULTS
    with tempfile.TemporaryDirectory() as directory:
        sys.modules["summarize"].RESULTS = Path(directory)
        assert len(plot_results(rows)) == 2
    sys.modules["summarize"].RESULTS = original_results
    print("steering checks passed")


if __name__ == "__main__":
    main()
