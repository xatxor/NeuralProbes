"""CPU-only checks for split determinism and GCG candidate sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gcg import sample_candidates  # noqa: E402
from pipeline import stratified_split  # noqa: E402


def main() -> None:
    rows = [{"id": str(i), "prompt": f"p{i}", "category": f"c{i % 4}"} for i in range(40)]
    first = stratified_split(rows, 8, 12, 42)
    second = stratified_split(rows, 8, 12, 42)
    assert first == second
    assert not {row["id"] for row in first["train"]} & {row["id"] for row in first["test"]}
    assert {row["category"] for row in first["train"]} == {"c0", "c1", "c2", "c3"}

    control = torch.tensor([1, 1, 1])
    gradient = torch.tensor([[0.1, -2.0, 0.5, 4.0], [0.3, 0.2, -1.0, 0.7], [1.0, 0.0, 0.2, -3.0]])
    distances = torch.zeros_like(gradient)
    candidates = sample_candidates(
        control, gradient, distances, torch.tensor([0, 1, 2, 3]),
        6, 4, 0.1, 10.0, 42, set(),
    )
    assert candidates and all(candidate.shape == control.shape for candidate in candidates)
    assert len({tuple(candidate.tolist()) for candidate in candidates}) == len(candidates)
    assert any(not torch.equal(candidate, control) for candidate in candidates)


if __name__ == "__main__":
    main()
