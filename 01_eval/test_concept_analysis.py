"""Run with: python 01_eval/test_concept_analysis.py"""

import tempfile
from pathlib import Path

import torch

from concept_analysis import AnalysisResult, AnalysisWriter, ConceptScorer, LAYERS, METHODS, thinking_span


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"<think>": [1], "</think>": [2]}[text]


def main() -> None:
    assert thinking_span(Tokenizer(), [9, 1, 3, 2], 4) == (2, 3, "closed_thinking")
    with tempfile.TemporaryDirectory() as directory:
        writer = AnalysisWriter(Path(directory), "-check")
        writer.add_highlights([{"benchmark": "x", "id": "0", "method": "diff", "layer": 11, "token_index": 0, "token": "x", "pair": 0, "cosine": 0.1, "polarity": "positive"}])
        writer.add(AnalysisResult(1, "closed_thinking", ["x"], {}, [{"benchmark": "x", "id": "0", "method": "diff", "layer": 11, "pair": 0, "mean_cosine": 0.1, "min_cosine": 0.1, "max_cosine": 0.1, "reasoning_tokens": 1}], []))
        writer.close()
        assert (Path(directory) / "concept_scores-check.parquet").exists()
        assert (Path(directory) / "token_highlights-check.parquet").exists()
    with tempfile.TemporaryDirectory() as directory:
        scorer = object.__new__(ConceptScorer)
        scorer.pair_ids = [0, 1]
        scorer.captured_tokens = 0
        scorer.captured = {layer: [torch.ones(4), torch.ones(4)] for layer in LAYERS}
        scorer.combined_vectors = {layer: torch.ones(6, 4) for layer in LAYERS}
        scorer.streams = {(method, layer): (Path(directory) / f"{method}-{layer}").open("wb") for method in METHODS for layer in LAYERS}
        scorer._flush_captured()
        scorer._close_streams()
        assert scorer.captured_tokens == 2
        assert all(not values for values in scorer.captured.values())
        assert all((Path(directory) / f"{method}-{layer}").stat().st_size == 2 * 2 * 2 for method in METHODS for layer in LAYERS)


if __name__ == "__main__":
    main()
