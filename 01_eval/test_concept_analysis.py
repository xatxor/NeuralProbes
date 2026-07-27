"""Run with: python 01_eval/test_concept_analysis.py"""

import tempfile
from pathlib import Path

from concept_analysis import AnalysisResult, AnalysisWriter, thinking_span


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


if __name__ == "__main__":
    main()
