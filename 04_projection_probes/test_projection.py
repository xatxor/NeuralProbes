"""Run with: python 04_projection_probes/test_projection.py"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("projection_pipeline", ROOT / "pipeline.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
sys.modules["pipeline"] = module
spec.loader.exec_module(module)


def main() -> None:
    assert module.allocate_counts([10, 20, 30], 12) == [2, 4, 6]
    assert module.find_sequence([1, 2, 3, 2, 3], [2, 3], 2) == 3
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    vectors = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 2.0]])
    probes = module.orthogonalize_vectors(vectors, basis)
    torch.testing.assert_close(probes @ basis, torch.zeros(2, 1))
    torch.testing.assert_close(probes.norm(dim=1), torch.ones(2))
    hidden = torch.tensor([[0.0, 2.0, 3.0]])
    torch.testing.assert_close(hidden @ probes.T, torch.tensor([[2.0, 3.0]]))

    report_spec = importlib.util.spec_from_file_location("projection_report", ROOT / "report.py")
    report = importlib.util.module_from_spec(report_spec)
    sys.modules[report_spec.name] = report
    report_spec.loader.exec_module(report)
    values = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    mean, low, high = report.bootstrap_statistics(values, 100, 7)
    np.testing.assert_allclose(mean, [2.0, 4.0])
    assert np.all(low <= mean) and np.all(mean <= high)
    scores = pd.DataFrame(
        [
            {
                "id": str(response),
                "correct": response == 0,
                "reasoning_tokens": 10,
                "layer": layer,
                "pair": pair,
                "mean_projection": response + pair / 1000,
            }
            for response in range(2)
            for layer in module.LAYERS
            for pair in range(module.PAIRS)
        ]
    )
    statistics = report.group_statistics(scores, 10, 7)
    assert len(statistics) == 3 * len(module.LAYERS) * module.PAIRS
    pairs = pd.DataFrame(
        {
            "pair": range(module.PAIRS),
            "concept": [f"concept {pair}" for pair in range(module.PAIRS)],
            "antagonist": [f"antagonist {pair}" for pair in range(module.PAIRS)],
            "class_name": ["test"] * module.PAIRS,
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        report.build_viewer(scores, pairs, Path(directory))
        index = json.loads((Path(directory) / "viewer" / "index.json").read_text())
        assert len(index["concepts"]) == module.PAIRS
        assert len(index["responses"]) == 2


if __name__ == "__main__":
    main()
