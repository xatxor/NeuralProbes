import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("neutral_pca", Path(__file__).with_name("pipeline.py"))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_fixed_sample_and_render_helpers():
    counts = pipeline.PROJECTION.allocate_counts([10, 20, 30, 40], 50)
    assert counts == [5, 10, 15, 20]
    basis = torch_basis = np.linalg.qr(np.random.default_rng(0).normal(size=(8, 2)))[0]
    vectors = pipeline.PROJECTION.orthogonalize_vectors(
        pipeline.torch.from_numpy(np.random.default_rng(1).normal(size=(3, 8))).float(),
        pipeline.torch.from_numpy(torch_basis).float(),
    )
    assert vectors.shape == (3, 8)
    assert np.allclose(vectors.numpy() @ basis, 0, atol=1e-5)


if __name__ == "__main__":
    test_fixed_sample_and_render_helpers()
