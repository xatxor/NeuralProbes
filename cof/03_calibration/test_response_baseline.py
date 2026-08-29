"""Run with: python 03_calibration/test_response_baseline.py"""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("response_baseline", Path(__file__).with_name("response_baseline.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    values = np.array([[[1.0, 3.0]], [[3.0, 7.0]], [[5.0, 11.0]]])
    mean, std = module.response_statistics(values)
    np.testing.assert_allclose(mean, [[3.0, 7.0]])
    np.testing.assert_allclose(std, [[2.0, 4.0]])


if __name__ == "__main__":
    main()
