"""Run with: python 03_calibration/test_response_analysis.py"""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).with_name("analyze_response_calibration.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    values = np.array([[2.0] + [0.0] * 19, [3.0] + [0.0] * 19])
    mean, lo, hi, stability = module.bootstrap_statistics(values, 50, 7)
    assert mean.shape == lo.shape == hi.shape == stability.shape == (20,)
    assert mean[0] == 2.5 and stability[0] == 1.0


if __name__ == "__main__":
    main()
