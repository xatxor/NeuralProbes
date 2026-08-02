"""Small no-model check for the two requested steering positions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import steering_mask


def main() -> None:
    first = steering_mask("assistant", True, 7, 7, [2, 5], 2, torch.device("cpu"))[:, :, 0]
    assert first.tolist() == [[False, False, True, False, False, False, False], [False, False, False, False, False, True, False]]
    assert not steering_mask("assistant", False, 1, 7, [2], 1, torch.device("cpu")).any()
    assert steering_mask("assistant_and_generated", False, 1, 7, [2], 1, torch.device("cpu")).all()


if __name__ == "__main__":
    main()
