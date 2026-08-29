"""Run with: python 03_calibration/test_calibrate.py"""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("calibrate.py")
spec = importlib.util.spec_from_file_location("calibrate", path)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class Tokenizer:
    eos_token = "!"

    def apply_chat_template(self, *_args, **_kwargs):
        return "user\n<|im_start|>assistant\n<think>\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def main() -> None:
    assert calibrate.shard_indices(10, 1, 4) == [1, 5, 9]
    token_ids, start, end = calibrate.model_input(
        Tokenizer(), {"problem": "p", "reasoning": "<think>abc</think>", "solution": "42"}
    )
    assert "".join(map(chr, token_ids[start:end])).strip() == "abc"
    first = np.array([[1.0, 2.0], [3.0, 5.0]])
    second = np.array([[3.0, 4.0], [7.0, 9.0]])
    values = np.stack((first, second))
    mean, variance, std = calibrate.finish_statistics(
        np.array([2, 2]), values.sum(axis=0), np.square(values).sum(axis=0)
    )
    np.testing.assert_allclose(mean, [[2, 3], [5, 7]])
    np.testing.assert_allclose(variance, [[2, 2], [8, 8]])
    np.testing.assert_allclose(std, np.sqrt(variance))


if __name__ == "__main__":
    main()
