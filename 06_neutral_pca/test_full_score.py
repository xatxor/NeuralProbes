import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("full_score", Path(__file__).with_name("full_score.py"))
full_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_score)


def test_find_subsequence():
    assert full_score.find_subsequence([1, 2, 3, 4], [2, 3]) == 1


if __name__ == "__main__":
    test_find_subsequence()
