"""MATH answer scoring without importing the model/evaluation stack."""

from __future__ import annotations

import os


def math_equal(prediction: str, reference: str) -> bool | None:
    """Compare a generated MATH answer with its reference using math-verify.

    ``None`` means that math-verify is not installed.  This is deliberately distinct
    from a wrong answer so a missing scorer cannot silently turn a whole benchmark into
    zero accuracy.
    """
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError:
        return None
    try:
        # math-verify implements its timeout with Unix signals.  On Windows the signal
        # setup raises PermissionError; Linux keeps the library's bounded default.
        parse_options = {"parsing_timeout": None} if os.name == "nt" else {}
        verify_options = {"timeout_seconds": None} if os.name == "nt" else {}
        gold = parse(reference, extraction_config=[LatexExtractionConfig()], **parse_options)
        pred = parse(prediction, extraction_config=[LatexExtractionConfig()], **parse_options)
        return bool(verify(gold, pred, **verify_options))
    except Exception:
        return False
