"""Evaluation statistics used across the project (Stage 4).

Deliberately dependency-light (just math) so results are easy to audit.
"""

import math
from collections import defaultdict


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(confidences, correct, n_bins: int = 10):
    """ECE = sum_b (n_b/N) * |acc_b - conf_b|.  confidences in [0,1], correct in {0,1}."""
    assert len(confidences) == len(correct)
    N = len(confidences)
    if N == 0:
        return 0.0
    bins = defaultdict(list)
    for c, y in zip(confidences, correct):
        b = min(n_bins - 1, int(c * n_bins))
        bins[b].append((c, y))
    ece = 0.0
    for items in bins.values():
        nb = len(items)
        conf = sum(c for c, _ in items) / nb
        acc = sum(y for _, y in items) / nb
        ece += (nb / N) * abs(acc - conf)
    return ece


def mcnemar(correct_a, correct_b):
    """Paired McNemar test on two models scored over the SAME items.

    correct_a, correct_b: equal-length sequences of {0,1}.
    Returns (b01, b10, chi2_cc, note). b01 = A wrong & B right; b10 = A right & B wrong.
    Use the discordant counts to judge whether a head-to-head accuracy gap is real.
    """
    assert len(correct_a) == len(correct_b)
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return (b01, b10, 0.0, "no discordant pairs")
    chi2_cc = (abs(b01 - b10) - 1) ** 2 / n  # continuity-corrected
    return (b01, b10, chi2_cc, "chi2>3.84 => p<0.05 (1 dof)")


def brier_score(confidences, correct):
    """Mean squared error between stated confidence and outcome. Lower is better."""
    if not confidences:
        return 0.0
    return sum((c - y) ** 2 for c, y in zip(confidences, correct)) / len(confidences)
