"""Grader unit tests. A buggy grader silently invalidates every result in the project,
so this is the one component that genuinely must be tested. Run: `pytest`.
"""

import sympy as sp

from lc_calc.grader import grade

x = sp.Symbol("x")


def diff_item(gold):
    return {"kind": "diff", "gold_srepr": sp.srepr(sp.expand(gold))}


def int_item(gold):
    return {"kind": "int", "gold_srepr": sp.srepr(sp.expand(gold))}


def test_diff_exact_match():
    assert grade(2 * x + 2, diff_item(2 * x + 2))


def test_diff_equivalent_forms():
    # 2(x+1) must grade equal to 2x+2
    assert grade(2 * (x + 1), diff_item(2 * x + 2))


def test_diff_wrong():
    assert not grade(2 * x + 3, diff_item(2 * x + 2))


def test_int_allows_constant_of_integration():
    # antiderivatives differing by a constant are both correct
    assert grade(x**2 + 5, int_item(x**2))
    assert grade(x**2, int_item(x**2 + 99))


def test_int_wrong():
    assert not grade(x**3, int_item(x**2))


def test_string_input_exam_notation():
    # candidate supplied as an exam-notation string
    assert grade("2*x + 2", diff_item(2 * x + 2))


def test_garbage_input_is_false_not_crash():
    assert not grade("not an expression $$", diff_item(2 * x + 2))
