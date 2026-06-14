"""Symbolic grader — the SINGLE source of truth for correctness.

Imported by the generator (self-check), the headroom test, and the Stage 4 eval
harness. Keeping one grader is the most important anti-footgun rule in the project:
if training and evaluation grade differently, every number you report is suspect.

  diff: correct iff (candidate - gold) simplifies to 0
  int : correct iff d/dx(candidate - gold) simplifies to 0   (allows the +C freedom)
"""

import sympy as sp

x = sp.Symbol("x")


def _to_expr(obj):
    """Coerce a SymPy expr, an srepr string, or a lenient student-style string to Expr."""
    if isinstance(obj, sp.Expr):
        return obj
    s = str(obj)
    # gold is stored as srepr (e.g. "Add(Mul(...))"); sympify handles that directly
    try:
        return sp.sympify(s)
    except Exception:
        pass
    # otherwise treat as exam notation and normalise
    try:
        return sp.sympify(s.replace("^", "**").replace("\u00b7", "*").replace("e**(", "exp("))
    except Exception:
        return None


def grade(candidate, item) -> bool:
    """candidate: SymPy expr or parseable string. item: a dataset record with gold_srepr + kind."""
    gold = _to_expr(item["gold_srepr"])
    cand = _to_expr(candidate)
    if cand is None or gold is None:
        return False
    try:
        if item["kind"] == "diff":
            return sp.simplify(cand - gold) == 0
        return sp.simplify(sp.diff(cand - gold, x)) == 0
    except Exception:
        return False
