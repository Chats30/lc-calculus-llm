r"""Robust model-output -> SymPy parser. Single source of truth for answer extraction.
Handles implicit multiplication (6x), LaTeX (\boxed, $, \frac, \cdot, \sin), and
exponentials (e^x, xe^x, e^{-x^2}). Rejects expressions containing non-x symbols.
"""
import re
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor)

x = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application, convert_xor)
_FUNCS = ("exp", "sin", "cos", "tan", "log", "sqrt")


def _extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1: return None
    i = text.find("{", idx)
    if i == -1: return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0: return text[i+1:j]
    return text[i+1:]


def _clean(s):
    s = s.replace("$"," ").replace("\\(", " ").replace("\\)"," ").replace("\\[", " ").replace("\\]"," ")
    s = s.replace("\\left","").replace("\\right","")
    for _ in range(4):
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\boxed","")
    for a,b in [("\\cdot"," * "),("\\times"," * "),("\\div"," / "),("\\sin"," sin "),("\\cos"," cos "),
                ("\\tan"," tan "),("\\ln"," log "),("\\log"," log "),("\\exp"," exp "),
                ("\\sqrt"," sqrt "),("\\pi"," pi ")]:
        s = s.replace(a,b)
    s = re.sub(r"e\s*\^\s*\{([^{}]*)\}", r"exp(\1)", s)
    s = re.sub(r"e\s*\^\s*\(([^()]*)\)", r"exp(\1)", s)
    s = re.sub(r"e\s*\^\s*([A-Za-z0-9]+)", r"exp(\1)", s)
    s = s.replace("\u00b7","*")
    s = re.sub(r"(?<=[0-9A-Za-z])(" + "|".join(_FUNCS) + r")\(", r"*\1(", s)
    return s.replace("{","(").replace("}",")")


def parse_answer(text):
    """Return (sympy_expr_or_None, parse_ok: bool)."""
    boxed = _extract_boxed(text)
    region = boxed if boxed is not None else (
        text.split("Answer:")[-1] if "Answer:" in text else
        (([l for l in text.strip().splitlines() if l.strip()] or [""])[-1]))
    line = next((l for l in region.strip().splitlines() if l.strip()), "")
    line = line.split("Answer:")[-1]
    line = _clean(line)
    if "=" in line: line = line.split("=")[-1]
    line = re.sub(r"\+\s*C\b","",line); line = re.sub(r"\bdx\b","",line)
    line = line.strip().strip(".,;:").strip()
    if not line: return None, False
    try: expr = parse_expr(line, transformations=_T)
    except Exception: return None, False
    if expr.free_symbols - {x}: return None, False
    return expr, True
