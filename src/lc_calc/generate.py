"""
Leaving Cert Higher Level — Calculus dataset generator (differentiation + integration).

Design principles (see project plan, Stage 1):
  * VERIFIABILITY: every problem has a SymPy ground-truth answer; grading uses symbolic
    equality, so equivalent forms (2(x+1) vs 2x+2) are graded equal. Integrals are graded
    up to an additive constant.
  * CONTAMINATION IMMUNITY: problems are freshly synthesised from a seed, so no frontier
    model has seen this exact eval set.
  * GOLD REASONING TRACES: each item carries a short worked solution for SFT (Stage 3) and
    intermediate ground-truth quantities for the faithfulness metric (Stage 4).
  * DIFFICULTY / SUBTYPE CONTROL: every item is tagged so results can be stratified.
  * FORWARD-GENERATION for integrals: we pick an antiderivative F, differentiate to get the
    integrand f, then ask "integrate f". Gold = F. This guarantees clean, elementary integrals
    and avoids non-integrable garbage.

Outputs JSONL splits: train / valid / test_id / test_ood.
  test_ood uses difficulty tiers strictly harder than anything in train (generalisation probe).

Usage:
  python lc_calculus_gen.py --n 3000 --out ./data --seed 0
  python lc_calculus_gen.py --self-check          # validate gold answers + print distributions
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp

x = sp.Symbol("x")

from lc_calc.grader import grade  # single source of truth

# ----------------------------------------------------------------------------------
# Pretty-printing helpers (LC-style problem text)
# ----------------------------------------------------------------------------------

def expr_to_text(e: sp.Expr) -> str:
    """Render a SymPy expression the way a student would read it on an exam paper.

    Uses '^' for powers, 'e^(...)' for the exponential, and a middot '·' for
    multiplication so that products like x*exp(3x) render as 'x·e^(3x)' rather than
    the ambiguous 'xexp(3x)'. Unambiguous notation matters: a model misreading the
    problem would otherwise look like a reasoning failure in the eval (Stage 4).
    """
    s = sp.sstr(e)
    s = s.replace("**", "^")
    s = s.replace("exp(", "e^(")
    s = s.replace("*", "\u00b7")  # middot
    return s


# ----------------------------------------------------------------------------------
# Random building blocks
# ----------------------------------------------------------------------------------

def nz(rng, lo, hi):
    """Random non-zero integer in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def rand_poly(rng, degree, max_coeff=6):
    """Random polynomial of exactly `degree` (leading coeff non-zero)."""
    terms = []
    for k in range(degree + 1):
        c = rng.randint(-max_coeff, max_coeff)
        if k == degree and c == 0:
            c = nz(rng, 1, max_coeff)
        terms.append(c * x**k)
    return sp.expand(sum(terms))


# ----------------------------------------------------------------------------------
# Differentiation problems  (gold = f')
# ----------------------------------------------------------------------------------

def make_diff(rng, tier):
    """Return (f, subtype). Difficulty rises with `tier`."""
    if tier == "easy":
        f = rand_poly(rng, rng.randint(2, 4))
        return f, "diff_poly"

    if tier == "medium":
        choice = rng.choice(["product", "chain_lin"])
        if choice == "product":
            f = rand_poly(rng, rng.randint(1, 2)) * rng.choice(
                [sp.sin(x), sp.cos(x), sp.exp(x)]
            )
            return sp.expand(f), "diff_product"
        a, b = nz(rng, 2, 5), rng.randint(-4, 4)
        inner = a * x + b
        f = rng.choice([sp.sin(inner), sp.cos(inner), sp.exp(inner), inner ** rng.randint(2, 4)])
        return f, "diff_chain"

    if tier == "hard":
        choice = rng.choice(["quotient", "chain_quad", "product_comp"])
        if choice == "quotient":
            num = rand_poly(rng, rng.randint(1, 2))
            den = rand_poly(rng, 1)
            return num / den, "diff_quotient"
        if choice == "chain_quad":
            inner = rand_poly(rng, 2, max_coeff=3)
            f = rng.choice([sp.sin(inner), sp.cos(inner), sp.exp(inner)])
            return f, "diff_chain"
        f = (rand_poly(rng, 1)) * rng.choice([sp.sin(x), sp.exp(x)]) * rng.choice([sp.cos(x), x])
        return sp.expand(f), "diff_product"

    # OOD: strictly harder than train (deeper composition / mixed transcendental products)
    if tier == "ood":
        choice = rng.choice(["nested", "triple_product", "quotient_comp"])
        if choice == "nested":
            inner = rand_poly(rng, 2, max_coeff=3)
            f = rng.choice([sp.sin(inner) ** 2, sp.exp(inner) * sp.cos(inner)])
            return f, "diff_nested"
        if choice == "triple_product":
            f = rand_poly(rng, 1) * sp.sin(x) * sp.exp(x)
            return sp.expand(f), "diff_triple_product"
        inner = rand_poly(rng, 2, max_coeff=3)
        return sp.sin(inner) / (rand_poly(rng, 1)), "diff_quotient_comp"

    raise ValueError(tier)


# ----------------------------------------------------------------------------------
# Integration problems  (forward-generated: choose F, differentiate to get integrand)
# gold = F  (graded up to +C)
# ----------------------------------------------------------------------------------

def make_int(rng, tier):
    """Return (integrand f, antiderivative F, subtype)."""
    if tier == "easy":
        F = rand_poly(rng, rng.randint(3, 5))
        return sp.diff(F, x), F, "int_poly"

    if tier == "medium":
        choice = rng.choice(["trig", "exp", "subst"])
        if choice == "trig":
            a, b = nz(rng, 1, 5), nz(rng, 2, 5)
            F = a * rng.choice([sp.sin(b * x), sp.cos(b * x)])
            return sp.diff(F, x), F, "int_trig"
        if choice == "exp":
            a, b = nz(rng, 1, 5), nz(rng, 2, 4)
            F = a * sp.exp(b * x)
            return sp.diff(F, x), F, "int_exp"
        a, b = nz(rng, 2, 5), rng.randint(-3, 3)
        n = rng.randint(2, 4)
        F = (a * x + b) ** n / (a * n)
        return sp.diff(F, x), sp.expand(F), "int_substitution"

    if tier == "hard":
        choice = rng.choice(["byparts", "subst_quad"])
        if choice == "byparts":
            a = nz(rng, 1, 3)
            F = x * rng.choice([sp.exp(a * x), sp.sin(a * x), sp.cos(a * x)])
            return sp.diff(F, x), F, "int_byparts"
        a, b = nz(rng, 1, 4), nz(rng, 1, 4)
        F = a * (x**2 + b) ** rng.randint(2, 3)
        return sp.expand(sp.diff(F, x)), sp.expand(F), "int_substitution"

    if tier == "ood":
        choice = rng.choice(["byparts_quad", "product_trig_exp"])
        if choice == "byparts_quad":
            F = (rand_poly(rng, 2, max_coeff=3)) * sp.exp(nz(rng, 1, 2) * x)
            return sp.diff(F, x), F, "int_byparts_quad"
        a = nz(rng, 1, 3)
        F = sp.exp(a * x) * rng.choice([sp.sin(x), sp.cos(x)])
        return sp.diff(F, x), F, "int_product_trig_exp"

    raise ValueError(tier)


# ----------------------------------------------------------------------------------
# Worked solution (gold reasoning trace) — kept concise; enrich in Stage 3 if desired
# ----------------------------------------------------------------------------------

def diff_solution(f, gold):
    return (
        f"We differentiate f(x) = {expr_to_text(f)} with respect to x, "
        f"applying the standard rules (power/product/quotient/chain as needed). "
        f"This gives f'(x) = {expr_to_text(sp.simplify(gold))}."
    )

def int_solution(f, F, subtype):
    technique = {
        "int_poly": "integrating term by term using the power rule",
        "int_trig": "reversing the derivative of sine/cosine",
        "int_exp": "reversing the derivative of the exponential",
        "int_substitution": "a linear/substitution recognition",
        "int_byparts": "integration by parts",
        "int_byparts_quad": "integration by parts (applied repeatedly)",
        "int_product_trig_exp": "integration by parts (cyclic)",
    }.get(subtype, "standard techniques")
    return (
        f"We integrate f(x) = {expr_to_text(f)} using {technique}. "
        f"The antiderivative is F(x) = {expr_to_text(sp.simplify(F))} + C, "
        f"which can be checked since F'(x) = f(x)."
    )


# ----------------------------------------------------------------------------------
# Problem assembly
# ----------------------------------------------------------------------------------

def build_item(rng, kind, tier, idx):
    if kind == "diff":
        f, subtype = make_diff(rng, tier)
        gold = sp.simplify(sp.diff(f, x))
        prompt = (
            "Differentiate the following with respect to x. "
            "Give your final answer after 'Answer:'.\n\n"
            f"f(x) = {expr_to_text(f)}"
        )
        steps = diff_solution(f, gold)
        gold_srepr = sp.srepr(sp.expand(gold))
    else:
        f, F, subtype = make_int(rng, tier)
        gold = sp.simplify(F)
        prompt = (
            "Find the following indefinite integral (you may omit the constant of "
            "integration or write + C). Give your final answer after 'Answer:'.\n\n"
            f"\u222b ( {expr_to_text(f)} ) dx"
        )
        steps = int_solution(f, F, subtype)
        gold_srepr = sp.srepr(sp.expand(gold))

    return {
        "id": f"{kind}_{tier}_{idx}",
        "kind": kind,                 # 'diff' | 'int'  -> grader needs this
        "subtype": subtype,
        "difficulty": tier,
        "prompt": prompt,
        "problem_expr": sp.srepr(f),  # the f(x) being differentiated/integrated
        "gold_answer": expr_to_text(gold),
        "gold_srepr": gold_srepr,     # machine-checkable ground truth
        "solution_steps": steps,
        "text": f"{prompt}\n\n{steps}\nAnswer: {expr_to_text(gold)}",  # SFT target
    }


# ----------------------------------------------------------------------------------
# Dataset generation with dedup + splits
# ----------------------------------------------------------------------------------

TRAIN_TIERS = ["easy", "medium", "hard"]

def generate(n, seed):
    rng = random.Random(seed)
    seen = set()
    items = []
    attempts = 0
    # in-distribution pool
    while len(items) < n and attempts < n * 40:
        attempts += 1
        kind = rng.choice(["diff", "int"])
        tier = rng.choice(TRAIN_TIERS)
        it = build_item(rng, kind, tier, len(items))
        key = (it["kind"], it["problem_expr"])
        if key in seen:
            continue
        seen.add(key)
        items.append(it)
    return items, seen, rng


def generate_ood(n, rng, seen):
    items = []
    attempts = 0
    while len(items) < n and attempts < n * 60:
        attempts += 1
        kind = rng.choice(["diff", "int"])
        it = build_item(rng, kind, "ood", len(items))
        key = (it["kind"], it["problem_expr"])
        if key in seen:
            continue
        seen.add(key)
        items.append(it)
    return items


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="number of in-distribution items")
    ap.add_argument("--out", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ood", type=int, default=300)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        rng = random.Random(0)
        n_ok = 0
        dist = {}
        sample = []
        for i in range(400):
            kind = rng.choice(["diff", "int"])
            tier = rng.choice(TRAIN_TIERS + ["ood"])
            it = build_item(rng, kind, tier, i)
            ok = grade(it["gold_srepr"], it)  # gold must grade itself correct
            n_ok += ok
            dist[(it["kind"], it["difficulty"])] = dist.get((it["kind"], it["difficulty"]), 0) + 1
            if len(sample) < 3 and tier in ("medium", "hard"):
                sample.append(it)
        print(f"self-check: {n_ok}/400 gold answers grade correctly")
        print("distribution:", dict(sorted(dist.items(), key=lambda kv: str(kv[0]))))
        for s in sample:
            print("\n---", s["id"], "|", s["subtype"], "|", s["difficulty"])
            print(s["prompt"])
            print("GOLD:", s["gold_answer"])
        return

    items, seen, rng = generate(args.n, args.seed)
    ood = generate_ood(args.ood, rng, seen)

    # split in-distribution: 80 / 10 / 10  (stratification by difficulty preserved by shuffle)
    rng.shuffle(items)
    n_val = max(1, len(items) // 10)
    n_test = max(1, len(items) // 10)
    valid = items[:n_val]
    test_id = items[n_val : n_val + n_test]
    train = items[n_val + n_test :]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test_id.jsonl", test_id)
    write_jsonl(out / "test_ood.jsonl", ood)
    print(f"train={len(train)}  valid={len(valid)}  test_id={len(test_id)}  test_ood={len(ood)}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
