"""Headroom test (the Stage 1 gate). Run on your Mac:

    pip install -e .            # installs the lc_calc package + deps
    python scripts/make_dataset.py --n 3000 --out ./data --seed 0
    python scripts/run_headroom.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --data ./data/test_id.jsonl

Want overall accuracy in ~30-65%: enough signal that the task is learnable, enough
headroom that fine-tuning can show a measurable lift.
"""

import argparse
import json
import re

import sympy as sp

from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci


def parse_answer(text: str):
    """First-pass extraction of the final answer. Stage 4 hardens this and logs
    parse-failure rate so notation misreads aren't miscounted as reasoning errors."""
    chunk = text.split("Answer:")[-1] if "Answer:" in text else text
    line = chunk.strip().splitlines()[0].strip() if chunk.strip() else ""
    line = line.replace("\u00b7", "*").replace("^", "**").replace("e**(", "exp(")
    line = re.sub(r"\+\s*C\b", "", line).strip().rstrip(".")
    if not line:
        return None, False
    try:
        return sp.sympify(line), True
    except Exception:
        return None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--data", default="./data/test_id.jsonl")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    from mlx_lm import load, generate as mlx_generate

    model, tok = load(args.model)
    rows = [json.loads(l) for l in open(args.data)]

    n_ok = n_parse_fail = 0
    by_diff = {}
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]
                 + "\n\nWork step by step, then give your final answer after 'Answer:'."}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        cand, parsed = parse_answer(out)
        n_parse_fail += (not parsed)
        ok = bool(cand is not None and grade(cand, r))
        n_ok += ok
        d = by_diff.setdefault(r["difficulty"], [0, 0]); d[0] += ok; d[1] += 1

    n = len(rows)
    lo, hi = wilson_ci(n_ok, n)
    acc = n_ok / n if n else 0
    print(f"\nOverall: {n_ok}/{n} = {acc:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
    print(f"Parse failures: {n_parse_fail}/{n}")
    for k, (o, t) in sorted(by_diff.items()):
        print(f"  {k:8s}: {o}/{t} = {o/t:.1%}")
    print("\nHEADROOM: " + ("GOOD — proceed." if 0.30 <= acc <= 0.65
                             else "OUT OF BAND — retune difficulty mix in the generator."))


if __name__ == "__main__":
    main()
