"""N-way comparison of eval runs, re-graded through ONE parser, with pairwise McNemar.

  # 3B (defaults):
  python scripts/compare.py
  # any pair/set by name=path:
  python scripts/compare.py --runs base=results/runs/base_7b.jsonl v2=results/runs/finetuned_7b_v2.jsonl
"""
import argparse, json
from itertools import combinations
from lc_calc.parsing import parse_answer
from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci, mcnemar


def load_correct(path, gold_by_id):
    out = {}
    for l in open(path):
        r = json.loads(l)
        it = gold_by_id.get(r["id"])
        if it is None: continue
        cand, _ = parse_answer(r["raw"])
        out[r["id"]] = int(bool(cand is not None and grade(cand, it)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/test_id.jsonl")
    ap.add_argument("--runs", nargs="+", default=[
        "base=results/runs/headroom.jsonl",
        "v1-thin=results/runs/finetuned.jsonl",
        "v2-reason=results/runs/finetuned_v2.jsonl"],
        help="space-separated name=path pairs")
    args = ap.parse_args()

    gold = {}
    for l in open(args.data):
        it = json.loads(l); gold[it["id"]] = it

    runs = dict(kv.split("=", 1) for kv in args.runs)
    correct = {n: load_correct(p, gold) for n, p in runs.items()}
    ids = sorted(set.intersection(*[set(c) for c in correct.values()]))
    print(f"Comparing {len(ids)} common items\n")
    for n in runs:
        c = [correct[n][i] for i in ids]; k = sum(c); lo, hi = wilson_ci(k, len(c))
        print(f"  {n:12s} {k/len(c):6.1%}  [{lo:.1%}, {hi:.1%}]  ({k}/{len(c)})")
    print("\nMcNemar (paired):")
    for a, b in combinations(runs, 2):
        ca = [correct[a][i] for i in ids]; cb = [correct[b][i] for i in ids]
        b01, b10, chi2, _ = mcnemar(ca, cb)
        sig = "SIGNIFICANT (p<0.05)" if chi2 > 3.84 else "not significant"
        print(f"  {a} vs {b}: chi2={chi2:.2f} -> {sig}  ({b} gains {b01}, {a} gains {b10})")


if __name__ == "__main__":
    main()
