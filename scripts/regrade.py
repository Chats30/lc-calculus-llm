"""Re-grade saved transcripts with the current parser. Joins to the dataset by id for true gold.
   python scripts/regrade.py --transcripts ./results/runs/headroom.jsonl --data ./data/test_id.jsonl
"""
import argparse, json
from lc_calc.parsing import parse_answer
from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="./results/runs/headroom.jsonl")
    ap.add_argument("--data", default="./data/test_id.jsonl")
    args = ap.parse_args()
    items = {}
    for l in open(args.data):
        it = json.loads(l); items[it["id"]] = it
    rows = [json.loads(l) for l in open(args.transcripts)]
    n_ok = n_pf = 0; by = {}
    for r in rows:
        it = items.get(r["id"])
        if it is None: continue
        cand, parsed = parse_answer(r["raw"])
        n_pf += (not parsed)
        ok = bool(cand is not None and grade(cand, it)); n_ok += ok
        d = by.setdefault(it["difficulty"], [0, 0]); d[0] += ok; d[1] += 1
    n = len(rows); lo, hi = wilson_ci(n_ok, n); acc = n_ok / n if n else 0
    print(f"Re-graded {n} transcripts with the fixed parser:")
    print(f"Overall: {n_ok}/{n} = {acc:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
    print(f"Parse failures: {n_pf}/{n}")
    for k, (o, t) in sorted(by.items()): print(f"  {k:8s}: {o}/{t} = {o/t:.1%}")

if __name__ == "__main__":
    main()
