"""Three-way comparison (base vs v1 vs v2), re-graded through one parser, with McNemar."""
import json
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
    gold = {}
    for l in open("./data/test_id.jsonl"):
        it = json.loads(l); gold[it["id"]] = it
    runs = {"base": "results/runs/headroom.jsonl",
            "v1-thin": "results/runs/finetuned.jsonl",
            "v2-reason": "results/runs/finetuned_v2.jsonl"}
    correct = {n: load_correct(p, gold) for n, p in runs.items()}
    ids = sorted(set.intersection(*[set(c) for c in correct.values()]))
    print(f"Comparing {len(ids)} common items\n")
    for n in runs:
        c = [correct[n][i] for i in ids]; k = sum(c); lo, hi = wilson_ci(k, len(c))
        print(f"  {n:11s} {k/len(c):6.1%}  [{lo:.1%}, {hi:.1%}]  ({k}/{len(c)})")
    print("\nMcNemar (paired):")
    for a, b in [("base","v1-thin"), ("base","v2-reason"), ("v1-thin","v2-reason")]:
        ca = [correct[a][i] for i in ids]; cb = [correct[b][i] for i in ids]
        b01, b10, chi2, _ = mcnemar(ca, cb)
        sig = "SIGNIFICANT (p<0.05)" if chi2 > 3.84 else "not significant"
        print(f"  {a} vs {b}: chi2={chi2:.2f} -> {sig}  ({b} gains {b01}, {a} gains {b10})")

if __name__ == "__main__":
    main()
