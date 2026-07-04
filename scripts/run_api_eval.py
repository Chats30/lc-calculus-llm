"""Stage 4: evaluate a frontier model (GitHub Models) on the SAME contamination-free
test_id set, through the SAME parse+grade path as the local models.

  source .env
  python scripts/run_api_eval.py --limit 5     # smoke test
  python scripts/run_api_eval.py               # full run (resumable)
"""
import argparse, json, os, time
from tqdm import tqdm
from lc_calc.prompts import user_content
from lc_calc.parsing import parse_answer
from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci
from lc_calc.models import GitHubModelsBackend


def load_done(out_path):
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line); done[rec["id"]] = rec
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5")
    ap.add_argument("--data", default="./data/test_id.jsonl")
    ap.add_argument("--out", default="./results/runs/gpt5.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--token-env", default="OPENAI_API_KEY")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = load_done(args.out)
    backend = GitHubModelsBackend(model=args.model, token_env=args.token_env,
                                  max_tokens=args.max_tokens)

    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to evaluate")

    fh = open(args.out, "a")
    for r in tqdm(todo, desc=args.model, unit="q"):
        msgs = [{"role": "user", "content": user_content(r["prompt"])}]
        text, usage, dt = backend.generate(msgs)
        cand, parsed = parse_answer(text)
        ok = bool(cand is not None and grade(cand, r))
        rec = {"id": r["id"], "difficulty": r["difficulty"], "gold": r["gold_answer"],
               "raw": text, "parsed": str(cand), "parse_ok": parsed, "correct": ok,
               "latency_s": round(dt, 2), "usage": usage}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        time.sleep(args.sleep)
    fh.close()

    allrec = load_done(args.out)
    ids = {r["id"] for r in rows}
    recs = [allrec[i] for i in ids if i in allrec]
    n = len(recs)
    if n == 0:
        print("no records yet"); return
    n_ok = sum(r["correct"] for r in recs)
    n_pf = sum(not r["parse_ok"] for r in recs)
    lo, hi = wilson_ci(n_ok, n)
    by = {}
    for rec in recs:
        d = by.setdefault(rec["difficulty"], [0, 0]); d[0] += rec["correct"]; d[1] += 1
    tot_c = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in recs)
    tot_p = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in recs)
    avg_lat = sum(r.get("latency_s", 0) for r in recs) / n
    print(f"\n{args.model}")
    print(f"Overall: {n_ok}/{n} = {n_ok/n:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
    print(f"Parse failures: {n_pf}/{n}")
    for k, (o, t) in sorted(by.items()):
        print(f"  {k:8s}: {o}/{t} = {o/t:.1%}")
    print(f"Tokens: prompt={tot_p} completion={tot_c} | avg latency {avg_lat:.1f}s/q")


if __name__ == "__main__":
    main()
