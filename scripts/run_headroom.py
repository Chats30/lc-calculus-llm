"""Headroom test (Stage 1 gate). Parser lives in lc_calc.parsing; this runs the model."""
import argparse, json, os
from tqdm import tqdm
from lc_calc.parsing import parse_answer
from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--data", default="./data/test_id.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--out", default="./results/runs/headroom.jsonl")
    args = ap.parse_args()
    from mlx_lm import load, generate as mlx_generate
    try:
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        _sampler = make_sampler(temp=0.0); _logits = make_logits_processors(repetition_penalty=1.15)
    except Exception:
        _sampler = _logits = None
    model, tok = load(args.model, adapter_path=args.adapter_path)
    def gen(prompt):
        kw = dict(max_tokens=args.max_tokens, verbose=False)
        if _sampler is not None:
            try: return mlx_generate(model, tok, prompt=prompt, sampler=_sampler, logits_processors=_logits, **kw)
            except TypeError: pass
        return mlx_generate(model, tok, prompt=prompt, **kw)
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit: rows = rows[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records, n_ok, n_pf, by = [], 0, 0, {}
    for r in tqdm(rows, desc="grading", unit="q"):
        msgs = [{"role":"user","content":r["prompt"]+"\n\nSolve it step by step, then on the final line write 'Answer:' followed by only the simplified expression."}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = gen(prompt)
        cand, parsed = parse_answer(out); n_pf += (not parsed)
        ok = bool(cand is not None and grade(cand, r)); n_ok += ok
        d = by.setdefault(r["difficulty"], [0,0]); d[0]+=ok; d[1]+=1
        records.append({"id":r["id"],"difficulty":r["difficulty"],"gold":r["gold_answer"],
                        "raw":out,"parsed":str(cand),"parse_ok":parsed,"correct":ok})
    with open(args.out,"w") as fh:
        for rec in records: fh.write(json.dumps(rec)+"\n")
    n=len(rows); lo,hi=wilson_ci(n_ok,n); acc=n_ok/n if n else 0
    print(f"\nOverall: {n_ok}/{n} = {acc:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
    print(f"Parse failures: {n_pf}/{n}")
    for k,(o,t) in sorted(by.items()): print(f"  {k:8s}: {o}/{t} = {o/t:.1%}")
    print("\nHEADROOM: " + ("GOOD" if 0.30<=acc<=0.65 else "OUT OF BAND") + f"  (parse failures: {n_pf}/{n})")

if __name__ == "__main__":
    main()
