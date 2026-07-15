"""Stage 6 figure: accuracy by model size, base vs QLoRA fine-tune, with Wilson CIs.
Re-grades every run through the single shared parser+grader (no stored-label trust),
and overlays GPT-4o as a frontier reference if results/runs/gpt4o.jsonl is present.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lc_calc.parsing import parse_answer
from lc_calc.grader import grade
from lc_calc.metrics import wilson_ci

GOLD = {json.loads(l)["id"]: json.loads(l) for l in open("data/test_id.jsonl")}


def acc(path):
    if not os.path.exists(path):
        return None
    k = n = 0
    for l in open(path):
        r = json.loads(l); it = GOLD.get(r["id"])
        if it is None: continue
        cand, _ = parse_answer(r["raw"])
        k += int(bool(cand is not None and grade(cand, it))); n += 1
    if n == 0:
        return None
    lo, hi = wilson_ci(k, n)
    return (100*k/n, 100*lo, 100*hi, n)


runs = {
    ("Qwen2.5-3B", "base"): "results/runs/headroom.jsonl",
    ("Qwen2.5-3B", "v2"):   "results/runs/finetuned_v2.jsonl",
    ("Qwen2.5-7B", "base"): "results/runs/base_7b.jsonl",
    ("Qwen2.5-7B", "v2"):   "results/runs/finetuned_7b_v2.jsonl",
}
vals = {k: acc(p) for k, p in runs.items()}
groups = ["Qwen2.5-3B", "Qwen2.5-7B"]
x = np.arange(len(groups)); w = 0.36


def bars(kind, offset, color, label):
    pts, lo, hi = [], [], []
    for g in groups:
        v = vals[(g, kind)]
        if v is None:
            pts.append(0); lo.append(0); hi.append(0); continue
        pts.append(v[0]); lo.append(v[0]-v[1]); hi.append(v[2]-v[0])
    plt.bar(x+offset, pts, w, yerr=[lo, hi], capsize=5, color=color,
            label=label, edgecolor="white", linewidth=0.5)
    for xi, p in zip(x+offset, pts):
        if p > 0:
            plt.text(xi, p+3.5, f"{p:.1f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")


plt.figure(figsize=(8, 5.2))
bars("base", -w/2, "#9aa5b1", "Base (zero-shot)")
bars("v2",   +w/2, "#2f6fed", "QLoRA fine-tune (v2)")

g4 = acc("results/runs/gpt4o.jsonl")
if g4 is not None:
    plt.axhline(g4[0], color="#e67e22", linestyle="--", linewidth=1.6,
                label=f"GPT-4o reference ({g4[0]:.1f}%, n={g4[3]})")

plt.xticks(x, groups, fontsize=11)
plt.ylabel("Accuracy on 300-item contamination-free test set (%)", fontsize=11)
plt.ylim(0, 100)
plt.title("Same QLoRA recipe, opposite outcomes:\n"
          "SFT efficacy is gated by base capability, not base zero-shot accuracy",
          fontsize=11.5, fontweight="bold")
plt.legend(frameon=False, fontsize=9.5, loc="upper left")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

os.makedirs("results/figures", exist_ok=True)
out = "results/figures/accuracy_by_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for k, v in vals.items():
    if v: print(f"  {k[0]:12s} {k[1]:5s}: {v[0]:.1f}%  [{v[1]:.1f}, {v[2]:.1f}]  n={v[3]}")
if g4: print(f"  GPT-4o           : {g4[0]:.1f}%  n={g4[3]}")
