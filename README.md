# Specialising a 3B LLM on Leaving Cert Higher Level Calculus

**TL;DR** - I fine tuned Qwen 2.5 (3B and 7B) with QLoRA 
on synthetically generated, symbolically verifiable Irish Leaving Certificate Higher Level calculus, entirely on a MacBook (Apple MLX, 4-bit). The identical recipe **regressed the 3B by 7 points but lifted the 7B by 52 points** despite the two base modelsbeing statistically indistinguishable zero shot (40.7% vs 37.7%, McNemar). The finding: **SFT efficacy here is gated by base capability, not base zero shot accuracy** The fine tuned 7B reaches **89.7%**, within 5 points of GPT-4o (94.7%) on a 300 item contamination free test set at zero marginal cost, offline on a laptop.

![Accuracy by model size]
(results/figures/accuracy_by_size.png)

## TL;DR (details)
- **Task:** differentiation + integration in the style of the Irish LC Higher Level syllabus.
- **Data:** synthetically generated with SymPy as the solver — exact ground truth, gold worked steps, tunable difficulty, and zero contamination risk.
  worked steps, tunable difficulty, and zero contamination risk.
- **Method:** 4-bit QLoRA (LoRA rank 16) on Qwen2.5-Instruct via MLX on Apple Silicon.
- **Evaluation:** symbolic-equality grading, 95% Wilson CIs, McNemar significance vs. baselines, parse-failure rate tracked separately, and a stratified failure taxonomy.

## Results

All models evaluated on the same 300 freshly-synthesised items, graded by the same SymPy symbolic grader, parsed by the same parser. 95% Wilson intervals.

| Model | Accuracy (95% CI) | Local |
|---|---|---|
| Qwen2.5-3B (zero-shot) | 40.7% [35.3–46.3] | ✓ |
| Qwen2.5-3B + QLoRA | 33.3% [28.2–38.8] | ✓ |
| Qwen2.5-7B (zero-shot) | 37.7% [32.4–43.3] | ✓ |
| **Qwen2.5-7B + QLoRA (ours)** | **89.7% [85.7–92.6]** | ✓ |
| GPT-4o | 94.7% [91.5–96.7] | ✗ |

**Key paired comparisons (McNemar, n=300):**

| Comparison | χ² | Verdict |
|---|---|---|
| 3B base vs 3B fine-tune | 5.25 | fine-tuning significantly **hurt** |
| 3B base vs 7B base | 0.70 | n.s. — **the two bases are equivalent** |
| 7B base vs 7B fine-tune | 139.68 | fine-tuning significantly **helped** (164 vs 8) |
| 7B fine-tune vs GPT-4o | 5.03 | GPT-4o ahead, narrowly (27 vs 12) |


## The finding

The 3B and 7B base models score the same zero-shot (χ²=0.70, n.s.). Given identical training data, LoRA rank, and hyperparameters, the 3B gets **worse** and the 7B gets **dramatically better**. So the lift is not "the 7B is a stronger base" — by the only measure that matters here, it isn't. The 7B's capacity lets it absorb reasoning traces the 3B cannot; below that threshold, the same supervision displaces the base model's own (better) free-form chain-of-thought.

## Why this domain

Calculus is the rare reasoning slice that is **programmatically verifiable**: SymPy checks symbolic equality, so `2(x+1)` grades equal to `2x+2`, and integrals are graded up to `+C`. That makes clean auto-grading, unlimited contamination free training data, and an honest frontier comparison all possible. Every test item is synthesised fresh from a seed, so no frontier model has memorised it.

## Reproduce

​```bash
pip install -e .
python scripts/make_dataset.py --n 3000 --seed 0
mlx_lm.lora -c configs/mlx_lora.yaml
python scripts/run_headroom.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ./adapters/qwen7b-v2 --out results/runs/finetuned_7b_v2.jsonl
python scripts/run_api_eval.py --model openai/gpt-4o --out results/runs/gpt4o.jsonl --sleep 5
python scripts/compare.py --runs base3b=results/runs/headroom.jsonl \
  v2_3b=results/runs/finetuned_v2.jsonl base7b=results/runs/base_7b.jsonl \
  v2_7b=results/runs/finetuned_7b_v2.jsonl gpt4o=results/runs/gpt4o.jsonl
python scripts/make_figures.py

## Method notes

- **QLoRA on Apple Silicon:** `bitsandbytes` is CUDA-only, so this project uses MLX's native 4-bit quantised LoRA. Fine-tuned on a MacBook, 6.3 GB peak memory, ~1 hr.
- **Config:** LoRA rank 16, 16 layers, lr 1e-4, ~0.5 epoch; LoRA = 0.30% of params (23M/7.6B); val loss 2.15 → 0.055.
- **One grader and one parser everywhere:** training, headroom, local eval, and API eval all import the same `grade()` and `parse_answer()`, so grading can't drift between train and eval or between my model and the baseline.
- **Deterministic decoding** (temp 0); every transcript saved so parser fixes apply offline via re-grading, no re-runs.

## Repository layout

​```
src/lc_calc/    generate · grader · parsing · prompts · metrics · models
scripts/        make_dataset · run_headroom · run_api_eval · compare · regrade · make_figures
configs/        data / train / eval hyperparameters (YAML)
results/runs/   committed evidence: per-run predictions + transcripts
tests/          grader unit tests
​```



## Limitations
- Synthetic problems dont capture full LC contextual ("Section B") questions; grading is final answer symbolic equality, not the official partial credit scheme.
- GPT-4o is significantly ahead (χ²=5.03) — a 5-point gap, not a tie. On 12 of 300 items the fine-tune wins where GPT-4o fails.
- n=1 per configuration: no seed variance estimate on training runs.
- The capability threshold result is domain specific;
nothing here shows it generalises beyond this task

## License
Code: MIT. Data: synthetic, original (see [`NOTICE.md`]).
