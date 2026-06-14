# Specialising a 3B LLM on Leaving Cert Higher Level Calculus

> **Headline result (fill in after Stage 4):** A 4-bit QLoRA fine-tune of Qwen2.5-3B,
> trained on a laptop, reaches **XX% ± Y%** on held-out Leaving Cert HL calculus —
> within Z points of GPT-4o — at roughly **1/N the inference cost** and running fully
> locally. Evaluation set is freshly synthesised, so the comparison is contamination-free.

<p align="center"><em>[ headline figure: accuracy-by-difficulty, your model vs base vs frontier ]</em></p>

## TL;DR
- **Task:** differentiation + integration in the style of the Irish LC Higher Level syllabus.
- **Data:** synthetically generated with SymPy as the solver — exact ground truth, gold
  worked steps, tunable difficulty, and zero contamination risk.
- **Method:** 4-bit QLoRA (LoRA r=16, α=32) on Qwen2.5-3B-Instruct via MLX on Apple Silicon.
- **Evaluation:** symbolic-equality grading, 95% Wilson CIs, McNemar significance vs. baselines,
  ECE/Brier calibration, reasoning-faithfulness, and a stratified failure taxonomy.

## Why this domain
Calculus is the rare reasoning slice that is **programmatically verifiable** (SymPy checks
symbolic equality, so `2(x+1)` grades equal to `2x+2`; integrals graded up to `+C`), which
makes clean auto-grading, unlimited clean training data, and an honest frontier comparison
all possible. See [`docs/design.md`] for the full rationale.

## Results
| Model | test_id acc (95% CI) | test_ood acc | ECE ↓ | cost / 100 Q | local |
|---|---|---|---|---|---|
| Qwen2.5-3B (zero-shot) | — | — | — | — | ✓ |
| Qwen2.5-3B (4-shot) | — | — | — | — | ✓ |
| **Qwen2.5-3B + QLoRA (ours)** | — | — | — | — | ✓ |
| GPT-4o | — | — | — | — | ✗ |
| Gemini 1.5 Pro | — | — | — | — | ✗ |

Breakdowns by sub-type (polynomials, chain rule, by-parts, …) and reliability diagrams
are in [`results/figures/`].

## Reproduce
```bash
pip install -e .
make data        # generate train/valid/test_id/test_ood from seed 0
make headroom    # Stage-1 gate: base-model accuracy + CI by difficulty
make train       # QLoRA fine-tune (Apple Silicon / MLX)
make eval        # run all models through the identical eval scaffold
make figures     # produce the tables + plots above
make test        # grader unit tests
```
Everything is seeded; `make all` regenerates every number and figure from scratch.

## Method notes
- **QLoRA on Apple Silicon:** `bitsandbytes` is CUDA-only, so this project uses MLX's native
  4-bit quantised LoRA. A CUDA `bitsandbytes` config is provided in `configs/` for cloud reproduction.
- **One grader everywhere:** training, the headroom test, and evaluation all import
  `lc_calc.grader.grade`, so grading can't drift between train and eval.

## Repository layout
```
src/lc_calc/    generate · grader · models · evaluate · metrics · plots
scripts/        thin CLI entry points
configs/        data / train / eval hyperparameters (YAML)
results/        committed evidence: per-run configs, metrics, predictions, figures
tests/          grader unit tests
```

## Limitations
*(Be specific and honest — this section is read closely.)* e.g. synthetic problems don't
capture full LC contextual ("Section B") questions; grading is final-answer + step-checking,
not the official partial-credit marking scheme; OOD split probes depth generalisation only.

## License
Code: MIT. Data: synthetic, original (see [`NOTICE.md`]).
