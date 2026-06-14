"""Eval harness (Stage 4): run a backend over a split, emit one rich record per item to JSONL.

Record schema (the unit of all downstream analysis):
  {item_id, kind, subtype, difficulty, prompt, raw_output, parsed_answer,
   gold, correct, confidence, parse_ok, latency_ms, tokens, cost}

Decouple generation (expensive, here) from analysis (cheap, metrics.py) so you can
re-analyse without re-calling APIs.
"""

def run(backend, items, scaffold, out_path):
    raise NotImplementedError("Stage 4: loop items -> backend -> grade -> write JSONL.")
