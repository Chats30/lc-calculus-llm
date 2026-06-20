"""Shared prompt format. Imported by BOTH training-data generation and evaluation so the
user turn is structurally identical at train and eval time (no silent format drift)."""

INSTRUCTION = ("Solve it step by step, then on the final line write 'Answer:' "
               "followed by only the simplified expression.")


def user_content(problem_prompt: str) -> str:
    return f"{problem_prompt}\n\n{INSTRUCTION}"


def build_messages(problem_prompt: str, solution_text: str):
    return [
        {"role": "user", "content": user_content(problem_prompt)},
        {"role": "assistant", "content": solution_text},
    ]
