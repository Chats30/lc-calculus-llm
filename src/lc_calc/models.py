"""Unified model backends (Stage 4): one generate(prompt) -> text signature for every model,
so the eval scaffold is identical across your fine-tune and the frontier APIs.

Fill these in when you reach Stage 4. Keep decoding params (temperature=0 for the headline
number) IDENTICAL across backends — apples-to-apples is the whole point.
"""

def mlx_backend(model_path: str):
    """Return fn(prompt:str)->str using mlx_lm. Used for base + fine-tuned local models."""
    raise NotImplementedError("Stage 4: wrap mlx_lm.load/generate here.")

def openai_backend(model: str = "gpt-4o"):
    raise NotImplementedError("Stage 4: call the OpenAI API at temperature=0, fixed seed.")

def gemini_backend(model: str = "gemini-1.5-pro"):
    raise NotImplementedError("Stage 4: call the Gemini API with matched decoding params.")
