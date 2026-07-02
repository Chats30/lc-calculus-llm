"""Model backends for evaluation.

Local MLX models are run directly inside the eval scripts; this module adds API
backends behind a uniform .generate(messages) -> (text, usage, latency) interface,
so the Stage 4 harness can push every model through the SAME parse+grade path.

GitHubModelsBackend talks to GitHub Models (https://models.github.ai/inference),
which is OpenAI-API-compatible, so we use the OpenAI SDK and just repoint base_url.
Auth is a GitHub token read from an env var (default OPENAI_API_KEY).
"""
import os
import time


class GitHubModelsBackend:
    def __init__(self, model="openai/gpt-5",
                 endpoint="https://models.github.ai/inference",
                 token_env="OPENAI_API_KEY", max_tokens=4096):
        from openai import OpenAI
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(
                f"No token found in ${token_env}. Did you run 'source .env'?")
        self.client = OpenAI(base_url=endpoint, api_key=token)
        self.model = model
        self.max_tokens = max_tokens

    def _create(self, messages):
        try:
            return self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens)
        except Exception as e:
            if "max_completion_tokens" in str(e) or "max_tokens" in str(e):
                return self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_completion_tokens=self.max_tokens)
            raise

    def generate(self, messages, max_retries=6):
        """Return (text, usage_dict, latency_s). Backs off on rate limits / transient errors."""
        delay = 4.0
        last_err = None
        for _ in range(max_retries):
            t0 = time.time()
            try:
                resp = self._create(messages)
                dt = time.time() - t0
                text = resp.choices[0].message.content or ""
                usage = {}
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                             "completion_tokens": getattr(u, "completion_tokens", None),
                             "total_tokens": getattr(u, "total_tokens", None)}
                return text, usage, dt
            except Exception as e:
                last_err = e
                m = str(e).lower()
                if any(s in m for s in ("429", "rate", "limit", "timeout", "503", "502", "overload")):
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
        raise RuntimeError(f"Giving up after {max_retries} retries: {last_err}")
