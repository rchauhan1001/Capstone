import json
import time
import os
from typing import Dict, Any
from .base_llm import BaseLLM


class LoggedLLM(BaseLLM):
    """
    Wraps any BaseLLM implementation to log every call.
    Works with OpenAILLM now, LocalVLLM later — no changes needed.
    """

    def __init__(self, inner_llm: BaseLLM, log_path: str = "inference_log.jsonl"):
        super().__init__(inner_llm.config)
        self.inner = inner_llm
        self.log_path = log_path
        self.call_count = 0

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        print(f"[LoggedLLM] Logging all LLM calls to: {self.log_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        call_id = self.call_count

        log_entry = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": getattr(self.inner, "model_name", "unknown"),
            "prompt": prompt,
            "params": {
                "max_tokens": kwargs.get("max_tokens", getattr(self.inner, "default_max_tokens", None)),
                "temperature": kwargs.get("temperature", getattr(self.inner, "default_temperature", None)),
            },
        }

        start = time.time()

        # Call the inner LLM but capture the full API response
        response_text, api_metadata = self._generate_with_metadata(prompt, **kwargs)

        elapsed = time.time() - start

        log_entry.update({
            "elapsed_seconds": round(elapsed, 3),
            "response": response_text,
            "status": "error" if response_text.startswith("Error generating") else "success",
            "finish_reason": api_metadata.get("finish_reason"),
            "usage": api_metadata.get("usage"),
        })

        self._write_log(log_entry)

        # Console output
        finish = api_metadata.get("finish_reason", "?")
        tokens = api_metadata.get("usage", {}).get("total_tokens", "?")
        truncated = " ⚠️ TRUNCATED" if finish == "length" else ""
        print(f"[LoggedLLM] Call #{call_id} | {elapsed:.1f}s | {len(response_text)} chars | tokens: {tokens} | finish: {finish}{truncated}")

        return response_text

    def _generate_with_metadata(self, prompt: str, **kwargs) -> tuple:
        """
        Call the inner LLM's client directly to get the full API response,
        including finish_reason and token usage.
        Falls back to inner.generate() if direct access isn't available.
        """
        metadata = {}

        # Try to access the inner LLM's OpenAI client directly
        if hasattr(self.inner, "client"):
            try:
                max_tokens = kwargs.get("max_tokens", getattr(self.inner, "default_max_tokens", 1024))
                temperature = kwargs.get("temperature", getattr(self.inner, "default_temperature", 0.7))

                response = self.inner.client.chat.completions.create(
                    model=self.inner.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]},
                )

                text = response.choices[0].message.content.strip()
                metadata["finish_reason"] = response.choices[0].finish_reason

                if response.usage:
                    metadata["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                return text, metadata

            except Exception as e:
                print(f"[LoggedLLM] Direct client call failed, falling back: {e}")

        # Fallback: use inner.generate() (no metadata available)
        text = self.inner.generate(prompt, **kwargs)
        return text, metadata

    def _write_log(self, entry: Dict[str, Any]):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LoggedLLM] WARNING: Failed to write log: {e}")