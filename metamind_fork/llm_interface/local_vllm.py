import openai
import json
import time
import os
from typing import Dict, Any
from .base_llm import BaseLLM


class LocalVLLM(BaseLLM):
    """
    LLM interface for a local vLLM server (OpenAI-compatible API).
    Captures every inference call: prompt, full response, CoT, tokens, timing.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:8000/v1")
        self.model_name = config.get("model_name", "openai/gpt-oss-120b")
        self.default_max_tokens = config.get("default_max_tokens", 2048)
        self.default_temperature = config.get("default_temperature", 0.7)

        # Bypass any proxy for local connections
        import os
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ["no_proxy"] = "localhost,127.0.0.1"

        # OpenAI client pointed at local vLLM server — no real API key needed
        import httpx
        self.client = openai.OpenAI(
            api_key=config.get("api_key", "not-needed"),
            base_url=self.base_url,
            http_client=httpx.Client(proxy=None),
        )

        # Logging setup
        self.log_path = config.get("log_path", "inference_log.jsonl")
        self.call_count = 0

        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        print(f"[LocalVLLM] Initialized — server: {self.base_url}, model: {self.model_name}")
        print(f"[LocalVLLM] Logging all calls to: {self.log_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text via local vLLM server. Logs everything.
        """
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        temperature = kwargs.get("temperature", self.default_temperature)
        self.call_count += 1
        call_id = self.call_count

        log_entry = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.model_name,
            "prompt": prompt,
            "params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]},
            },
        }

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]},
            )

            elapsed = time.time() - start_time
            content = response.choices[0].message.content or ""

            # Capture token usage
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Capture full response including any reasoning/CoT
            # vLLM may include reasoning in the content or in a separate field
            reasoning = None
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning = response.choices[0].message.reasoning_content
            elif hasattr(response.choices[0].message, "reasoning"):
                reasoning = response.choices[0].message.reasoning

            log_entry.update({
                "status": "success",
                "elapsed_seconds": round(elapsed, 3),
                "response_content": content,
                "reasoning_content": reasoning,
                "full_raw_response": content if reasoning is None else f"[REASONING]\n{reasoning}\n[/REASONING]\n\n[RESPONSE]\n{content}\n[/RESPONSE]",
                "usage": usage,
                "finish_reason": response.choices[0].finish_reason,
            })

            self._write_log(log_entry)

            print(f"[LocalVLLM] Call #{call_id} completed in {elapsed:.1f}s | "
                  f"tokens: {usage.get('total_tokens', '?')} | "
                  f"finish: {response.choices[0].finish_reason}")

            return content.strip()

        except Exception as e:
            elapsed = time.time() - start_time
            log_entry.update({
                "status": "error",
                "elapsed_seconds": round(elapsed, 3),
                "error": str(e),
            })
            self._write_log(log_entry)

            print(f"[LocalVLLM] Call #{call_id} FAILED after {elapsed:.1f}s: {e}")
            return f"Error generating response: {e}"

    def _write_log(self, entry: Dict[str, Any]):
        """Append a log entry to the JSONL log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LocalVLLM] WARNING: Failed to write log: {e}")