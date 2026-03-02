import json
import time
import os
from typing import Dict, Any
from .base_llm import BaseLLM


class DirectVLLM(BaseLLM):
    """
    LLM interface using vLLM's Python API directly — no HTTP server, no proxy issues.
    Drop-in replacement for LocalVLLM.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.model_name = config.get("model_name", "openai/gpt-oss-120b")
        self.default_max_tokens = config.get("default_max_tokens", 2048)
        self.default_temperature = config.get("default_temperature", 0.7)
        self.log_path = config.get("log_path", "inference_log.jsonl")
        self.call_count = 0

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        print(f"[DirectVLLM] Loading model from: {self.model_path}")
        print(f"[DirectVLLM] This may take 5-10 minutes...")

        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=self.model_path,
            served_model_name=self.model_name,
            dtype="auto",
            max_model_len=config.get("max_model_len", 8192),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
            enforce_eager=config.get("enforce_eager", True),
        )

        print(f"[DirectVLLM] Model loaded. Logging to: {self.log_path}")

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        temperature = kwargs.get("temperature", self.default_temperature)
        self.call_count += 1
        call_id = self.call_count

        log_entry = {
            "call_id": call_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.model_name,
            "prompt": prompt,
            "params": {"max_tokens": max_tokens, "temperature": temperature},
        }

        start_time = time.time()

        try:
            sampling_params = self.SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outputs = self.llm.generate([prompt], sampling_params)
            content = outputs[0].outputs[0].text.strip()
            finish_reason = outputs[0].outputs[0].finish_reason

            elapsed = time.time() - start_time
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(outputs[0].outputs[0].token_ids)

            log_entry.update({
                "status": "success",
                "elapsed_seconds": round(elapsed, 3),
                "response_content": content,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
            self._write_log(log_entry)

            print(f"[DirectVLLM] Call #{call_id} completed in {elapsed:.1f}s | "
                  f"tokens: {prompt_tokens + completion_tokens} | finish: {finish_reason}")

            return content

        except Exception as e:
            elapsed = time.time() - start_time
            log_entry.update({"status": "error", "elapsed_seconds": round(elapsed, 3), "error": str(e)})
            self._write_log(log_entry)
            print(f"[DirectVLLM] Call #{call_id} FAILED after {elapsed:.1f}s: {e}")
            return f"Error generating response: {e}"

    def _write_log(self, entry: Dict[str, Any]):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[DirectVLLM] WARNING: Failed to write log: {e}")