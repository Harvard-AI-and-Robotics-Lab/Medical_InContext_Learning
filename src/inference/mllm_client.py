import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from PIL import Image


@dataclass
class InferenceRecord:
    query_id: str
    method: str
    model: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    temperature: float = 0.0
    seed: Optional[int] = None
    finish_reason: str = ""

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "method": self.method,
            "model": self.model,
            "raw_response": self.raw_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "temperature": self.temperature,
            "seed": self.seed,
            "finish_reason": self.finish_reason,
        }


class MLLMClient:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int = 42, top_p: Optional[float] = None,
                 api_key_env: str = "OPENAI_API_KEY", max_retries: int = 3,
                 retry_delay: float = 5.0, base_url: str = "",
                 response_format: Optional[dict | str] = None,
                 extra_body: Optional[dict] = None,
                 timeout: float = 300.0,
                 parallel_requests: int = 1,
                 batch_delay: float = 0.5,
                 image_max_side: Optional[int] = None,
                 image_quality: int = 95):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.response_format = response_format
        self.extra_body = extra_body or {}
        self.parallel_requests = max(1, int(parallel_requests or 1))
        self.batch_delay = batch_delay
        self.image_max_side = int(image_max_side) if image_max_side else None
        self.image_quality = int(image_quality)

        api_key = os.environ.get(api_key_env)
        if not api_key:
            if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
                api_key = "EMPTY"
            else:
                raise ValueError(f"API key not found in environment variable: {api_key_env}")
        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        if not self.image_max_side:
            with open(image_path, "rb") as f:
                ext = Path(image_path).suffix.lower().replace(".", "")
                if ext == "jpg":
                    ext = "jpeg"
                return base64.b64encode(f.read()).decode("utf-8"), ext

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            max_side = max(image.size)
            if max_side > self.image_max_side:
                image.thumbnail(
                    (self.image_max_side, self.image_max_side),
                    resample=Image.Resampling.LANCZOS,
                )
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.image_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "jpeg"

    def _prepare_messages(self, messages: list) -> list:
        prepared = []
        for msg in messages:
            role = msg["role"]
            content_items = msg["content"]
            prepared_content = []
            for item in content_items:
                if item["type"] == "text":
                    prepared_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    img_path = item["image_url"]["url"]
                    if os.path.exists(img_path):
                        b64, ext = self._encode_image(img_path)
                        prepared_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{b64}"}
                        })
                    else:
                        prepared_content.append(item)
            prepared.append({"role": role, "content": prepared_content})
        return prepared

    def infer(self, messages: list, query_id: str = "", method: str = "") -> InferenceRecord:
        prepared = self._prepare_messages(messages)
        request_kwargs = {
            "model": self.model,
            "messages": prepared,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.response_format:
            if isinstance(self.response_format, str):
                request_kwargs["response_format"] = {"type": self.response_format}
            else:
                request_kwargs["response_format"] = self.response_format
        if self.extra_body:
            request_kwargs["extra_body"] = self.extra_body

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(**request_kwargs)
                elapsed = (time.time() - start) * 1000

                choice = response.choices[0]
                usage = response.usage
                return InferenceRecord(
                    query_id=query_id,
                    method=method,
                    model=self.model,
                    raw_response=choice.message.content,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=elapsed,
                    temperature=self.temperature,
                    seed=self.seed,
                    finish_reason=choice.finish_reason,
                )
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APIStatusError,
            ) as exc:
                status_code = getattr(exc, "status_code", None)
                retryable_status = status_code in {500, 502, 503, 504}
                retryable_transport = isinstance(
                    exc,
                    (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError),
                )
                if not (retryable_transport or retryable_status):
                    raise
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def infer_batch(self, batch: list, delay: float | None = None) -> list:
        if delay is None:
            delay = self.batch_delay
        if self.parallel_requests > 1 and len(batch) > 1:
            results = [None] * len(batch)
            workers = min(self.parallel_requests, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self.infer,
                        messages=item["messages"],
                        query_id=item.get("query_id", ""),
                        method=item.get("method", ""),
                    ): idx
                    for idx, item in enumerate(batch)
                }
                for future in as_completed(future_to_idx):
                    results[future_to_idx[future]] = future.result()
            return results

        results = []
        for item in batch:
            result = self.infer(
                messages=item["messages"],
                query_id=item.get("query_id", ""),
                method=item.get("method", ""),
            )
            results.append(result)
            if delay > 0:
                time.sleep(delay)
        return results
