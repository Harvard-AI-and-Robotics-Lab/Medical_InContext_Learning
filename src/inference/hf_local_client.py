import os
import json
import time

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .mllm_client import InferenceRecord


def _contains_complete_json_object(text: str) -> bool:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and {"label", "confidence", "evidence"} <= set(obj):
            return True
    return False


class _JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_width: int, batch_size: int, response_prefix: str = ""):
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width
        self.response_prefix = response_prefix
        self.finished = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for idx in range(input_ids.shape[0]):
            if self.finished[idx]:
                continue
            completion_ids = input_ids[idx][self.prompt_width:]
            text = self.response_prefix + self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.finished[idx] = _contains_complete_json_object(text)
        return all(self.finished)


class HFLocalClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        top_p: float | None = None,
        load_in_8bit: bool = True,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_memory: dict | None = None,
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        disable_allocator_warmup: bool = False,
        processor_min_pixels: int | None = None,
        processor_max_pixels: int | None = None,
        enable_thinking: bool | None = None,
        assistant_prefill: str = "",
        stop_on_json: bool = False,
        disable_fla_fast_path: bool = False,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.max_memory = max_memory
        self.disable_allocator_warmup = disable_allocator_warmup
        self.enable_thinking = enable_thinking
        self.assistant_prefill = assistant_prefill or ""
        self.stop_on_json = stop_on_json
        self.disable_fla_fast_path = disable_fla_fast_path
        self.dtype = getattr(torch, torch_dtype)

        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        processor_kwargs = {"trust_remote_code": trust_remote_code}
        if processor_min_pixels is not None:
            processor_kwargs["min_pixels"] = processor_min_pixels
        if processor_max_pixels is not None:
            processor_kwargs["max_pixels"] = processor_max_pixels
        self.processor = AutoProcessor.from_pretrained(model, **processor_kwargs)

        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": self.dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if disable_allocator_warmup:
            try:
                import transformers.modeling_utils as modeling_utils

                modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            except Exception:
                pass
        if disable_fla_fast_path:
            try:
                import transformers.utils.import_utils as import_utils

                if hasattr(import_utils.is_flash_linear_attention_available, "cache_clear"):
                    import_utils.is_flash_linear_attention_available.cache_clear()
                import_utils.is_flash_linear_attention_available = lambda: False
            except Exception:
                pass

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model, **load_kwargs)
        except Exception as first_exc:
            try:
                from transformers import AutoModelForVision2Seq
            except ImportError:
                raise first_exc
            self.model = AutoModelForVision2Seq.from_pretrained(model, **load_kwargs)

        self.model.eval()
        self.processor_name = self.processor.__class__.__name__
        self._disable_batch_infer = False

    def _first_device(self):
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device
        for param in self.model.parameters():
            return param.device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def _to_conversation(self, messages: list):
        conversation = []
        images = []
        for msg in messages:
            content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    image_path = item["image_url"]["url"]
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image path not found: {image_path}")
                    images.append(self._load_image(image_path))
                    content.append({"type": "image"})
            conversation.append({"role": msg["role"], "content": content})
        return conversation, images

    def _build_inputs(self, messages: list):
        if self.processor_name == "Gemma4Processor":
            return self._build_gemma4_inputs(messages)
        conversation, images = self._to_conversation(messages)
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        prompt_text = self.processor.apply_chat_template(conversation, **template_kwargs)
        if self.assistant_prefill:
            prompt_text += self.assistant_prefill
        inputs = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
        )
        device = self._first_device()
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs

    def _build_gemma4_inputs(self, messages: list):
        gemma_messages = self._to_gemma4_messages(messages)
        inputs = self.processor.apply_chat_template(
            gemma_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        device = self._first_device()
        return inputs.to(device)

    def _to_gemma4_messages(self, messages: list):
        gemma_messages = []
        for msg in messages:
            content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    image_path = item["image_url"]["url"]
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image path not found: {image_path}")
                    content.append({"type": "image", "url": image_path})
            gemma_messages.append({"role": msg["role"], "content": content})
        return gemma_messages

    def _to_qwen_messages(self, messages: list):
        qwen_messages = []
        for msg in messages:
            content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    image_path = item["image_url"]["url"]
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image path not found: {image_path}")
                    content.append({"type": "image", "image": image_path})
            qwen_messages.append({"role": msg["role"], "content": content})
        return qwen_messages

    def infer(self, messages: list, query_id: str = "", method: str = "") -> InferenceRecord:
        inputs = self._build_inputs(messages)
        input_token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generate_kwargs["top_p"] = self.top_p
        if self.stop_on_json:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _JsonObjectStoppingCriteria(
                        tokenizer=self.processor,
                        prompt_width=input_token_count,
                        batch_size=1,
                        response_prefix=self.assistant_prefill,
                    )
                ]
            )

        start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        latency_ms = (time.time() - start) * 1000

        completion_ids = output_ids[:, input_token_count:] if input_token_count > 0 else output_ids
        raw_response = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()
        if self.assistant_prefill:
            raw_response = self.assistant_prefill + raw_response

        completion_token_count = int(completion_ids.shape[-1]) if hasattr(completion_ids, "shape") else 0
        return InferenceRecord(
            query_id=query_id,
            method=method,
            model=self.model_name,
            raw_response=raw_response,
            prompt_tokens=input_token_count,
            completion_tokens=completion_token_count,
            latency_ms=latency_ms,
            temperature=self.temperature,
            seed=self.seed,
            finish_reason="stop",
        )

    def infer_batch(self, batch: list[dict]) -> list[InferenceRecord]:
        if not batch:
            return []

        if self._disable_batch_infer:
            return [
                self.infer(
                    messages=item["messages"],
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                )
                for item in batch
            ]

        if self.processor_name == "Qwen3VLProcessor":
            return self._infer_qwen_batch(batch)

        if self.processor_name != "Gemma4Processor":
            return [
                self.infer(
                    messages=item["messages"],
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                )
                for item in batch
            ]

        batched_messages = [self._to_gemma4_messages(item["messages"]) for item in batch]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        device = self._first_device()
        inputs = inputs.to(device)
        input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generate_kwargs["top_p"] = self.top_p
        if self.stop_on_json:
            prompt_width = int(inputs["input_ids"].shape[-1])
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _JsonObjectStoppingCriteria(
                        tokenizer=self.processor,
                        prompt_width=prompt_width,
                        batch_size=len(batch),
                        response_prefix=self.assistant_prefill,
                    )
                ]
            )

        start = time.time()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
        except ValueError as exc:
            # Gemma4 batched multimodal generation can fail on mixed real-image
            # batches with a token/feature alignment error. Fall back to stable
            # single-sample inference for the rest of the run.
            if "Image features and image tokens do not match" not in str(exc):
                raise
            self._disable_batch_infer = True
            return [
                self.infer(
                    messages=item["messages"],
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                )
                for item in batch
            ]
        latency_ms = (time.time() - start) * 1000

        records = []
        for idx, item in enumerate(batch):
            completion_ids = output_ids[idx][int(input_token_counts[idx]):]
            raw_response = self.processor.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            records.append(
                InferenceRecord(
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                    model=self.model_name,
                    raw_response=raw_response,
                    prompt_tokens=int(input_token_counts[idx]),
                    completion_tokens=int(completion_ids.shape[-1]),
                    latency_ms=latency_ms,
                    temperature=self.temperature,
                    seed=self.seed,
                    finish_reason="stop",
                )
            )
        return records

    def _infer_qwen_batch(self, batch: list[dict]) -> list[InferenceRecord]:
        conversations = []
        batch_images = []
        for item in batch:
            conversation, images = self._to_conversation(item["messages"])
            conversations.append(conversation)
            batch_images.append(images)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        prompt_texts = [
            self.processor.apply_chat_template(conversation, **template_kwargs)
            for conversation in conversations
        ]
        if self.assistant_prefill:
            prompt_texts = [text + self.assistant_prefill for text in prompt_texts]
        inputs = self.processor(
            text=prompt_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        device = self._first_device()
        inputs = inputs.to(device)
        prompt_width = int(inputs["input_ids"].shape[-1])
        input_token_counts = inputs["attention_mask"].sum(dim=1).tolist()

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generate_kwargs["top_p"] = self.top_p
        if self.stop_on_json:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _JsonObjectStoppingCriteria(
                        tokenizer=self.processor,
                        prompt_width=prompt_width,
                        batch_size=len(batch),
                        response_prefix=self.assistant_prefill,
                    )
                ]
            )

        start = time.time()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
        except Exception:
            self._disable_batch_infer = True
            return [
                self.infer(
                    messages=item["messages"],
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                )
                for item in batch
            ]
        latency_ms = (time.time() - start) * 1000

        records = []
        for idx, item in enumerate(batch):
            completion_ids = output_ids[idx][prompt_width:]
            raw_response = self.processor.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if self.assistant_prefill:
                raw_response = self.assistant_prefill + raw_response
            records.append(
                InferenceRecord(
                    query_id=item.get("query_id", ""),
                    method=item.get("method", ""),
                    model=self.model_name,
                    raw_response=raw_response,
                    prompt_tokens=int(input_token_counts[idx]),
                    completion_tokens=int(completion_ids.shape[-1]),
                    latency_ms=latency_ms,
                    temperature=self.temperature,
                    seed=self.seed,
                    finish_reason="stop",
                )
            )
        return records
