import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PreferenceOutcome(Enum):
    WIN_A = "win_a"
    WIN_B = "win_b"
    TIE = "tie"


@dataclass
class JudgeResult:
    query_id: str
    question: str
    method_a: str
    method_b: str
    answer_a: str
    answer_b: str
    presented_order: str
    preference: PreferenceOutcome = PreferenceOutcome.TIE
    scores_a: dict = field(default_factory=dict)
    scores_b: dict = field(default_factory=dict)
    judge_model: str = ""
    judge_prompt_version: str = "v1"
    judge_temperature: float = 0.0
    judge_seed: Optional[int] = None
    raw_judge_response: str = ""
    latency_ms: float = 0.0

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "question": self.question,
            "method_a": self.method_a,
            "method_b": self.method_b,
            "answer_a": self.answer_a,
            "answer_b": self.answer_b,
            "presented_order": self.presented_order,
            "preference": self.preference.value,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "judge_model": self.judge_model,
            "judge_prompt_version": self.judge_prompt_version,
            "judge_temperature": self.judge_temperature,
            "judge_seed": self.judge_seed,
            "raw_judge_response": self.raw_judge_response,
            "latency_ms": self.latency_ms,
        }


RUBRIC_PROMPT = """You are an expert medical evaluator. You will be given a medical image question and two candidate answers (Answer A and Answer B). Evaluate each answer independently on the following dimensions using a 0-5 scale:

1. Clinical Correctness (0-5): Is the answer medically accurate given the visible image evidence and question context?
2. Evidence Grounding (0-5): Is the answer supported by visible image evidence?
3. Completeness (0-5): Does the answer address the question appropriately without unnecessary speculation?
4. Uncertainty Acknowledgement (0-5): Does the answer clearly express uncertainty when appropriate?

After scoring both answers independently, state your overall preference: which answer is better (A, B, or Tie).

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Respond in the following JSON format exactly:
{{
  "scores_a": {{
    "clinical_correctness": <0-5>,
    "evidence_grounding": <0-5>,
    "completeness": <0-5>,
    "uncertainty_acknowledgement": <0-5>
  }},
  "scores_b": {{
    "clinical_correctness": <0-5>,
    "evidence_grounding": <0-5>,
    "completeness": <0-5>,
    "uncertainty_acknowledgement": <0-5>
  }},
  "preference": "<A|B|Tie>",
  "reasoning": "<brief explanation>"
}}"""


class LLMJudge:
    def __init__(self, model: str = "gpt-5", temperature: float = 0.0,
                 seed: int = 42, api_key_env: str = "OPENAI_API_KEY",
                 base_url: str = "",
                 max_retries: int = 3, retry_delay: float = 5.0):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        import openai
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)

    def _randomize_order(self, answer_a: str, answer_b: str,
                         method_a: str, method_b: str,
                         rng: np.random.RandomState) -> tuple:
        if rng.random() < 0.5:
            return answer_a, answer_b, method_a, method_b, "a_first"
        return answer_b, answer_a, method_b, method_a, "b_first"

    def _parse_judge_response(self, raw: str, presented_order: str,
                              original_method_a: str, original_method_b: str) -> tuple:
        scores_a = {}
        scores_b = {}
        preference = PreferenceOutcome.TIE

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])

                presented_scores_a = parsed.get("scores_a", {})
                presented_scores_b = parsed.get("scores_b", {})

                if presented_order == "a_first":
                    scores_a = presented_scores_a
                    scores_b = presented_scores_b
                else:
                    scores_a = presented_scores_b
                    scores_b = presented_scores_a

                pref_str = parsed.get("preference", "Tie").strip().upper()
                if pref_str == "A":
                    if presented_order == "a_first":
                        preference = PreferenceOutcome.WIN_A
                    else:
                        preference = PreferenceOutcome.WIN_B
                elif pref_str == "B":
                    if presented_order == "a_first":
                        preference = PreferenceOutcome.WIN_B
                    else:
                        preference = PreferenceOutcome.WIN_A
                else:
                    preference = PreferenceOutcome.TIE
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        return scores_a, scores_b, preference

    def evaluate(self, query_id: str, question: str,
                 answer_a: str, answer_b: str,
                 method_a: str, method_b: str,
                 rng_seed: int = None) -> JudgeResult:
        rng = np.random.RandomState(rng_seed if rng_seed is not None else self.seed)

        pres_a, pres_b, pres_method_a, pres_method_b, order = self._randomize_order(
            answer_a, answer_b, method_a, method_b, rng
        )

        prompt = RUBRIC_PROMPT.format(
            question=question,
            answer_a=pres_a,
            answer_b=pres_b,
        )

        raw_response = ""
        latency = 0.0

        for attempt in range(self.max_retries):
            try:
                import openai as oai
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    seed=self.seed,
                )
                latency = (time.time() - start) * 1000
                raw_response = response.choices[0].message.content
                break
            except (oai.RateLimitError, oai.APITimeoutError, oai.APIConnectionError):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        scores_a, scores_b, preference = self._parse_judge_response(
            raw_response, order, method_a, method_b
        )

        return JudgeResult(
            query_id=query_id,
            question=question,
            method_a=method_a,
            method_b=method_b,
            answer_a=answer_a,
            answer_b=answer_b,
            presented_order=order,
            preference=preference,
            scores_a=scores_a,
            scores_b=scores_b,
            judge_model=self.model,
            judge_prompt_version="v1",
            judge_temperature=self.temperature,
            judge_seed=self.seed,
            raw_judge_response=raw_response,
            latency_ms=latency,
        )

    def evaluate_batch(self, items: list, delay: float = 1.0) -> list:
        results = []
        for i, item in enumerate(items):
            result = self.evaluate(
                query_id=item["query_id"],
                question=item["question"],
                answer_a=item["answer_a"],
                answer_b=item["answer_b"],
                method_a=item["method_a"],
                method_b=item["method_b"],
                rng_seed=self.seed + i,
            )
            results.append(result)
            if delay > 0:
                time.sleep(delay)
        return results

    @staticmethod
    def aggregate_results(results: list) -> dict:
        if not results:
            return {}

        wins_a = sum(1 for r in results if r.preference == PreferenceOutcome.WIN_A)
        wins_b = sum(1 for r in results if r.preference == PreferenceOutcome.WIN_B)
        ties = sum(1 for r in results if r.preference == PreferenceOutcome.TIE)
        total = len(results)

        dimensions = ["clinical_correctness", "evidence_grounding",
                       "completeness", "uncertainty_acknowledgement"]
        mean_scores_a = {}
        mean_scores_b = {}
        for dim in dimensions:
            vals_a = [r.scores_a.get(dim, 0) for r in results if dim in r.scores_a]
            vals_b = [r.scores_b.get(dim, 0) for r in results if dim in r.scores_b]
            mean_scores_a[dim] = float(np.mean(vals_a)) if vals_a else 0.0
            mean_scores_b[dim] = float(np.mean(vals_b)) if vals_b else 0.0

        return {
            "n_total": total,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": (wins_a + 0.5 * ties) / total if total > 0 else 0,
            "win_rate_b": (wins_b + 0.5 * ties) / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
            "mean_scores_a": mean_scores_a,
            "mean_scores_b": mean_scores_b,
            "method_a": results[0].method_a if results else "",
            "method_b": results[0].method_b if results else "",
        }
