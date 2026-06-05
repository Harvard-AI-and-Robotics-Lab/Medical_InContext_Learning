import numpy as np
from dataclasses import dataclass, field
from .templates import ClassificationTemplate, VQATemplate, _text_content
from .zero_shot import PromptRecord


class NaiveICLPrompter:
    def __init__(self, k: int = 6, seed: int = 42, **kwargs):
        self.k = k
        self.seed = seed

    def _sample_references(self, reference_pool: list, k: int, rng_seed: int) -> list:
        rng = np.random.RandomState(rng_seed)
        n = len(reference_pool)
        if n <= k:
            return list(reference_pool)
        indices = rng.choice(n, size=k, replace=False)
        return [reference_pool[i] for i in indices]

    def build_classification_prompt(self, query_sample, reference_pool: list,
                                     label_names: list, is_multi_label: bool = False,
                                     dataset_name: str = "", k: int = None,
                                     rng_seed: int = None) -> PromptRecord:
        k = k if k is not None else self.k
        rng_seed = rng_seed if rng_seed is not None else self.seed

        refs = self._sample_references(reference_pool, k, rng_seed)

        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []

        for idx, ref in enumerate(refs):
            ref_content = ClassificationTemplate.format_reference(
                image_path=ref.image_path,
                label_name=ref.label_name,
                is_multi_label=is_multi_label,
                multi_label=getattr(ref, 'multi_label', None),
                label_names=label_names,
                dataset_name=dataset_name,
            )
            user_content.extend(ref_content)
            ref_ids.append(ref.id)
            ref_labels.append(ref.label_name if not is_multi_label else str(ref.multi_label))
            ref_order.append(idx)

        query_content = ClassificationTemplate.format_query(
            image_path=query_sample.image_path,
            label_names=label_names,
            is_multi_label=is_multi_label,
            dataset_name=dataset_name,
            method="naive_icl",
        )
        user_content.extend(query_content)

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="naive_icl",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )

    def build_vqa_prompt(self, query_sample, reference_pool: list,
                         k: int = None, rng_seed: int = None) -> PromptRecord:
        k = k if k is not None else self.k
        rng_seed = rng_seed if rng_seed is not None else self.seed

        refs = self._sample_references(reference_pool, k, rng_seed)

        system = VQATemplate.SYSTEM_PROMPT

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []

        for idx, ref in enumerate(refs):
            ref_content = VQATemplate.format_reference(
                image_path=ref.image_path,
                question=ref.question,
                answer=ref.answer,
            )
            user_content.extend(ref_content)
            ref_ids.append(ref.id)
            ref_labels.append(ref.answer)
            ref_order.append(idx)

        query_content = VQATemplate.format_query(
            image_path=query_sample.image_path,
            question=query_sample.question,
        )
        user_content.extend(query_content)

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="naive_icl",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )
