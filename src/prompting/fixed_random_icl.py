from .templates import ClassificationTemplate, VQATemplate, _text_content
from .zero_shot import PromptRecord


class FixedRandomICLPrompter:
    def __init__(self, fixed_references: list | None = None, **kwargs):
        self.fixed_references = list(fixed_references or [])

    def build_classification_prompt(
        self,
        query_sample,
        fixed_references: list | None = None,
        label_names: list | None = None,
        is_multi_label: bool = False,
        dataset_name: str = "",
        **kwargs,
    ) -> PromptRecord:
        refs = list(fixed_references or self.fixed_references)
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
                multi_label=getattr(ref, "multi_label", None),
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
            method="fixed_random_6",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )

    def build_vqa_prompt(self, query_sample, fixed_references: list | None = None, **kwargs) -> PromptRecord:
        refs = list(fixed_references or self.fixed_references)
        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []

        for idx, ref in enumerate(refs):
            user_content.extend(
                VQATemplate.format_reference(
                    image_path=ref.image_path,
                    question=ref.question,
                    answer=ref.answer,
                )
            )
            ref_ids.append(ref.id)
            ref_labels.append(ref.answer)
            ref_order.append(idx)

        user_content.extend(
            VQATemplate.format_query(
                image_path=query_sample.image_path,
                question=query_sample.question,
            )
        )

        messages = [
            {"role": "system", "content": [_text_content(VQATemplate.SYSTEM_PROMPT)]},
            {"role": "user", "content": user_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="fixed_random_6",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )
