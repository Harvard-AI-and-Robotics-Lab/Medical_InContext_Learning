from dataclasses import dataclass, field
from .templates import ClassificationTemplate, VQATemplate, PromptMessage, _text_content


@dataclass
class PromptRecord:
    query_id: str
    method: str
    reference_ids: list = field(default_factory=list)
    reference_labels: list = field(default_factory=list)
    reference_order: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "method": self.method,
            "reference_ids": self.reference_ids,
            "reference_labels": self.reference_labels,
            "reference_order": self.reference_order,
            "metadata": self.metadata,
        }


class ZeroShotPrompter:
    def __init__(self, **kwargs):
        pass

    def build_classification_prompt(self, query_sample, label_names: list,
                                     is_multi_label: bool = False,
                                     dataset_name: str = "") -> PromptRecord:
        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)

        query_content = ClassificationTemplate.format_query(
            image_path=query_sample.image_path,
            label_names=label_names,
            is_multi_label=is_multi_label,
            dataset_name=dataset_name,
            method="zero_shot",
        )

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": query_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="zero_shot",
            messages=messages,
        )

    def build_vqa_prompt(self, query_sample) -> PromptRecord:
        system = VQATemplate.SYSTEM_PROMPT

        query_content = VQATemplate.format_query(
            image_path=query_sample.image_path,
            question=query_sample.question,
        )

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": query_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="zero_shot",
            messages=messages,
        )
