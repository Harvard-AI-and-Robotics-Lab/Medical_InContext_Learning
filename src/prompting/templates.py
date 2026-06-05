from dataclasses import dataclass
from typing import Optional
from .dataset_contracts import (
    DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    get_classification_contract,
)


@dataclass
class PromptMessage:
    role: str
    content: list


def _image_content(image_path: str):
    return {"type": "image_url", "image_url": {"url": image_path}}


def _text_content(text: str):
    return {"type": "text", "text": text}


VQA_SYSTEM_PROMPT = (
    "You are a board-certified medical specialist with extensive experience interpreting medical images.\n"
    "Your task is to answer clinical visual questions accurately and conservatively based solely on "
    "the provided image and question."
)

class ClassificationTemplate:
    @staticmethod
    def get_system_prompt(dataset_name: str = "", is_multi_label: bool = False) -> str:
        if is_multi_label:
            return (
                "You are a clinical chest X-ray interpretation assistant.\n"
                "You must detect each possible condition independently."
            )
        return get_classification_contract(dataset_name).system_prompt

    @staticmethod
    def get_method_instruction(method: str = "zero_shot", dataset_name: str = "") -> str:
        return get_classification_contract(dataset_name).method_instruction(method)

    SYSTEM_PROMPT = DEFAULT_CLASSIFICATION_SYSTEM_PROMPT

    MULTI_LABEL_SYSTEM_PROMPT = (
        "You are a clinical chest X-ray interpretation assistant.\n"
        "You must detect each possible condition independently."
    )

    @staticmethod
    def format_query(image_path: str, label_names: list, is_multi_label: bool = False,
                     dataset_name: str = "", method: str = "zero_shot") -> list:
        instruction = ClassificationTemplate.get_method_instruction(method, dataset_name)
        if is_multi_label:
            labels_str = ", ".join(label_names)
            label_schema = ", ".join([f'"{name}":{{"present":false,"probability":0.0}}' for name in label_names])
            text = (
                f"{instruction}\n\n"
                "Independently assess each listed CheXpert finding as present or absent. "
                "Use probability as calibrated probability that the finding is present.\n"
                f"Findings: {labels_str}\n\n"
                "Return only one compact JSON object, no markdown, no extra text, exactly in this schema:\n"
                f'{{"findings":{{{label_schema}}},"evidence":"eight words max"}}'
            )
        else:
            contract = get_classification_contract(dataset_name)
            text = contract.format_query_text(label_names, method)
        return [_image_content(image_path), _text_content(text)]

    @staticmethod
    def format_reference(image_path: str, label_name: str, is_multi_label: bool = False,
                         multi_label: list = None, label_names: list = None,
                         dataset_name: str = "") -> list:
        if is_multi_label and multi_label is not None and label_names is not None:
            findings = []
            for name, val in zip(label_names, multi_label):
                status = "present" if val == 1 else "absent"
                findings.append(f"{name}: {status}")
            label_text = "; ".join(findings)
        else:
            label_text = label_name
        text = get_classification_contract(dataset_name).reference_text(label_text)
        return [_image_content(image_path), _text_content(text)]


class VQATemplate:
    SYSTEM_PROMPT = VQA_SYSTEM_PROMPT

    @staticmethod
    def format_query(image_path: str, question: str) -> list:
        text = f"Question: {question}\nProvide your answer based on the medical image shown."
        return [_image_content(image_path), _text_content(text)]

    @staticmethod
    def format_reference(image_path: str, question: str, answer: str) -> list:
        text = f"Reference — Question: {question}\nAnswer: {answer}"
        return [_image_content(image_path), _text_content(text)]
