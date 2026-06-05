from collections import Counter
from dataclasses import dataclass, field
from .dataset_contracts import get_classification_contract
from .templates import ClassificationTemplate, VQATemplate, _image_content, _text_content
from .zero_shot import PromptRecord


def _compact_label_counts(retrieved_refs: list) -> dict:
    return dict(Counter(ref.label_name for ref in retrieved_refs))


def _format_multilabel_findings(sample, label_names: list) -> str:
    values = getattr(sample, "multi_label", None) or []
    present = [name for name, value in zip(label_names or [], values) if int(value) == 1]
    if not present:
        return "no listed findings present"
    return ", ".join(present)


def _multilabel_query_instruction(label_names: list, n_references: int, retrieval_description: str, score_name: str) -> str:
    labels_str = ", ".join(label_names)
    label_schema = ", ".join([f'"{name}":{{"present":false,"probability":0.0}}' for name in label_names])
    return (
        f"You are given the {n_references} most similar training-set chest X-ray reference images retrieved by "
        f"{retrieval_description}, each with known CheXpert findings and {score_name}.\n"
        "Use the retrieved references as visual context, and combine that context with visual evidence in the query image. "
        "Do not force a single diagnosis; multiple findings may be present.\n\n"
        f"Findings: {labels_str}\n"
        "Return only one compact JSON object, no markdown, no extra text, exactly in this schema:\n"
        f'{{"findings":{{{label_schema}}},"evidence":"eight words max"}}'
    )


class RGICLPrompter:
    def __init__(self, k: int = 6, **kwargs):
        self.k = k

    def build_classification_prompt(self, query_sample, retrieved_refs: list,
                                     retrieval_result=None, label_names: list = None,
                                     is_multi_label: bool = False,
                                     dataset_name: str = "") -> PromptRecord:
        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []

        for idx, ref in enumerate(retrieved_refs):
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

        method_name = "rg_icl_global_spatial"
        if retrieval_result is not None:
            method_name = f"rg_icl_{retrieval_result.method}"

        query_content = ClassificationTemplate.format_query(
            image_path=query_sample.image_path,
            label_names=label_names,
            is_multi_label=is_multi_label,
            dataset_name=dataset_name,
            method=method_name,
        )
        user_content.extend(query_content)

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method=method_name,
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )

    def build_vqa_prompt(self, query_sample, retrieved_refs: list,
                         retrieval_result=None) -> PromptRecord:
        system = VQATemplate.SYSTEM_PROMPT

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []

        for idx, ref in enumerate(retrieved_refs):
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

        method_name = "rg_icl_global_spatial"
        if retrieval_result is not None:
            method_name = f"rg_icl_{retrieval_result.method}"

        return PromptRecord(
            query_id=query_sample.id,
            method=method_name,
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            messages=messages,
        )


class KNNCorrectionPrompter:
    def __init__(self, k: int = 6, **kwargs):
        self.k = k

    def build_classification_prompt(
        self,
        query_sample,
        retrieved_refs: list,
        retrieval_result=None,
        label_names: list = None,
        is_multi_label: bool = False,
        dataset_name: str = "",
    ) -> PromptRecord:
        if (dataset_name or "").lower() != "lag":
            raise ValueError("KNN correction prompting is currently defined only by the LAG dataset contract.")
        if is_multi_label:
            raise ValueError("KNN correction prompting is only implemented for binary classification.")
        if not label_names or len(label_names) != 2:
            raise ValueError("KNN correction prompting expects exactly two class names.")
        if retrieval_result is None:
            raise ValueError("KNN correction prompting requires retrieval_result with neighbor scores.")

        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)
        contract = get_classification_contract(dataset_name)
        negative_label = label_names[0]
        positive_label = label_names[-1]

        ref_ids = []
        ref_labels = []
        ref_order = []
        ref_content = []
        positive_count = 0

        scores_by_id = {
            str(ref_id): float(score)
            for ref_id, score in zip(retrieval_result.neighbor_ids, retrieval_result.neighbor_scores)
        }

        for idx, ref in enumerate(retrieved_refs, start=1):
            ref_id = str(ref.id)
            score = scores_by_id.get(ref_id, 0.0)
            label = ref.label_name
            if label == positive_label:
                positive_count += 1
            ref_content.append(_image_content(ref.image_path))
            ref_content.append(
                _text_content(
                    f"Reference {idx}: diagnosis = {label}; CLIP similarity = {score:.6f}."
                )
            )
            ref_ids.append(ref.id)
            ref_labels.append(label)
            ref_order.append(idx - 1)

        k = len(retrieved_refs)
        p_knn = positive_count / k if k else 0.0
        knn_prediction = positive_label if positive_count > (k / 2) else negative_label
        vote_summary = f"{positive_count}/{k} {positive_label}, {k - positive_count}/{k} {negative_label}"

        query_instruction = (
            "A CLIP nearest-neighbor classifier has already made a preliminary diagnosis.\n"
            f"kNN prediction: {knn_prediction}\n"
            f"kNN vote summary: {vote_summary}\n"
            f"kNN p_glaucoma from neighbor labels: {p_knn:.6f}\n\n"
            "Your task is to audit this kNN prediction, not merely repeat it.\n"
            "Review the query fundus image and the retrieved references. Decide whether the nearest neighbors are "
            "diagnostically valid, or whether they may be only superficially similar due to color, brightness, image "
            "quality, field of view, or camera style. Check whether the query image itself has glaucomatous evidence "
            "such as enlarged cup-to-disc ratio, neuroretinal rim thinning/notching, RNFL defects, disc hemorrhage, "
            "or suspicious optic nerve appearance. If the kNN prediction is misleading, override it using visual evidence.\n\n"
            f"Classify the query image into one of: {', '.join(label_names)}.\n"
            "Return only one compact JSON object with exactly these keys:\n"
            '{"label":"glaucoma|non_glaucoma","confidence":0.0,"probability":0.0,'
            '"override_knn":true,"knn_diagnostically_valid":true,"evidence":"eight words max"}\n'
            "The confidence field is confidence in the chosen final label. "
            "The probability field MUST be the probability that the query image is glaucoma."
        )

        user_content = []
        user_content.extend(ref_content)
        user_content.append(_image_content(query_sample.image_path))
        user_content.append(_text_content(query_instruction))

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        return PromptRecord(
            query_id=query_sample.id,
            method="rg_icl_global_knn_correction",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            metadata={
                "knn_prediction": knn_prediction,
                "knn_positive_count": positive_count,
                "knn_k": k,
                "knn_p_glaucoma": p_knn,
                "neighbor_scores": [
                    scores_by_id.get(str(ref.id), 0.0) for ref in retrieved_refs
                ],
            },
            messages=messages,
        )


class GlobalSimilarityPrompter:
    def __init__(self, k: int = 6, **kwargs):
        self.k = k

    def build_classification_prompt(
        self,
        query_sample,
        retrieved_refs: list,
        retrieval_result=None,
        label_names: list = None,
        is_multi_label: bool = False,
        dataset_name: str = "",
    ) -> PromptRecord:
        if not label_names or len(label_names) < 2:
            raise ValueError("Global similarity prompting expects at least two class names.")
        if retrieval_result is None:
            raise ValueError("Global similarity prompting requires retrieval_result with neighbor scores.")

        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)
        contract = get_classification_contract(dataset_name)
        scores_by_id = {
            str(ref_id): float(score)
            for ref_id, score in zip(retrieval_result.neighbor_ids, retrieval_result.neighbor_scores)
        }

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []
        neighbor_scores = []

        for idx, ref in enumerate(retrieved_refs, start=1):
            score = scores_by_id.get(str(ref.id), 0.0)
            user_content.append(_image_content(ref.image_path))
            user_content.append(
                _text_content(
                    (
                        f"Reference {idx}: CheXpert findings present = {_format_multilabel_findings(ref, label_names)}; "
                        f"retrieval similarity = {score:.6f}."
                    ) if is_multi_label else (
                        f"Reference {idx}: {contract.reference_label_name.lower()} = {ref.label_name}; "
                        f"retrieval similarity = {score:.6f}."
                    )
                )
            )
            ref_ids.append(ref.id)
            ref_labels.append(_format_multilabel_findings(ref, label_names) if is_multi_label else ref.label_name)
            ref_order.append(idx - 1)
            neighbor_scores.append(float(score))

        if is_multi_label:
            query_instruction = _multilabel_query_instruction(
                label_names=label_names,
                n_references=len(retrieved_refs),
                retrieval_description="global image-embedding similarity",
                score_name="retrieval similarity score",
            )
        else:
            query_instruction = contract.global_similarity_query_text(
                label_names=label_names,
                n_references=len(retrieved_refs),
                retrieval_description="global image-embedding similarity",
                score_name="retrieval similarity score",
            )

        user_content.append(_image_content(query_sample.image_path))
        user_content.append(_text_content(query_instruction))

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        label_counts = _compact_label_counts(retrieved_refs)
        positive_count = sum(1 for ref in retrieved_refs if ref.label_name == label_names[-1])
        return PromptRecord(
            query_id=query_sample.id,
            method="rg_icl_global_similarity",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            metadata={
                "reference_label_counts": label_counts,
                "global_positive_count": positive_count,
                "global_negative_count": len(retrieved_refs) - positive_count,
                "neighbor_scores": neighbor_scores,
            },
            messages=messages,
        )


class DualGlobalSimilarityPrompter:
    def __init__(self, k: int = 6, **kwargs):
        self.k = k

    def build_classification_prompt(
        self,
        query_sample,
        retrieved_refs: list,
        retrieval_result=None,
        label_names: list = None,
        is_multi_label: bool = False,
        dataset_name: str = "",
    ) -> PromptRecord:
        if not label_names or len(label_names) < 2:
            raise ValueError("Dual global similarity prompting expects at least two class names.")
        if retrieval_result is None:
            raise ValueError("Dual global similarity prompting requires retrieval_result with neighbor scores.")

        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)
        contract = get_classification_contract(dataset_name)
        scores_by_id = {
            str(ref_id): float(score)
            for ref_id, score in zip(retrieval_result.neighbor_ids, retrieval_result.neighbor_scores)
        }

        user_content = []
        ref_ids = []
        ref_labels = []
        ref_order = []
        neighbor_scores = []

        for idx, ref in enumerate(retrieved_refs, start=1):
            score = scores_by_id.get(str(ref.id), 0.0)
            user_content.append(_image_content(ref.image_path))
            user_content.append(
                _text_content(
                    (
                        f"Reference {idx}: CheXpert findings present = {_format_multilabel_findings(ref, label_names)}; "
                        f"combined similarity = 0.5*CLIP + 0.5*DINOv3 embedding = {score:.6f}."
                    ) if is_multi_label else (
                        f"Reference {idx}: {contract.reference_label_name.lower()} = {ref.label_name}; "
                        f"combined similarity = 0.5*CLIP + 0.5*DINOv3 embedding = {score:.6f}."
                    )
                )
            )
            ref_ids.append(ref.id)
            ref_labels.append(_format_multilabel_findings(ref, label_names) if is_multi_label else ref.label_name)
            ref_order.append(idx - 1)
            neighbor_scores.append(float(score))

        if is_multi_label:
            query_instruction = _multilabel_query_instruction(
                label_names=label_names,
                n_references=len(retrieved_refs),
                retrieval_description=(
                    "a combined global visual similarity score: 0.5 times CLIP global cosine similarity "
                    "plus 0.5 times DINOv3 embedding cosine similarity"
                ),
                score_name="combined similarity score",
            )
        else:
            query_instruction = contract.global_similarity_query_text(
                label_names=label_names,
                n_references=len(retrieved_refs),
                retrieval_description=(
                    "a combined global visual similarity score: 0.5 times CLIP global cosine similarity "
                    "plus 0.5 times DINOv3 embedding cosine similarity"
                ),
                score_name="combined similarity score",
            )

        user_content.append(_image_content(query_sample.image_path))
        user_content.append(_text_content(query_instruction))

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        label_counts = _compact_label_counts(retrieved_refs)
        positive_count = sum(1 for ref in retrieved_refs if ref.label_name == label_names[-1])
        return PromptRecord(
            query_id=query_sample.id,
            method="rg_icl_dual_global_similarity",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            metadata={
                "reference_label_counts": label_counts,
                "dual_global_positive_count": positive_count,
                "dual_global_negative_count": len(retrieved_refs) - positive_count,
                "neighbor_scores": neighbor_scores,
                "score_definition": "0.5*CLIP_global_cosine + 0.5*DINOv3_embedding_cosine",
            },
            messages=messages,
        )


class BalancedSimilarityPrompter:
    def __init__(self, k: int = 6, **kwargs):
        self.k = k

    def build_classification_prompt(
        self,
        query_sample,
        retrieved_refs: list,
        retrieval_result=None,
        label_names: list = None,
        is_multi_label: bool = False,
        dataset_name: str = "",
        reference_scores: list | None = None,
    ) -> PromptRecord:
        if (dataset_name or "").lower() != "lag":
            raise ValueError("Balanced similarity prompting is currently defined only by the LAG dataset contract.")
        if is_multi_label:
            raise ValueError("Balanced similarity prompting is only implemented for binary classification.")
        if not label_names or len(label_names) != 2:
            raise ValueError("Balanced similarity prompting expects exactly two class names.")

        system = ClassificationTemplate.get_system_prompt(dataset_name, is_multi_label)
        ref_ids = []
        ref_labels = []
        ref_order = []
        user_content = []
        scores = list(reference_scores or [0.0] * len(retrieved_refs))

        for idx, (ref, score) in enumerate(zip(retrieved_refs, scores), start=1):
            user_content.append(_image_content(ref.image_path))
            user_content.append(
                _text_content(
                    f"Reference {idx}: diagnosis = {ref.label_name}; CLIP similarity = {float(score):.6f}."
                )
            )
            ref_ids.append(ref.id)
            ref_labels.append(ref.label_name)
            ref_order.append(idx - 1)

        query_instruction = (
            "You are given a class-balanced retrieved reference set: the 3 most similar glaucoma references and "
            "the 3 most similar non-glaucoma references from the training set, each with its CLIP similarity score.\n"
            "Compare the query fundus image against both groups. Use the similarity scores as retrieval context, "
            "but make the final diagnosis from diagnostic visual evidence in the query image and the references.\n\n"
            f"Classify the query image into one of: {', '.join(label_names)}.\n"
            "Return only one compact JSON object with exactly these keys:\n"
            '{"label":"glaucoma|non_glaucoma","confidence":0.0,"probability":0.0,'
            '"evidence":"eight words max"}\n'
            "The confidence field is confidence in the chosen final label. "
            "The probability field MUST be the probability that the query image is glaucoma."
        )

        user_content.append(_image_content(query_sample.image_path))
        user_content.append(_text_content(query_instruction))

        messages = [
            {"role": "system", "content": [_text_content(system)]},
            {"role": "user", "content": user_content},
        ]

        positive_count = sum(1 for ref in retrieved_refs if ref.label_name == label_names[-1])
        return PromptRecord(
            query_id=query_sample.id,
            method="rg_icl_global_balanced",
            reference_ids=ref_ids,
            reference_labels=ref_labels,
            reference_order=ref_order,
            metadata={
                "balanced_positive_count": positive_count,
                "balanced_negative_count": len(retrieved_refs) - positive_count,
                "neighbor_scores": [float(score) for score in scores],
            },
            messages=messages,
        )
