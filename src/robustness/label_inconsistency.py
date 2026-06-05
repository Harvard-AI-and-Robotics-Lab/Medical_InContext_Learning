import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InconsistencyResult:
    query_id: str = ""
    original_ref_id: str = ""
    replaced_ref_id: str = ""
    original_ref_label: str = ""
    replaced_ref_label: str = ""
    similarity_score: float = 0.0
    auc_before: float = 0.0
    auc_after: float = 0.0
    auc_drop: float = 0.0
    method: str = ""

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "original_ref_id": self.original_ref_id,
            "replaced_ref_id": self.replaced_ref_id,
            "original_ref_label": self.original_ref_label,
            "replaced_ref_label": self.replaced_ref_label,
            "similarity_score": self.similarity_score,
            "auc_before": self.auc_before,
            "auc_after": self.auc_after,
            "auc_drop": self.auc_drop,
            "method": self.method,
        }


class LabelInconsistencyExperiment:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def find_visually_similar_inconsistent(self, target_sample, reference_pool: list,
                                            global_embeddings: dict,
                                            top_n: int = 10) -> list:
        target_emb = global_embeddings.get(target_sample.id)
        if target_emb is None:
            return []

        target_label = target_sample.label
        target_norm = np.linalg.norm(target_emb)
        if target_norm > 0:
            target_emb_n = target_emb / target_norm
        else:
            target_emb_n = target_emb

        candidates = []
        for ref in reference_pool:
            if ref.label == target_label:
                continue
            ref_emb = global_embeddings.get(ref.id)
            if ref_emb is None:
                continue
            ref_norm = np.linalg.norm(ref_emb)
            if ref_norm > 0:
                ref_emb_n = ref_emb / ref_norm
            else:
                ref_emb_n = ref_emb
            sim = float(np.dot(target_emb_n, ref_emb_n))
            candidates.append((ref, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def inject_inconsistent_reference(self, references: list, position: int,
                                       inconsistent_sample, reference_pool: list,
                                       global_embeddings: dict) -> tuple:
        original_ref = references[position]

        candidates = self.find_visually_similar_inconsistent(
            original_ref, reference_pool, global_embeddings
        )

        if not candidates:
            return references, None, 0.0

        replacement, similarity = candidates[0]

        new_refs = references.copy()
        new_refs[position] = replacement

        return new_refs, replacement, similarity

    def run_stress_test(self, references: list, reference_pool: list,
                        global_embeddings: dict) -> list:
        injection_results = []
        for pos in range(len(references)):
            new_refs, replacement, sim = self.inject_inconsistent_reference(
                references, pos, None, reference_pool, global_embeddings
            )
            if replacement is not None:
                injection_results.append({
                    "position": pos,
                    "original_id": references[pos].id,
                    "original_label": references[pos].label_name,
                    "replacement_id": replacement.id,
                    "replacement_label": replacement.label_name,
                    "similarity": sim,
                    "modified_references": new_refs,
                })
        return injection_results
