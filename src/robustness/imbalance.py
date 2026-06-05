import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImbalanceResult:
    ratio_name: str = ""
    neg_count: int = 0
    pos_count: int = 0
    reference_ids: list = field(default_factory=list)
    reference_labels: list = field(default_factory=list)
    sens_at_spec90: float = 0.0
    spec_at_sens90: float = 0.0
    auc: float = 0.0
    method: str = ""

    def to_dict(self):
        return {
            "ratio_name": self.ratio_name,
            "neg_count": self.neg_count,
            "pos_count": self.pos_count,
            "reference_ids": self.reference_ids,
            "reference_labels": self.reference_labels,
            "sens_at_spec90": self.sens_at_spec90,
            "spec_at_sens90": self.spec_at_sens90,
            "auc": self.auc,
            "method": self.method,
        }


class ImbalanceExperiment:
    def __init__(self, k: int = 6, seed: int = 42):
        self.k = k
        self.seed = seed

    def construct_imbalanced_set(self, reference_pool: list, neg_ratio: int,
                                  pos_ratio: int, positive_label: int = 1) -> list:
        total = self.k
        pos_count = int(round(total * pos_ratio / (neg_ratio + pos_ratio)))
        neg_count = total - pos_count

        positives = [s for s in reference_pool if s.label == positive_label]
        negatives = [s for s in reference_pool if s.label != positive_label]

        rng = np.random.RandomState(self.seed)

        if len(positives) < pos_count:
            selected_pos = positives[:]
        else:
            idx = rng.choice(len(positives), size=pos_count, replace=False)
            selected_pos = [positives[i] for i in idx]

        if len(negatives) < neg_count:
            selected_neg = negatives[:]
        else:
            idx = rng.choice(len(negatives), size=neg_count, replace=False)
            selected_neg = [negatives[i] for i in idx]

        combined = selected_neg + selected_pos
        rng.shuffle(combined)

        return combined

    def get_screening_set(self, reference_pool: list, positive_label: int = 1) -> list:
        return self.construct_imbalanced_set(reference_pool, neg_ratio=5, pos_ratio=1,
                                              positive_label=positive_label)

    def get_balanced_set(self, reference_pool: list, positive_label: int = 1) -> list:
        return self.construct_imbalanced_set(reference_pool, neg_ratio=3, pos_ratio=3,
                                              positive_label=positive_label)

    def get_specialty_set(self, reference_pool: list, positive_label: int = 1) -> list:
        return self.construct_imbalanced_set(reference_pool, neg_ratio=1, pos_ratio=5,
                                              positive_label=positive_label)

    def run_all_ratios(self, reference_pool: list, ratios: list,
                       positive_label: int = 1) -> list:
        results = []
        for ratio in ratios:
            neg_r = ratio["neg"]
            pos_r = ratio["pos"]
            refs = self.construct_imbalanced_set(
                reference_pool, neg_ratio=neg_r, pos_ratio=pos_r,
                positive_label=positive_label,
            )
            ratio_name = f"{neg_r}N:{pos_r}P"
            results.append({
                "ratio_name": ratio_name,
                "references": refs,
                "neg_count": sum(1 for s in refs if s.label != positive_label),
                "pos_count": sum(1 for s in refs if s.label == positive_label),
            })
        return results
