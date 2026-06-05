import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrderingResult:
    permutation_idx: int = 0
    reference_ids: list = field(default_factory=list)
    reference_order: list = field(default_factory=list)
    auc: float = 0.0
    method: str = ""

    def to_dict(self):
        return {
            "permutation_idx": self.permutation_idx,
            "reference_ids": self.reference_ids,
            "reference_order": self.reference_order,
            "auc": self.auc,
            "method": self.method,
        }


@dataclass
class OrderingSummary:
    method: str = ""
    mean_auc: float = 0.0
    std_auc: float = 0.0
    min_auc: float = 0.0
    max_auc: float = 0.0
    n_permutations: int = 0
    per_permutation: list = field(default_factory=list)

    def to_dict(self):
        return {
            "method": self.method,
            "mean_auc": self.mean_auc,
            "std_auc": self.std_auc,
            "min_auc": self.min_auc,
            "max_auc": self.max_auc,
            "n_permutations": self.n_permutations,
            "per_permutation": [p.to_dict() for p in self.per_permutation],
        }


class OrderingExperiment:
    def __init__(self, n_permutations: int = 10, seed: int = 42):
        self.n_permutations = n_permutations
        self.seed = seed

    def generate_permutations(self, reference_ids: list) -> list:
        rng = np.random.RandomState(self.seed)
        k = len(reference_ids)
        permutations = []
        seen = set()

        original = tuple(range(k))
        permutations.append(list(original))
        seen.add(original)

        max_attempts = self.n_permutations * 100
        attempts = 0
        while len(permutations) < self.n_permutations and attempts < max_attempts:
            perm = rng.permutation(k).tolist()
            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                permutations.append(perm)
                seen.add(perm_tuple)
            attempts += 1

        return permutations

    def reorder_references(self, references: list, permutation: list) -> list:
        return [references[i] for i in permutation]

    def summarize(self, results: list, method: str = "") -> OrderingSummary:
        if not results:
            return OrderingSummary(method=method)

        aucs = [r.auc for r in results]
        return OrderingSummary(
            method=method,
            mean_auc=float(np.mean(aucs)),
            std_auc=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            min_auc=float(np.min(aucs)),
            max_auc=float(np.max(aucs)),
            n_permutations=len(results),
            per_permutation=results,
        )
