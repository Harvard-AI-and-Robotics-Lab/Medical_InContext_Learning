from .templates import ClassificationTemplate, VQATemplate
from .zero_shot import ZeroShotPrompter
from .naive_icl import NaiveICLPrompter
from .fixed_random_icl import FixedRandomICLPrompter
from .rg_icl import (
    BalancedSimilarityPrompter,
    DualGlobalSimilarityPrompter,
    GlobalSimilarityPrompter,
    KNNCorrectionPrompter,
    RGICLPrompter,
)

PROMPTERS = {
    "zero_shot": ZeroShotPrompter,
    "naive_icl": NaiveICLPrompter,
    "fixed_random_6": FixedRandomICLPrompter,
    "rg_icl_global": RGICLPrompter,
    "rg_icl_spatial": RGICLPrompter,
    "rg_icl_global_spatial": RGICLPrompter,
    "rg_icl_global_knn_correction": KNNCorrectionPrompter,
    "rg_icl_global_balanced": BalancedSimilarityPrompter,
    "rg_icl_global_similarity": GlobalSimilarityPrompter,
    "rg_icl_dual_global_similarity": DualGlobalSimilarityPrompter,
}


def get_prompter(method: str, **kwargs):
    if method not in PROMPTERS:
        raise ValueError(f"Unknown method: {method}. Available: {list(PROMPTERS.keys())}")
    return PROMPTERS[method](**kwargs)
