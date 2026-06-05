from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import yaml


@dataclass
class EncoderConfig:
    name: str = "dinov3"
    model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    device: str = "cuda"
    batch_size: int = 32
    image_size: int = 512
    normalize: bool = True


@dataclass
class RetrievalConfig:
    method: str = "global_spatial"
    k: int = 6
    alpha: float = 0.5
    similarity_metric: str = "cosine"
    exclude_query: bool = True
    exclude_test_set: bool = True


@dataclass
class InferenceConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 42
    top_p: Optional[float] = None
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class MetricsConfig:
    classification: list = field(default_factory=lambda: [
        "auc", "sens_at_spec90", "accuracy", "brier", "ece", "nll"
    ])
    vqa_lexical: list = field(default_factory=lambda: [
        "bleu4", "rouge_l", "meteor"
    ])
    bootstrap_n: int = 2000
    bootstrap_seed: int = 42
    ci_level: float = 0.95


@dataclass
class JudgeConfig:
    model: str = "gpt-5"
    temperature: float = 0.0
    seed: int = 42
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = ""
    n_samples_per_dataset: int = 300
    dimensions: list = field(default_factory=lambda: [
        "clinical_correctness",
        "evidence_grounding",
        "completeness",
        "uncertainty_acknowledgement",
    ])
    score_range: tuple = (0, 5)


@dataclass
class RobustnessConfig:
    imbalance_ratios: list = field(default_factory=lambda: [
        {"neg": 5, "pos": 1},
        {"neg": 3, "pos": 3},
        {"neg": 1, "pos": 5},
    ])
    ordering_permutations: int = 10
    ordering_seed: int = 42
    label_inconsistency_encoder: str = "dinov3"


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    data_root: str = "data"
    output_root: str = "outputs"
    output_file: str = "predictions.json"
    features_root: str = "outputs/features"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    datasets: list = field(default_factory=list)
    methods: list = field(default_factory=lambda: [
        "zero_shot", "naive_icl", "rg_icl_global", "rg_icl_global_spatial"
    ])


def _merge_dict_into_dataclass(dc, d):
    for k, v in d.items():
        if hasattr(dc, k):
            current = getattr(dc, k)
            if hasattr(current, '__dataclass_fields__') and isinstance(v, dict):
                _merge_dict_into_dataclass(current, v)
            else:
                setattr(dc, k, v)
    return dc


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig()
    if raw:
        _merge_dict_into_dataclass(cfg, raw)
    return cfg


def save_config(cfg: ExperimentConfig, path: str):
    from dataclasses import asdict
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
