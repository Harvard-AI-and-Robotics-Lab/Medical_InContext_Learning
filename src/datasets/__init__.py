from .base import BaseDataset, ClassificationDataset, VQADataset
from .lag import LAGDataset
from .project_lag import ProjectLAGDataset
from .ddr import DDRDataset
from .chexpert import CheXpertDataset
from .breakhis import BreakHisBinaryDataset, BreakHisDataset
from .tbx11k import TBX11KDataset
from .medical_cxr_vqa import MedicalCXRVQADataset
from .vqa_rad import VQARADDataset
from .pathvqa import PathVQADataset
from .pathmmu import PathMMUDataset
from .pmc_vqa import PMCVQADataset
from .slake import SLAKEDataset
from .vqamed2019 import VQAMed2019Dataset

CLASSIFICATION_DATASETS = {
    "lag": LAGDataset,
    "lag_project": ProjectLAGDataset,
    "ddr": DDRDataset,
    "chexpert": CheXpertDataset,
    "breakhis": BreakHisDataset,
    "breakhis_binary": BreakHisBinaryDataset,
    "tbx11k": TBX11KDataset,
}

VQA_DATASETS = {
    "medical_cxr_vqa": MedicalCXRVQADataset,
    "vqa_rad": VQARADDataset,
    "pathvqa": PathVQADataset,
    "pathmmu": PathMMUDataset,
    "pmc_vqa": PMCVQADataset,
    "slake": SLAKEDataset,
    "vqamed2019": VQAMed2019Dataset,
}

ALL_DATASETS = {**CLASSIFICATION_DATASETS, **VQA_DATASETS}


def get_dataset(name: str, data_root: str, split: str = "test", **kwargs):
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(ALL_DATASETS.keys())}")
    return ALL_DATASETS[name](data_root=data_root, split=split, **kwargs)
