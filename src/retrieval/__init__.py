from .global_retrieval import GlobalRetriever
from .spatial_retrieval import SpatialRetriever
from .combined_retrieval import CombinedRetriever

RETRIEVERS = {
    "global": GlobalRetriever,
    "spatial": SpatialRetriever,
    "global_spatial": CombinedRetriever,
}


def get_retriever(method: str, **kwargs):
    if method not in RETRIEVERS:
        raise ValueError(f"Unknown retrieval method: {method}. Available: {list(RETRIEVERS.keys())}")
    return RETRIEVERS[method](**kwargs)
