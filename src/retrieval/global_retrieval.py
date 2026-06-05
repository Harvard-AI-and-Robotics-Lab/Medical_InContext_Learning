import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import hashlib


@dataclass
class RetrievalResult:
    query_id: str
    query_embedding_hash: str
    neighbor_ids: list
    neighbor_scores: list
    neighbor_labels: list
    encoder_name: str
    encoder_version: str
    preprocessing_hash: str
    method: str


class GlobalRetriever:
    def __init__(self, similarity_metric: str = "cosine", exclude_query: bool = True,
                 exclude_test_set: bool = True):
        self.similarity_metric = similarity_metric
        self.exclude_query = exclude_query
        self.exclude_test_set = exclude_test_set
        self._index_embeddings = None
        self._index_ids = None
        self._index_labels = None
        self._index_splits = None

    def build_index(self, ids: list, embeddings: np.ndarray, labels: list, splits: list):
        self._index_ids = np.array(ids)
        self._index_embeddings = embeddings.copy()
        self._index_labels = np.array(labels)
        self._index_splits = np.array(splits)

        norms = np.linalg.norm(self._index_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._index_embeddings = self._index_embeddings / norms

    def retrieve(self, query_id: str, query_embedding: np.ndarray, k: int = 6,
                 encoder_name: str = "", encoder_version: str = "",
                 preprocessing_hash: str = "") -> RetrievalResult:
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        similarities = self._index_embeddings @ query_normalized

        mask = np.ones(len(self._index_ids), dtype=bool)
        if self.exclude_query:
            mask &= self._index_ids != query_id
        if self.exclude_test_set:
            mask &= self._index_splits != "test"

        similarities[~mask] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:k]

        neighbor_ids = self._index_ids[top_indices].tolist()
        neighbor_scores = similarities[top_indices].tolist()
        neighbor_labels = self._index_labels[top_indices].tolist()

        emb_hash = hashlib.sha256(query_embedding.tobytes()).hexdigest()[:16]

        return RetrievalResult(
            query_id=query_id,
            query_embedding_hash=emb_hash,
            neighbor_ids=neighbor_ids,
            neighbor_scores=neighbor_scores,
            neighbor_labels=neighbor_labels,
            encoder_name=encoder_name,
            encoder_version=encoder_version,
            preprocessing_hash=preprocessing_hash,
            method="global",
        )

    def retrieve_batch(self, query_ids: list, query_embeddings: np.ndarray, k: int = 6,
                       encoder_name: str = "", encoder_version: str = "",
                       preprocessing_hash: str = "") -> list:
        results = []
        for i in range(len(query_ids)):
            result = self.retrieve(
                query_id=query_ids[i],
                query_embedding=query_embeddings[i],
                k=k,
                encoder_name=encoder_name,
                encoder_version=encoder_version,
                preprocessing_hash=preprocessing_hash,
            )
            results.append(result)
        return results
