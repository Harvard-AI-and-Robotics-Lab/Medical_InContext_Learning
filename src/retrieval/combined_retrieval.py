import numpy as np
import hashlib
from .global_retrieval import RetrievalResult


class CombinedRetriever:
    def __init__(self, alpha: float = 0.5, similarity_metric: str = "cosine",
                 exclude_query: bool = True, exclude_test_set: bool = True):
        self.alpha = alpha
        self.similarity_metric = similarity_metric
        self.exclude_query = exclude_query
        self.exclude_test_set = exclude_test_set
        self._index_global = None
        self._index_spatial = None
        self._index_ids = None
        self._index_labels = None
        self._index_splits = None

    def build_index(self, ids: list, global_embeddings: np.ndarray, spatial_features: list,
                    labels: list, splits: list):
        self._index_ids = np.array(ids)
        self._index_labels = np.array(labels)
        self._index_splits = np.array(splits)

        self._index_global = global_embeddings.copy()
        norms = np.linalg.norm(self._index_global, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._index_global = self._index_global / norms

        self._index_spatial = []
        for feat in spatial_features:
            feat_norms = np.linalg.norm(feat, axis=1, keepdims=True)
            feat_norms = np.maximum(feat_norms, 1e-8)
            self._index_spatial.append(feat / feat_norms)

    def _compute_spatial_similarity(self, query_spatial: np.ndarray, ref_spatial: np.ndarray) -> float:
        q_norms = np.linalg.norm(query_spatial, axis=1, keepdims=True)
        q_norms = np.maximum(q_norms, 1e-8)
        query_normalized = query_spatial / q_norms

        sim_matrix = query_normalized @ ref_spatial.T
        max_per_query_patch = sim_matrix.max(axis=1)
        return float(max_per_query_patch.mean())

    def retrieve(self, query_id: str, query_global: np.ndarray, query_spatial: np.ndarray,
                 k: int = 6, encoder_name: str = "", encoder_version: str = "",
                 preprocessing_hash: str = "") -> RetrievalResult:
        query_g_norm = np.linalg.norm(query_global)
        if query_g_norm > 0:
            query_g_normalized = query_global / query_g_norm
        else:
            query_g_normalized = query_global

        global_sims = self._index_global @ query_g_normalized

        mask = np.ones(len(self._index_ids), dtype=bool)
        if self.exclude_query:
            mask &= self._index_ids != query_id
        if self.exclude_test_set:
            mask &= self._index_splits != "test"

        spatial_sims = np.full(len(self._index_ids), -np.inf)
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            spatial_sims[idx] = self._compute_spatial_similarity(query_spatial, self._index_spatial[idx])

        combined_scores = np.full(len(self._index_ids), -np.inf)
        combined_scores[valid_indices] = (
            self.alpha * global_sims[valid_indices] +
            (1 - self.alpha) * spatial_sims[valid_indices]
        )

        top_indices = np.argsort(combined_scores)[::-1][:k]

        neighbor_ids = self._index_ids[top_indices].tolist()
        neighbor_scores = combined_scores[top_indices].tolist()
        neighbor_labels = self._index_labels[top_indices].tolist()

        emb_bytes = query_global.tobytes() + query_spatial.tobytes()
        emb_hash = hashlib.sha256(emb_bytes).hexdigest()[:16]

        return RetrievalResult(
            query_id=query_id,
            query_embedding_hash=emb_hash,
            neighbor_ids=neighbor_ids,
            neighbor_scores=neighbor_scores,
            neighbor_labels=neighbor_labels,
            encoder_name=encoder_name,
            encoder_version=encoder_version,
            preprocessing_hash=preprocessing_hash,
            method="global_spatial",
        )

    def retrieve_batch(self, query_ids: list, query_globals: np.ndarray,
                       query_spatials: list, k: int = 6,
                       encoder_name: str = "", encoder_version: str = "",
                       preprocessing_hash: str = "") -> list:
        results = []
        for i in range(len(query_ids)):
            result = self.retrieve(
                query_id=query_ids[i],
                query_global=query_globals[i],
                query_spatial=query_spatials[i],
                k=k,
                encoder_name=encoder_name,
                encoder_version=encoder_version,
                preprocessing_hash=preprocessing_hash,
            )
            results.append(result)
        return results
