import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class DeLongResult:
    auc_a: float = 0.0
    auc_b: float = 0.0
    auc_diff: float = 0.0
    z_statistic: float = 0.0
    p_value: float = 1.0
    se_a: float = 0.0
    se_b: float = 0.0
    se_diff: float = 0.0
    method_a_name: str = ""
    method_b_name: str = ""
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self):
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "auc_diff": self.auc_diff,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "se_a": self.se_a,
            "se_b": self.se_b,
            "se_diff": self.se_diff,
            "method_a_name": self.method_a_name,
            "method_b_name": self.method_b_name,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


class DeLongTest:
    def __init__(self):
        pass

    def _compute_midrank(self, x):
        n = len(x)
        order = np.argsort(x)
        ranks = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and x[order[j]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j + 1)
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    def _fast_delong(self, y_true, scores):
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        m = len(pos_idx)
        n = len(neg_idx)

        if m == 0 or n == 0:
            return 0.5, np.zeros(m), np.zeros(n)

        all_scores = np.concatenate([scores[pos_idx], scores[neg_idx]])
        ranks = self._compute_midrank(all_scores)

        pos_ranks = ranks[:m]
        auc = (np.sum(pos_ranks) - m * (m + 1) / 2.0) / (m * n)

        v_pos = np.zeros(m)
        for i in range(m):
            v_pos[i] = np.mean((scores[pos_idx[i]] > scores[neg_idx]).astype(float) +
                               0.5 * (scores[pos_idx[i]] == scores[neg_idx]).astype(float))

        v_neg = np.zeros(n)
        for j in range(n):
            v_neg[j] = np.mean((scores[pos_idx] > scores[neg_idx[j]]).astype(float) +
                               0.5 * (scores[pos_idx] == scores[neg_idx[j]]).astype(float))

        return auc, v_pos, v_neg

    def test(self, y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
             method_a_name: str = "A", method_b_name: str = "B") -> DeLongResult:
        y_true = np.asarray(y_true, dtype=int)
        scores_a = np.asarray(scores_a, dtype=float)
        scores_b = np.asarray(scores_b, dtype=float)

        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        m = len(pos_idx)
        n = len(neg_idx)

        if m < 2 or n < 2:
            return DeLongResult(
                method_a_name=method_a_name,
                method_b_name=method_b_name,
                n_positive=m,
                n_negative=n,
            )

        auc_a, v_pos_a, v_neg_a = self._fast_delong(y_true, scores_a)
        auc_b, v_pos_b, v_neg_b = self._fast_delong(y_true, scores_b)

        var_a = np.var(v_pos_a, ddof=1) / m + np.var(v_neg_a, ddof=1) / n
        var_b = np.var(v_pos_b, ddof=1) / m + np.var(v_neg_b, ddof=1) / n

        cov_pos = np.cov(v_pos_a, v_pos_b, ddof=1)[0, 1] if m > 1 else 0.0
        cov_neg = np.cov(v_neg_a, v_neg_b, ddof=1)[0, 1] if n > 1 else 0.0
        covar = cov_pos / m + cov_neg / n

        var_diff = var_a + var_b - 2.0 * covar
        var_diff = max(var_diff, 1e-15)

        se_a = float(np.sqrt(max(var_a, 0)))
        se_b = float(np.sqrt(max(var_b, 0)))
        se_diff = float(np.sqrt(var_diff))

        z = (auc_a - auc_b) / se_diff
        p_value = 2.0 * stats.norm.sf(abs(z))

        return DeLongResult(
            auc_a=float(auc_a),
            auc_b=float(auc_b),
            auc_diff=float(auc_a - auc_b),
            z_statistic=float(z),
            p_value=float(p_value),
            se_a=se_a,
            se_b=se_b,
            se_diff=se_diff,
            method_a_name=method_a_name,
            method_b_name=method_b_name,
            n_positive=m,
            n_negative=n,
        )

    def test_multilabel(self, y_true: np.ndarray, scores_a: np.ndarray,
                        scores_b: np.ndarray, n_labels: int,
                        method_a_name: str = "A",
                        method_b_name: str = "B") -> list:
        results = []
        for c in range(n_labels):
            result = self.test(
                y_true[:, c], scores_a[:, c], scores_b[:, c],
                method_a_name=method_a_name,
                method_b_name=method_b_name,
            )
            results.append(result)
        return results
