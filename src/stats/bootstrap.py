import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from sklearn.metrics import roc_auc_score


@dataclass
class BootstrapResult:
    metric_name: str = ""
    method_a_name: str = ""
    method_b_name: str = ""
    observed_a: float = 0.0
    observed_b: float = 0.0
    observed_diff: float = 0.0
    ci_lower_a: float = 0.0
    ci_upper_a: float = 0.0
    ci_lower_b: float = 0.0
    ci_upper_b: float = 0.0
    ci_lower_diff: float = 0.0
    ci_upper_diff: float = 0.0
    p_value: float = 1.0
    n_resamples: int = 0
    seed: int = 0
    resample_indices: Optional[list] = None

    def to_dict(self):
        return {
            "metric_name": self.metric_name,
            "method_a_name": self.method_a_name,
            "method_b_name": self.method_b_name,
            "observed_a": self.observed_a,
            "observed_b": self.observed_b,
            "observed_diff": self.observed_diff,
            "ci_lower_a": self.ci_lower_a,
            "ci_upper_a": self.ci_upper_a,
            "ci_lower_b": self.ci_lower_b,
            "ci_upper_b": self.ci_upper_b,
            "ci_lower_diff": self.ci_lower_diff,
            "ci_upper_diff": self.ci_upper_diff,
            "p_value": self.p_value,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
        }


class PairedBootstrap:
    def __init__(self, n_resamples: int = 2000, seed: int = 42, ci_level: float = 0.95,
                 store_indices: bool = True):
        self.n_resamples = n_resamples
        self.seed = seed
        self.ci_level = ci_level
        self.store_indices = store_indices

    def _percentile_ci(self, samples: np.ndarray) -> tuple:
        alpha = (1.0 - self.ci_level) / 2.0
        lower = float(np.percentile(samples, 100 * alpha))
        upper = float(np.percentile(samples, 100 * (1.0 - alpha)))
        return lower, upper

    def _two_sided_percentile_p(self, boot_diffs: np.ndarray) -> float:
        n_pos = np.sum(boot_diffs > 0)
        n_neg = np.sum(boot_diffs < 0)
        n_total = len(boot_diffs)
        if n_total == 0:
            return 1.0
        smaller_tail = min(n_pos, n_neg)
        p = 2.0 * smaller_tail / n_total
        return min(p, 1.0)

    def test(self, y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
             metric_fn: Callable, metric_name: str = "",
             method_a_name: str = "A", method_b_name: str = "B") -> BootstrapResult:
        y_true = np.asarray(y_true)
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        n = len(y_true)

        observed_a = metric_fn(y_true, scores_a)
        observed_b = metric_fn(y_true, scores_b)
        observed_diff = observed_b - observed_a

        rng = np.random.RandomState(self.seed)
        boot_a = np.zeros(self.n_resamples)
        boot_b = np.zeros(self.n_resamples)
        boot_diff = np.zeros(self.n_resamples)
        all_indices = []

        for i in range(self.n_resamples):
            idx = rng.choice(n, size=n, replace=True)
            if self.store_indices:
                all_indices.append(idx.tolist())
            try:
                boot_a[i] = metric_fn(y_true[idx], scores_a[idx])
                boot_b[i] = metric_fn(y_true[idx], scores_b[idx])
            except (ValueError, ZeroDivisionError):
                boot_a[i] = np.nan
                boot_b[i] = np.nan
            boot_diff[i] = boot_b[i] - boot_a[i]

        valid = ~np.isnan(boot_diff)
        boot_a_valid = boot_a[valid]
        boot_b_valid = boot_b[valid]
        boot_diff_valid = boot_diff[valid]

        ci_a = self._percentile_ci(boot_a_valid)
        ci_b = self._percentile_ci(boot_b_valid)
        ci_diff = self._percentile_ci(boot_diff_valid)
        p_value = self._two_sided_percentile_p(boot_diff_valid)

        return BootstrapResult(
            metric_name=metric_name,
            method_a_name=method_a_name,
            method_b_name=method_b_name,
            observed_a=float(observed_a),
            observed_b=float(observed_b),
            observed_diff=float(observed_diff),
            ci_lower_a=ci_a[0],
            ci_upper_a=ci_a[1],
            ci_lower_b=ci_b[0],
            ci_upper_b=ci_b[1],
            ci_lower_diff=ci_diff[0],
            ci_upper_diff=ci_diff[1],
            p_value=p_value,
            n_resamples=len(boot_diff_valid),
            seed=self.seed,
            resample_indices=all_indices if self.store_indices else None,
        )

    def test_auc(self, y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                 method_a_name: str = "A", method_b_name: str = "B") -> BootstrapResult:
        def auc_fn(y, s):
            if len(np.unique(y)) < 2:
                return 0.5
            return roc_auc_score(y, s)
        return self.test(y_true, scores_a, scores_b, auc_fn, "auc", method_a_name, method_b_name)

    def test_accuracy(self, y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                      method_a_name: str = "A", method_b_name: str = "B") -> BootstrapResult:
        def acc_fn(y, p):
            return float(np.mean(y == p))
        return self.test(y_true, preds_a, preds_b, acc_fn, "accuracy", method_a_name, method_b_name)
