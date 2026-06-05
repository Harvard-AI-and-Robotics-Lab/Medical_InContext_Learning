import numpy as np
from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    ece: float = 0.0
    mce: float = 0.0
    brier: float = 0.0
    nll: float = 0.0
    bin_accuracies: list = field(default_factory=list)
    bin_confidences: list = field(default_factory=list)
    bin_counts: list = field(default_factory=list)
    correct_confidences: list = field(default_factory=list)
    incorrect_confidences: list = field(default_factory=list)

    def to_dict(self):
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "nll": self.nll,
            "bin_accuracies": self.bin_accuracies,
            "bin_confidences": self.bin_confidences,
            "bin_counts": self.bin_counts,
        }


class CalibrationMetrics:
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray,
                y_pred: np.ndarray = None) -> CalibrationResult:
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        if y_pred is None:
            y_pred = (y_prob >= 0.5).astype(int) if y_prob.ndim == 1 else np.argmax(y_prob, axis=1)

        y_pred = np.asarray(y_pred)
        n = len(y_true)
        if n == 0:
            return CalibrationResult()

        if y_prob.ndim == 1:
            confidences = y_prob.copy()
            correctness = (y_pred == y_true).astype(float)
        else:
            confidences = np.max(y_prob, axis=1)
            correctness = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []
        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            if i == 0:
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            else:
                mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            count = int(np.sum(mask))
            bin_counts.append(count)

            if count == 0:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                continue

            acc = float(np.mean(correctness[mask]))
            conf = float(np.mean(confidences[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)

            gap = abs(acc - conf)
            ece += (count / n) * gap
            mce = max(mce, gap)

        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

        if y_prob.ndim == 1:
            brier = float(np.mean((y_prob - y_true) ** 2))
            nll = float(-np.mean(
                y_true * np.log(y_prob_clipped) +
                (1 - y_true) * np.log(1 - y_prob_clipped)
            ))
        else:
            n_classes = y_prob.shape[1]
            y_true_int = y_true.astype(int)
            y_onehot = np.zeros_like(y_prob)
            for i in range(n):
                y_onehot[i, y_true_int[i]] = 1.0
            brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))
            nll_vals = []
            for i in range(n):
                nll_vals.append(-np.log(y_prob_clipped[i, y_true_int[i]]))
            nll = float(np.mean(nll_vals))

        correct_mask = correctness == 1.0
        correct_confs = confidences[correct_mask].tolist() if np.any(correct_mask) else []
        incorrect_confs = confidences[~correct_mask].tolist() if np.any(~correct_mask) else []

        return CalibrationResult(
            ece=ece,
            mce=mce,
            brier=brier,
            nll=nll,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            correct_confidences=correct_confs,
            incorrect_confidences=incorrect_confs,
        )

    def compute_multilabel(self, y_true: np.ndarray, y_prob: np.ndarray,
                           n_labels: int) -> list:
        results = []
        for c in range(n_labels):
            y_c = y_true[:, c]
            p_c = y_prob[:, c]
            d_c = (p_c >= 0.5).astype(int)
            results.append(self.compute(y_c, p_c, d_c))
        return results
