import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationResult:
    auc: float = 0.0
    auc_per_label: list = field(default_factory=list)
    sens_at_spec90: float = 0.0
    sens_at_spec90_per_label: list = field(default_factory=list)
    spec_at_sens90: float = 0.0
    spec_at_sens90_per_label: list = field(default_factory=list)
    accuracy: float = 0.0
    brier: float = 0.0
    ece: float = 0.0
    nll: float = 0.0
    confusion: Optional[np.ndarray] = None
    n_samples: int = 0

    def to_dict(self):
        result = {
            "auc": self.auc,
            "auc_per_label": self.auc_per_label,
            "sens_at_spec90": self.sens_at_spec90,
            "sens_at_spec90_per_label": self.sens_at_spec90_per_label,
            "spec_at_sens90": self.spec_at_sens90,
            "spec_at_sens90_per_label": self.spec_at_sens90_per_label,
            "accuracy": self.accuracy,
            "brier": self.brier,
            "ece": self.ece,
            "nll": self.nll,
            "n_samples": self.n_samples,
        }
        if self.confusion is not None:
            result["confusion"] = self.confusion.tolist()
        return result


class ClassificationMetrics:
    def __init__(self, n_ece_bins: int = 15):
        self.n_ece_bins = n_ece_bins

    def _sens_at_spec(self, y_true, y_score, target_spec=0.90):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        spec = 1.0 - fpr
        valid = spec >= target_spec
        if not np.any(valid):
            return 0.0
        return float(tpr[valid].max())

    def _spec_at_sens(self, y_true, y_score, target_sens=0.90):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        spec = 1.0 - fpr
        valid = tpr >= target_sens
        if not np.any(valid):
            return 0.0
        return float(spec[valid].max())

    def _brier_score(self, y_true, y_prob):
        return float(np.mean((y_prob - y_true) ** 2))

    def _ece(self, y_true, y_prob, n_bins=None):
        if n_bins is None:
            n_bins = self.n_ece_bins
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if i == 0:
                mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
        return float(ece)

    def _nll(self, y_true, y_prob):
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    def _has_pos_and_neg(self, y_true_binary) -> bool:
        values = np.unique(np.asarray(y_true_binary))
        return len(values) == 2

    def _nanmean_or_zero(self, values) -> float:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return 0.0
        return float(np.mean(values[finite]))

    def compute_binary(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: np.ndarray) -> ClassificationResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        n = len(y_true)
        if n == 0:
            return ClassificationResult()

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.0

        sens90 = self._sens_at_spec(y_true, y_prob, 0.90)
        spec90 = self._spec_at_sens(y_true, y_prob, 0.90)
        acc = float(accuracy_score(y_true, y_pred))
        brier = self._brier_score(y_true, y_prob)
        ece = self._ece(y_true, y_prob)
        nll = self._nll(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)

        return ClassificationResult(
            auc=auc,
            sens_at_spec90=sens90,
            spec_at_sens90=spec90,
            accuracy=acc,
            brier=brier,
            ece=ece,
            nll=nll,
            confusion=cm,
            n_samples=n,
        )

    def compute_multiclass(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray, n_classes: int) -> ClassificationResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        n = len(y_true)
        if n == 0:
            return ClassificationResult()

        auc_per_label = []
        sens90_per_label = []
        spec90_per_label = []
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            if y_prob.ndim == 2:
                y_score_c = y_prob[:, c]
            else:
                y_score_c = (y_pred == c).astype(float)
            if self._has_pos_and_neg(y_bin):
                auc_per_label.append(float(roc_auc_score(y_bin, y_score_c)))
                sens90_per_label.append(self._sens_at_spec(y_bin, y_score_c, 0.90))
                spec90_per_label.append(self._spec_at_sens(y_bin, y_score_c, 0.90))
            else:
                auc_per_label.append(float("nan"))
                sens90_per_label.append(float("nan"))
                spec90_per_label.append(float("nan"))

        auc = self._nanmean_or_zero(auc_per_label)

        acc = float(accuracy_score(y_true, y_pred))

        if y_prob.ndim == 2:
            y_true_onehot = np.zeros_like(y_prob)
            for i in range(n):
                y_true_onehot[i, y_true[i]] = 1.0
            brier = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))
            ece_vals = []
            nll_vals = []
            for c in range(n_classes):
                ece_vals.append(self._ece(y_true_onehot[:, c], y_prob[:, c]))
                nll_vals.append(self._nll(y_true_onehot[:, c], y_prob[:, c]))
            ece = float(np.mean(ece_vals))
            nll = float(np.mean(nll_vals))
        else:
            brier = 0.0
            ece = 0.0
            nll = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        return ClassificationResult(
            auc=auc,
            auc_per_label=auc_per_label,
            sens_at_spec90=self._nanmean_or_zero(sens90_per_label),
            sens_at_spec90_per_label=sens90_per_label,
            spec_at_sens90=self._nanmean_or_zero(spec90_per_label),
            spec_at_sens90_per_label=spec90_per_label,
            accuracy=acc,
            brier=brier,
            ece=ece,
            nll=nll,
            confusion=cm,
            n_samples=n,
        )

    def compute_multilabel(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray, n_labels: int) -> ClassificationResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        n = len(y_true)
        if n == 0:
            return ClassificationResult()

        auc_per_label = []
        sens90_per_label = []
        spec90_per_label = []
        brier_per_label = []
        ece_per_label = []
        nll_per_label = []

        for c in range(n_labels):
            y_c = y_true[:, c]
            p_c = y_prob[:, c]
            d_c = y_pred[:, c]

            if len(np.unique(y_c)) < 2:
                auc_per_label.append(0.0)
                sens90_per_label.append(0.0)
                spec90_per_label.append(0.0)
            else:
                try:
                    auc_per_label.append(float(roc_auc_score(y_c, p_c)))
                except ValueError:
                    auc_per_label.append(0.0)
                sens90_per_label.append(self._sens_at_spec(y_c, p_c, 0.90))
                spec90_per_label.append(self._spec_at_sens(y_c, p_c, 0.90))

            brier_per_label.append(self._brier_score(y_c, p_c))
            ece_per_label.append(self._ece(y_c, p_c))
            nll_per_label.append(self._nll(y_c, p_c))

        valid_aucs = [a for a in auc_per_label if a > 0]
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        acc_per_label = []
        for c in range(n_labels):
            acc_per_label.append(float(accuracy_score(y_true[:, c], y_pred[:, c])))
        acc = float(np.mean(acc_per_label))

        return ClassificationResult(
            auc=macro_auc,
            auc_per_label=auc_per_label,
            sens_at_spec90=float(np.mean(sens90_per_label)),
            sens_at_spec90_per_label=sens90_per_label,
            spec_at_sens90=float(np.mean(spec90_per_label)),
            spec_at_sens90_per_label=spec90_per_label,
            accuracy=acc,
            brier=float(np.mean(brier_per_label)),
            ece=float(np.mean(ece_per_label)),
            nll=float(np.mean(nll_per_label)),
            n_samples=n,
        )
