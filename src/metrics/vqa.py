import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VQAResult:
    bleu4: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0
    exact_match: float = 0.0
    n_samples: int = 0
    per_sample_bleu4: list = field(default_factory=list)
    per_sample_rouge_l: list = field(default_factory=list)
    per_sample_meteor: list = field(default_factory=list)

    def to_dict(self):
        return {
            "bleu4": self.bleu4,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
            "exact_match": self.exact_match,
            "n_samples": self.n_samples,
        }


class VQAMetrics:
    def __init__(self):
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)

    def _compute_bleu4(self, reference: str, hypothesis: str) -> float:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0

        smoothie = SmoothingFunction().method1
        try:
            score = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
        except (ValueError, ZeroDivisionError):
            score = 0.0
        return float(score)

    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return float(scores['rougeL'].fmeasure)

    def _compute_meteor(self, reference: str, hypothesis: str) -> float:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize

        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0

        try:
            score = meteor_score([ref_tokens], hyp_tokens)
        except (ValueError, ZeroDivisionError):
            score = 0.0
        return float(score)

    def _exact_match(self, reference: str, hypothesis: str) -> float:
        return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0

    def compute(self, references: list, hypotheses: list) -> VQAResult:
        n = len(references)
        if n == 0:
            return VQAResult()

        bleu4_scores = []
        rouge_l_scores = []
        meteor_scores = []
        em_scores = []

        for ref, hyp in zip(references, hypotheses):
            if not hyp or not hyp.strip():
                bleu4_scores.append(0.0)
                rouge_l_scores.append(0.0)
                meteor_scores.append(0.0)
                em_scores.append(0.0)
                continue

            bleu4_scores.append(self._compute_bleu4(ref, hyp))
            rouge_l_scores.append(self._compute_rouge_l(ref, hyp))
            meteor_scores.append(self._compute_meteor(ref, hyp))
            em_scores.append(self._exact_match(ref, hyp))

        return VQAResult(
            bleu4=float(np.mean(bleu4_scores)),
            rouge_l=float(np.mean(rouge_l_scores)),
            meteor=float(np.mean(meteor_scores)),
            exact_match=float(np.mean(em_scores)),
            n_samples=n,
            per_sample_bleu4=bleu4_scores,
            per_sample_rouge_l=rouge_l_scores,
            per_sample_meteor=meteor_scores,
        )
