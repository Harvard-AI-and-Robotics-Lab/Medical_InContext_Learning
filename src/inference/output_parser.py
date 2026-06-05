import json
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from prompting.dataset_contracts import get_classification_contract
except ImportError:
    from src.prompting.dataset_contracts import get_classification_contract


@dataclass
class ClassificationParsedOutput:
    query_id: str
    predicted_label: str = ""
    predicted_label_idx: int = -1
    subtype: str = ""
    confidence: float = 0.0
    probability: float = 0.0
    class_probabilities: list = field(default_factory=list)
    multi_label_predictions: list = field(default_factory=list)
    multi_label_confidences: list = field(default_factory=list)
    evidence: str = ""
    raw_response: str = ""
    parse_success: bool = False

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "predicted_label": self.predicted_label,
            "predicted_label_idx": self.predicted_label_idx,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "probability": self.probability,
            "class_probabilities": self.class_probabilities,
            "multi_label_predictions": self.multi_label_predictions,
            "multi_label_confidences": self.multi_label_confidences,
            "evidence": self.evidence,
            "parse_success": self.parse_success,
        }


@dataclass
class VQAParsedOutput:
    query_id: str
    answer: str = ""
    raw_response: str = ""
    parse_success: bool = False

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "parse_success": self.parse_success,
        }


class OutputParser:
    def __init__(self):
        pass

    def _normalize_label(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9_\s]', '', text)
        text = re.sub(r'\s+', '_', text)
        return text

    def _find_best_label_match(self, text: str, label_names: list) -> tuple:
        text_normalized = self._normalize_label(text)

        for idx, label in enumerate(label_names):
            if self._normalize_label(label) == text_normalized:
                return idx, label

        for idx, label in enumerate(label_names):
            if self._normalize_label(label) in text_normalized:
                return idx, label

        for idx, label in enumerate(label_names):
            if text_normalized in self._normalize_label(label):
                return idx, label

        return -1, ""

    def _extract_confidence(self, text: str) -> float:
        patterns = [
            r'[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
            r'[Pp]robability[:\s]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*(?:%|percent)',
            r'\b(0\.\d+|1\.0|1\.00?)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                if val > 1.0:
                    val = val / 100.0
                return min(max(val, 0.0), 1.0)
        return 0.5

    def _extract_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return ""

    def _clamp_probability(self, value, default: float = 0.5) -> float:
        try:
            prob = float(value)
            if prob > 1.0:
                prob = prob / 100.0
            return min(max(prob, 0.0), 1.0)
        except (ValueError, TypeError):
            return default


    def _extract_json_dict(self, raw_response: str) -> dict:
        text = raw_response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        json_text = self._extract_first_json_object(text)
        if not json_text:
            return {}
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _extract_string_field(self, raw_response: str, field_names: list[str]) -> str:
        data = self._extract_json_dict(raw_response)
        for field_name in field_names:
            value = data.get(field_name)
            if value is not None:
                return str(value).strip()
        for field_name in field_names:
            match = re.search(
                rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)"',
                raw_response,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        return ""

    def _attach_optional_fields(self, output: ClassificationParsedOutput, raw_response: str):
        output.subtype = self._extract_string_field(
            raw_response,
            ["subtype", "tumor_subtype", "predicted_subtype", "subtype_label"],
        )
        output.evidence = self._extract_string_field(raw_response, ["evidence", "rationale"])

    def _fallback_class_probabilities(
        self,
        label_idx: int,
        confidence: float,
        probability: float,
        n_classes: int,
        label_names: list | None = None,
        dataset_name: str = "",
    ) -> list:
        if n_classes <= 0:
            return []
        if n_classes == 2:
            prob = self._clamp_probability(probability, confidence)
            if label_names:
                contract = get_classification_contract(dataset_name)
                class_probs = contract.scalar_probability_to_class_probs(prob, label_names)
                if class_probs is not None:
                    return class_probs
            return [1.0 - prob, prob]
        if label_idx < 0:
            return [1.0 / n_classes] * n_classes
        confidence = self._clamp_probability(confidence, 0.5)
        remainder = max(0.0, 1.0 - confidence)
        other = remainder / max(n_classes - 1, 1)
        probs = [other] * n_classes
        probs[label_idx] = confidence
        return probs

    def _extract_class_probabilities(
        self,
        data: dict,
        label_names: list,
        label_idx: int,
        confidence: float,
        probability: float,
        dataset_name: str = "",
    ) -> list:
        n_classes = len(label_names)
        probabilities = data.get("probabilities", data.get("class_probabilities", None))
        if isinstance(probabilities, dict):
            normalized_probs = {
                self._normalize_label(str(key)): value for key, value in probabilities.items()
            }
            class_probs = []
            found_any = False
            for label in label_names:
                key = self._normalize_label(label)
                if key in normalized_probs:
                    class_probs.append(self._clamp_probability(normalized_probs[key], 0.0))
                    found_any = True
                else:
                    class_probs.append(0.0)
            total = sum(class_probs)
            if found_any and total > 0:
                return [float(p / total) for p in class_probs]
        if isinstance(probabilities, list) and len(probabilities) == n_classes:
            class_probs = [self._clamp_probability(p, 0.0) for p in probabilities]
            total = sum(class_probs)
            if total > 0:
                return [float(p / total) for p in class_probs]
        return self._fallback_class_probabilities(
            label_idx,
            confidence,
            probability,
            n_classes,
            label_names=label_names,
            dataset_name=dataset_name,
        )

    def _parse_json_response(self, raw_response: str, label_names: list, dataset_name: str = "") -> tuple:
        """Try to extract label, confidence, scalar probability, and class probabilities."""
        # Strip markdown code fences if present
        text = raw_response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        # Find first JSON object in the response
        json_text = self._extract_first_json_object(text)
        if not json_text:
            return self._parse_partial_json_response(text, label_names, dataset_name)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return self._parse_partial_json_response(text, label_names, dataset_name)
        label_val = data.get('label', '')
        confidence_val = data.get('confidence', 0.5)
        probability_val = (
            data.get('probability')
            if 'probability' in data
            else data.get('p_glaucoma', data.get('glaucoma_probability', data.get('probability_glaucoma', confidence_val)))
        )
        idx, matched_label = self._find_best_label_match(str(label_val), label_names)
        conf = self._clamp_probability(confidence_val, 0.5)
        prob = self._clamp_probability(probability_val, conf)
        class_probs = self._extract_class_probabilities(data, label_names, idx, conf, prob, dataset_name)
        if len(label_names) > 2 and 0 <= idx < len(class_probs):
            prob = class_probs[idx]
        return idx, matched_label, conf, prob, class_probs

    def _parse_partial_json_response(self, text: str, label_names: list, dataset_name: str = "") -> tuple:
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if not label_match:
            return -1, '', 0.5, 0.5, self._fallback_class_probabilities(
                -1, 0.5, 0.5, len(label_names), label_names=label_names, dataset_name=dataset_name
            )
        idx, matched_label = self._find_best_label_match(label_match.group(1), label_names)
        if idx < 0:
            return -1, '', 0.5, 0.5, self._fallback_class_probabilities(
                -1, 0.5, 0.5, len(label_names), label_names=label_names, dataset_name=dataset_name
            )
        confidence_match = re.search(
            r'"confidence"\s*:\s*"?([0-9]*\.?[0-9]+)"?',
            text,
            flags=re.IGNORECASE,
        )
        probability_match = re.search(
            r'"(?:probability|p_glaucoma|glaucoma_probability|probability_glaucoma)"\s*:\s*"?([0-9]*\.?[0-9]+)"?',
            text,
            flags=re.IGNORECASE,
        )
        conf = 0.5
        if confidence_match:
            conf = self._clamp_probability(confidence_match.group(1), 0.5)
        prob = conf
        if probability_match:
            prob = self._clamp_probability(probability_match.group(1), conf)
        class_probs = self._fallback_class_probabilities(
            idx, conf, prob, len(label_names), label_names=label_names, dataset_name=dataset_name
        )
        return idx, matched_label, conf, prob, class_probs

    def parse_classification(self, raw_response: str, query_id: str,
                              label_names: list, is_multi_label: bool = False,
                              dataset_name: str = "") -> ClassificationParsedOutput:
        output = ClassificationParsedOutput(query_id=query_id, raw_response=raw_response)

        if not raw_response or not raw_response.strip():
            return output

        if is_multi_label:
            return self._parse_multi_label(raw_response, query_id, label_names)

        # 1. Try JSON parsing first (gpt-4o returns JSON reliably)
        idx, matched_label, conf, prob, class_probs = self._parse_json_response(raw_response, label_names, dataset_name)
        if idx >= 0:
            output.predicted_label = matched_label
            output.predicted_label_idx = idx
            self._attach_optional_fields(output, raw_response)
            output.confidence = conf
            output.probability = prob
            output.class_probabilities = class_probs
            output.parse_success = True
            return output

        # 2. Try structured "Label: <value>" pattern
        label_match = re.search(r'[Ll]abel[:\s]+(.+?)(?:\n|$)', raw_response)
        if label_match:
            label_text = label_match.group(1).strip()
            idx, matched_label = self._find_best_label_match(label_text, label_names)
            if idx >= 0:
                output.predicted_label = matched_label
                output.predicted_label_idx = idx
                self._attach_optional_fields(output, raw_response)
                output.confidence = self._extract_confidence(raw_response)
                output.probability = output.confidence
                output.class_probabilities = self._fallback_class_probabilities(
                    idx,
                    output.confidence,
                    output.probability,
                    len(label_names),
                    label_names=label_names,
                    dataset_name=dataset_name,
                )
                output.parse_success = True
                return output

        # 3. Substring fallback — sort by length descending to avoid false partial matches
        #    e.g. avoid matching 'non glaucoma' inside 'no signs of glaucoma'
        ranked = sorted(enumerate(label_names), key=lambda x: len(x[1]), reverse=True)
        for idx, label in ranked:
            label_lower = label.lower().replace('_', ' ')
            # Require word-boundary match to avoid 'non glaucoma' matching inside 'glaucoma'
            if re.search(r'\b' + re.escape(label_lower) + r'\b', raw_response.lower()):
                output.predicted_label = label
                output.predicted_label_idx = idx
                self._attach_optional_fields(output, raw_response)
                output.confidence = self._extract_confidence(raw_response)
                output.probability = output.confidence
                output.class_probabilities = self._fallback_class_probabilities(
                    idx,
                    output.confidence,
                    output.probability,
                    len(label_names),
                    label_names=label_names,
                    dataset_name=dataset_name,
                )
                output.parse_success = True
                return output

        output.confidence = self._extract_confidence(raw_response)
        output.probability = output.confidence
        output.class_probabilities = self._fallback_class_probabilities(
            -1,
            output.confidence,
            output.probability,
            len(label_names),
            label_names=label_names,
            dataset_name=dataset_name,
        )
        return output

    def _parse_multi_label(self, raw_response: str, query_id: str,
                           label_names: list) -> ClassificationParsedOutput:
        output = ClassificationParsedOutput(
            query_id=query_id,
            raw_response=raw_response,
            multi_label_predictions=[0] * len(label_names),
            multi_label_confidences=[0.0] * len(label_names),
        )

        data = self._extract_json_dict(raw_response)
        findings = data.get("findings", data.get("labels", data.get("predictions", None)))
        if isinstance(findings, dict):
            normalized = {self._normalize_label(str(k)): v for k, v in findings.items()}
            matched_any = False
            for idx, label in enumerate(label_names):
                value = normalized.get(self._normalize_label(label))
                if value is None:
                    continue
                present = None
                probability = None
                if isinstance(value, dict):
                    present = value.get("present", value.get("positive", value.get("label", None)))
                    probability = value.get("probability", value.get("confidence", value.get("prob", None)))
                elif isinstance(value, bool):
                    present = value
                    probability = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    probability = self._clamp_probability(value, 0.0)
                    present = probability >= 0.5
                elif isinstance(value, str):
                    lower = value.strip().lower()
                    present = lower in {"present", "positive", "yes", "true", "1"}

                prob = self._clamp_probability(probability, 0.5)
                if isinstance(present, str):
                    present_norm = present.strip().lower()
                    pred = 1 if present_norm in {"present", "positive", "yes", "true", "1"} else 0
                elif present is None:
                    pred = 1 if prob >= 0.5 else 0
                else:
                    pred = 1 if bool(present) else 0
                output.multi_label_predictions[idx] = pred
                output.multi_label_confidences[idx] = prob
                matched_any = True
            if matched_any:
                output.evidence = self._extract_string_field(raw_response, ["evidence", "rationale"])
                output.parse_success = True
                return output

        lines = raw_response.strip().split("\n")
        matched_any = False
        for line in lines:
            line_lower = line.lower().strip()
            for idx, label in enumerate(label_names):
                label_variants = [label.lower(), label.lower().replace("_", " ")]
                if not any(variant in line_lower for variant in label_variants):
                    continue
                present_patterns = [r'\bpresent\b', r'\byes\b', r'\bpositive\b', r'\bfound\b', r'\bdetected\b', r'\b1\b']
                absent_patterns = [r'\babsent\b', r'\bno\b', r'\bnegative\b', r'\bnot\s+found\b', r'\bnot\s+detected\b', r'\b0\b']
                is_present = any(re.search(pat, line_lower) for pat in present_patterns)
                is_absent = any(re.search(pat, line_lower) for pat in absent_patterns)
                if is_present and not is_absent:
                    output.multi_label_predictions[idx] = 1
                elif is_absent:
                    output.multi_label_predictions[idx] = 0
                conf_match = re.search(r'([0-9]*\.?[0-9]+)', line.split(",")[-1] if "," in line else line)
                output.multi_label_confidences[idx] = self._clamp_probability(conf_match.group(1), 0.5) if conf_match else 0.5
                matched_any = True
                break

        output.parse_success = matched_any
        return output

    def parse_vqa(self, raw_response: str, query_id: str) -> VQAParsedOutput:
        output = VQAParsedOutput(query_id=query_id, raw_response=raw_response)

        if not raw_response or not raw_response.strip():
            return output

        raw = raw_response.strip()
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                answer = parsed.get("answer", "")
                if answer:
                    output.answer = str(answer).strip()
                    output.parse_success = True
                    return output
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        answer_match = re.search(r'[Aa]nswer[:\s]+(.+?)(?:\n\n|$)', raw, re.DOTALL)
        output.answer = answer_match.group(1).strip() if answer_match else raw

        output.parse_success = bool(output.answer)
        return output
