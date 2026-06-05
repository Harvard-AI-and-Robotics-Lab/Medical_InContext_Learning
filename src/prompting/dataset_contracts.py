from dataclasses import dataclass, field


def _method_group(method: str = "zero_shot") -> str:
    method = method or "zero_shot"
    if method in ("naive_icl", "fixed_random_6", "random_icl"):
        return "naive_icl"
    if method.startswith("rg_icl"):
        return "rg_icl"
    return "zero_shot"


@dataclass(frozen=True)
class ClassificationDatasetContract:
    name: str
    system_prompt: str
    method_instructions: dict = field(default_factory=dict)
    probability_mode: str = "auto"
    positive_label: str | None = None
    reference_label_name: str = "Diagnosis"
    task_name: str = "diagnostic classification"
    visual_evidence_name: str = "diagnostic visual evidence"

    def method_instruction(self, method: str = "zero_shot") -> str:
        method_key = _method_group(method)
        if method_key in self.method_instructions:
            return self.method_instructions[method_key]
        return DEFAULT_METHOD_INSTRUCTIONS[method_key]

    def reference_text(self, label_text: str) -> str:
        return f"Reference image - {self.reference_label_name}: {label_text}"

    def probability_schema(self, label_names: list) -> str:
        if self.is_binary_positive_probability(label_names):
            positive_label = self.binary_positive_label(label_names)
            return (
                f'{{"label": "<one of: {", ".join(label_names)}>", '
                '"confidence": <confidence in the chosen final label, float 0.0-1.0>, '
                f'"probability": <probability that the image is {positive_label}, float 0.0-1.0>, '
                '"evidence": "<one sentence>"}'
            )

        probability_schema = ", ".join([f'"{label}": 0.0' for label in label_names])
        return (
            f'{{"label": "<one of: {", ".join(label_names)}>", '
            '"confidence": <confidence in the chosen final label, float 0.0-1.0>, '
            f'"probabilities": {{{probability_schema}}}, '
            '"evidence": "<one sentence>"}'
        )

    def compact_probability_schema(self, label_names: list) -> str:
        if self.is_binary_positive_probability(label_names):
            positive_label = self.binary_positive_label(label_names)
            return (
                "Return only one compact JSON object with exactly these keys:\n"
                f'{{"label":"<one of: {", ".join(label_names)}>","confidence":0.0,'
                '"probability":0.0,"evidence":"eight words max"}\n'
                "The confidence field is confidence in the chosen final label. "
                f"The probability field MUST be the probability that the query image is {positive_label}."
            )

        probability_schema = ", ".join([f'"{label}": 0.0' for label in label_names])
        return (
            "Return only one compact JSON object with exactly these keys:\n"
            f'{{"label":"<one of: {", ".join(label_names)}>","confidence":0.0,'
            f'"probabilities":{{{probability_schema}}},"evidence":"eight words max"}}\n'
            "The confidence field is confidence in the chosen final label. "
            "The probabilities field MUST be a calibrated distribution over all listed classes and sum to 1.0."
        )

    def format_query_text(self, label_names: list, method: str = "zero_shot") -> str:
        instruction = self.method_instruction(method)
        labels_str = ", ".join(label_names)
        if self.is_binary_positive_probability(label_names):
            return (
                f"{instruction}\n\n"
                f"Classify this image into one of: {labels_str}\n\n"
                "You MUST respond using ONLY this exact JSON format, no other text:\n"
                f"{self.probability_schema(label_names)}"
            )

        return (
            f"{instruction}\n\n"
            f"Classify this image into exactly one of: {labels_str}\n\n"
            "You MUST respond using ONLY this exact JSON format, no other text:\n"
            f"{self.probability_schema(label_names)}\n"
            "The probabilities must be a calibrated distribution over all classes and sum to 1.0."
        )

    def global_similarity_query_text(
        self,
        label_names: list,
        n_references: int,
        retrieval_description: str,
        score_name: str,
    ) -> str:
        return (
            f"You are given the {n_references} most similar training-set reference images retrieved by "
            f"{retrieval_description}, each with its {self.reference_label_name.lower()} and {score_name}.\n"
            f"Use the retrieved references and their similarity scores as {self.task_name} context, "
            f"and combine that context with {self.visual_evidence_name} in the query image.\n\n"
            f"Classify the query image into one of: {', '.join(label_names)}.\n"
            f"{self.compact_probability_schema(label_names)}"
        )

    def is_binary_positive_probability(self, label_names: list) -> bool:
        if self.probability_mode == "binary_positive":
            return True
        if self.probability_mode == "multiclass_distribution":
            return False
        return len(label_names) == 2

    def binary_positive_label(self, label_names: list) -> str:
        return self.positive_label or label_names[-1]

    def scalar_probability_to_class_probs(
        self,
        probability: float,
        label_names: list,
    ) -> list | None:
        if not self.is_binary_positive_probability(label_names):
            return None
        positive_label = self.binary_positive_label(label_names)
        try:
            positive_idx = label_names.index(positive_label)
        except ValueError:
            positive_idx = len(label_names) - 1
        probs = [0.0] * len(label_names)
        if len(label_names) == 2:
            probs[positive_idx] = probability
            probs[1 - positive_idx] = 1.0 - probability
        return probs


DEFAULT_METHOD_INSTRUCTIONS = {
    "zero_shot": "Given the medical image, make the requested diagnostic classification.",
    "naive_icl": (
        "You will be shown several example cases with images and labels.\n"
        "Learn the visual patterns associated with each label.\n"
        "Then classify the query image accordingly."
    ),
    "rg_icl": (
        "You will be given:\n"
        "1. The query image\n"
        "2. A set of retrieved reference images most similar in anatomical structure and pathology.\n"
        "Use the reference images to guide your classification of the query image."
    ),
}


DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a medical imaging assistant specialized in medical image classification.\n"
    "You must make a diagnostic decision from the provided medical image.\n"
    "You must reason only from the image content provided and not assume patient history.\n"
    "If visual evidence is insufficient, you must output low confidence rather than guess.\n"
    "You must output calibrated probabilities."
)


CLASSIFICATION_CONTRACTS = {
    "lag": ClassificationDatasetContract(
        name="lag",
        system_prompt=(
            "You are a glaucoma specialist. Based on the provided fundus photograph, determine whether "
            "there is evidence of glaucoma or suspected glaucoma, or whether the eye appears non-glaucomatous.\n"
        ),
        method_instructions={
            "zero_shot": (
                "Given the image, determine whether it represents a case of glaucoma.\n"
                "Return:\n"
                "- Binary label (0 = non-glaucoma, 1 = glaucoma)\n"
                "- Probability that it represents a case of glaucoma\n"
                "- One sentence describing the strongest visual evidence"
            ),
            "naive_icl": (
                "You will be shown several example cases with images and labels.\n"
                "Learn the visual patterns associated with glaucoma and non-glaucoma conditions.\n"
                "Then classify the query image accordingly."
            ),
            "rg_icl": DEFAULT_METHOD_INSTRUCTIONS["rg_icl"],
        },
        probability_mode="binary_positive",
        positive_label="glaucoma",
        reference_label_name="Diagnosis",
        task_name="glaucoma classification",
        visual_evidence_name="diagnostic fundus visual evidence",
    ),
    "breakhis": ClassificationDatasetContract(
        name="breakhis",
        system_prompt=(
            "You are a histopathology classification assistant.\n"
            "You must classify microscopic breast tissue images into tumor subtypes.\n"
            "Use cellular morphology, gland formation, nuclear atypia, and tissue architecture."
        ),
        method_instructions={
            "zero_shot": "Given the histopathology image, classify the breast tumor subtype.",
            "naive_icl": (
                "You will be shown several histopathology reference cases with images and subtype labels.\n"
                "Compare cellular morphology, gland formation, nuclear atypia, and tissue architecture.\n"
                "Then classify the query image into one tumor subtype."
            ),
            "rg_icl": (
                "You will be given the query histopathology image and retrieved training-set reference images.\n"
                "Use the retrieved references as visual context, and combine that context with "
                "diagnostic histopathology evidence in the query image."
            ),
        },
        probability_mode="multiclass_distribution",
        reference_label_name="Tumor subtype",
        task_name="breast histopathology tumor-subtype classification",
        visual_evidence_name="histopathology tumor-subtype visual evidence",
    ),
    "breakhis_binary": ClassificationDatasetContract(
        name="breakhis_binary",
        system_prompt=(
            "You are a breast histopathology classification assistant.\n"
            "Classify microscopic breast tissue images as benign or malignant.\n"
            "Use cellular morphology, nuclear atypia, gland formation, stromal pattern, and tissue architecture."
        ),
        method_instructions={
            "zero_shot": (
                "Given the histopathology image, classify the breast lesion as benign or malignant."
            ),
            "naive_icl": (
                "You will be shown histopathology reference cases with benign or malignant labels.\n"
                "Compare cellular morphology, nuclear atypia, gland formation, stromal pattern, and tissue architecture.\n"
                "Then classify the query image as benign or malignant."
            ),
            "rg_icl": (
                "You will be given the query histopathology image and retrieved training-set reference images.\n"
                "Use the retrieved references and their similarity scores as benign-versus-malignant classification context, "
                "and combine that context with histopathology evidence in the query image."
            ),
        },
        probability_mode="binary_positive",
        positive_label="malignant",
        reference_label_name="Diagnosis",
        task_name="breast histopathology benign-versus-malignant classification",
        visual_evidence_name="histopathology benign-versus-malignant visual evidence",
    ),

    "chexpert": ClassificationDatasetContract(
        name="chexpert",
        system_prompt=(
            "You are a clinical chest X-ray interpretation assistant.\n"
            "Detect each CheXpert finding independently from the frontal or lateral radiograph.\n"
            "Use radiographic evidence only; do not infer from demographics or prevalence."
        ),
        method_instructions={
            "zero_shot": (
                "Given the chest X-ray, independently assess every listed CheXpert finding. "
                "A finding can be present even when other findings are absent."
            ),
            "naive_icl": (
                "You will be shown chest X-ray reference cases with multi-label findings.\n"
                "Use them as examples of radiographic appearance, then independently assess every listed finding in the query image."
            ),
            "rg_icl": (
                "You will be given the query chest X-ray and retrieved training-set reference images with multi-label findings.\n"
                "Use the references and similarity scores as context, and combine that context with "
                "radiographic evidence in the query image for each listed finding."
            ),
        },
        reference_label_name="CheXpert findings",
        task_name="multi-label chest X-ray finding detection",
        visual_evidence_name="chest radiograph visual evidence",
    ),
    "ddr": ClassificationDatasetContract(
        name="ddr",
        system_prompt=(
            "You are an ophthalmic image grading assistant.\n"
            "You must assign one diabetic retinopathy severity grade.\n"
            "Use microaneurysms, hemorrhages, exudates, neovascularization, and vessel abnormalities."
        ),
        method_instructions={
            "zero_shot": (
                "Given the fundus photograph, assign the diabetic retinopathy severity grade.\n"
                "Use retinal lesions such as microaneurysms, hemorrhages, hard exudates, cotton-wool spots, "
                "venous beading, intraretinal microvascular abnormalities, neovascularization, and image "
                "gradability."
            ),
            "naive_icl": (
                "You will be shown fundus reference cases with diabetic retinopathy grade labels.\n"
                "Compare lesion type, lesion burden, vascular abnormalities, proliferative findings, and "
                "image gradability. Then assign one severity grade to the query image."
            ),
            "rg_icl": (
                "You will be given the query fundus photograph and retrieved training-set reference images.\n"
                "Use the retrieved references and their similarity scores as grading context, and combine "
                "that context with diabetic-retinopathy visual evidence in the query image."
            ),
        },
        probability_mode="multiclass_distribution",
        reference_label_name="DR grade",
        task_name="diabetic retinopathy grading",
        visual_evidence_name="ophthalmic visual evidence",
    ),
    "tbx11k": ClassificationDatasetContract(
        name="tbx11k",
        system_prompt=(
            "You are a chest X-ray classification assistant for tuberculosis screening.\n"
            "You must classify the frontal chest radiograph into exactly one of three categories: "
            "healthy, sick but non-TB, or TB.\n"
            "Use radiographic evidence such as lung opacities, cavitation, consolidation, pleural changes, "
            "fibrotic scarring, and overall cardiopulmonary abnormality. Do not assume clinical history."
        ),
        method_instructions={
            "zero_shot": (
                "Given the chest X-ray, classify it as healthy, sick but non-TB, or TB based only on image evidence."
            ),
            "naive_icl": (
                "You will be shown chest X-ray reference cases with known labels.\n"
                "Use these references as classification context for the three label categories, and combine "
                "that context with radiographic evidence in the query image.\n"
                "Compare lung fields, focal opacities, cavitary or fibrotic changes, pleural abnormalities, "
                "and other radiographic disease patterns.\n"
                "Then classify the query image into exactly one category."
            ),
            "rg_icl": (
                "You will be given the query chest X-ray and retrieved training-set reference chest X-rays.\n"
                "Use the retrieved references and their similarity scores as classification context, and combine "
                "that context with radiographic evidence in the query image."
            ),
        },
        probability_mode="multiclass_distribution",
        reference_label_name="CXR diagnosis",
        task_name="TBX11K chest X-ray three-class classification",
        visual_evidence_name="chest X-ray visual evidence",
    ),
}


def get_classification_contract(dataset_name: str = "") -> ClassificationDatasetContract:
    dataset_key = (dataset_name or "").lower()
    if dataset_key in CLASSIFICATION_CONTRACTS:
        return CLASSIFICATION_CONTRACTS[dataset_key]
    return ClassificationDatasetContract(
        name=dataset_key or "default",
        system_prompt=DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    )
