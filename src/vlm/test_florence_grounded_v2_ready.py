from pathlib import Path
import json
import re
import time

import csv
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_FILENAME = "colonca1207.jpeg"
IMAGE_PATH = (
    PROJECT_ROOT / "data" / "lung_colon_image_set"
    / "colon_image_sets" / "colon_aca" / IMAGE_FILENAME
)

GRADCAM_PATH = (
    PROJECT_ROOT / "results" / "explainability"
    / "residual_multiscale_cnn"
    / "correct_colon_aca_index_0_true_colon_aca_pred_colon_aca.png"
)

QUANTITATIVE_CSV = (
    PROJECT_ROOT / "results" / "explainability"
    / "residual_multiscale_cnn"
    / "quantitative_gradcam_metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "vlm" / "florence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT = OUTPUT_DIR / "colonca1207_florence_grounded_report.txt"
OUTPUT_JSON = OUTPUT_DIR / "colonca1207_florence_grounded_report.json"

MODEL_NAME = "microsoft/Florence-2-base-ft"
DEVICE = torch.device("cpu")

# Ground truth/project result for this representative case.
# The classifier is the source of the prediction; Florence must not change it.
CLASSIFIER_NAME = "Residual Multi-Scale CNN"
PREDICTED_CLASS = "colon_aca"
PREDICTED_CLASS_NAME = "Colon adenocarcinoma"
CONFIDENCE = 0.9984
TRUE_CLASS = "colon_aca"


def find_column(fieldnames, candidates):
    exact = {str(c).lower().strip(): c for c in fieldnames}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in exact:
            return exact[key]

    for c in fieldnames:
        c_low = str(c).lower()
        for candidate in candidates:
            if candidate.lower() in c_low:
                return c
    return None


def get_case_row(csv_path):
    # Quantitative CSV stores representative cases using category + index.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        category_col = find_column(fieldnames, ["category"])
        index_col = find_column(fieldnames, ["index"])

        if category_col is None or index_col is None:
            raise ValueError(
                "Expected category/index columns in quantitative Grad-CAM CSV. "
                f"Found: {fieldnames}"
            )

        for row in reader:
            if (
                str(row.get(category_col, "")).strip().lower()
                == "correct_colon_aca"
                and str(row.get(index_col, "")).strip() == "0"
            ):
                return row

    raise ValueError(
        "Could not find representative correct_colon_aca index 0 "
        "in quantitative Grad-CAM CSV."
    )


def metric(row, candidates):
    if not row:
        return None

    col = find_column(row.keys(), candidates)
    if col is None:
        return None

    try:
        return float(row[col])
    except (TypeError, ValueError):
        return None


def fmt(value, percent=False):
    if value is None:
        return "not available"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def clean_generated(text):
    text = text.strip()
    text = re.sub(r"^<MORE_DETAILED_CAPTION>\s*", "", text, flags=re.I)
    text = re.sub(r"^assistant\s*:?\s*", "", text, flags=re.I)
    return text.strip()


def generate_caption(processor, model, image, prompt):
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: (v.to(DEVICE) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
        )

    # Florence may return prompt tokens + generated tokens.
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        input_length = input_ids.shape[1]
        new_tokens = generated_ids[:, input_length:]
    else:
        new_tokens = generated_ids

    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0]

    return clean_generated(text)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("PROJECT-GROUNDED FLORENCE-2 CRC REPORT")
print("=" * 75)

required = {
    "Original image": IMAGE_PATH,
    "Grad-CAM image": GRADCAM_PATH,
    "Quantitative Grad-CAM CSV": QUANTITATIVE_CSV,
}

for name, path in required.items():
    print(f"{name}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

# ============================================================
# LOAD PROJECT EVIDENCE
# ============================================================

row = get_case_row(QUANTITATIVE_CSV)

mean_cam = metric(row, ["mean_activation", "mean_cam", "mean"])
median_cam = metric(row, ["median_activation", "median_cam", "median"])
std_cam = metric(row, ["std_activation", "std_cam", "std"])
coverage_50 = metric(row, ["coverage_50", "coverage_0.5", "coverage_at_0.5"])
coverage_80 = metric(row, ["coverage_80", "coverage_0.8", "coverage_at_0.8"])
coverage_90 = metric(row, ["coverage_90", "coverage_0.9", "coverage_at_0.9"])
top20 = metric(row, ["top20_activation_concentration", "top20"])
entropy = metric(row, ["normalized_activation_entropy", "normalized_entropy", "entropy"])

print("\nCLASSIFIER EVIDENCE")
print(f"Model       : {CLASSIFIER_NAME}")
print(f"Prediction  : {PREDICTED_CLASS_NAME} ({PREDICTED_CLASS})")
print(f"Confidence  : {CONFIDENCE * 100:.2f}%")

print("\nQUANTITATIVE GRAD-CAM EVIDENCE")
print(f"Mean activation       : {fmt(mean_cam)}")
print(f"Median activation     : {fmt(median_cam)}")
print(f"Std activation        : {fmt(std_cam)}")
print(f"Coverage >= 0.5       : {fmt(coverage_50, True)}")
print(f"Coverage >= 0.8       : {fmt(coverage_80, True)}")
print(f"Coverage >= 0.9       : {fmt(coverage_90, True)}")
print(f"Top-20 concentration  : {fmt(top20)}")
print(f"Normalized entropy    : {fmt(entropy)}")

original = Image.open(IMAGE_PATH).convert("RGB")
gradcam = Image.open(GRADCAM_PATH).convert("RGB")

# ============================================================
# LOAD FLORENCE
# ============================================================

print("\n" + "=" * 75)
print("LOADING FLORENCE-2")
print("=" * 75)

start = time.time()

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    _attn_implementation="eager",
)
model.to(DEVICE)
model.eval()

print(f"Model loaded in {time.time() - start:.2f} seconds")
print("Parameters:", sum(p.numel() for p in model.parameters()))

# ============================================================
# GROUNDED PROMPTS
# ============================================================

base_evidence = f"""
PROJECT EVIDENCE — DO NOT MODIFY THESE FACTS

Classification model: {CLASSIFIER_NAME}
Prediction: {PREDICTED_CLASS_NAME} ({PREDICTED_CLASS})
Confidence: {CONFIDENCE * 100:.2f}%
True class for this evaluated representative case: {TRUE_CLASS}

Grad-CAM quantitative measurements:
Mean activation: {fmt(mean_cam)}
Median activation: {fmt(median_cam)}
Standard deviation: {fmt(std_cam)}
Activation coverage >= 0.5: {fmt(coverage_50, True)}
Activation coverage >= 0.8: {fmt(coverage_80, True)}
Activation coverage >= 0.9: {fmt(coverage_90, True)}
Top-20% activation concentration: {fmt(top20)}
Normalized activation entropy: {fmt(entropy)}

Interpretation rules:
- The CNN is the classifier.
- Grad-CAM shows regions that contributed to the CNN decision.
- Quantitative Grad-CAM values describe activation distribution.
- Do not treat Grad-CAM as a pathology segmentation mask.
- Do not invent malignant cells, mutations, genes, biomarkers, treatment,
  prognosis, or microscopic findings that are not clearly visible.
- Do not override or contradict the supplied CNN prediction.
- Clearly distinguish visual observation from model-derived evidence.
- This is an AI-assisted research interpretation, not a clinical diagnosis.
"""

original_prompt = f"""
You are the visual-description component of a colorectal
histopathology research system.

{base_evidence}

Look at the ORIGINAL HISTOPATHOLOGY IMAGE.

Write a concise scientific visual description (3–5 sentences).
Describe only visible image characteristics such as:
- overall staining/color appearance,
- tissue organization,
- cellular density,
- visible architectural patterns,
- distribution of microscopic structures.

Do not diagnose the image yourself.
Do not claim a specific pathological finding unless it is visibly
supported by the image.
Do not mention objects such as a microscope, camera, or generic
"circular objects" unless they are genuinely relevant to the tissue.
"""

gradcam_prompt = f"""
You are the explainability-reporting component of a colorectal
histopathology research system.

{base_evidence}

Look at the supplied GRAD-CAM VISUALIZATION.

Write a concise evidence description (3–5 sentences).
Explain:
1. Where the classifier's attention appears concentrated.
2. Whether activation is localized or broadly distributed.
3. How the visual activation relates to the supplied classifier prediction.
4. What the quantitative activation values indicate.

Important:
- Grad-CAM identifies image regions contributing to the model decision.
- It does NOT prove that a highlighted region is cancerous.
- Do not identify individual malignant cells.
- Do not invent pathology.
- Do not change the supplied prediction.
"""

print("\n" + "=" * 75)
print("GENERATING PROJECT-GROUNDED DESCRIPTIONS")
print("=" * 75)

start = time.time()
original_description = generate_caption(
    processor, model, original, original_prompt
)
original_time = time.time() - start

print("\nORIGINAL IMAGE DESCRIPTION")
print("-" * 75)
print(original_description)
print(f"\nTime: {original_time:.2f} seconds")

start = time.time()
gradcam_description = generate_caption(
    processor, model, gradcam, gradcam_prompt
)
gradcam_time = time.time() - start

print("\nGRAD-CAM EVIDENCE DESCRIPTION")
print("-" * 75)
print(gradcam_description)
print(f"\nTime: {gradcam_time:.2f} seconds")

# ============================================================
# DETERMINISTIC PROJECT INTERPRETATION
# ============================================================

if coverage_50 is not None:
    coverage_statement = (
        f"{coverage_50 * 100:.2f}% of the Grad-CAM map has activation "
        f"at or above 0.5."
    )
else:
    coverage_statement = "The 0.5 activation coverage is not available."

if coverage_80 is not None:
    coverage_80_statement = (
        f"Only {coverage_80 * 100:.2f}% reaches or exceeds 0.8, "
        "indicating that the strongest activation is concentrated in "
        "a smaller subset of the image."
    )
else:
    coverage_80_statement = "The 0.8 activation coverage is not available."

project_interpretation = (
    f"The Residual Multi-Scale CNN predicts {PREDICTED_CLASS_NAME} "
    f"with {CONFIDENCE * 100:.2f}% confidence. "
    f"The Grad-CAM provides model-derived spatial evidence for this "
    f"prediction. {coverage_statement} {coverage_80_statement} "
    f"The mean activation is {fmt(mean_cam)}, while the top-20% "
    f"activation concentration is {fmt(top20)}. These measurements "
    f"describe the distribution of model attention and should not be "
    f"interpreted as a direct pathological segmentation."
)

final_report = f"""
PROJECT-GROUNDED EXPLAINABLE CRC REPORT
=======================================

CASE
----
Image: {IMAGE_FILENAME}
True class: {TRUE_CLASS}

CLASSIFICATION RESULT
---------------------
Model: {CLASSIFIER_NAME}
Prediction: {PREDICTED_CLASS_NAME} ({PREDICTED_CLASS})
Confidence: {CONFIDENCE * 100:.2f}%

VISUAL DESCRIPTION
------------------
{original_description}

GRAD-CAM EVIDENCE DESCRIPTION
-----------------------------
{gradcam_description}

QUANTITATIVE GRAD-CAM EVIDENCE
------------------------------
Mean activation: {fmt(mean_cam)}
Median activation: {fmt(median_cam)}
Standard deviation: {fmt(std_cam)}
Coverage >= 0.5: {fmt(coverage_50, True)}
Coverage >= 0.8: {fmt(coverage_80, True)}
Coverage >= 0.9: {fmt(coverage_90, True)}
Top-20% activation concentration: {fmt(top20)}
Normalized activation entropy: {fmt(entropy)}

PROJECT-GROUNDED INTERPRETATION
-------------------------------
{project_interpretation}

ROLE OF FLORENCE-2
------------------
Florence-2 is used as an auxiliary natural-language visual
description module. It does not perform the primary CRC
classification and does not replace Grad-CAM or the trained CNN.

LIMITATIONS
-----------
Grad-CAM indicates regions contributing to the model prediction;
it is not a pathological segmentation mask and does not establish
the presence of individual malignant cells.

This output is an AI-assisted research interpretation and is not
a clinical diagnosis.
""".strip()

report_data = {
    "case": {
        "image": IMAGE_FILENAME,
        "true_class": TRUE_CLASS,
    },
    "classifier": {
        "model": CLASSIFIER_NAME,
        "prediction": PREDICTED_CLASS,
        "prediction_name": PREDICTED_CLASS_NAME,
        "confidence": CONFIDENCE,
    },
    "gradcam_metrics": {
        "mean_activation": mean_cam,
        "median_activation": median_cam,
        "std_activation": std_cam,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "top20_activation_concentration": top20,
        "normalized_activation_entropy": entropy,
    },
    "florence": {
        "model": MODEL_NAME,
        "parameters": sum(p.numel() for p in model.parameters()),
        "original_description": original_description,
        "gradcam_description": gradcam_description,
        "original_generation_seconds": original_time,
        "gradcam_generation_seconds": gradcam_time,
        "total_generation_seconds": original_time + gradcam_time,
    },
    "report": final_report,
}

OUTPUT_REPORT.write_text(final_report, encoding="utf-8")
OUTPUT_JSON.write_text(
    json.dumps(report_data, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n" + "=" * 75)
print("FINAL PROJECT-GROUNDED REPORT")
print("=" * 75)
print(final_report)

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)
print(OUTPUT_REPORT)
print(OUTPUT_JSON)
print("\nIntegration completed successfully.")
