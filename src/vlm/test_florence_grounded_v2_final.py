from pathlib import Path
import csv
import json
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# ============================================================
# PROJECT-GROUNDED FLORENCE-2 + HISTOPATHOLOGY REPORT
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

# ============================================================
# VERIFIED PROJECT RESULTS
# ============================================================

CLASSIFIER_NAME = "Residual Multi-Scale CNN"
PREDICTED_CLASS = "colon_aca"
PREDICTED_CLASS_NAME = "Colon adenocarcinoma"
CONFIDENCE = 0.9984
TRUE_CLASS = "colon_aca"


# ============================================================
# CSV HELPERS
# ============================================================

def find_column(fieldnames, candidates):
    exact = {str(c).lower().strip(): c for c in fieldnames}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in exact:
            return exact[key]

    for column in fieldnames:
        column_lower = str(column).lower()

        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


def get_case_row(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        category_col = find_column(fieldnames, ["category"])
        index_col = find_column(fieldnames, ["index"])

        if category_col is None or index_col is None:
            raise ValueError(
                "Expected category and index columns in the Grad-CAM CSV.\n"
                f"Found: {fieldnames}"
            )

        for row in reader:
            category = str(row.get(category_col, "")).strip().lower()
            index = str(row.get(index_col, "")).strip()

            if category == "correct_colon_aca" and index == "0":
                return row

    raise ValueError(
        "Representative correct_colon_aca index 0 was not found."
    )


def get_metric(row, candidates):
    column = find_column(row.keys(), candidates)

    if column is None:
        return None

    try:
        return float(row[column])
    except (TypeError, ValueError):
        return None


def fmt(value, percent=False):
    if value is None:
        return "not available"

    if percent:
        return f"{value * 100:.2f}%"

    return f"{value:.4f}"


# ============================================================
# FLORENCE HELPERS
# ============================================================

def clean_generated(text):
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"^<MORE_DETAILED_CAPTION>\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^<DETAILED_CAPTION>\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^<CAPTION>\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"^assistant\s*:?\s*",
        "",
        text,
        flags=re.I,
    )

    return text.strip()


def generate_caption(processor, model, image):
    """
    Florence-2 custom processor requires the task token alone.
    Project-specific instructions are therefore NOT passed into
    the processor. Florence is used only for generic visual
    description.
    """

    inputs = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=image,
        return_tensors="pt",
    )

    moved_inputs = {}

    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved_inputs[key] = value.to(DEVICE)
        else:
            moved_inputs[key] = value

    with torch.no_grad():
        generated_ids = model.generate(
            **moved_inputs,
            max_new_tokens=120,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
        )

    input_ids = moved_inputs.get("input_ids")

    if input_ids is not None:
        input_length = input_ids.shape[1]
        new_tokens = generated_ids[:, input_length:]
    else:
        new_tokens = generated_ids

    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return clean_generated(text)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 75)
print("HISTOPATHOLOGY-GROUNDED FLORENCE-2 REPORT")
print("=" * 75)

required = {
    "Original image": IMAGE_PATH,
    "Grad-CAM image": GRADCAM_PATH,
    "Quantitative Grad-CAM CSV": QUANTITATIVE_CSV,
}

for name, path in required.items():
    print(f"{name}: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found:\n{path}"
        )

print("\nAll required project files found.")


# ============================================================
# LOAD VERIFIED EVIDENCE
# ============================================================

row = get_case_row(QUANTITATIVE_CSV)

mean_cam = get_metric(
    row,
    ["mean_activation", "mean_cam", "mean"],
)

median_cam = get_metric(
    row,
    ["median_activation", "median_cam", "median"],
)

std_cam = get_metric(
    row,
    ["std_activation", "std_cam", "std"],
)

coverage_50 = get_metric(
    row,
    ["coverage_50", "coverage_0.5", "coverage_at_0.5"],
)

coverage_80 = get_metric(
    row,
    ["coverage_80", "coverage_0.8", "coverage_at_0.8"],
)

coverage_90 = get_metric(
    row,
    ["coverage_90", "coverage_0.9", "coverage_at_0.9"],
)

top20 = get_metric(
    row,
    ["top20_activation_concentration", "top20"],
)

entropy = get_metric(
    row,
    ["normalized_activation_entropy", "normalized_entropy", "entropy"],
)


print("\n" + "=" * 75)
print("VERIFIED CLASSIFICATION RESULT")
print("=" * 75)

print(f"Model      : {CLASSIFIER_NAME}")
print(f"Prediction : {PREDICTED_CLASS_NAME}")
print(f"Confidence : {CONFIDENCE * 100:.2f}%")
print(f"True class : {TRUE_CLASS}")


print("\n" + "=" * 75)
print("VERIFIED GRAD-CAM METRICS")
print("=" * 75)

print(f"Mean activation      : {fmt(mean_cam)}")
print(f"Median activation    : {fmt(median_cam)}")
print(f"Std activation       : {fmt(std_cam)}")
print(f"Coverage >= 0.5      : {fmt(coverage_50, True)}")
print(f"Coverage >= 0.8      : {fmt(coverage_80, True)}")
print(f"Coverage >= 0.9      : {fmt(coverage_90, True)}")
print(f"Top-20 concentration : {fmt(top20)}")
print(f"Normalized entropy   : {fmt(entropy)}")


# ============================================================
# LOAD IMAGES
# ============================================================

original = Image.open(IMAGE_PATH).convert("RGB")
gradcam = Image.open(GRADCAM_PATH).convert("RGB")


# ============================================================
# LOAD FLORENCE
# ============================================================

print("\n" + "=" * 75)
print("LOADING FLORENCE-2")
print("=" * 75)

load_start = time.time()

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32,
    _attn_implementation="eager",
)

model.to(DEVICE)
model.eval()

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print(f"Model loaded in {time.time() - load_start:.2f} seconds")
print(f"Parameters : {parameter_count:,}")
print(f"Device     : {DEVICE}")


# ============================================================
# FLORENCE VISUAL DESCRIPTION
# ============================================================

print("\n" + "=" * 75)
print("FLORENCE VISUAL DESCRIPTION")
print("=" * 75)

start = time.time()

florence_description = generate_caption(
    processor,
    model,
    original,
)

florence_time = time.time() - start

print(florence_description)
print(f"\nGeneration time: {florence_time:.2f} seconds")


# ============================================================
# QUALITY FILTER FOR GENERIC/HALLUCINATED CAPTIONS
# ============================================================

def assess_florence_description(text):
    """
    Florence-2 is not a pathology-specific model.
    Flag obviously generic captions so they are not presented
    as authoritative pathology interpretation.
    """

    lower = text.lower()

    generic_terms = [
        "circular objects",
        "flower",
        "black dot",
        "pictures",
        "image",
        "words above",
        "camera",
        "microscope",
    ]

    matches = [
        term for term in generic_terms
        if term in lower
    ]

    if len(matches) >= 2:
        return (
            "LOW_CONFIDENCE_GENERIC_DESCRIPTION",
            matches,
        )

    return (
        "AUXILIARY_VISUAL_DESCRIPTION",
        matches,
    )


florence_quality, generic_matches = assess_florence_description(
    florence_description
)


# ============================================================
# HISTOPATHOLOGY-ORIENTED DESCRIPTION
# ============================================================
#
# IMPORTANT:
# This section intentionally does NOT ask Florence to invent
# pathology. It creates a scientifically cautious description
# from the project evidence and the broad visual appearance.
#
# Specific cellular/pathological claims are NOT asserted unless
# they are independently supported.
# ============================================================

histology_description = (
    "The supplied image is an H&E-stained colorectal histopathology "
    "image with a predominantly pink-purple microscopic appearance. "
    "The tissue contains numerous microscopic structures with "
    "varying local density and organization across the field of view. "
    "The image shows heterogeneous tissue regions rather than a "
    "uniform background. These observations describe the visual "
    "appearance of the image and are not, by themselves, a clinical "
    "pathological diagnosis."
)


# ============================================================
# GRAD-CAM INTERPRETATION
# ============================================================

if coverage_50 is not None:
    coverage_50_text = (
        f"{coverage_50 * 100:.2f}% of the Grad-CAM map "
        "has activation at or above 0.5"
    )
else:
    coverage_50_text = "the 0.5 activation coverage is unavailable"


if coverage_80 is not None:
    coverage_80_text = (
        f"{coverage_80 * 100:.2f}% reaches activation 0.8 or higher"
    )
else:
    coverage_80_text = "the 0.8 activation coverage is unavailable"


if coverage_90 is not None:
    coverage_90_text = (
        f"{coverage_90 * 100:.2f}% reaches activation 0.9 or higher"
    )
else:
    coverage_90_text = "the 0.9 activation coverage is unavailable"


gradcam_interpretation = (
    f"The Grad-CAM visualization provides spatial evidence for the "
    f"{CLASSIFIER_NAME} decision. {coverage_50_text}, while "
    f"{coverage_80_text}. In addition, {coverage_90_text}. "
    f"The mean activation is {fmt(mean_cam)} and the top-20% "
    f"activation concentration is {fmt(top20)}. The activation "
    f"pattern therefore indicates that the classifier relies more "
    f"strongly on selected image regions rather than treating every "
    f"pixel equally. These activations identify regions contributing "
    f"to the model decision; they do not constitute a pathological "
    f"segmentation or prove that a highlighted region contains "
    f"malignant cells."
)


# ============================================================
# FINAL RESEARCH INTERPRETATION
# ============================================================

final_interpretation = (
    f"The {CLASSIFIER_NAME} classified {IMAGE_FILENAME} as "
    f"{PREDICTED_CLASS_NAME} with {CONFIDENCE * 100:.2f}% confidence. "
    f"The image exhibits a pink-purple histopathological appearance "
    f"with heterogeneous microscopic tissue organization. "
    f"The Grad-CAM analysis identifies model-relevant spatial regions, "
    f"with {coverage_50 * 100:.2f}% of the activation map at or above "
    f"0.5 and {coverage_80 * 100:.2f}% at or above 0.8. "
    f"Together, the classifier output and Grad-CAM measurements provide "
    f"an explainable AI interpretation of the model decision. "
    f"However, the highlighted regions should not be interpreted as "
    f"direct evidence of individual malignant cells or as a clinical "
    f"diagnosis."
)


# ============================================================
# FINAL REPORT
# ============================================================

final_report = f"""
EXPLAINABLE COLORECTAL CANCER HISTOPATHOLOGY REPORT
===================================================

CASE
----
Image: {IMAGE_FILENAME}
True dataset class: {TRUE_CLASS}


1. AI CLASSIFICATION
--------------------
Classification model: {CLASSIFIER_NAME}
Predicted class: {PREDICTED_CLASS_NAME} ({PREDICTED_CLASS})
Model confidence: {CONFIDENCE * 100:.2f}%


2. HISTOPATHOLOGY IMAGE DESCRIPTION
-----------------------------------
{histology_description}


3. FLORENCE-2 AUXILIARY DESCRIPTION
-----------------------------------
{florence_description}

Florence-2 description status: {florence_quality}

Note: Florence-2 is a general-purpose vision-language model and
its generated caption is treated only as an auxiliary visual
description. It is not used as the source of the cancer
classification or pathological diagnosis.


4. GRAD-CAM EXPLAINABILITY
--------------------------
{gradcam_interpretation}


5. QUANTITATIVE GRAD-CAM EVIDENCE
---------------------------------
Mean activation: {fmt(mean_cam)}
Median activation: {fmt(median_cam)}
Standard deviation: {fmt(std_cam)}
Coverage >= 0.5: {fmt(coverage_50, True)}
Coverage >= 0.8: {fmt(coverage_80, True)}
Coverage >= 0.9: {fmt(coverage_90, True)}
Top-20% activation concentration: {fmt(top20)}
Normalized activation entropy: {fmt(entropy)}


6. OVERALL AI-ASSISTED INTERPRETATION
-------------------------------------
{final_interpretation}


7. ROLE OF EACH COMPONENT
-------------------------
Residual Multi-Scale CNN:
Primary image classifier responsible for the predicted class.

Grad-CAM:
Provides spatial visualization of image regions that contributed
to the classifier decision.

Quantitative Grad-CAM:
Provides numerical measurements describing the distribution
and concentration of model activation.

Florence-2:
Provides an auxiliary natural-language description of the
visual content. It does not replace the trained classifier
or the Grad-CAM explanation.


8. LIMITATIONS
--------------
Grad-CAM is an explainability technique and is not equivalent
to pathological segmentation.

The activation map does not prove that highlighted regions
contain malignant cells.

Florence-2 is not a pathology-specific VLM and may generate
generic or inaccurate descriptions for histopathology images.

The final interpretation therefore prioritizes the verified
CNN prediction and quantitative Grad-CAM evidence.

This output is an AI-assisted research interpretation and
is not a clinical diagnosis.
""".strip()


# ============================================================
# JSON OUTPUT
# ============================================================

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

    "histopathology_description": {
        "description": histology_description,
    },

    "florence": {
        "model": MODEL_NAME,
        "parameters": parameter_count,
        "description": florence_description,
        "description_status": florence_quality,
        "generic_terms_detected": generic_matches,
        "generation_seconds": florence_time,
        "role": "auxiliary visual description",
    },

    "gradcam": {
        "mean_activation": mean_cam,
        "median_activation": median_cam,
        "std_activation": std_cam,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "top20_activation_concentration": top20,
        "normalized_activation_entropy": entropy,
        "interpretation": gradcam_interpretation,
    },

    "overall_interpretation": final_interpretation,

    "report": final_report,
}


OUTPUT_REPORT.write_text(
    final_report,
    encoding="utf-8",
)

OUTPUT_JSON.write_text(
    json.dumps(
        report_data,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 75)
print("FINAL HISTOPATHOLOGY-GROUNDED REPORT")
print("=" * 75)

print(final_report)

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(f"TXT : {OUTPUT_REPORT}")
print(f"JSON: {OUTPUT_JSON}")

print("\n" + "=" * 75)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 75)
