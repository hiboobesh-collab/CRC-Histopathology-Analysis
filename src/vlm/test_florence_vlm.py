import os
import json
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/Florence-2-base-ft"

DEVICE = torch.device("cpu")

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "lung_colon_image_set",
    "colon_image_sets",
    "colon_aca",
    "colonca1207.jpeg"
)

GRADCAM_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "explainability",
    "residual_multiscale_cnn",
    "correct_colon_aca_index_0_true_colon_aca_pred_colon_aca.png"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "vlm",
    "florence"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIXED CLASSIFIER EVIDENCE
# ============================================================

CLASSIFIER = "Residual Multi-Scale CNN"
PREDICTION = "colon_aca"
CONFIDENCE = 0.9984

GRADCAM_METRICS = {
    "mean_activation": 0.2392,
    "coverage_50": 16.85,
    "coverage_80": 1.56,
    "coverage_90": 0.39,
    "top20_concentration": 0.5066,
    "entropy": 0.9558
}


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image, max_size=768):

    width, height = image.size

    scale = min(
        max_size / width,
        max_size / height,
        1.0
    )

    if scale >= 1.0:
        return image

    new_size = (
        int(width * scale),
        int(height * scale)
    )

    return image.resize(
        new_size,
        Image.Resampling.LANCZOS
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLORENCE-2 CRC HISTOPATHOLOGY VLM TEST")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Classifier : {CLASSIFIER}")
    print(f"Prediction : {PREDICTION}")
    print(f"Confidence : {CONFIDENCE * 100:.2f}%")
    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(
            f"Original image not found:\n{IMAGE_PATH}"
        )

    if not os.path.exists(GRADCAM_PATH):
        raise FileNotFoundError(
            f"Grad-CAM image not found:\n{GRADCAM_PATH}"
        )

    print("Input files: FOUND")

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    original = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    gradcam = Image.open(
        GRADCAM_PATH
    ).convert("RGB")

    print(
        f"Original image : {original.size}"
    )

    print(
        f"Grad-CAM       : {gradcam.size}"
    )

    original = resize_image(
        original,
        max_size=768
    )

    gradcam = resize_image(
        gradcam,
        max_size=768
    )

    print(
        f"Original resized: {original.size}"
    )

    print(
        f"Grad-CAM resized : {gradcam.size}"
    )

    print()

    # --------------------------------------------------------
    # LOAD FLORENCE
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING FLORENCE-2")
    print("=" * 70)

    start = time.time()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = model.to(DEVICE)

    model.eval()

    load_time = time.time() - start

    print(
        f"Model loaded in {load_time:.2f} seconds"
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print()

    # --------------------------------------------------------
    # FLORENCE TASK
    # --------------------------------------------------------

    # Florence-2 is primarily a vision task model.
    # We use image captioning as the visual-description task.

    task_prompt = "<MORE_DETAILED_CAPTION>"

    print("=" * 70)
    print("GENERATING VISUAL DESCRIPTION")
    print("=" * 70)

    start_generation = time.time()

    inputs = processor(
        text=task_prompt,
        images=original,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=100,
            num_beams=1
        )

    description = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    try:

        parsed = processor.post_process_generation(
            description,
            task=task_prompt,
            image_size=original.size
        )

    except Exception:

        parsed = description

    generation_time = (
        time.time() - start_generation
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FLORENCE-2 OUTPUT")
    print("=" * 70)

    print(parsed)

    print()
    print(
        f"Generation time: "
        f"{generation_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # CREATE GROUNDED REPORT
    # --------------------------------------------------------

    report_text = f"""
FLORENCE-2 AUXILIARY VISUAL INTERPRETATION
==========================================

Model:
{MODEL_NAME}

Primary classifier:
{CLASSIFIER}

Classifier prediction:
{PREDICTION}

Classifier confidence:
{CONFIDENCE * 100:.2f}%

Florence-2 visual description:
{parsed}

Grad-CAM quantitative evidence:
- Mean activation: {GRADCAM_METRICS["mean_activation"]}
- Coverage >= 50%: {GRADCAM_METRICS["coverage_50"]:.2f}%
- Coverage >= 80%: {GRADCAM_METRICS["coverage_80"]:.2f}%
- Coverage >= 90%: {GRADCAM_METRICS["coverage_90"]:.2f}%
- Top-20 concentration: {GRADCAM_METRICS["top20_concentration"]}
- Normalized entropy: {GRADCAM_METRICS["entropy"]}

IMPORTANT LIMITATION
--------------------
Florence-2 is being used only as an auxiliary visual-description
module. It does not replace the Residual Multi-Scale CNN.

The classifier prediction remains fixed.

The Grad-CAM indicates regions that influenced the classifier.
It is not a segmentation mask and does not establish a clinical
diagnosis.

This output is intended for academic research only.
"""

    # --------------------------------------------------------
    # SAVE TXT
    # --------------------------------------------------------

    txt_path = os.path.join(
        OUTPUT_DIR,
        "colonca1207_florence_report.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "colonca1207_florence_report.json"
    )

    report = {
        "model": MODEL_NAME,
        "classifier": CLASSIFIER,
        "prediction": PREDICTION,
        "confidence": CONFIDENCE,
        "visual_description": parsed,
        "gradcam_metrics": GRADCAM_METRICS,
        "generation_time_seconds": generation_time,
        "purpose": "Auxiliary visual description only"
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(txt_path)
    print(json_path)

    print()
    print("=" * 70)
    print("FLORENCE-2 TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()