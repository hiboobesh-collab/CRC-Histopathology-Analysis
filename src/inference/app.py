"""CRC Histopathology Explainable AI — Streamlit interface.



Run from the project root:

    streamlit run src/inference/app.py

"""



from __future__ import annotations



import sys

import csv

from pathlib import Path



import numpy as np

import streamlit as st

import torch

import torch.nn.functional as F

from PIL import Image, ImageFilter

from torchvision import transforms



from reportlab.lib import colors

from reportlab.lib.enums import TA_CENTER

from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.lib.units import inch

from reportlab.platypus import (

    Image as ReportLabImage,

    PageBreak,

    Paragraph,

    SimpleDocTemplate,

    Spacer,

    Table,

    TableStyle,

)





# ============================================================

# PROJECT PATHS

# ============================================================



ROOT = Path(__file__).resolve().parents[2]



if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



CHECKPOINT = (

    ROOT

    / "models"

    / "classification"

    / "residual_multiscale_cnn_groupaware"

    / "best"

    / "residual_multiscale_cnn_best.pth"

)



CLASS_NAMES = (

    "Normal Colon Tissue",

    "Colon Adenocarcinoma",

)



# Exact preprocessing used for validation/test inference.

TRANSFORM = transforms.Compose(

    [

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406],

            [0.229, 0.224, 0.225],

        ),

    ]

)





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="CRC Histopathology Analysis",

    page_icon="🔬",

    layout="wide",

    initial_sidebar_state="collapsed",

)





# ============================================================

# CLEAN PROFESSIONAL UI

# ============================================================



st.markdown(

    """

    <style>

        .block-container {

            max-width: 1060px;

            padding-top: 1.8rem;

            padding-bottom: 3rem;

        }



        h1 {

            font-size: 2rem !important;

            letter-spacing: -0.04em;

            margin-bottom: 0.15rem !important;

        }



        h2 {

            font-size: 1.25rem !important;

            letter-spacing: -0.02em;

        }



        .app-subtitle {

            color: #94a3b8;

            font-size: 0.92rem;

            margin-bottom: 1.4rem;

        }



        .upload-note {

            color: #94a3b8;

            font-size: 0.84rem;

            margin-top: -0.25rem;

            margin-bottom: 0.8rem;

        }



        /* Main prediction card */

        .arch-card {

            background: #171b22;

            border: 1px solid #303743;

            border-radius: 14px;

            padding: 20px;

            margin: 10px 0 20px;

        }



        .arch-title {

            color: #ffffff;

            font-size: 1.15rem;

            font-weight: 700;

            margin-bottom: 4px;

        }



        .arch-subtitle {

            color: #94a3b8;

            font-size: 0.84rem;

            margin-bottom: 16px;

        }



        .arch-flow {

            max-width: 780px;

            margin: 0 auto;

        }



        .arch-step {

            background: #202631;

            border: 1px solid #3b4553;

            border-radius: 9px;

            padding: 11px 14px;

            color: #e5e7eb;

            text-align: center;

            font-size: 0.84rem;

        }



        .arch-arrow {

            color: #94a3b8;

            text-align: center;

            line-height: 22px;

            font-size: 1rem;

        }



        .result-card {

            display: flex;

            justify-content: space-between;

            align-items: center;

            gap: 28px;

            padding: 22px 24px;

            margin: 10px 0 24px;

            border-radius: 14px;

            background: #171b22;

            border: 1px solid #303743;

        }



        .result-left {

            flex: 1;

            min-width: 0;

        }



        .result-label {

            font-size: 0.70rem;

            font-weight: 700;

            letter-spacing: 0.09em;

            color: #aab4c2;

            margin-bottom: 7px;

        }



        .result-title {

            font-size: 1.55rem;

            font-weight: 700;

            color: #ffffff !important;

            line-height: 1.2;

        }



        .result-note {

            margin-top: 9px;

            font-size: 0.80rem;

            color: #9aa5b4;

        }



        .confidence-box {

            min-width: 175px;

            padding: 14px 20px;

            border-radius: 12px;

            background: #202631;

            border: 1px solid #3b4553;

            text-align: center;

        }



        .confidence-label {

            font-size: 0.68rem;

            font-weight: 700;

            letter-spacing: 0.09em;

            color: #b7c0cc !important;

            margin-bottom: 5px;

        }



        .confidence-value {

            font-size: 1.75rem;

            font-weight: 750;

            color: #ffffff !important;

            line-height: 1.1;

        }



        /* Image captions */

        .image-caption {

            font-size: 0.82rem;

            font-weight: 600;

            color: #d5dbe3;

            margin-bottom: 7px;

        }



        /* Explanation cards */

        .explanation-box {

            padding: 14px 16px;

            border-radius: 11px;

            background: #151920;

            border: 1px solid #2d3440;

            margin-top: 12px;

            margin-bottom: 10px;

        }



        .explanation-title {

            font-size: 0.82rem;

            font-weight: 700;

            color: #e5e7eb;

            margin-bottom: 5px;

        }



        .explanation-text {

            font-size: 0.88rem;

            line-height: 1.55;

            color: #aeb7c4;

        }



        /* Probability labels */

        .probability-label {

            font-size: 0.86rem;

            color: #d7dce4;

            margin-bottom: 5px;

        }



        /* Summary */

        .summary-box {

            padding: 16px 18px;

            border-radius: 11px;

            background: #151920;

            border: 1px solid #2d3440;

            line-height: 1.6;

        }



        .disclaimer {

            text-align: center;

            color: #7f8a99;

            font-size: 0.76rem;

            margin-top: 24px;

        }



        @media (max-width: 700px) {

            .result-card {

                flex-direction: column;

                align-items: stretch;

            }



            .confidence-box {

                min-width: unset;

            }

        }

    </style>

    """,

    unsafe_allow_html=True,

)





# ============================================================

# MODEL

# ============================================================



@st.cache_resource(show_spinner="Loading classification model...")

def load_model():

    from src.models.residual_multiscale_cnn import ResidualMultiScaleCNN



    model = ResidualMultiScaleCNN(num_classes=2)



    try:

        checkpoint = torch.load(

            CHECKPOINT,

            map_location="cpu",

            weights_only=True,

        )

    except Exception as exc:

        raise RuntimeError(

            f"Unable to load checkpoint: {exc}"

        ) from exc



    if isinstance(checkpoint, dict):

        for key in ("model_state_dict", "state_dict"):

            if key in checkpoint:

                checkpoint = checkpoint[key]

                break



    if not isinstance(checkpoint, dict):

        raise ValueError(

            "Checkpoint does not contain a model state dictionary."

        )



    state = {

        key.removeprefix("module."): value

        for key, value in checkpoint.items()

    }



    model.load_state_dict(state, strict=True)

    model.eval()



    return model





# ============================================================

# HISTOLOGY INPUT VALIDATION / OOD GATE

# ============================================================

# The CRC classifier is a closed-set model. Before allowing an image

# into the classifier, this lightweight gate checks whether the image

# has visual characteristics consistent with H&E histopathology.

# It is intentionally separate from the CRC classifier and does not

# claim clinical tissue identification.



GATE_SIZE = (128, 128)

# Input-suitability gate for the CRC classifier. The classifier itself is
# closed-set; this gate prevents ordinary photos, screenshots, posters,
# cartoons, collages, etc. from reaching the CRC model.
GATE_MIN_SCORE = 0.58
# Secondary safety thresholds are applied only to the input gate.
GATE_MIN_CONFIDENCE = 0.55
GATE_MIN_COVERAGE50 = 0.005

def _histology_features(image):
    """Fast microscopy-vs-photograph features used only for input validation."""
    rgb_img = image.convert("RGB")
    w, h = rgb_img.size
    small_img = rgb_img.resize(GATE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(small_img, dtype=np.float32) / 255.0
    hsv = np.asarray(small_img.convert("HSV"), dtype=np.float32) / 255.0
    hch, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    pink = (((hch <= 0.10) | (hch >= 0.91)) & (sat >= 0.12) & (val >= 0.18))
    purple = ((hch >= 0.54) & (hch <= 0.88) & (sat >= 0.12) & (val >= 0.16))
    he_like = pink | purple
    tissue = (sat >= 0.06) & (val <= 0.995)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gradient = 0.5 * (gx[:-1, :] + gy[:, :-1])

    smooth = np.asarray(small_img.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32) / 255.0
    smooth_gray = 0.299 * smooth[..., 0] + 0.587 * smooth[..., 1] + 0.114 * smooth[..., 2]
    fine_detail = np.abs(gray - smooth_gray)

    patch_gradient, patch_fine, patch_he, patch_purple, patch_std = [], [], [], [], []
    n_grid = 8
    for pi in range(n_grid):
        for pj in range(n_grid):
            y0, y1 = pi * 128 // n_grid, (pi + 1) * 128 // n_grid
            x0, x1 = pj * 128 // n_grid, (pj + 1) * 128 // n_grid
            patch_gradient.append(float(gradient[y0:y1-1, x0:x1-1].mean()))
            patch_fine.append(float(fine_detail[y0:y1, x0:x1].mean()))
            patch_he.append(float(he_like[y0:y1, x0:x1].mean()))
            patch_purple.append(float(purple[y0:y1, x0:x1].mean()))
            patch_std.append(float(gray[y0:y1, x0:x1].std()))

    patch_gradient = np.asarray(patch_gradient)
    patch_fine = np.asarray(patch_fine)
    patch_he = np.asarray(patch_he)
    patch_purple = np.asarray(patch_purple)
    patch_std = np.asarray(patch_std)

    return {
        "aspect_ratio": float(w / max(h, 1)),
        "he_fraction": float(he_like.mean()),
        "purple_fraction": float(purple.mean()),
        "tissue_fraction": float(tissue.mean()),
        "texture": float(gradient.mean()),
        "edge_fraction": float(np.mean(gradient >= 0.035)),
        "stain_field_coverage": float(np.mean(patch_he >= 0.10)),
        "purple_field_coverage": float(np.mean(patch_purple >= 0.015)),
        "fine_detail_mean": float(fine_detail.mean()),
        "fine_patch_coverage": float(np.mean(patch_fine >= 0.018)),
        "gradient_patch_coverage": float(np.mean(patch_gradient >= 0.020)),
        "textured_patch_coverage": float(np.mean(patch_std >= 0.035)),
    }

@st.cache_resource(show_spinner="Preparing histopathology input validator...")
def build_histology_gate_profile():
    candidates = [ROOT / "data" / "splits" / "group_val.csv", ROOT / "data" / "splits" / "val.csv", ROOT / "data" / "splits" / "group_test.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None
    paths = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative = row.get("filepath") or row.get("file_path") or row.get("path")
                if not relative:
                    continue
                path = Path(relative)
                if not path.is_absolute():
                    path = ROOT / path
                if path.exists():
                    paths.append(path)
                if len(paths) >= 250:
                    break
    except Exception:
        return None
    if len(paths) < 20:
        return None
    feats = []
    for path in paths:
        try:
            with Image.open(path) as img:
                f = _histology_features(img)
                feats.append([f[k] for k in ("aspect_ratio","he_fraction","purple_fraction","tissue_fraction","texture","edge_fraction","fine_detail_mean","fine_patch_coverage","gradient_patch_coverage","textured_patch_coverage","stain_field_coverage","purple_field_coverage")])
        except Exception:
            pass
    if len(feats) < 20:
        return None
    m = np.asarray(feats, dtype=np.float64)
    med = np.median(m, axis=0)
    mad = np.median(np.abs(m-med), axis=0)
    return {"median": med, "scale": np.maximum(1.4826*mad, 0.01), "q10": np.percentile(m,10,axis=0), "q90": np.percentile(m,90,axis=0), "n_reference": len(feats)}

def validate_histology_input(image):
    f = _histology_features(image)
    profile = build_histology_gate_profile()
    reasons = []
    ar = f["aspect_ratio"]
    if ar < 0.82 or ar > 1.22:
        reasons.append("image format is not consistent with a histology tile")
    if f["he_fraction"] < 0.14:
        reasons.append("insufficient H&E-like staining")
    if f["purple_fraction"] < 0.018:
        reasons.append("insufficient hematoxylin-like nuclear staining")
    if f["tissue_fraction"] < 0.28:
        reasons.append("insufficient tissue-like coverage")
    if f["fine_patch_coverage"] < 0.35:
        reasons.append("insufficient microscopic fine-scale texture")
    if f["gradient_patch_coverage"] < 0.35:
        reasons.append("insufficient distributed microscopic structure")
    if f["textured_patch_coverage"] < 0.35:
        reasons.append("image does not contain tissue-like texture across the field")
    if f["stain_field_coverage"] < 0.55 and f["purple_field_coverage"] < 0.35:
        reasons.append("staining is not distributed like a microscope tissue field")

    reference_similarity = 0.0
    if profile is not None:
        keys = ("aspect_ratio","he_fraction","purple_fraction","tissue_fraction","texture","edge_fraction","fine_detail_mean","fine_patch_coverage","gradient_patch_coverage","textured_patch_coverage","stain_field_coverage","purple_field_coverage")
        current = np.asarray([f[k] for k in keys], dtype=np.float64)
        z = np.abs((current-profile["median"])/profile["scale"])
        reference_similarity = float(np.mean(np.exp(-0.35*np.minimum(z,8.0))))
        key_idx = [4,6,7,8,9]
        overlap = [profile["q10"][i] <= current[i] <= profile["q90"][i] for i in key_idx]
        if sum(overlap) < 2:
            reasons.append("image structure is outside the trained histopathology appearance range")

    score = (0.18*min(1,f["he_fraction"]/0.30) + 0.12*min(1,f["purple_fraction"]/0.07) + 0.12*min(1,f["tissue_fraction"]/0.65) + 0.16*min(1,f["fine_patch_coverage"]/0.70) + 0.14*min(1,f["gradient_patch_coverage"]/0.70) + 0.12*min(1,f["textured_patch_coverage"]/0.70) + 0.08*max(min(1,f["stain_field_coverage"]/0.75),min(1,f["purple_field_coverage"]/0.55)) + 0.08*reference_similarity)
    return len(reasons) == 0 and score >= GATE_MIN_SCORE, float(score), reasons


# GRAD-CAM

# ============================================================



class GradCAM:

    def __init__(self, model):

        self.activations = None

        self.gradients = None



        self.handles = [

            model.block3.register_forward_hook(self._forward),

            model.block3.register_full_backward_hook(self._backward),

        ]



    def _forward(self, _module, _inputs, output):

        self.activations = output.detach()



    def _backward(self, _module, _grad_input, grad_output):

        self.gradients = grad_output[0].detach()



    def generate(self, model, tensor, class_index):

        model.zero_grad(set_to_none=True)



        logits = model(tensor)



        if isinstance(logits, (tuple, list)):

            logits = logits[0]



        logits[:, class_index].sum().backward()



        weights = self.gradients.mean(

            dim=(2, 3),

            keepdim=True,

        )



        cam = F.relu(

            (weights * self.activations).sum(

                dim=1,

                keepdim=True,

            )

        )



        cam = F.interpolate(

            cam,

            size=(224, 224),

            mode="bilinear",

            align_corners=False,

        )



        cam = cam[0, 0].detach().cpu().numpy()



        span = float(cam.max() - cam.min())



        if span > 1e-12:

            cam = (cam - cam.min()) / span

        else:

            cam = np.zeros_like(cam)



        return cam.astype(np.float32)



    def close(self):

        for handle in self.handles:

            handle.remove()





# ============================================================

# GRAD-CAM METRICS

# ============================================================



def metrics_for(cam):

    flat = cam.reshape(-1).astype(np.float64)



    ordered = np.sort(flat)[::-1]

    k = max(1, int(len(flat) * 0.20))



    total = float(flat.sum())



    if total > 0:

        probabilities = flat / total

        positive = probabilities[probabilities > 0]



        entropy = float(

            -np.sum(

                positive * np.log(positive)

            )

            / np.log(len(flat))

        )



        top20 = float(

            ordered[:k].sum() / total

        )

    else:

        entropy = 0.0

        top20 = 0.0



    return {

        "mean": float(flat.mean()),

        "median": float(np.median(flat)),

        "std": float(flat.std()),

        "max": float(flat.max()),

        "coverage50": float(np.mean(flat >= 0.5)),

        "coverage80": float(np.mean(flat >= 0.8)),

        "coverage90": float(np.mean(flat >= 0.9)),

        "top20": top20,

        "entropy": entropy,

    }





# ============================================================

# IMAGE GENERATION

# ============================================================



def render_images(original, cam):

    """Create original, Grad-CAM heatmap and overlay images."""



    display_original = original.resize(

        (224, 224),

        Image.Resampling.BILINEAR,

    )



    from matplotlib import colormaps



    heat = (

        colormaps["jet"](cam)[..., :3] * 255

    ).astype(np.uint8)



    heatmap = Image.fromarray(

        heat,

        "RGB",

    )



    overlay = Image.blend(

        display_original,

        heatmap,

        alpha=0.42,

    )



    return display_original, heatmap, overlay





# ============================================================

# HUMAN-READABLE EXPLANATIONS

# ============================================================



def describe_original():

    return (

        "The uploaded image is a microscopic colorectal tissue "

        "image with predominantly pink and purple staining and "

        "visible variation in tissue texture and structure. "

        "This description refers only to the visual appearance "

        "of the image and does not independently determine "

        "pathological status."

    )





def describe_heatmap(metrics):

    coverage = metrics["coverage50"] * 100



    if coverage < 5:

        pattern = "a relatively localized portion"

    elif coverage < 20:

        pattern = "selected regions"

    else:

        pattern = "a broader portion"



    return (

        f"The Grad-CAM heatmap shows stronger relative model "

        f"activation across {pattern} of the image. "

        f"Approximately {coverage:.2f}% of the heatmap reaches "

        f"at least half of its peak activation. Warmer colours "

        f"represent higher relative activation and indicate "

        f"regions that contributed more strongly to the model's "

        f"prediction. They do not identify confirmed cancer cells "

        f"or define a tumour boundary."

    )





def describe_overlay():

    return (

        "The Grad-CAM overlay places the model's activation map "

        "on top of the original tissue image. This makes it easier "

        "to see where the highlighted regions occur within the "

        "original field of view. The overlay represents model "

        "attribution rather than cell segmentation or a clinical "

        "pathology finding."

    )





# ============================================================

# TECHNICAL REPORT

# ============================================================



def report_text(

    filename,

    original_size,

    pred,

    probabilities,

    metrics,

    original_description,

    heatmap_description,

    overlay_description,

):

    confidence = float(probabilities[pred]) * 100



    return f"""CRC HISTOPATHOLOGY — EXPLAINABLE AI RESEARCH REPORT

==========================================================



1. ANALYSIS SUMMARY

----------------------------------------------------------

Input image: {filename}

Original image dimensions: {original_size[0]} × {original_size[1]} pixels

Inference image size: 224 × 224 pixels

Predicted class: {CLASS_NAMES[pred]}

Model confidence: {confidence:.4f}%





2. CLASS PROBABILITIES

----------------------------------------------------------

Normal Colon Tissue: {probabilities[0] * 100:.4f}%

Colon Adenocarcinoma: {probabilities[1] * 100:.4f}%





3. ORIGINAL IMAGE DESCRIPTION

----------------------------------------------------------

{original_description}





4. GRAD-CAM HEATMAP DESCRIPTION

----------------------------------------------------------

{heatmap_description}





5. GRAD-CAM OVERLAY DESCRIPTION

----------------------------------------------------------

{overlay_description}





6. MODEL AND INFERENCE DETAILS

----------------------------------------------------------

Architecture: Residual Multi-Scale CNN

Model parameters: 2,156,322

Classes: colon_n (index 0), colon_aca (index 1)

Input size: 224 × 224 RGB

Preprocessing:

- Resize to 224 × 224

- Convert to tensor

- ImageNet normalization

- Mean: [0.485, 0.456, 0.406]

- Standard deviation: [0.229, 0.224, 0.225]



Inference:

- Evaluation mode

- CPU

- Softmax class probabilities



Grad-CAM target layer: model.block3

Grad-CAM target class: predicted class (index {pred})





7. QUANTITATIVE GRAD-CAM ANALYSIS

----------------------------------------------------------

Mean activation: {metrics["mean"]:.4f}

Median activation: {metrics["median"]:.4f}

Standard deviation: {metrics["std"]:.4f}

Maximum normalized activation: {metrics["max"]:.4f}



Coverage ≥ 50% of peak activation:

{metrics["coverage50"] * 100:.2f}%



Coverage ≥ 80% of peak activation:

{metrics["coverage80"] * 100:.2f}%



Coverage ≥ 90% of peak activation:

{metrics["coverage90"] * 100:.2f}%



Top-20% activation concentration:

{metrics["top20"]:.4f}



Normalized activation entropy:

{metrics["entropy"]:.4f}





8. METRIC INTERPRETATION

----------------------------------------------------------

Coverage values describe the fraction of heatmap pixels

above each relative activation threshold.



Top-20% activation concentration measures the proportion

of total activation contained within the highest-activation

20% of pixels.



Normalized activation entropy describes how broadly the

activation is distributed across the heatmap.



These measurements are descriptive model-attribution

statistics. They are not validated clinical thresholds.





9. OVERALL INTERPRETATION

----------------------------------------------------------

The Residual Multi-Scale CNN predicts {CLASS_NAMES[pred]}

with {confidence:.4f}% model confidence.



The Grad-CAM heatmap identifies image regions that influenced

this prediction, while the overlay shows their locations on

the original image.



The prediction and explanation describe the behaviour of the

trained model. They do not independently establish whether

the prediction is pathologically correct.





10. LIMITATIONS AND RESEARCH DISCLAIMER

----------------------------------------------------------

This system is an academic research prototype.



The reported confidence is a model probability and is not

a clinically calibrated certainty. The model may make

incorrect predictions, including high-confidence errors.



Grad-CAM is an attribution visualization. It is not a cell

detector, tumour segmentation method, or independently

verified pathological annotation.



The system has not been clinically validated and must not

be used to diagnose disease or guide patient care.

"""







# ============================================================

# PDF REPORT GENERATION

# ============================================================



def create_pdf_report(

    filename,

    original_size,

    pred,

    probabilities,

    metrics,

    original_description,

    heatmap_description,

    overlay_description,

    original_image,

    heatmap_image,

    overlay_image,

):

    """Create a professional technical PDF report in memory."""



    from io import BytesIO



    buffer = BytesIO()



    styles = getSampleStyleSheet()



    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        fontName="Helvetica-Bold",

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        spaceAfter=8,

        textColor=colors.HexColor("#1f2937"),

    )



    subtitle_style = ParagraphStyle(

        "ReportSubtitle",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=9,

        leading=12,

        alignment=TA_CENTER,

        textColor=colors.HexColor("#64748b"),

        spaceAfter=18,

    )



    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontName="Helvetica-Bold",

        fontSize=12,

        leading=15,

        spaceBefore=12,

        spaceAfter=7,

        textColor=colors.HexColor("#1f2937"),

    )



    body_style = ParagraphStyle(

        "Body",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=9.5,

        leading=14,

        spaceAfter=7,

        textColor=colors.HexColor("#374151"),

    )



    small_style = ParagraphStyle(

        "Small",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=8,

        leading=11,

        textColor=colors.HexColor("#64748b"),

    )



    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=42,

        leftMargin=42,

        topMargin=42,

        bottomMargin=42,

        title="CRC Histopathology Explainable AI Report",

        author="CRC Histopathology Explainable AI",

    )



    story = []



    confidence = float(probabilities[pred]) * 100



    story.append(

        Paragraph(

            "CRC Histopathology<br/>Explainable AI Research Report",

            title_style,

        )

    )

    story.append(

        Paragraph(

            "Residual Multi-Scale CNN with Grad-CAM",

            subtitle_style,

        )

    )



    # Summary table.

    summary_data = [

        ["Input image", filename],

        [

            "Original dimensions",

            f"{original_size[0]} × {original_size[1]} pixels",

        ],

        ["Predicted class", CLASS_NAMES[pred]],

        ["Model confidence", f"{confidence:.4f}%"],

        [

            "Normal Colon Tissue",

            f"{probabilities[0] * 100:.4f}%",

        ],

        [

            "Colon Adenocarcinoma",

            f"{probabilities[1] * 100:.4f}%",

        ],

    ]



    summary_table = Table(

        summary_data,

        colWidths=[145, 350],

        hAlign="LEFT",

    )



    summary_table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),

                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#334155")),

                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111827")),

                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),

                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),

                ("FONTSIZE", (0, 0), (-1, -1), 8.5),

                ("LEADING", (0, 0), (-1, -1), 11),

                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ec")),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("TOPPADDING", (0, 0), (-1, -1), 7),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),

            ]

        )

    )



    story.append(Paragraph("1. Analysis Summary", section_style))

    story.append(summary_table)



    # Visual evidence.

    story.append(Paragraph("2. Visual Evidence", section_style))

    story.append(

        Paragraph(

            "The following images show the original input, the Grad-CAM "

            "activation map, and the Grad-CAM overlay used to interpret "

            "the model prediction.",

            body_style,

        )

    )



    image_buffer = []

    for image in [original_image, heatmap_image, overlay_image]:

        img_io = BytesIO()

        image.save(img_io, format="PNG")

        img_io.seek(0)

        image_buffer.append(img_io)



    visual_table = Table(

        [

            [

                ReportLabImage(image_buffer[0], width=1.65 * inch, height=1.65 * inch),

                ReportLabImage(image_buffer[1], width=1.65 * inch, height=1.65 * inch),

                ReportLabImage(image_buffer[2], width=1.65 * inch, height=1.65 * inch),

            ],

            [

                Paragraph("Original image", small_style),

                Paragraph("Grad-CAM heatmap", small_style),

                Paragraph("Grad-CAM overlay", small_style),

            ],

        ],

        colWidths=[1.7 * inch, 1.7 * inch, 1.7 * inch],

        hAlign="CENTER",

    )



    visual_table.setStyle(

        TableStyle(

            [

                ("ALIGN", (0, 0), (-1, -1), "CENTER"),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("LEFTPADDING", (0, 0), (-1, -1), 4),

                ("RIGHTPADDING", (0, 0), (-1, -1), 4),

                ("TOPPADDING", (0, 0), (-1, -1), 4),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),

            ]

        )

    )



    story.append(visual_table)



    story.append(Paragraph("3. Original Image Description", section_style))

    story.append(Paragraph(original_description, body_style))



    story.append(Paragraph("4. Grad-CAM Heatmap Description", section_style))

    story.append(Paragraph(heatmap_description, body_style))



    story.append(Paragraph("5. Grad-CAM Overlay Description", section_style))

    story.append(Paragraph(overlay_description, body_style))



    story.append(PageBreak())



    story.append(Paragraph("6. Model and Inference Details", section_style))



    model_details = [

        ["Architecture", "Residual Multi-Scale CNN"],

        ["Model parameters", "2,156,322"],

        ["Classes", "colon_n (0), colon_aca (1)"],

        ["Input size", "224 × 224 RGB"],

        ["Preprocessing", "Resize → Tensor → ImageNet normalization"],

        [

            "Normalization mean",

            "[0.485, 0.456, 0.406]",

        ],

        [

            "Normalization standard deviation",

            "[0.229, 0.224, 0.225]",

        ],

        ["Inference device", "CPU"],

        ["Grad-CAM target layer", "model.block3"],

        ["Grad-CAM target class", CLASS_NAMES[pred]],

    ]



    model_table = Table(

        model_details,

        colWidths=[170, 325],

        hAlign="LEFT",

    )



    model_table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),

                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),

                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),

                ("FONTSIZE", (0, 0), (-1, -1), 8.5),

                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),

                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ec")),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("TOPPADDING", (0, 0), (-1, -1), 7),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),

            ]

        )

    )



    story.append(model_table)



    story.append(Paragraph("7. Quantitative Grad-CAM Analysis", section_style))



    metric_data = [

        ["Metric", "Value"],

        ["Mean activation", f"{metrics['mean']:.4f}"],

        ["Median activation", f"{metrics['median']:.4f}"],

        ["Standard deviation", f"{metrics['std']:.4f}"],

        ["Maximum normalized activation", f"{metrics['max']:.4f}"],

        [

            "Coverage ≥ 50%",

            f"{metrics['coverage50'] * 100:.2f}%",

        ],

        [

            "Coverage ≥ 80%",

            f"{metrics['coverage80'] * 100:.2f}%",

        ],

        [

            "Coverage ≥ 90%",

            f"{metrics['coverage90'] * 100:.2f}%",

        ],

        [

            "Top-20% activation concentration",

            f"{metrics['top20']:.4f}",

        ],

        [

            "Normalized activation entropy",

            f"{metrics['entropy']:.4f}",

        ],

    ]



    metric_table = Table(

        metric_data,

        colWidths=[340, 155],

        hAlign="LEFT",

    )



    metric_table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),

                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),

                ("FONTSIZE", (0, 0), (-1, -1), 8.5),

                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),

                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ec")),

                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                ("TOPPADDING", (0, 0), (-1, -1), 7),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),

            ]

        )

    )



    story.append(metric_table)



    story.append(Paragraph("8. Metric Interpretation", section_style))

    story.append(

        Paragraph(

            "Coverage values describe the fraction of heatmap pixels "

            "above each relative activation threshold. Top-20% activation "

            "concentration measures the proportion of total activation "

            "contained within the highest-activation 20% of pixels. "

            "Normalized activation entropy describes how broadly activation "

            "is distributed across the heatmap. These measurements are "

            "descriptive model-attribution statistics and are not validated "

            "clinical thresholds.",

            body_style,

        )

    )



    story.append(Paragraph("9. Overall Interpretation", section_style))

    story.append(

        Paragraph(

            f"The Residual Multi-Scale CNN predicts "

            f"<b>{CLASS_NAMES[pred]}</b> with "

            f"<b>{confidence:.4f}% model confidence</b>. "

            "Grad-CAM highlights image regions that influenced this "

            "prediction, while the overlay shows their locations on the "

            "original image. The prediction and explanation describe the "

            "behaviour of the trained model and do not independently "

            "establish pathological correctness.",

            body_style,

        )

    )



    story.append(Paragraph("10. Limitations and Research Disclaimer", section_style))

    story.append(

        Paragraph(

            "This system is an academic research prototype. The reported "

            "confidence is a model probability and is not a clinically "

            "calibrated certainty. The model may make incorrect predictions, "

            "including high-confidence errors. Grad-CAM is an attribution "

            "visualization; it is not a cell detector, tumour segmentation "

            "method, or independently verified pathological annotation. "

            "The system has not been clinically validated and must not be "

            "used to diagnose disease or guide patient care.",

            body_style,

        )

    )



    doc.build(story)



    buffer.seek(0)

    return buffer.getvalue()





# ============================================================

# INFERENCE

# ============================================================



def analyze(image):

    model = load_model()



    tensor = TRANSFORM(image).unsqueeze(0)



    with torch.no_grad():

        logits = model(tensor)



        if isinstance(logits, (tuple, list)):

            logits = logits[0]



        probs = (

            torch.softmax(logits, dim=1)[0]

            .cpu()

            .numpy()

        )



    predicted = int(np.argmax(probs))



    cam_engine = GradCAM(model)



    try:

        cam = cam_engine.generate(

            model,

            tensor,

            predicted,

        )

    finally:

        cam_engine.close()



    metrics = metrics_for(cam)



    rendered = render_images(

        image,

        cam,

    )



    return (

        predicted,

        probs,

        metrics,

        rendered,

    )









def render_image_analysis():


    # ============================================================

    # IMAGE UPLOAD

    # ============================================================



    upload = st.file_uploader(

        "Upload a histopathology image",

        type=["png", "jpg", "jpeg"],

    )



    st.caption("Upload a PNG or JPEG histopathology image to begin analysis.")



    if upload is None:

        return





    # ============================================================

    # OPEN IMAGE

    # ============================================================



    try:

        source_image = Image.open(upload)

        source_image.load()

        source_image = source_image.convert("RGB")

    except Exception:

        st.error(

            "This image could not be opened. "

            "Please upload a valid PNG or JPEG."

        )

        return





    # Small preview.

    st.image(

        source_image,

        caption="Selected image",

        width=210,

    )





    # ============================================================

    # ANALYZE BUTTON

    # ============================================================



    if st.button(

        "Analyze image",

        type="primary",

        use_container_width=True,

    ):

        with st.spinner("Checking image suitability..."):

            try:

                valid_input, gate_score, gate_reasons = validate_histology_input(source_image)



                if not valid_input:

                    st.session_state.pop("analysis", None)

                    st.session_state["invalid_input"] = {
                        "fingerprint": (upload.name, upload.size, upload.getvalue()),
                        "score": gate_score,
                        "reasons": gate_reasons,
                        "filename": upload.name,
                    }
                    st.toast("🚫 Invalid input: please upload a histopathology image.", icon="🚫")
                    st.error("🚫 Invalid input: please upload a histopathology image.")
                    with st.container(border=True):
                        st.markdown("### Invalid / Not a suitable histopathology image")
                        st.write("The uploaded image was rejected before CRC classification. No cancer prediction or Grad-CAM analysis was generated.")
                        st.metric("Input suitability score", f"{gate_score:.2f}")
                        if gate_reasons:
                            st.caption("Validation findings: " + "; ".join(gate_reasons[:3]) + ".")
                    return



                with st.spinner("Running CRC analysis..."):

                    result = analyze(source_image)



                st.session_state.pop("invalid_input", None)



                st.session_state["analysis"] = {

                    "fingerprint": (

                        upload.name,

                        upload.size,

                        upload.getvalue(),

                    ),

                    "result": result,

                    "filename": upload.name,

                    "dimensions": source_image.size,

                }



            except Exception as exc:

                st.error(

                    f"Analysis could not be completed: {exc}"

                )

                return





    # ============================================================

    # INVALID / OUT-OF-DISTRIBUTION INPUT

    # ============================================================



    invalid = st.session_state.get("invalid_input")



    if invalid:

        invalid_fingerprint = (

            upload.name,

            upload.size,

            upload.getvalue(),

        )



        if invalid.get("fingerprint") == invalid_fingerprint:

            st.divider()

            st.subheader("Analysis result")
            st.error("🚫 Invalid input: please upload a histopathology image.")
            with st.container(border=True):
                st.markdown("### Invalid / Not a suitable histopathology image")
                st.write(
                    "The uploaded image was rejected before CRC classification. "
                    "No cancer prediction or Grad-CAM interpretation was generated."
                )
                st.metric("Input suitability score", f"{invalid.get('score', 0.0):.2f}")



            reasons = invalid.get("reasons") or []

            if reasons:

                st.caption("Input validation findings: " + "; ".join(reasons[:3]) + ".")



            st.info(

                "Upload a microscope/histopathology image (for example, an H&E-stained tissue field) to run the CRC classifier. "

                "This validator is an input-suitability safeguard, not a clinical tissue-identification system."

            )

            return



    # ============================================================

    # DISPLAY SAVED ANALYSIS

    # ============================================================



    saved = st.session_state.get("analysis")



    if not saved:

        return



    current_fingerprint = (

        upload.name,

        upload.size,

        upload.getvalue(),

    )



    if saved["fingerprint"] != current_fingerprint:

        return



    (

        pred,

        probs,

        metrics,

        (original, heatmap, overlay),

    ) = saved["result"]



    confidence = float(

        probs[pred]

    ) * 100





    # ============================================================

    # RESULT
    # ============================================================

    st.divider()
    st.subheader("Analysis result")

    result_left, result_right = st.columns([3.2, 1.2], gap="large")

    with result_left:
        with st.container(border=True):
            st.caption("MODEL PREDICTION")
            st.markdown(f"### {CLASS_NAMES[pred]}")
            st.caption("Model confidence is not the same as clinical certainty.")

    with result_right:
        with st.container(border=True):
            st.metric("CONFIDENCE", f"{confidence:.2f}%")

    # VISUAL EXPLANATION
    st.subheader("Visual explanation")
    cols = st.columns(3, gap="medium")
    for col, title, image in zip(
        cols,
        ["Original image", "Grad-CAM heatmap", "Grad-CAM overlay"],
        [original, heatmap, overlay],
    ):
        with col:
            st.caption(title)
            st.image(image, width=210)

    original_description = describe_original()
    heatmap_description = describe_heatmap(metrics)
    overlay_description = describe_overlay()

    with st.container(border=True):
        st.markdown("**Original image**")
        st.write(original_description)
    with st.container(border=True):
        st.markdown("**Grad-CAM heatmap**")
        st.write(heatmap_description)
    with st.container(border=True):
        st.markdown("**Grad-CAM overlay**")
        st.write(overlay_description)

    # CLASS PROBABILITIES
    st.subheader("Class probabilities")
    for label, probability in zip(CLASS_NAMES, probs):
        probability_pct = float(probability) * 100.0
        left, right = st.columns([5, 1])
        with left:
            st.write(label)
        with right:
            st.write(f"{probability_pct:.4f}%")
        st.progress(float(probability))

    # MODEL INTERPRETATION
    st.subheader("Model interpretation")

    if pred == 1:
        interpretation = (
            "The model classified this image as **Colon Adenocarcinoma**. "
            "The Grad-CAM visualization highlights tissue regions that contributed most strongly "
            "to this prediction. These highlighted regions represent model attribution and should "
            "not be interpreted as confirmed cancer cells, a tumour boundary, or a clinical diagnosis."
        )
    else:
        interpretation = (
            "The model classified this image as **Normal Colon Tissue**. "
            "The Grad-CAM visualization highlights tissue regions that contributed most strongly "
            "to this prediction. These highlighted regions represent model attribution and should "
            "not be interpreted as confirmed normal tissue findings or a clinical diagnosis."
        )

    with st.container(border=True):
        st.markdown("**🎯 Why did the model make this prediction?**")
        st.write(interpretation)

    interpretation_cols = st.columns(3)
    with interpretation_cols[0]:
        st.metric("Prediction", CLASS_NAMES[pred])
    with interpretation_cols[1]:
        st.metric("Model confidence", f"{confidence:.2f}%")
    with interpretation_cols[2]:
        st.metric("Explanation method", "Grad-CAM")

    with st.container(border=True):
        st.markdown("**🔥 How to read the Grad-CAM**")
        st.write(
            "Warmer regions indicate relatively stronger model activation, while cooler regions "
            "indicate lower relative activation. The overlay connects these highlighted regions "
            "to the original tissue image so the viewer can see where the model's decision was "
            "influenced. Grad-CAM explains model behaviour; it is not cell segmentation or a clinical finding."
        )

    st.info(
        "The detailed numerical Grad-CAM measurements are retained in the technical TXT/PDF report "
        "for research documentation, but are intentionally hidden from the main demo interface."
    )

    # SUMMARY
    st.subheader("Summary")
    with st.container(border=True):
        st.write(
            f"The model predicts **{CLASS_NAMES[pred]}** with **{confidence:.2f}%** model confidence. "
            "The Grad-CAM heatmap and overlay show the image regions that contributed to this prediction. "
            "This explanation describes model behaviour and is not a clinical diagnosis."
        )


    # TECHNICAL REPORT DOWNLOAD

    # ============================================================



    report = report_text(

        saved["filename"],

        saved["dimensions"],

        pred,

        probs,

        metrics,

        original_description,

        heatmap_description,

        overlay_description,

    )



    report_filename = Path(saved["filename"]).stem



    st.download_button(

        "Download technical report (TXT)",

        data=report,

        file_name=f"{report_filename}_technical_report.txt",

        mime="text/plain",

        use_container_width=True,

    )



    pdf_report = create_pdf_report(

        saved["filename"],

        saved["dimensions"],

        pred,

        probs,

        metrics,

        original_description,

        heatmap_description,

        overlay_description,

        original,

        heatmap,

        overlay,

    )



    st.download_button(

        "Download technical report (PDF)",

        data=pdf_report,

        file_name=f"{report_filename}_technical_report.pdf",

        mime="application/pdf",

        use_container_width=True,

    )





    # ============================================================

    # DISCLAIMER

    # ============================================================



    st.caption("Research prototype · Not for clinical diagnosis")





# ============================================================

# MODEL COMPARISON TAB

# ============================================================



def architecture_card(name, subtitle, lines):
    """Render architecture diagram using native Streamlit components."""
    with st.container(border=True):
        st.markdown(f"### {name}")
        st.caption(subtitle)
        for i, line in enumerate(lines):
            st.markdown(f"**{line}**")
            if i < len(lines) - 1:
                st.caption("↓")


def render_model_comparison():

    st.header("Model Comparison & Architecture Analysis")

    st.caption("Primary group-aware LC25000 colon classification experiments. InceptionV3 is intentionally excluded.")

    st.subheader("Model Comparison Highlights")



    models = [

        "Basic CNN", "VGG16", "ResNet18", "DenseNet121",

        "EfficientNet-B0", "GhostNet", "Residual Multi-Scale CNN (Proposed)"

    ]



    performance = {

        "Model": models,

        "Accuracy (%)": [92.98, 100.00, 99.93, 100.00, 100.00, 100.00, 99.9338],

        "Precision (%)": [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.8708],

        "Sensitivity (%)": [86.29, 100.00, 99.87, 100.00, 100.00, 100.00, 100.00],

        "Specificity (%)": [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.8643],

        "F1 (%)": [92.64, 100.00, 99.94, 100.00, 100.00, 100.00, 99.9354],

        "ROC-AUC (%)": [99.90, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99982447],

    }

    import pandas as pd

    import matplotlib.pyplot as plt

    df = pd.DataFrame(performance)



    st.subheader("1. Models Evaluated")

    st.dataframe(df[["Model"]], use_container_width=True, hide_index=True)

    st.info("The primary comparison contains seven CNN classifiers. ResNet18 Split B, handcrafted SVM, fusion experiments, and InceptionV3 are excluded from this primary table to keep the comparison scientifically consistent.")



    st.subheader("2. Architecture Comparison")

    architecture_rows = [

        ["Basic CNN", "Sequential convolutional blocks", "No", "No", "Yes"],

        ["VGG16", "Stacked 3×3 convolution blocks", "No", "No", "No"],

        ["ResNet18", "Residual blocks with skip connections", "Residual", "No", "Moderate"],

        ["DenseNet121", "Dense feature reuse across layers", "Dense", "No", "Moderate"],

        ["EfficientNet-B0", "MBConv-based efficient CNN", "Block-level", "Limited", "Yes"],

        ["GhostNet", "Ghost feature generation", "Block-level", "Limited", "Yes"],

        ["Residual Multi-Scale CNN", "Parallel multi-scale residual blocks", "Residual", "Yes", "Yes"],

    ]

    arch_df = pd.DataFrame(architecture_rows, columns=["Model","Core design","Connection strategy","Multi-scale processing","Efficiency focus"])

    st.dataframe(arch_df, use_container_width=True, hide_index=True)



    st.subheader("3. Architecture Visualization")

    selected = st.selectbox("Select a model", models, index=6)

    diagrams = {

        "Basic CNN": ("Sequential feature extraction", ["Input 224×224×3", "Conv + BN + ReLU + Pool", "Conv + BN + ReLU + Pool", "Conv + BN + ReLU + Pool", "Conv + BN + ReLU", "Global Average Pool → Dropout → FC(2)"]),

        "VGG16": ("Deep stacked 3×3 convolution architecture", ["Input 224×224×3", "VGG convolution blocks", "Repeated 3×3 Conv + ReLU", "Max Pooling between blocks", "Adaptive/flattened classifier head", "FC → 2 classes"]),

        "ResNet18": ("Residual learning", ["Input 224×224×3", "Stem convolution + pooling", "Residual Block × 2", "Residual Block × 2", "Residual Block × 2", "Residual Block × 2", "Global Average Pool → FC(2)"]),

        "DenseNet121": ("Dense feature reuse", ["Input 224×224×3", "Stem convolution", "Dense Block 1", "Transition 1", "Dense Blocks 2–4 + transitions", "Global Average Pool → Classifier(2)"]),

        "EfficientNet-B0": ("Efficient MBConv-based design", ["Input 224×224×3", "Stem convolution", "MBConv stages", "Squeeze-and-Excitation inside blocks", "Final convolution + global pooling", "Dropout → FC(2)"]),

        "GhostNet": ("Efficient Ghost feature generation", ["Input 224×224×3", "Stem convolution", "Ghost bottleneck stages", "Ghost feature generation + cheap operations", "Final convolution + global pooling", "Classifier → 2 classes"]),

        "Residual Multi-Scale CNN (Proposed)": ("Parallel multi-scale residual feature extraction", ["Input 224×224×3", "Stem: Conv 3×3, 3→32", "Block 1: 3×3 + 5×5 + Dilated 3×3 → 1×1 fusion + residual", "Block 2: 64→128, stride 2", "Block 3: 128→256, stride 2", "Global Average Pool → Dropout 0.3", "FC 256→2"]),

    }

    subtitle, lines = diagrams[selected]

    architecture_card(selected, subtitle, lines)

    if selected == "Residual Multi-Scale CNN (Proposed)":

        st.success("Proposed architecture: multi-scale convolution branches are fused and combined with a residual shortcut before activation.")



    st.subheader("4. Classification Performance Comparison")

    st.dataframe(df.style.format({c:"{:.4f}" for c in df.columns[1:]}), use_container_width=True, hide_index=True)



    st.markdown("**Accuracy Comparison**")

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(df["Model"], df["Accuracy (%)"])

    ax.set_ylim(85, 101)

    ax.set_ylabel("Accuracy (%)")

    ax.tick_params(axis="x", rotation=25)

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)



    st.markdown("**Overall Performance Metrics**")

    metric_cols = ["Accuracy (%)", "Precision (%)", "Sensitivity (%)", "Specificity (%)", "F1 (%)"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(df))

    width = 0.16

    for i, col in enumerate(metric_cols):

        ax.bar([v + (i-2)*width for v in x], df[col], width, label=col.replace(" (%)", ""))

    ax.set_xticks(list(x))

    ax.set_xticklabels(df["Model"], rotation=25, ha="right")

    ax.set_ylim(80, 101)

    ax.set_ylabel("Score (%)")

    ax.legend(ncol=3, fontsize=8)

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)



    st.markdown("**ROC-AUC Comparison**")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(df["Model"], df["ROC-AUC (%)"])

    ax.set_ylim(98, 100.2)

    ax.set_ylabel("ROC-AUC (%)")

    ax.tick_params(axis="x", rotation=25)

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)



    st.subheader("5. Parameter Efficiency & Computational Complexity")

    complexity = pd.DataFrame([

        ["Basic CNN", 93730, 0.505774336, 1.011548672, 15.475516, 64.6182, "Measured"],

        ["ResNet18", 11177538, 1.813562368, 3.627124736, 27.807094, 35.9620462, "Measured"],

        ["Residual Multi-Scale CNN", 2156322, 3.896869376, 7.793738752, 62.067296, 16.1115445, "Measured"],

    ], columns=["Model","Parameters","MACs (G)","FLOPs (G)","Inference (ms)","Images/s","Status"])

    st.dataframe(complexity.style.format({"Parameters":"{:,.0f}","MACs (G)":"{:.4f}","FLOPs (G)":"{:.4f}","Inference (ms)":"{:.3f}","Images/s":"{:.2f}"}), use_container_width=True, hide_index=True)

    st.caption("Complexity values are shown only where they were measured or directly verified. Missing values are intentionally not estimated.")



    st.markdown("**Parameter Count Comparison — All Primary Models**")

    # Parameter counts are shown for every primary model where the exact
    # project-specific count is available. GhostNet is intentionally left
    # unestimated because its project-specific parameter result was not
    # recorded in the complexity run.
    parameter_comparison = pd.DataFrame([
        ["Basic CNN", 93730],
        ["VGG16", 134268738],
        ["ResNet18", 11177538],
        ["DenseNet121", 6955906],
        ["EfficientNet-B0", 4010110],
        ["GhostNet", None],
        ["Residual Multi-Scale CNN", 2156322],
    ], columns=["Model", "Parameters"])

    plot_df = parameter_comparison.dropna(subset=["Parameters"]).copy()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(
        plot_df["Model"],
        plot_df["Parameters"] / 1e6,
    )

    ax.set_ylabel("Parameters (millions)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)

    # Display exact parameter values above every measured bar.
    for bar, value in zip(bars, plot_df["Parameters"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    # GhostNet remains part of the primary model set, but no project-specific
    # parameter count was recorded, so it is explicitly marked as N/A rather
    # than assigning an external or estimated value.
    ghost_index = list(parameter_comparison["Model"]).index("GhostNet")
    ax.text(
        ghost_index,
        0.15,
        "N/A",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(range(len(parameter_comparison)))
    ax.set_xticklabels(parameter_comparison["Model"])
    ax.set_xlim(-0.6, len(parameter_comparison) - 0.4)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Parameter counts are shown for all primary models where project-specific values are available. "
        "GhostNet is marked N/A because its project-specific parameter count was not recorded; no external value is substituted."
    )



    st.subheader("6. Architecture–Performance Interpretation")

    st.markdown("""

    **Basic CNN:** The shallow sequential design is computationally light, but its limited representation capacity is reflected in lower sensitivity and F1-score.



    **VGG16:** Uses deep stacked 3×3 convolutions and achieves perfect performance on the evaluated test set, but has a substantially larger parameter count than the proposed model.



    **ResNet18:** Residual skip connections support strong optimization and near-perfect classification with 11.18M parameters.



    **DenseNet121:** Dense feature reuse provides strong representation capability and achieves perfect performance on the evaluated test set.



    **EfficientNet-B0 and GhostNet:** These architectures emphasize efficient feature extraction and achieve perfect performance in the evaluated experiment.



    **Residual Multi-Scale CNN — Proposed:** Combines residual learning with parallel 3×3, 5×5, and dilated 3×3 branches. It achieves 99.93% accuracy and 99.94% F1-score using 2.16M parameters. Compared with ResNet18, this is approximately 80.7% fewer parameters, although the parallel branches increase FLOPs and CPU inference time.

    """)



    st.subheader("⭐ Proposed Model Highlight")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Accuracy", "99.93%")

    c2.metric("F1 Score", "99.94%")

    c3.metric("Sensitivity", "100.00%")

    c4.metric("ROC-AUC", "99.9998%")

    c5.metric("Parameters", "2.16M")

    st.info("The proposed model is not presented as the highest-accuracy model. Its contribution is the combination of multi-scale residual feature extraction and a favorable parameter-efficiency trade-off.")





# ============================================================

# APPLICATION TABS

# ============================================================



st.title("CRC Histopathology Analysis")

st.caption("Explainable image classification for colorectal histopathology research")



analysis_tab, comparison_tab = st.tabs(["🔬 Image Analysis", "📊 Model Comparison"])



with analysis_tab:

    render_image_analysis()



with comparison_tab:

    render_model_comparison()
