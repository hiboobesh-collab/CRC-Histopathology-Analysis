from pathlib import Path
import sys
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

from src.models.residual_multiscale_cnn import (
    build_residual_multiscale_cnn
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/classification/"
    "residual_multiscale_cnn_groupaware/"
    "best/residual_multiscale_cnn_best.pth"
)

OUTPUT_ROOT = Path("results/inference")

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "colon_n",
    1: "colon_aca",
}

CLASS_DISPLAY_NAMES = {
    0: "Normal Colon Tissue",
    1: "Colon Adenocarcinoma",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Exact validation/test preprocessing used during evaluation
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# UTILITIES
# ============================================================

def load_model():

    print("\nLoading Residual Multi-Scale CNN...")

    model = build_residual_multiscale_cnn(
        num_classes=2,
        dropout=0.3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def denormalize_image(tensor):

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device
    ).view(3, 1, 1)

    image = tensor * std + mean

    return torch.clamp(image, 0, 1)


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image):

    image_tensor = TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    return (
        image_tensor,
        probabilities[0].cpu().numpy(),
        predicted_class,
        confidence
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activation = None
        self.gradient = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activation = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradient = grad_output[0]

    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image_tensor
        )

        target_score = logits[
            0,
            target_class
        ]

        target_score.backward()

        activation = self.activation
        gradient = self.gradient

        if activation is None:
            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if gradient is None:
            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )

        # Global average pooling of gradients
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam.detach().cpu().numpy()

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# QUANTITATIVE GRAD-CAM
# ============================================================

def calculate_gradcam_metrics(cam):

    flat = cam.flatten()

    mean_activation = float(
        np.mean(flat)
    )

    median_activation = float(
        np.median(flat)
    )

    std_activation = float(
        np.std(flat)
    )

    max_activation = float(
        np.max(flat)
    )

    coverage_50 = float(
        np.mean(cam >= 0.50)
    )

    coverage_80 = float(
        np.mean(cam >= 0.80)
    )

    coverage_90 = float(
        np.mean(cam >= 0.90)
    )

    # Top 20% activation concentration
    sorted_values = np.sort(flat)

    top_count = max(
        1,
        int(
            len(sorted_values) * 0.20
        )
    )

    top20_sum = np.sum(
        sorted_values[-top_count:]
    )

    total_sum = np.sum(
        sorted_values
    )

    if total_sum > 0:
        top20_concentration = (
            top20_sum / total_sum
        )
    else:
        top20_concentration = 0.0

    # Normalized activation entropy
    positive = flat + 1e-12

    probabilities = (
        positive /
        np.sum(positive)
    )

    entropy = -np.sum(
        probabilities *
        np.log(
            probabilities
        )
    )

    max_entropy = np.log(
        len(probabilities)
    )

    normalized_entropy = (
        entropy /
        max_entropy
    )

    return {
        "mean_activation": mean_activation,
        "median_activation": median_activation,
        "std_activation": std_activation,
        "max_activation": max_activation,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "top20_activation_concentration":
            float(top20_concentration),
        "normalized_activation_entropy":
            float(normalized_entropy),
    }


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def save_gradcam_visualization(
    image,
    cam,
    output_path
):

    image_array = np.asarray(
        image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
    )

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image_array
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )

    # Heatmap
    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis(
        "off"
    )

    # Overlay
    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        image_array
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        "Grad-CAM Overlay"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FLORENCE-2
# ============================================================

def generate_florence_description(image):

    try:

        from transformers import (
            AutoProcessor,
            AutoModelForCausalLM
        )

        model_name = (
            "microsoft/Florence-2-base-ft"
        )

        print(
            "\nLoading Florence-2..."
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )

        florence_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )

        florence_model.to(DEVICE)
        florence_model.eval()

        task_prompt = (
            "<MORE_DETAILED_CAPTION>"
        )

        inputs = processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():

            generated_ids = (
                florence_model.generate(
                    input_ids=inputs.get(
                        "input_ids"
                    ),
                    pixel_values=inputs.get(
                        "pixel_values"
                    ),
                    max_new_tokens=100,
                    num_beams=1
                )
            )

        generated_text = (
            processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
        )

        try:

            parsed = (
                processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=image.size
                )
            )

            if isinstance(
                parsed,
                dict
            ):

                description = str(
                    parsed.get(
                        task_prompt,
                        parsed
                    )
                )

            else:

                description = str(
                    parsed
                )

        except Exception:

            description = generated_text

        return description.strip()

    except Exception as error:

        return (
            "Florence-2 auxiliary description "
            f"was unavailable: {error}"
        )


# ============================================================
# GROUNDED REPORT
# ============================================================

def create_report(
    image_path,
    probabilities,
    predicted_class,
    confidence,
    metrics,
    florence_description
):

    normal_probability = (
        probabilities[0] * 100
    )

    cancer_probability = (
        probabilities[1] * 100
    )

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "CRC HISTOPATHOLOGY AI INFERENCE REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Input image       : {image_path.name}"
    )

    report.append(
        f"Device            : {DEVICE}"
    )

    report.append("")

    report.append(
        "MODEL"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Model             : Residual Multi-Scale CNN"
    )

    report.append(
        "Parameters        : 2,156,322"
    )

    report.append("")

    report.append(
        "PREDICTION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Predicted class   : "
        f"{CLASS_DISPLAY_NAMES[predicted_class]}"
    )

    report.append(
        f"Confidence        : {confidence * 100:.4f}%"
    )

    report.append("")

    report.append(
        "Class probabilities:"
    )

    report.append(
        f"  Normal colon tissue       : "
        f"{normal_probability:.4f}%"
    )

    report.append(
        f"  Colon adenocarcinoma     : "
        f"{cancer_probability:.4f}%"
    )

    report.append("")

    report.append(
        "GRAD-CAM"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Target layer      : block3"
    )

    report.append(
        f"Mean activation   : "
        f"{metrics['mean_activation']:.4f}"
    )

    report.append(
        f"Median activation : "
        f"{metrics['median_activation']:.4f}"
    )

    report.append(
        f"Std activation    : "
        f"{metrics['std_activation']:.4f}"
    )

    report.append(
        f"Coverage >= 50%   : "
        f"{metrics['coverage_50'] * 100:.2f}%"
    )

    report.append(
        f"Coverage >= 80%   : "
        f"{metrics['coverage_80'] * 100:.2f}%"
    )

    report.append(
        f"Coverage >= 90%   : "
        f"{metrics['coverage_90'] * 100:.2f}%"
    )

    report.append(
        f"Top-20 concentration : "
        f"{metrics['top20_activation_concentration'] * 100:.2f}%"
    )

    report.append(
        f"Normalized entropy : "
        f"{metrics['normalized_activation_entropy']:.4f}"
    )

    report.append("")

    report.append(
        "FLORENCE-2 AUXILIARY DESCRIPTION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        florence_description
    )

    report.append("")

    report.append(
        "GROUNDED AI INTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    if predicted_class == 1:

        report.append(
            "The Residual Multi-Scale CNN predicts "
            "colon adenocarcinoma for the supplied image "
            f"with {confidence * 100:.2f}% confidence."
        )

    else:

        report.append(
            "The Residual Multi-Scale CNN predicts "
            "normal colon tissue for the supplied image "
            f"with {confidence * 100:.2f}% confidence."
        )

    report.append(
        "Grad-CAM highlights image regions that "
        "contributed to the model prediction."
    )

    report.append(
        "The quantitative Grad-CAM measurements "
        "describe the spatial distribution and "
        "activation strength of the model's visual evidence."
    )

    report.append(
        "Florence-2 is used only as an auxiliary "
        "visual-description component and does not "
        "change the CNN prediction."
    )

    report.append("")

    report.append(
        "IMPORTANT RESEARCH NOTE"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "This is an AI-assisted research result."
    )

    report.append(
        "Grad-CAM identifies regions influential "
        "to the model prediction; it does not establish "
        "specific pathological structures."
    )

    report.append(
        "The system is not a clinical diagnostic tool "
        "and should not replace expert pathological assessment."
    )

    report.append(
        "=" * 70
    )

    return "\n".join(report)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python -m src.inference.predict_single_image '
            '"path\\to\\image.jpeg"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"\nERROR: Image not found:\n"
            f"{image_path}"
        )

        sys.exit(1)

    if not MODEL_PATH.exists():

        print(
            f"\nERROR: Model checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

        sys.exit(1)

    print(
        "\n" + "=" * 70
    )

    print(
        "CRC HISTOPATHOLOGY SINGLE-IMAGE INFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        f"Image : {image_path}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"Original image size: {image.size}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    image_tensor, probabilities, predicted_class, confidence = (
        predict(
            model,
            image
        )
    )

    print("\nPrediction:")
    print(
        f"  Class      : "
        f"{CLASS_DISPLAY_NAMES[predicted_class]}"
    )

    print(
        f"  Confidence : "
        f"{confidence * 100:.4f}%"
    )

    print(
        f"  Normal     : "
        f"{probabilities[0] * 100:.4f}%"
    )

    print(
        f"  Colon ACA  : "
        f"{probabilities[1] * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT /
        image_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print(
        "\nGenerating Grad-CAM..."
    )

    gradcam = GradCAM(
        model,
        model.block3
    )

    cam = gradcam.generate(
        image_tensor,
        predicted_class
    )

    gradcam.close()

    metrics = calculate_gradcam_metrics(
        cam
    )

    print(
        "Grad-CAM generated."
    )

    print(
        f"  Mean activation : "
        f"{metrics['mean_activation']:.4f}"
    )

    print(
        f"  Coverage >= 50% : "
        f"{metrics['coverage_50'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save Grad-CAM
    # --------------------------------------------------------

    gradcam_path = (
        output_dir /
        "gradcam.png"
    )

    save_gradcam_visualization(
        image,
        cam,
        gradcam_path
    )

    # --------------------------------------------------------
    # Florence-2
    # --------------------------------------------------------

    print(
        "\nGenerating auxiliary Florence-2 description..."
    )

    florence_description = (
        generate_florence_description(
            image
        )
    )

    print(
        "\nFlorence-2 description:"
    )

    print(
        florence_description
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = create_report(
        image_path,
        probabilities,
        predicted_class,
        confidence,
        metrics,
        florence_description
    )

    report_path = (
        output_dir /
        "report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_data = {

        "input_image": str(
            image_path
        ),

        "model": {
            "name":
                "Residual Multi-Scale CNN",

            "parameters":
                2156322,

            "checkpoint":
                str(MODEL_PATH),

            "device":
                str(DEVICE),
        },

        "prediction": {

            "class_index":
                predicted_class,

            "class":
                CLASS_NAMES[predicted_class],

            "display_name":
                CLASS_DISPLAY_NAMES[
                    predicted_class
                ],

            "confidence":
                confidence,

            "probabilities": {

                "colon_n":
                    float(probabilities[0]),

                "colon_aca":
                    float(probabilities[1]),
            }
        },

        "gradcam": {

            "target_layer":
                "block3",

            **metrics,

            "visualization":
                str(gradcam_path),
        },

        "florence_2": {

            "description":
                florence_description,

            "role":
                "auxiliary_visual_description_only",
        },

        "research_note":
            "AI-assisted research result; "
            "not a clinical diagnosis."
    }

    json_path = (
        output_dir /
        "prediction.json"
    )

    json_path.write_text(
        json.dumps(
            json_data,
            indent=4
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INFERENCE COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Output directory:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        f"\nGrad-CAM:"
    )

    print(
        f"  {gradcam_path}"
    )

    print(
        f"\nReport:"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"\nJSON:"
    )

    print(
        f"  {json_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()