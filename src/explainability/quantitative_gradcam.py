from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import ColonHistopathologyDataset
from src.data.transforms import val_test_transform
from src.models.residual_multiscale_cnn import (
    build_residual_multiscale_cnn,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_CSV = Path(
    "data/splits/group_test.csv"
)

MODEL_PATH = Path(
    "models/classification/"
    "residual_multiscale_cnn_groupaware/"
    "best/residual_multiscale_cnn_best.pth"
)

OUTPUT_DIR = Path(
    "results/explainability/"
    "residual_multiscale_cnn"
)

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    0: "colon_n",
    1: "colon_aca",
}


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(image)

        score = output[:, target_class]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                image.shape[2],
                image.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam, output

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# FIND REPRESENTATIVE EXAMPLES
# ============================================================

def find_examples(
    model,
    dataset
):

    examples = {
        "correct_colon_n": None,
        "correct_colon_aca": None,
        "false_positive": None,
        "false_negative": None,
    }

    model.eval()

    for index in range(
        len(dataset)
    ):

        image, label = dataset[index]

        input_tensor = (
            image
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = int(
                probabilities.argmax(
                    dim=1
                ).item()
            )

            probability = float(
                probabilities[
                    0,
                    prediction
                ].item()
            )

        if (
            label == 0
            and prediction == 0
            and examples["correct_colon_n"] is None
        ):

            examples["correct_colon_n"] = (
                index,
                label,
                prediction,
                probability
            )

        elif (
            label == 1
            and prediction == 1
            and examples["correct_colon_aca"] is None
        ):

            examples["correct_colon_aca"] = (
                index,
                label,
                prediction,
                probability
            )

        elif (
            label == 0
            and prediction == 1
            and examples["false_positive"] is None
        ):

            examples["false_positive"] = (
                index,
                label,
                prediction,
                probability
            )

        elif (
            label == 1
            and prediction == 0
            and examples["false_negative"] is None
        ):

            examples["false_negative"] = (
                index,
                label,
                prediction,
                probability
            )

        if all(
            value is not None
            for value in examples.values()
        ):

            break

    return examples


# ============================================================
# QUANTITATIVE CAM METRICS
# ============================================================

def calculate_cam_metrics(cam):

    # --------------------------------------------------------
    # Basic activation statistics
    # --------------------------------------------------------

    mean_activation = float(
        np.mean(cam)
    )

    max_activation = float(
        np.max(cam)
    )

    median_activation = float(
        np.median(cam)
    )

    std_activation = float(
        np.std(cam)
    )

    # --------------------------------------------------------
    # High activation regions
    # --------------------------------------------------------

    coverage_50 = float(
        np.mean(
            cam >= 0.50
        )
    )

    coverage_80 = float(
        np.mean(
            cam >= 0.80
        )
    )

    coverage_90 = float(
        np.mean(
            cam >= 0.90
        )
    )

    # --------------------------------------------------------
    # Activation concentration
    #
    # Fraction of total activation contained
    # in the top 20% most activated pixels.
    # --------------------------------------------------------

    flattened = cam.flatten()

    sorted_values = np.sort(
        flattened
    )[::-1]

    top_count = max(
        1,
        int(
            len(sorted_values) * 0.20
        )
    )

    total_activation = (
        np.sum(sorted_values)
    )

    top20_activation = (
        np.sum(
            sorted_values[:top_count]
        )
    )

    if total_activation > 0:

        concentration_top20 = (
            top20_activation /
            total_activation
        )

    else:

        concentration_top20 = 0.0

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    probabilities = (
        flattened /
        (np.sum(flattened) + 1e-12)
    )

    entropy = -np.sum(
        probabilities *
        np.log(
            probabilities + 1e-12
        )
    )

    normalized_entropy = (
        entropy /
        np.log(len(probabilities))
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
            float(concentration_top20),
        "normalized_activation_entropy":
            float(normalized_entropy),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "QUANTITATIVE GRAD-CAM ANALYSIS"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = build_residual_multiscale_cnn(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Checkpoint : {MODEL_PATH}"
    )

    print(
        f"Epoch      : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ColonHistopathologyDataset(
        csv_path=TEST_CSV,
        transform=val_test_transform
    )

    print(
        f"Test samples : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Grad-CAM target
    # --------------------------------------------------------

    target_layer = model.block3

    gradcam = GradCAM(
        model,
        target_layer
    )

    print(
        "Target layer : model.block3"
    )

    # --------------------------------------------------------
    # Find examples
    # --------------------------------------------------------

    examples = find_examples(
        model,
        dataset
    )

    results = []

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    for category, result in examples.items():

        if result is None:

            print(
                f"[NOT FOUND] {category}"
            )

            continue

        index, true_class, predicted_class, probability = result

        image, label = dataset[index]

        input_tensor = (
            image
            .unsqueeze(0)
            .to(DEVICE)
        )

        cam, output = gradcam.generate(
            input_tensor,
            predicted_class
        )

        metrics = calculate_cam_metrics(
            cam
        )

        row = {
            "category": category,
            "index": index,
            "true_class":
                CLASS_NAMES[true_class],
            "predicted_class":
                CLASS_NAMES[predicted_class],
            "confidence":
                probability,
        }

        row.update(
            metrics
        )

        results.append(row)

        print()
        print(
            "-" * 70
        )

        print(
            f"Category       : {category}"
        )

        print(
            f"True class     : "
            f"{CLASS_NAMES[true_class]}"
        )

        print(
            f"Predicted      : "
            f"{CLASS_NAMES[predicted_class]}"
        )

        print(
            f"Confidence     : "
            f"{probability:.4f}"
        )

        print(
            f"Mean CAM       : "
            f"{metrics['mean_activation']:.4f}"
        )

        print(
            f"Max CAM        : "
            f"{metrics['max_activation']:.4f}"
        )

        print(
            f"Coverage @ 0.5 : "
            f"{metrics['coverage_50'] * 100:.2f}%"
        )

        print(
            f"Coverage @ 0.8 : "
            f"{metrics['coverage_80'] * 100:.2f}%"
        )

        print(
            f"Coverage @ 0.9 : "
            f"{metrics['coverage_90'] * 100:.2f}%"
        )

        print(
            f"Top-20% concentration : "
            f"{metrics['top20_activation_concentration']:.4f}"
        )

        print(
            f"Normalized entropy    : "
            f"{metrics['normalized_activation_entropy']:.4f}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_csv = (
        OUTPUT_DIR /
        "quantitative_gradcam_metrics.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save readable report
    # --------------------------------------------------------

    output_txt = (
        OUTPUT_DIR /
        "quantitative_gradcam_report.txt"
    )

    with open(
        output_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "QUANTITATIVE GRAD-CAM ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        for row in results:

            f.write(
                f"Category: {row['category']}\n"
            )

            f.write(
                f"True class: {row['true_class']}\n"
            )

            f.write(
                f"Predicted class: "
                f"{row['predicted_class']}\n"
            )

            f.write(
                f"Confidence: "
                f"{row['confidence']:.4f}\n"
            )

            f.write(
                f"Mean activation: "
                f"{row['mean_activation']:.4f}\n"
            )

            f.write(
                f"Max activation: "
                f"{row['max_activation']:.4f}\n"
            )

            f.write(
                f"Coverage @ 0.5: "
                f"{row['coverage_50'] * 100:.2f}%\n"
            )

            f.write(
                f"Coverage @ 0.8: "
                f"{row['coverage_80'] * 100:.2f}%\n"
            )

            f.write(
                f"Coverage @ 0.9: "
                f"{row['coverage_90'] * 100:.2f}%\n"
            )

            f.write(
                f"Top-20% concentration: "
                f"{row['top20_activation_concentration']:.4f}\n"
            )

            f.write(
                f"Normalized entropy: "
                f"{row['normalized_activation_entropy']:.4f}\n"
            )

            f.write(
                "\n" + "-" * 70 + "\n\n"
            )

    gradcam.close()

    print()
    print("=" * 70)
    print(
        "QUANTITATIVE GRAD-CAM COMPLETE"
    )
    print("=" * 70)

    print(
        f"CSV report : {output_csv}"
    )

    print(
        f"Text report: {output_txt}"
    )


if __name__ == "__main__":
    main()