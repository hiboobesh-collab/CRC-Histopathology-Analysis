from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

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
                self._save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def _save_gradient(
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

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Remove negative activations
        cam = F.relu(cam)

        # Resize CAM to input image size
        cam = F.interpolate(
            cam,
            size=(
                image.shape[2],
                image.shape[3]
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, output

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# IMAGE NORMALIZATION FOR DISPLAY
# ============================================================

def denormalize_image(tensor):

    image = tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    # ImageNet normalization
    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0,
        1
    )

    return image


# ============================================================
# GENERATE GRAD-CAM FIGURE
# ============================================================

def save_gradcam_figure(
    original_image,
    cam,
    true_class,
    predicted_class,
    probability,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        original_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(
        original_image
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        "Grad-CAM Overlay"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"True: {CLASS_NAMES[true_class]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | "
        f"Probability: {probability:.4f}",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print(
        "LOADING RESIDUAL MULTI-SCALE CNN"
    )
    print("=" * 70)

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
        f"Device           : {DEVICE}"
    )

    print(
        f"Checkpoint       : {MODEL_PATH}"
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation loss  : "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )

    print(
        f"Validation acc   : "
        f"{checkpoint.get('val_accuracy', 'unknown')}"
    )

    return model


# ============================================================
# FIND EXAMPLES
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

        # Correct colon_n
        if (
            label == 0
            and prediction == 0
            and examples[
                "correct_colon_n"
            ] is None
        ):

            examples[
                "correct_colon_n"
            ] = (
                index,
                label,
                prediction,
                probability
            )

        # Correct colon_aca
        elif (
            label == 1
            and prediction == 1
            and examples[
                "correct_colon_aca"
            ] is None
        ):

            examples[
                "correct_colon_aca"
            ] = (
                index,
                label,
                prediction,
                probability
            )

        # False positive
        elif (
            label == 0
            and prediction == 1
            and examples[
                "false_positive"
            ] is None
        ):

            examples[
                "false_positive"
            ] = (
                index,
                label,
                prediction,
                probability
            )

        # False negative
        elif (
            label == 1
            and prediction == 0
            and examples[
                "false_negative"
            ] is None
        ):

            examples[
                "false_negative"
            ] = (
                index,
                label,
                prediction,
                probability
            )

        # Stop if all categories found
        if all(
            value is not None
            for value in examples.values()
        ):

            break

    return examples


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RESIDUAL MULTI-SCALE CNN - GRAD-CAM"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ColonHistopathologyDataset(
        csv_path=TEST_CSV,
        transform=val_test_transform,
    )

    print()
    print(
        f"Test samples : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.block3

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    print(
        "Grad-CAM target layer : model.block3"
    )

    # --------------------------------------------------------
    # Find representative examples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SEARCHING FOR REPRESENTATIVE EXAMPLES"
    )
    print("=" * 70)

    examples = find_examples(
        model,
        dataset
    )

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------

    summary = []

    for name, result in examples.items():

        if result is None:

            print(
                f"[NOT FOUND] {name}"
            )

            continue

        index, true_class, predicted_class, probability = result

        image_tensor, label = dataset[index]

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        row = dataset.data.iloc[index]

        image_path = Path(
            row["filepath"]
        )

        original = Image.open(
            image_path
        ).convert("RGB")

        original = original.resize(
            (224, 224)
        )

        original_np = np.array(
            original
        ) / 255.0

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Grad-CAM for predicted class
        cam, output = gradcam.generate(
            input_tensor,
            predicted_class
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        filename = (
            f"{name}_"
            f"index_{index}_"
            f"true_{CLASS_NAMES[true_class]}_"
            f"pred_{CLASS_NAMES[predicted_class]}.png"
        )

        output_path = (
            OUTPUT_DIR /
            filename
        )

        save_gradcam_figure(
            original_image=original_np,
            cam=cam,
            true_class=true_class,
            predicted_class=predicted_class,
            probability=probability,
            output_path=output_path,
        )

        summary.append({
            "category": name,
            "index": index,
            "filepath": str(image_path),
            "true_class": CLASS_NAMES[
                true_class
            ],
            "predicted_class": CLASS_NAMES[
                predicted_class
            ],
            "probability": probability,
            "gradcam_target": (
                CLASS_NAMES[predicted_class]
            ),
            "output_file": str(
                output_path
            ),
        })

        print(
            f"[SAVED] {name}"
        )

        print(
            f"        Image : {image_path}"
        )

        print(
            f"        True  : "
            f"{CLASS_NAMES[true_class]}"
        )

        print(
            f"        Pred  : "
            f"{CLASS_NAMES[predicted_class]}"
        )

        print(
            f"        Prob  : "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR /
        "gradcam_summary.csv"
    )

    pd.DataFrame(
        summary
    ).to_csv(
        summary_path,
        index=False
    )

    gradcam.close()

    print()
    print("=" * 70)
    print(
        "GRAD-CAM COMPLETE"
    )
    print("=" * 70)

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print(
        f"Summary CSV      : {summary_path}"
    )


if __name__ == "__main__":

    main()