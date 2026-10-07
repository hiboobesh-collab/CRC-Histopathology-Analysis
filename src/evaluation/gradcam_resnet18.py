from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from src.models.resnet18_model import build_resnet18
from src.data.transforms import val_test_transform


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    "data/lung_colon_image_set/"
    "colon_image_sets/colon_aca/"
    "colonca1907.jpeg"
)

MODEL_PATH = Path(
    "models/classification/"
    "resnet18_groupaware/"
    "best/"
    "resnet18_best.pth"
)

OUTPUT_DIR = Path(
    "results/classification/"
    "error_analysis/gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INFORMATION
# ============================================================

TRUE_CLASS = "colon_aca"
GROUP_ID = 171

CLASS_NAMES = [
    "colon_n",
    "colon_aca"
]


# ============================================================
# VERIFY FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RESNET18 — GRAD-CAM")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Image  : {IMAGE_PATH}")
print(f"Model  : {MODEL_PATH}")


model = build_resnet18(
    num_classes=2,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# PREPROCESS
# ============================================================

input_tensor = val_test_transform(
    original_image
)

input_tensor = input_tensor.unsqueeze(
    0
).to(DEVICE)


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


# ============================================================
# HOOK FUNCTIONS
# ============================================================

def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# ============================================================
# TARGET LAYER
# ============================================================

# Final convolutional layer of ResNet18.
#
# This contains high-level spatial features immediately
# before the average pooling and classification layers.

target_layer = model.layer4[-1]


forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# FORWARD PASS
# ============================================================

model.zero_grad()

outputs = model(
    input_tensor
)

probabilities = torch.softmax(
    outputs,
    dim=1
)

predicted_class = int(
    probabilities.argmax(
        dim=1
    ).item()
)

predicted_probability = float(
    probabilities[
        0,
        predicted_class
    ].item()
)

aca_probability = float(
    probabilities[
        0,
        1
    ].item()
)


print()
print(
    f"True class        : {TRUE_CLASS}"
)

print(
    f"Predicted class   : "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"P(colon_aca)      : "
    f"{aca_probability * 100:.2f}%"
)

print(
    f"Predicted prob.   : "
    f"{predicted_probability * 100:.2f}%"
)

print(
    f"Group ID          : {GROUP_ID}"
)


# ============================================================
# BACKWARD PASS
# ============================================================

target_score = outputs[
    0,
    predicted_class
]

target_score.backward()


# ============================================================
# CHECK ACTIVATIONS / GRADIENTS
# ============================================================

if activations is None:
    raise RuntimeError(
        "Forward activations were not captured."
    )

if gradients is None:
    raise RuntimeError(
        "Gradients were not captured."
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

activation_maps = (
    activations.detach()
    .cpu()
    .numpy()[0]
)

gradient_maps = (
    gradients.detach()
    .cpu()
    .numpy()[0]
)


# ============================================================
# GRAD-CAM WEIGHTS
# ============================================================

weights = np.mean(
    gradient_maps,
    axis=(1, 2)
)


# ============================================================
# WEIGHTED ACTIVATION MAP
# ============================================================

cam = np.zeros(
    activation_maps.shape[1:],
    dtype=np.float32
)

for channel in range(
    activation_maps.shape[0]
):

    cam += (
        weights[channel]
        * activation_maps[channel]
    )


# ============================================================
# RELU
# ============================================================

cam = np.maximum(
    cam,
    0
)


# ============================================================
# NORMALIZE
# ============================================================

if cam.max() > 0:

    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min()
    )

else:

    cam = np.zeros_like(
        cam
    )


# ============================================================
# RESIZE CAM
# ============================================================

original_width, original_height = (
    original_image.size
)

cam_image = Image.fromarray(
    np.uint8(
        cam * 255
    )
)

cam_image = cam_image.resize(
    (
        original_width,
        original_height
    ),
    Image.Resampling.BILINEAR
)

cam_array = np.asarray(
    cam_image
) / 255.0


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# ============================================================
# ORIGINAL
# ============================================================

axes[0].imshow(
    original_image
)

axes[0].set_title(
    "Original Histopathology Image",
    fontsize=14,
    fontweight="bold"
)

axes[0].axis("off")


# ============================================================
# GRAD-CAM
# ============================================================

axes[1].imshow(
    cam_array,
    cmap="jet"
)

axes[1].set_title(
    "Grad-CAM Activation Map",
    fontsize=14,
    fontweight="bold"
)

axes[1].axis("off")


# ============================================================
# OVERLAY
# ============================================================

axes[2].imshow(
    original_image
)

axes[2].imshow(
    cam_array,
    cmap="jet",
    alpha=0.45
)

axes[2].set_title(
    "Grad-CAM Overlay",
    fontsize=14,
    fontweight="bold"
)

axes[2].axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "ResNet18 — Grad-CAM Explainability",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# INFORMATION
# ============================================================

fig.text(
    0.5,
    0.02,
    (
        f"True: {TRUE_CLASS}  |  "
        f"Predicted: {CLASS_NAMES[predicted_class]}  |  "
        f"P(colon_aca): {aca_probability * 100:.2f}%  |  "
        f"Group ID: {GROUP_ID}"
    ),
    ha="center",
    fontsize=11
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        0.94
    ]
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR /
    "resnet18_gradcam_colonca1907.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CLEANUP
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("GRAD-CAM GENERATION COMPLETE")
print("=" * 70)

print(
    f"Saved to: {output_path.resolve()}"
)

print(
    f"File exists: {output_path.exists()}"
)