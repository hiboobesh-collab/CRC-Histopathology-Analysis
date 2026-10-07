import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from src.models.resnet18_model import build_resnet18


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "models/classification/resnet18_groupaware/best/resnet18_best.pth"
TEST_CSV = "data/splits/group_test.csv"
OUTPUT_DIR = "results/explainability/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "colon_n",
    1: "colon_aca"
}


# ============================================================
# Load model
# ============================================================

print("Device:", DEVICE)
print("Loading model...")

model = build_resnet18(
    num_classes=2,
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Best epoch:", checkpoint.get("epoch", "unknown"))
print("Model loaded.")


# ============================================================
# Image transformation
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Grad-CAM hooks
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer = model.layer4[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ============================================================
# Generate Grad-CAM
# ============================================================

def generate_gradcam(image_path):

    global activations, gradients

    image = Image.open(image_path).convert("RGB")
    original = image.copy()

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.zero_grad()

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_class].item())

    # Backpropagate predicted class
    score = output[0, predicted_class]
    score.backward()

    # Get activations and gradients
    acts = activations.detach()
    grads = gradients.detach()

    # Global average pooling of gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # Weighted feature maps
    cam = (weights * acts).sum(dim=1, keepdim=True)

    cam = F.relu(cam)

    # Resize CAM to original image dimensions
    cam = F.interpolate(
        cam,
        size=(original.height, original.width),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return original, cam, predicted_class, confidence, probabilities[0].detach().cpu().numpy()


# ============================================================
# Save visualization
# ============================================================

def save_gradcam(image_path, output_name):

    original, cam, predicted_class, confidence, probabilities = \
        generate_gradcam(image_path)

    original_np = np.array(original)

    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.axis("off")
    plt.title("Original Image")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(original_np)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(
        f"Grad-CAM\nPredicted: {CLASS_NAMES[predicted_class]}"
    )

    # Heatmap only
    plt.subplot(1, 3, 3)
    plt.imshow(cam, cmap="jet")
    plt.axis("off")
    plt.title("Activation Map")

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print()
    print("=" * 70)
    print("Grad-CAM result")
    print("=" * 70)
    print("Image:", image_path)
    print("Predicted:", CLASS_NAMES[predicted_class])
    print("Predicted confidence:", f"{confidence:.4f}")
    print("Probability colon_n:", f"{probabilities[0]:.4f}")
    print("Probability colon_aca:", f"{probabilities[1]:.4f}")
    print("Saved:", output_path)


# ============================================================
# Find examples
# ============================================================

df = pd.read_csv(TEST_CSV)

# Misclassified image
misclassified = df[
    df["filename"] == "colonca1907.jpeg"
]

if len(misclassified) == 0:
    raise FileNotFoundError(
        "colonca1907.jpeg was not found in group_test.csv"
    )

misclassified_path = misclassified.iloc[0]["filepath"]

# Find a correctly classified colon_aca image
# Prefer the first test example with colon_aca
correct = df[
    (df["label"] == "colon_aca") &
    (df["filename"] != "colonca1907.jpeg")
]

correct_path = correct.iloc[0]["filepath"]


# ============================================================
# Generate both visualizations
# ============================================================

print()
print("Generating Grad-CAM for misclassified image...")

save_gradcam(
    misclassified_path,
    "misclassified_colonca1907_gradcam.png"
)

print()
print("Generating Grad-CAM for correctly classified image...")

save_gradcam(
    correct_path,
    "correct_colon_aca_gradcam.png"
)

print()
print("=" * 70)
print("Grad-CAM generation complete.")
print("=" * 70)
