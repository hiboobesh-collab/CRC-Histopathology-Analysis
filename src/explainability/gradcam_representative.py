import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.resnet18_model import build_resnet18

CHECKPOINT = "models/classification/resnet18_groupaware/best/resnet18_best.pth"
TEST_CSV = "data/splits/group_test.csv"
PREDICTIONS = "results/classification/resnet18_groupaware/test_predictions.npz"

OUTPUT_DIR = "results/explainability/gradcam/representative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "colon_n",
    1: "colon_aca"
}

print("Device:", DEVICE)
print("Loading model...")

model = build_resnet18(num_classes=2, pretrained=False)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Best epoch:", checkpoint.get("epoch", "unknown"))
print("Model loaded.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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


def get_probability_class1(y_prob):
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        return y_prob

    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_prob[:, 1]

    return y_prob.reshape(-1)


def generate_gradcam(image_path, target_class):

    global activations, gradients

    image = Image.open(image_path).convert("RGB")
    original = image.copy()

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.zero_grad(set_to_none=True)

    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)

    predicted_class = int(
        torch.argmax(probabilities, dim=1).item()
    )

    target_score = output[0, target_class]
    target_score.backward()

    acts = activations.detach()
    grads = gradients.detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)

    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(original.height, original.width),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().cpu().numpy()

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    probs = probabilities[0].detach().cpu().numpy()

    return (
        original,
        cam,
        predicted_class,
        float(probs[0]),
        float(probs[1])
    )


def save_gradcam(row, index):

    image_path = row["filepath"]
    target_class = int(row["predicted_class"])

    original, cam, predicted, p_normal, p_aca = \
        generate_gradcam(image_path, target_class)

    original_np = np.array(original)

    status = (
        "correct"
        if row["true_class"] == row["predicted_class"]
        else "misclassified"
    )

    filename = (
        f"{index:02d}_{status}_"
        f"true-{CLASS_NAMES[row['true_class']]}_"
        f"pred-{CLASS_NAMES[row['predicted_class']]}_"
        f"{row['filename'].replace('.jpeg', '')}_gradcam.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.axis("off")
    plt.title(
        f"Original\nTrue: {CLASS_NAMES[row['true_class']]}"
    )

    plt.subplot(1, 3, 2)
    plt.imshow(original_np)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(
        f"Grad-CAM\nTarget: {CLASS_NAMES[target_class]}"
    )

    plt.subplot(1, 3, 3)
    plt.imshow(cam, cmap="jet")
    plt.axis("off")
    plt.title(
        f"Activation Map\nPred: {CLASS_NAMES[predicted]}"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return output_path


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

df = pd.read_csv(TEST_CSV)
pred = np.load(PREDICTIONS)

y_true = pred["y_true"].astype(int)
y_pred = pred["y_pred"].astype(int)
p1 = get_probability_class1(pred["y_prob"])

if not (len(df) == len(y_true) == len(y_pred) == len(p1)):
    raise ValueError("Prediction/CSV length mismatch.")

work = df.copy()

work["true_class"] = y_true
work["predicted_class"] = y_pred
work["prob_colon_aca"] = p1

work["confidence"] = np.where(
    y_pred == 1,
    p1,
    1.0 - p1
)

work["correct"] = y_true == y_pred


# ------------------------------------------------------------
# Select representative samples
# ------------------------------------------------------------

selected = []

for class_id in [0, 1]:

    correct = work[
        (work["correct"]) &
        (work["true_class"] == class_id)
    ].copy()

    high = correct.sort_values(
        "confidence",
        ascending=False
    ).head(5)

    low = correct.sort_values(
        "confidence",
        ascending=True
    ).head(5)

    high["selection"] = "correct_high_confidence"
    low["selection"] = "correct_low_confidence"

    selected.append(high)
    selected.append(low)


# Include ALL misclassified samples
misclassified = work[
    ~work["correct"]
].copy()

misclassified["selection"] = "misclassified"

selected.append(misclassified)

selection = pd.concat(
    selected,
    ignore_index=True
)

selection = selection.drop_duplicates(
    subset=["filepath"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Save selection manifest
# ------------------------------------------------------------

manifest = os.path.join(
    OUTPUT_DIR,
    "selection_manifest.csv"
)

selection.to_csv(
    manifest,
    index=False
)

print()
print("=" * 75)
print("REPRESENTATIVE GRAD-CAM SELECTION")
print("=" * 75)

print("Total selected:", len(selection))
print()

print(
    selection[
        [
            "selection",
            "filename",
            "true_class",
            "predicted_class",
            "confidence"
        ]
    ].to_string(index=False)
)

print()
print("Manifest:", manifest)
print()


# ------------------------------------------------------------
# Generate Grad-CAM
# ------------------------------------------------------------

for i, (_, row) in enumerate(
    selection.iterrows(),
    start=1
):

    print(
        f"[{i:02d}/{len(selection)}] "
        f"{row['filename']} | "
        f"true={CLASS_NAMES[row['true_class']]} | "
        f"pred={CLASS_NAMES[row['predicted_class']]} | "
        f"confidence={row['confidence']:.4f}"
    )

    output = save_gradcam(row, i)

    print("    Saved:", output)


print()
print("=" * 75)
print("REPRESENTATIVE GRAD-CAM GENERATION COMPLETE")
print("=" * 75)
print("Output:", OUTPUT_DIR)
print("Manifest:", manifest)
