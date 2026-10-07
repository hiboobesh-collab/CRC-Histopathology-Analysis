import torch
from PIL import Image

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


CHECKPOINT = "models/classification/inception_v3_groupaware/best/inception_v3_best.pth"

IMAGE_PATH = "data/lung_colon_image_set/colon_image_sets/colon_aca/colonca1.jpeg"

DEVICE = torch.device("cpu")


print("=" * 70)
print("INCEPTION V3 SINGLE IMAGE DEBUG")
print("=" * 70)

# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

print("Checkpoint epoch:", checkpoint["epoch"])
print("Saved validation accuracy:", checkpoint["val_accuracy"])
print("Saved validation loss:", checkpoint["val_loss"])


# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

transform = get_val_test_transform(299)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(DEVICE)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

with torch.no_grad():

    output = model(image_tensor)

    if hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

print()
print("Image:", IMAGE_PATH)
print("Expected class: colon_aca (1)")
print()
print("Logits:")
print(logits)

print()
print("Probabilities:")
print(probabilities)

print()
print("Predicted class:", prediction)

if prediction == 0:
    print("Prediction: colon_n")
else:
    print("Prediction: colon_aca")