import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

VAL_CSV = "data/splits/group_val.csv"

IMAGE_SIZE = 299
BATCH_SIZE = 16

device = torch.device("cpu")


class HistopathologyDataset(Dataset):

    def __init__(self, csv_path, transform):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.label_map = {
            "colon_n": 0,
            "colon_aca": 1,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        label = self.label_map[row["label"]]

        image = self.transform(image)

        return image, label


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

print("=" * 70)
print("INCEPTION V3 VALIDATION - TRAIN MODE DIAGNOSTIC")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

print("Checkpoint epoch:", checkpoint["epoch"])
print("Saved validation accuracy:", checkpoint["val_accuracy"])
print("Saved validation loss:", checkpoint["val_loss"])


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

transform = get_val_test_transform(
    IMAGE_SIZE
)

dataset = HistopathologyDataset(
    VAL_CSV,
    transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)


# ------------------------------------------------------------
# IMPORTANT:
# Keep model in TRAIN MODE so BatchNorm uses
# current batch statistics.
# No gradients are calculated.
# ------------------------------------------------------------

model.train()

correct = 0
total = 0

predicted_counts = [0, 0]
true_counts = [0, 0]

prob_class_1 = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        for p in predictions.cpu().tolist():
            predicted_counts[p] += 1

        for y in labels.cpu().tolist():
            true_counts[y] += 1

        prob_class_1.extend(
            probabilities[:, 1].cpu().tolist()
        )

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Processed "
                f"{batch_idx + 1}/{len(loader)} batches"
            )


accuracy = correct / total

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("Samples:", total)
print(f"Accuracy: {accuracy:.6f}")

print(
    "Predicted class counts:",
    predicted_counts
)

print(
    "True class counts:",
    true_counts
)

print(
    f"Mean P(colon_aca): "
    f"{sum(prob_class_1) / len(prob_class_1):.6f}"
)

print(
    f"Min P(colon_aca): "
    f"{min(prob_class_1):.6f}"
)

print(
    f"Max P(colon_aca): "
    f"{max(prob_class_1):.6f}"
)

print()
print("NOTE:")
print(
    "This evaluation intentionally uses model.train() "
    "with gradients disabled, so BatchNorm uses "
    "current batch statistics."
)
