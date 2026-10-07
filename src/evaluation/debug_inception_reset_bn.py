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

DEVICE = torch.device("cpu")
IMAGE_SIZE = 299
BATCH_SIZE = 16


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


def reset_batchnorm_running_stats(model):

    count = 0

    for module in model.modules():

        if isinstance(module, torch.nn.BatchNorm2d):

            module.reset_running_stats()

            count += 1

    return count


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    predicted_counts = [0, 0]

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(images)

            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for p in predictions.cpu().tolist():
                predicted_counts[p] += 1

    return (
        correct / total,
        predicted_counts
    )


print("=" * 70)
print("INCEPTION V3 BATCHNORM RESET DIAGNOSTIC")
print("=" * 70)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

dataset = HistopathologyDataset(
    VAL_CSV,
    get_val_test_transform(IMAGE_SIZE)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)


# ------------------------------------------------------------
# Original checkpoint
# ------------------------------------------------------------

print()
print("1. ORIGINAL CHECKPOINT")
print("-" * 70)

accuracy_original, counts_original = evaluate(
    model,
    loader
)

print(
    f"Accuracy: {accuracy_original:.6f}"
)

print(
    "Predicted class counts:",
    counts_original
)


# ------------------------------------------------------------
# Reset ONLY BatchNorm running statistics
# ------------------------------------------------------------

num_bn = reset_batchnorm_running_stats(
    model
)

print()
print(
    f"Reset running statistics for {num_bn} BatchNorm layers."
)


# ------------------------------------------------------------
# Evaluate after BN reset
# ------------------------------------------------------------

print()
print("2. AFTER BATCHNORM RUNNING-STAT RESET")
print("-" * 70)

accuracy_reset, counts_reset = evaluate(
    model,
    loader
)

print(
    f"Accuracy: {accuracy_reset:.6f}"
)

print(
    "Predicted class counts:",
    counts_reset
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Original checkpoint accuracy : "
    f"{accuracy_original:.6f}"
)

print(
    f"After BN reset accuracy      : "
    f"{accuracy_reset:.6f}"
)

print(
    f"Accuracy change              : "
    f"{accuracy_reset - accuracy_original:+.6f}"
)