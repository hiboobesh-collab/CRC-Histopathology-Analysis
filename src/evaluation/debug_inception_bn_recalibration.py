import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from src.models.classifier import build_model
from src.data.comparison_transforms import (
    get_train_transform,
    get_val_test_transform
)

DEVICE = torch.device("cpu")

TRAIN_CSV = "data/splits/group_train.csv"
VAL_CSV = "data/splits/group_val.csv"

CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)


class HistopathologyDataset(Dataset):

    def __init__(self, csv_path, transform=None):

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

        label = self.label_map[
            row["label"]
        ]

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = HistopathologyDataset(
    TRAIN_CSV,
    get_val_test_transform(299)
)

val_dataset = HistopathologyDataset(
    VAL_CSV,
    get_val_test_transform(299)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)


def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            preds = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


print("=" * 80)
print("INCEPTION V3 BATCHNORM RECALIBRATION DIAGNOSTIC")
print("=" * 80)

print()
print("BEFORE BN RECALIBRATION")

accuracy_before = evaluate()

print(
    "Validation accuracy:",
    accuracy_before
)


print()
print("Recalibrating BatchNorm statistics...")
print("No gradients. No optimizer. No weight updates.")

model.train()

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(train_loader):

        model(images)

        if (batch_idx + 1) % 100 == 0:
            print(
                "Processed training batches:",
                batch_idx + 1
            )


print()
print("BN recalibration complete.")

print()
print("AFTER BN RECALIBRATION")

accuracy_after = evaluate()

print(
    "Validation accuracy:",
    accuracy_after
)

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print(
    "Before:",
    accuracy_before
)

print(
    "After:",
    accuracy_after
)

print("=" * 80)
print("DONE")
print("=" * 80)
