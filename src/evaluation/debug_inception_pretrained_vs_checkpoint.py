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

        return self.transform(image), label


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    counts = [0, 0]

    with torch.no_grad():

        for images, labels in loader:

            output = model(images)

            logits = (
                output.logits
                if hasattr(output, "logits")
                else output
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for p in predictions.tolist():
                counts[p] += 1

    return correct / total, counts


print("=" * 70)
print("INCEPTION V3 PRETRAINED VS CHECKPOINT")
print("=" * 70)


dataset = HistopathologyDataset(
    VAL_CSV,
    get_val_test_transform(IMAGE_SIZE)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# 1. Fresh pretrained model
# ------------------------------------------------------------

print()
print("1. FRESH PRETRAINED MODEL")
print("-" * 70)

model = build_model(
    "inception_v3",
    2,
    True
)

model.eval()

acc, counts = evaluate(
    model,
    loader
)

print(f"Accuracy: {acc:.6f}")
print("Predicted counts:", counts)


# ------------------------------------------------------------
# 2. Load checkpoint into SAME model
# ------------------------------------------------------------

print()
print("2. SAME MODEL AFTER CHECKPOINT LOAD")
print("-" * 70)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

model.eval()

acc, counts = evaluate(
    model,
    loader
)

print(f"Accuracy: {acc:.6f}")
print("Predicted counts:", counts)