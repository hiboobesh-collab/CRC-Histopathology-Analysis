import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")
VAL_CSV = Path("data/splits/group_val.csv")
CHECKPOINT = Path(
    "models/classification/inception_v3_groupaware/best/inception_v3_best.pth"
)

IMAGE_SIZE = 299
BATCH_SIZE = 16


# ============================================================
# EXACT DATASET LOGIC FROM TRAINING SCRIPT
# ============================================================

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


# ============================================================
# DATA
# ============================================================

transform = get_val_test_transform(IMAGE_SIZE)

dataset = HistopathologyDataset(
    VAL_CSV,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# MODEL
# ============================================================

print("=" * 80)
print("INCEPTION V3 VALIDATION-LOOP DIAGNOSTIC")
print("=" * 80)

print("Validation samples:", len(dataset))
print("Checkpoint:", CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Recorded validation accuracy:",
      checkpoint.get("val_accuracy"))
print("Recorded validation loss:",
      checkpoint.get("val_loss"))

model = build_model(
    "inception_v3",
    2,
    False
)

missing, unexpected = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model = model.to(DEVICE)


# ============================================================
# EXACT VALIDATION MODE
# ============================================================

model.eval()

criterion = torch.nn.CrossEntropyLoss()

val_loss = 0.0
val_correct = 0
val_total = 0

all_preds = []
all_labels = []

first_batch = True

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        print_type = False

        if hasattr(outputs, "logits"):

            logits = outputs.logits

            if hasattr(outputs, "aux_logits"):
                aux_logits = outputs.aux_logits
            else:
                aux_logits = None

        else:

            logits = outputs
            aux_logits = None

        loss = criterion(
            logits,
            labels
        )

        val_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        val_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        val_total += labels.size(0)

        all_preds.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        # ----------------------------------------------------
        # PRINT FIRST BATCH ONLY
        # ----------------------------------------------------

        if first_batch:

            print()
            print("-" * 80)
            print("FIRST VALIDATION BATCH")
            print("-" * 80)

            print("Output type:", type(outputs))
            print("Images shape:", images.shape)
            print("Labels shape:", labels.shape)
            print("Logits shape:", logits.shape)

            if aux_logits is not None:
                print(
                    "Aux logits shape:",
                    aux_logits.shape
                )
            else:
                print("Aux logits: None")

            print()
            print("Labels:")
            print(labels.cpu().tolist())

            print()
            print("Predictions:")
            print(predictions.cpu().tolist())

            print()
            print("First logits:")
            print(logits[:5].cpu())

            print()
            print("First probabilities:")
            print(
                torch.softmax(
                    logits[:5],
                    dim=1
                ).cpu()
            )

            print("-" * 80)

            first_batch = False


# ============================================================
# FINAL METRICS
# ============================================================

val_loss /= val_total
val_accuracy = val_correct / val_total

pred_tensor = torch.tensor(all_preds)
label_tensor = torch.tensor(all_labels)

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print("Actual validation loss:", val_loss)
print("Actual validation accuracy:", val_accuracy)

print(
    "Predicted counts:",
    torch.bincount(
        pred_tensor,
        minlength=2
    ).tolist()
)

print(
    "True counts:",
    torch.bincount(
        label_tensor,
        minlength=2
    ).tolist()
)

print()
print("Correct predictions:", val_correct)
print("Total predictions:", val_total)

print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
