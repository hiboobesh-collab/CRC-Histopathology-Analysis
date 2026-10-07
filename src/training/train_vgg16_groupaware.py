import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.classifier import build_model


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = ROOT / "data" / "splits" / "group_train.csv"
VAL_CSV = ROOT / "data" / "splits" / "group_val.csv"

MODEL_DIR = ROOT / "models" / "classification" / "vgg16_groupaware"
BEST_DIR = MODEL_DIR / "best"
FINAL_DIR = MODEL_DIR / "final"

RESULTS_DIR = ROOT / "results" / "classification"

BEST_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "vgg16"
NUM_CLASSES = 2

IMAGE_SIZE = 224
BATCH_SIZE = 16

MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 3
SEED = 42

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("VGG16 GROUP-AWARE TRAINING")
print("=" * 70)

print("Device:", DEVICE)
print("Model:", MODEL_NAME)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Max epochs:", MAX_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Seed:", SEED)


# ============================================================
# DATASET
# ============================================================

LABEL_MAP = {
    "colon_n": 0,
    "colon_aca": 1,
}


class ColonDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(row["filepath"])

        image = Image.open(image_path).convert("RGB")

        label = LABEL_MAP[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# LOAD CSVs
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nDataset:")
print("Train:", len(train_df))
print("Validation:", len(val_df))

print("\nTrain classes:")
print(train_df["label"].value_counts())

print("\nValidation classes:")
print(val_df["label"].value_counts())

print("\nTrain groups:", train_df["group_id"].nunique())
print("Validation groups:", val_df["group_id"].nunique())


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = ColonDataset(
    train_df,
    transform=train_transform,
)

val_dataset = ColonDataset(
    val_df,
    transform=val_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# MODEL
# ============================================================

print("\nBuilding model...")

model = build_model(
    MODEL_NAME,
    NUM_CLASSES,
    True,
)

model = model.to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAINING
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_time_minutes": [],
}


best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0

patience_counter = 0

training_start = time.time()


for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    train_predictions = []
    train_labels = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

    train_loss = running_loss / len(train_dataset)

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions,
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_predictions = []
    val_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

    val_loss = val_running_loss / len(val_dataset)

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions,
    )


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    epoch_time = (
        time.time() - epoch_start
    ) / 60

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["epoch_time_minutes"].append(
        epoch_time
    )


    print(
        f"\nEpoch {epoch}/{MAX_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.6f} | "
        f"Train Acc: {train_accuracy:.4%}"
    )

    print(
        f"Val Loss:   {val_loss:.6f} | "
        f"Val Acc:   {val_accuracy:.4%}"
    )

    print(
        f"Time: {epoch_time:.2f} min"
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    is_better = (
        val_loss < best_val_loss
        or (
            val_loss == best_val_loss
            and val_accuracy > best_val_accuracy
        )
    )

    if is_better:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch

        patience_counter = 0

        checkpoint = {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

        torch.save(
            checkpoint,
            BEST_DIR / "vgg16_best.pth",
        )

        print("✓ New best checkpoint saved")

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# FINAL CHECKPOINT
# ============================================================

total_training_time = (
    time.time() - training_start
) / 60

torch.save(
    {
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "completed_epochs": len(
            history["train_loss"]
        ),
    },
    FINAL_DIR / "vgg16_final.pth",
)


# ============================================================
# SAVE HISTORY
# ============================================================

history["best_epoch"] = best_epoch
history["best_val_loss"] = best_val_loss
history["best_val_accuracy"] = best_val_accuracy
history["total_training_time_minutes"] = (
    total_training_time
)
history["completed_epochs"] = len(
    history["train_loss"]
)

history_path = (
    RESULTS_DIR /
    "vgg16_groupaware_training_history.json"
)

with open(history_path, "w") as f:

    json.dump(
        history,
        f,
        indent=4,
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.10f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4%}"
)

print(
    f"Completed epochs: "
    f"{len(history['train_loss'])}"
)

print(
    f"Training time: "
    f"{total_training_time:.2f} minutes"
)

print(
    "\nBest checkpoint:"
)

print(
    BEST_DIR / "vgg16_best.pth"
)

print(
    "\nHistory:"
)

print(history_path)