import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.classifier import build_model
from src.data.comparison_transforms import (
    get_train_transform,
    get_val_test_transform,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "inception_v3"

NUM_CLASSES = 2
IMAGE_SIZE = 299

BATCH_SIZE = 16
MAX_EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 3
SEED = 42

NUM_WORKERS = 0

TRAIN_CSV = Path("data/splits/group_train.csv")
VAL_CSV = Path("data/splits/group_val.csv")

OUTPUT_DIR = Path(
    "models/classification/inception_v3_groupaware"
)

BEST_DIR = OUTPUT_DIR / "best"
FINAL_DIR = OUTPUT_DIR / "final"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

HISTORY_PATH = Path(
    "results/classification/"
    "inception_v3_groupaware_training_history.json"
)

BEST_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATASET
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

train_transform = get_train_transform(
    IMAGE_SIZE
)

val_transform = get_val_test_transform(
    IMAGE_SIZE
)

train_dataset = HistopathologyDataset(
    TRAIN_CSV,
    transform=train_transform,
)

val_dataset = HistopathologyDataset(
    VAL_CSV,
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

print("=" * 70)
print("INCEPTIONV3 GROUP-AWARE TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Seed: {SEED}")

print("\nDataset:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")

print("\nTrain classes:")
print(
    train_dataset.df["label"].value_counts()
)

print("\nValidation classes:")
print(
    val_dataset.df["label"].value_counts()
)

print(
    f"\nTrain groups: "
    f"{train_dataset.df['group_id'].nunique()}"
)

print(
    f"Validation groups: "
    f"{val_dataset.df['group_id'].nunique()}"
)

print("\nBuilding model...")

model = build_model(
    MODEL_NAME,
    NUM_CLASSES,
    True,
)

model = model.to(DEVICE)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: "
    f"{total_params}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params}"
)


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

history = []

best_val_loss = float("inf")
best_val_acc = 0.0
best_epoch = 0

patience_counter = 0

training_start = time.time()


for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        # InceptionV3 returns:
        # InceptionOutputs(logits, aux_logits)
        if hasattr(outputs, "logits"):
            main_logits = outputs.logits
            aux_logits = outputs.aux_logits
        else:
            main_logits = outputs
            aux_logits = None

        loss = criterion(
            main_logits,
            labels,
        )

        # Auxiliary classifier loss
        if aux_logits is not None:
            aux_loss = criterion(
                aux_logits,
                labels,
            )

            loss = loss + 0.4 * aux_loss

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            main_logits,
            dim=1,
        )

        train_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        train_total += labels.size(0)

    train_loss /= train_total

    train_acc = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            loss = criterion(
                logits,
                labels,
            )

            val_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            val_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            val_total += labels.size(0)

    val_loss /= val_total

    val_acc = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # EPOCH OUTPUT
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        - epoch_start
    )

    print(
        f"\nEpoch {epoch}/{MAX_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.6f} | "
        f"Train Acc: {train_acc * 100:.4f}%"
    )

    print(
        f"Val Loss:   {val_loss:.6f} | "
        f"Val Acc:   {val_acc * 100:.4f}%"
    )

    print(
        f"Time: {epoch_time / 60:.2f} min"
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "epoch_time_seconds": epoch_time,
    }

    history.append(epoch_record)


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    is_better = (
        val_loss < best_val_loss
        or (
            val_loss == best_val_loss
            and val_acc > best_val_acc
        )
    )

    if is_better:

        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch = epoch

        patience_counter = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        }

        torch.save(
            checkpoint,
            BEST_DIR / "inception_v3_best.pth",
        )

        print(
            "✓ New best checkpoint saved"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: "
            f"{patience_counter}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # LATEST CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "model_name": MODEL_NAME,
        },
        CHECKPOINT_DIR / "latest.pth",
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
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
# FINAL MODEL
# ============================================================

training_time = (
    time.time()
    - training_start
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
    },
    FINAL_DIR / "inception_v3_final.pth",
)


# ============================================================
# COMPLETE
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
    f"{best_val_acc * 100:.4f}%"
)

print(
    f"Completed epochs: "
    f"{len(history)}"
)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

print("\nBest checkpoint:")

print(
    BEST_DIR /
    "inception_v3_best.pth"
)

print("\nHistory:")

print(
    HISTORY_PATH
)