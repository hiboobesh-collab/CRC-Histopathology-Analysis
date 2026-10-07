import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from src.data.dataset import create_datasets
from src.models.classifier import build_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "resnet18"

BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 3

NUM_WORKERS = 0

SEED = 42

CHECKPOINT_DIR = Path("models")
RESULTS_DIR = Path("results")

BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet18_best.pth"
HISTORY_PATH = RESULTS_DIR / "resnet18_training_history.json"


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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 65)
print("CRC HISTOPATHOLOGY - RESNET18 TRAINING")
print("=" * 65)

print(f"Model          : {MODEL_NAME}")
print(f"Device         : {DEVICE}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Maximum epochs : {MAX_EPOCHS}")
print(f"Learning rate  : {LEARNING_RATE}")
print(f"Weight decay   : {WEIGHT_DECAY}")
print(f"Early stopping : patience={PATIENCE}")
print("=" * 65)


# ============================================================
# DATASET
# ============================================================

train_dataset, val_dataset, test_dataset = create_datasets()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("\nDataset:")
print(f"  Train      : {len(train_dataset)}")
print(f"  Validation : {len(val_dataset)}")
print(f"  Test       : {len(test_dataset)}")


# ============================================================
# MODEL
# ============================================================

model = build_model(
    name=MODEL_NAME,
    num_classes=2,
    pretrained=True
)

model = model.to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_time_minutes": []
}


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

training_start = time.time()

for epoch in range(MAX_EPOCHS):

    epoch_start = time.time()

    print("\n" + "=" * 65)
    print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
    print("=" * 65)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 25 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total
    train_accuracy = correct / total

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    epoch_time = time.time() - epoch_start
    epoch_minutes = epoch_time / 60

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    history["epoch_time_minutes"].append(epoch_minutes)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nResults:")
    print(f"Train Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Val Loss       : {val_loss:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.4f}")
    print(f"Epoch Time     : {epoch_minutes:.2f} minutes")

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    improved = (
        val_loss < best_val_loss
        or (
            abs(val_loss - best_val_loss) < 1e-8
            and val_accuracy > best_val_accuracy
        )
    )

    if improved:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch + 1,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "class_mapping": {
                "0": "colon_n",
                "1": "colon_aca"
            }
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print("\n*** BEST MODEL SAVED ***")
        print(f"Checkpoint : {BEST_MODEL_PATH}")
        print(f"Best epoch : {best_epoch}")
        print(f"Best val loss : {best_val_loss:.4f}")
        print(f"Best val accuracy : {best_val_accuracy:.4f}")

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo validation improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        print(
            f"Validation did not improve for "
            f"{PATIENCE} consecutive epochs."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

training_time = (time.time() - training_start) / 60

history["total_training_time_minutes"] = training_time
history["best_epoch"] = best_epoch
history["best_val_loss"] = best_val_loss
history["best_val_accuracy"] = best_val_accuracy

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(f"Best epoch          : {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation acc : {best_val_accuracy:.4f}")
print(f"Training time       : {training_time:.2f} minutes")

print(f"\nBest checkpoint:")
print(BEST_MODEL_PATH)

print(f"\nTraining history:")
print(HISTORY_PATH)

print("=" * 65)
