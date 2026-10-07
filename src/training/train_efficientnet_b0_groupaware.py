import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset import ColonHistopathologyDataset
from src.data.transforms import train_transform, val_test_transform
from src.models.classifier import build_model


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnet_b0"

# Dataset
TRAIN_CSV = Path(
    "data/splits/group_train.csv"
)

VAL_CSV = Path(
    "data/splits/group_val.csv"
)

# Training
BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3

NUM_WORKERS = 0
SEED = 42

# Model storage
MODEL_DIR = Path(
    "models/classification/efficientnet_b0_groupaware"
)

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_DIR = MODEL_DIR / "best"
FINAL_DIR = MODEL_DIR / "final"

# Results
RESULTS_DIR = Path(
    "results/classification/efficientnet_b0_groupaware"
)

HISTORY_PATH = (
    RESULTS_DIR /
    "training_history.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)
print("GROUP-AWARE EFFICIENTNET-B0 TRAINING")
print("=" * 70)

print(f"Device        : {DEVICE}")
print(f"Model         : {MODEL_NAME}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Max epochs    : {MAX_EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Weight decay  : {WEIGHT_DECAY}")
print(f"Patience      : {PATIENCE}")
print(f"Seed          : {SEED}")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    CHECKPOINT_DIR,
    BEST_DIR,
    FINAL_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 70)
print("LOADING GROUP-AWARE DATASET")
print("=" * 70)

train_dataset = ColonHistopathologyDataset(
    csv_path=TRAIN_CSV,
    transform=train_transform,
)

val_dataset = ColonHistopathologyDataset(
    csv_path=VAL_CSV,
    transform=val_test_transform,
)

print(
    f"Training samples   : {len(train_dataset)}"
)

print(
    f"Validation samples : {len(val_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# BUILD EFFICIENTNET-B0
# ============================================================

print()
print("=" * 70)
print("BUILDING EFFICIENTNET-B0")
print("=" * 70)

model = build_model(
    name=MODEL_NAME,
    num_classes=2,
    pretrained=True,
)

model = model.to(DEVICE)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Total parameters : {parameter_count:,}"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "model_name": MODEL_NAME,
    "dataset": "LC25000 Colon",
    "split_type": "group-aware",
    "train_csv": str(TRAIN_CSV),
    "val_csv": str(VAL_CSV),
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "patience": PATIENCE,
    "seed": SEED,

    "train_loss": [],
    "train_accuracy": [],

    "val_loss": [],
    "val_accuracy": [],

    "epoch_time_minutes": [],
}


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_loss = float("inf")

best_val_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0


total_start = time.time()


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch}/{MAX_EPOCHS}"
    )
    print("=" * 70)

    epoch_start = time.time()


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # TIMING
    # ========================================================

    epoch_time = (
        time.time() - epoch_start
    ) / 60


    # ========================================================
    # UPDATE HISTORY
    # ========================================================

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


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    print(
        f"Train Loss      : "
        f"{train_loss:.6f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy * 100:.4f}%"
    )

    print(
        f"Val Loss        : "
        f"{val_loss:.6f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy * 100:.4f}%"
    )

    print(
        f"Epoch Time      : "
        f"{epoch_time:.2f} minutes"
    )


    # ========================================================
    # CHECKPOINT DATA
    # ========================================================

    checkpoint = {

        "epoch": epoch,

        "model_name": MODEL_NAME,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "seed":
            SEED,

    }


    # ========================================================
    # SAVE EVERY EPOCH
    # ========================================================

    epoch_path = (
        CHECKPOINT_DIR /
        f"epoch_{epoch:02d}.pth"
    )

    torch.save(
        checkpoint,
        epoch_path
    )

    print(
        f"Checkpoint saved: "
        f"{epoch_path}"
    )


    # ========================================================
    # SAVE LATEST
    # ========================================================

    latest_path = (
        CHECKPOINT_DIR /
        "latest.pth"
    )

    torch.save(
        checkpoint,
        latest_path
    )


    # ========================================================
    # CHECK FOR BEST MODEL
    # ========================================================

    improved = False

    # Primary criterion:
    # Lower validation loss.

    if val_loss < best_val_loss:

        improved = True

    # Tie breaker:
    # Higher validation accuracy.

    elif (
        val_loss == best_val_loss
        and val_accuracy > best_val_accuracy
    ):

        improved = True


    if improved:

        best_val_loss = val_loss

        best_val_accuracy = (
            val_accuracy
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        best_path = (
            BEST_DIR /
            "efficientnet_b0_best.pth"
        )

        torch.save(
            checkpoint,
            best_path
        )

        print()
        print(
            f"Best epoch       : "
            f"{best_epoch}"
        )

        print(
            f"Best val loss    : "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Best val accuracy: "
            f"{best_val_accuracy * 100:.4f}%"
        )

        print(
            f"Best model saved : "
            f"{best_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )


    # ========================================================
    # SAVE HISTORY AFTER EVERY EPOCH
    # ========================================================

    history["best_epoch"] = best_epoch

    history["best_val_loss"] = (
        best_val_loss
    )

    history["best_val_accuracy"] = (
        best_val_accuracy
    )

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_time = (
    time.time() - total_start
) / 60


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_path = (
    FINAL_DIR /
    "efficientnet_b0_final.pth"
)

final_checkpoint = {

    "epoch": epoch,

    "model_name": MODEL_NAME,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "train_loss":
        train_loss,

    "train_accuracy":
        train_accuracy,

    "val_loss":
        val_loss,

    "val_accuracy":
        val_accuracy,

}

torch.save(
    final_checkpoint,
    final_path
)


# ============================================================
# FINAL HISTORY
# ============================================================

history["completed_epochs"] = epoch

history["total_training_time_minutes"] = (
    total_time
)

history["best_epoch"] = best_epoch

history["best_val_loss"] = (
    best_val_loss
)

history["best_val_accuracy"] = (
    best_val_accuracy
)

with open(
    HISTORY_PATH,
    "w"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("GROUP-AWARE EFFICIENTNET-B0 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Completed epochs : {epoch}"
)

print(
    f"Best epoch       : {best_epoch}"
)

print(
    f"Best val loss    : "
    f"{best_val_loss:.6f}"
)

print(
    f"Best val accuracy: "
    f"{best_val_accuracy * 100:.4f}%"
)

print(
    f"Training time    : "
    f"{total_time:.2f} minutes"
)

print()
print("SAVED FILES")
print("-" * 70)

print(
    f"Checkpoints : "
    f"{CHECKPOINT_DIR}"
)

print(
    f"Best model  : "
    f"{BEST_DIR / 'efficientnet_b0_best.pth'}"
)

print(
    f"Final model : "
    f"{final_path}"
)

print(
    f"History     : "
    f"{HISTORY_PATH}"
)

print("=" * 70)