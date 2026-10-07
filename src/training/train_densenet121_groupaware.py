"""
DenseNet121 baseline training for CRC histopathology.

Uses the same group-aware Split A protocol as the validated
ResNet18 experiment.

Model:
    DenseNet121

Task:
    colon_n vs colon_aca

Split:
    data/splits/group_train.csv
    data/splits/group_val.csv
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset import ColonHistopathologyDataset
from src.data.transforms import (
    train_transform,
    val_test_transform,
)
from src.models.classifier import build_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "densenet121"

TRAIN_CSV = Path(
    "data/splits/group_train.csv"
)

VAL_CSV = Path(
    "data/splits/group_val.csv"
)

MODEL_DIR = Path(
    "models/classification/densenet121_groupaware"
)

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_DIR = MODEL_DIR / "best"
FINAL_DIR = MODEL_DIR / "final"

RESULTS_DIR = Path(
    "results/classification"
)

HISTORY_PATH = (
    RESULTS_DIR /
    "densenet121_groupaware_training_history.json"
)

BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3

NUM_WORKERS = 0
SEED = 42

NUM_CLASSES = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("CRC HISTOPATHOLOGY")
    print("DenseNet121 GROUP-AWARE BASELINE TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Patience: {PATIENCE}")

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = ColonHistopathologyDataset(
        csv_path=TRAIN_CSV,
        transform=train_transform,
    )

    val_dataset = ColonHistopathologyDataset(
        csv_path=VAL_CSV,
        transform=val_test_transform,
    )

    print()
    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
    MODEL_NAME,
    NUM_CLASSES,
    True,
)

    model = model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    epochs_without_improvement = 0

    training_start = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        epoch_start = time.time()

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch}/{MAX_EPOCHS}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # Some architectures may return
            # auxiliary outputs.
            if isinstance(outputs, tuple):

                outputs = outputs[0]

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= train_total

        train_accuracy = (
            train_correct /
            train_total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                if isinstance(outputs, tuple):

                    outputs = outputs[0]

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total

        val_accuracy = (
            val_correct /
            val_total
        )

        epoch_time = (
            time.time() -
            epoch_start
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print()
        print(
            f"Train Loss     : "
            f"{train_loss:.6f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy * 100:.4f}%"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.6f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_accuracy * 100:.4f}%"
        )

        print(
            f"Epoch Time     : "
            f"{epoch_time / 60:.2f} minutes"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time_seconds": epoch_time,
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        checkpoint_path = (
            CHECKPOINT_DIR /
            f"epoch_{epoch:02d}.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        # ----------------------------------------------------
        # Best model
        #
        # Primary criterion:
        # lower validation loss
        #
        # Tie breaker:
        # higher validation accuracy
        # ----------------------------------------------------

        improved = False

        if val_loss < best_val_loss:

            improved = True

        elif (
            np.isclose(
                val_loss,
                best_val_loss
            )
            and
            val_accuracy > best_val_accuracy
        ):

            improved = True

        if improved:

            best_val_loss = val_loss

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            epochs_without_improvement = 0

            best_path = (
                BEST_DIR /
                "densenet121_best.pth"
            )

            torch.save(
                checkpoint,
                best_path
            )

            print()
            print("✓ NEW BEST MODEL")

            print(
                f"  Best epoch: {best_epoch}"
            )

            print(
                f"  Best val loss: "
                f"{best_val_loss:.6f}"
            )

            print(
                f"  Best val accuracy: "
                f"{best_val_accuracy * 100:.4f}%"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # FINAL MODEL
    # ========================================================

    final_path = (
        FINAL_DIR /
        "densenet121_final.pth"
    )

    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
        },
        final_path
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model_name": MODEL_NAME,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
            },
            f,
            indent=4,
        )

    total_time = (
        time.time() -
        training_start
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Completed epochs: "
        f"{len(history)}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.4f}%"
    )

    print(
        f"Training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()
    print(
        f"Best model: {BEST_DIR / 'densenet121_best.pth'}"
    )

    print(
        f"Final model: {final_path}"
    )

    print(
        f"History: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()