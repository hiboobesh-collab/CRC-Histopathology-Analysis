"""
Evaluate the trained DenseNet121 model on the untouched
group-aware Split A test set.

Outputs:
- test_metrics.json
- classification_report.txt
- confusion_matrix.png
- roc_curve.png
- test_predictions.npz
"""

from pathlib import Path

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from torch.utils.data import DataLoader

from src.data.dataset import ColonHistopathologyDataset
from src.data.transforms import val_test_transform
from src.models.classifier import build_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "densenet121"

TEST_CSV = Path(
    "data/splits/group_test.csv"
)

CHECKPOINT_PATH = Path(
    "models/classification/"
    "densenet121_groupaware/best/"
    "densenet121_best.pth"
)

RESULTS_DIR = Path(
    "results/classification/"
    "densenet121_groupaware"
)

BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRC HISTOPATHOLOGY")
    print("DenseNet121 TEST EVALUATION")
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

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = ColonHistopathologyDataset(
        csv_path=TEST_CSV,
        transform=val_test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print()
    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        MODEL_NAME,
        NUM_CLASSES,
        False,
    )

    model = model.to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch",
        checkpoint.get("best_epoch", None)
    )

    checkpoint_val_loss = checkpoint.get(
        "val_loss",
        checkpoint.get("best_val_loss", None)
    )

    checkpoint_val_accuracy = checkpoint.get(
        "val_accuracy",
        checkpoint.get("best_val_accuracy", None)
    )

    print()
    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint_val_loss}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint_val_accuracy}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            # Safety for models returning auxiliary outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.asarray(
        all_labels,
        dtype=np.int64
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int64
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64
    )

    # Probability of colon_aca
    y_score = probabilities[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "colon_n",
            "colon_aca",
        ],
        digits=4,
        zero_division=0,
    )

    print()
    print("Classification Report:")
    print(report)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": MODEL_NAME,
        "test_samples": int(len(y_true)),
        "checkpoint_epoch": (
            int(checkpoint_epoch)
            if checkpoint_epoch is not None
            else None
        ),
        "validation_loss": (
            float(checkpoint_val_loss)
            if checkpoint_val_loss is not None
            else None
        ),
        "validation_accuracy": (
            float(checkpoint_val_accuracy)
            if checkpoint_val_accuracy is not None
            else None
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    metrics_path = (
        RESULTS_DIR /
        "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # Confusion matrix figure
    # --------------------------------------------------------

    cm_path = (
        RESULTS_DIR /
        "confusion_matrix.png"
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        "DenseNet121 Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["colon_n", "colon_aca"]
    )

    plt.yticks(
        [0, 1],
        ["colon_n", "colon_aca"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    roc_path = (
        RESULTS_DIR /
        "roc_curve.png"
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "DenseNet121 ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = (
        RESULTS_DIR /
        "test_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION FILES SAVED")
    print("=" * 70)

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Report: {report_path}"
    )

    print(
        f"Confusion matrix: {cm_path}"
    )

    print(
        f"ROC curve: {roc_path}"
    )

    print(
        f"Predictions: {predictions_path}"
    )


if __name__ == "__main__":
    main()