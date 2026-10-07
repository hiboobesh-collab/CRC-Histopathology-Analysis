import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from src.data.dataset import ColonHistopathologyDataset
from src.data.transforms import val_test_transform
from src.models.basic_cnn import build_basic_cnn


# ============================================================
# CONFIGURATION
# ============================================================

TEST_CSV = Path(
    "data/splits/group_test.csv"
)

MODEL_PATH = Path(
    "models/classification/"
    "basic_cnn_groupaware/"
    "best/basic_cnn_best.pth"
)

RESULTS_DIR = Path(
    "results/classification/"
    "basic_cnn_groupaware"
)

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print(
    "GROUP-AWARE BASIC CNN TEST EVALUATION"
)
print("=" * 70)

print(f"Device      : {DEVICE}")
print(f"Test CSV    : {TEST_CSV}")
print(f"Model       : {MODEL_PATH}")
print(f"Batch size  : {BATCH_SIZE}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = ColonHistopathologyDataset(
    csv_path=TEST_CSV,
    transform=val_test_transform,
)

print(
    f"Test samples : {len(test_dataset)}"
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BASIC CNN")
print("=" * 70)

model = build_basic_cnn(
    num_classes=2,
    dropout=0.3,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# PARAMETER COUNTS
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

non_trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if not p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Trainable parameters     : "
    f"{trainable_params:,}"
)

print(
    f"Non-trainable parameters : "
    f"{non_trainable_params:,}"
)

print(
    f"Total parameters         : "
    f"{total_params:,}"
)

print(
    f"Checkpoint epoch         : "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Validation loss          : "
    f"{checkpoint.get('val_loss', 'unknown')}"
)

print(
    f"Validation accuracy      : "
    f"{checkpoint.get('val_accuracy', 'unknown')}"
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING TEST INFERENCE")
print("=" * 70)

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "colon_n",
        "colon_aca"
    ],
    digits=4,
    zero_division=0,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy       : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Sensitivity    : {recall:.4f}"
)

print(
    f"Specificity    : {specificity:.4f}"
)

print(
    f"F1 Score       : {f1:.4f}"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print()
print("Confusion Matrix")
print("----------------")
print(cm)

print()
print("Classification Report")
print("---------------------")
print(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Basic CNN",
    "dataset": "LC25000 Colon",
    "split_type": "group-aware",
    "test_csv": str(TEST_CSV),
    "checkpoint": str(MODEL_PATH),

    "checkpoint_epoch": checkpoint.get(
        "epoch",
        None
    ),

    "test_samples": len(test_dataset),

    "trainable_parameters": int(
        trainable_params
    ),

    "non_trainable_parameters": int(
        non_trainable_params
    ),

    "total_parameters": int(
        total_params
    ),

    "accuracy": float(accuracy),
    "precision": float(precision),
    "sensitivity": float(recall),
    "specificity": float(specificity),
    "f1_score": float(f1),
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


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Basic CNN - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["colon_n", "colon_aca"]
)

plt.yticks(
    [0, 1],
    ["colon_n", "colon_aca"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = (
    RESULTS_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(
    figsize=(6, 5)
)

plt.plot(
    fpr,
    tpr,
    label=(
        "Basic CNN "
        f"(AUC = {roc_auc:.4f})"
    )
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
    "Basic CNN ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_path = (
    RESULTS_DIR /
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

prediction_path = (
    RESULTS_DIR /
    "test_predictions.npz"
)

np.savez(
    prediction_path,
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Metrics saved      : {metrics_path}"
)

print(
    f"Report saved       : {report_path}"
)

print(
    f"Confusion matrix   : {cm_path}"
)

print(
    f"ROC curve          : {roc_path}"
)

print(
    f"Predictions saved  : {prediction_path}"
)