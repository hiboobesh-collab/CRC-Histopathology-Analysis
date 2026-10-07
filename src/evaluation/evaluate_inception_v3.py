import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
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
from torch.utils.data import Dataset, DataLoader

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "inception_v3"
NUM_CLASSES = 2
IMAGE_SIZE = 299
BATCH_SIZE = 16

TEST_CSV = Path("data/splits/group_test.csv")

CHECKPOINT = Path(
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

OUTPUT_DIR = Path(
    "results/classification/inception_v3_groupaware"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABEL_MAP = {
    "colon_n": 0,
    "colon_aca": 1,
}


# ============================================================
# DATASET
# ============================================================

class HistopathologyDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        label = LABEL_MAP[row["label"]]

        if self.transform:
            image = self.transform(image)

        return (
            image,
            label,
            row["filename"],
            row["group_id"],
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("INCEPTIONV3 TEST EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Test CSV: {TEST_CSV}")


# ============================================================
# DATA
# ============================================================

transform = get_val_test_transform(
    IMAGE_SIZE
)

test_dataset = HistopathologyDataset(
    TEST_CSV,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(
    f"Test samples: {len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading InceptionV3...")

model = build_model(
    MODEL_NAME,
    NUM_CLASSES,
    False,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print(
    "Checkpoint loaded successfully."
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Validation loss: "
    f"{checkpoint.get('val_loss', 'N/A')}"
)

print(
    f"Validation accuracy: "
    f"{checkpoint.get('val_accuracy', 'N/A')}"
)


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning test inference...")

y_true = []
y_pred = []
y_prob = []

filenames = []
group_ids = []

with torch.no_grad():

    for (
        images,
        labels,
        batch_filenames,
        batch_groups,
    ) in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        # InceptionV3 returns InceptionOutputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_prob.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        filenames.extend(
            batch_filenames
        )

        group_ids.extend(
            batch_groups
        )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_true,
    y_prob,
)

cm = confusion_matrix(
    y_true,
    y_pred,
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INCEPTIONV3 TEST RESULTS")
print("=" * 70)

print(
    f"Test samples: {len(y_true)}"
)

print(
    f"Accuracy:     {accuracy:.4f}"
)

print(
    f"Precision:    {precision:.4f}"
)

print(
    f"Sensitivity:  {sensitivity:.4f}"
)

print(
    f"Specificity:  {specificity:.4f}"
)

print(
    f"F1 Score:     {f1:.4f}"
)

print(
    f"ROC-AUC:      {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "colon_n",
            "colon_aca",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": MODEL_NAME,
    "test_samples": int(len(y_true)),
    "checkpoint_epoch": checkpoint.get("epoch"),
    "validation_loss": checkpoint.get("val_loss"),
    "validation_accuracy": checkpoint.get(
        "val_accuracy"
    ),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
}

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
) as f:

    f.write(report)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.savez(
    OUTPUT_DIR / "test_predictions.npz",
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    filenames=np.array(filenames),
    group_ids=np.array(group_ids),
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

import matplotlib.pyplot as plt

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "InceptionV3 Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.xticks(
    [0, 1],
    ["colon_n", "colon_aca"],
)

plt.yticks(
    [0, 1],
    ["colon_n", "colon_aca"],
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob,
)

plt.figure(
    figsize=(6, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"InceptionV3 (AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "InceptionV3 ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Metrics:\n"
    f"{OUTPUT_DIR / 'test_metrics.json'}"
)

print(
    f"Classification report:\n"
    f"{OUTPUT_DIR / 'classification_report.txt'}"
)

print(
    f"Confusion matrix:\n"
    f"{OUTPUT_DIR / 'confusion_matrix.png'}"
)

print(
    f"ROC curve:\n"
    f"{OUTPUT_DIR / 'roc_curve.png'}"
)

print(
    f"Predictions:\n"
    f"{OUTPUT_DIR / 'test_predictions.npz'}"
)