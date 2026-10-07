import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from src.models.classifier import build_model
from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform


DEVICE = torch.device("cpu")

CHECKPOINT = Path(
    "models/classification/ghostnet_groupaware/best/ghostnet_best.pth"
)

TEST_CSV = Path(
    "data/splits/group_test.csv"
)

OUTPUT_DIR = Path(
    "results/classification/ghostnet_groupaware"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 70)
print("GHOSTNET TEST EVALUATION")
print("=" * 70)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model = build_model(
    "ghostnet",
    2,
    False,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model = model.to(DEVICE)
model.eval()


dataset = ColonHistopathologyDataset(
    csv_path=TEST_CSV,
    transform=get_val_test_transform(224),
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


y_true = []
y_pred = []
y_prob = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )

        y_prob.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)


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

cm = confusion_matrix(
    y_true,
    y_pred,
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob,
)


print()
print("Checkpoint epoch:", checkpoint["epoch"])
print(
    "Recorded validation accuracy:",
    checkpoint["val_accuracy"],
)
print(
    "Recorded validation loss:",
    checkpoint["val_loss"],
)

print()
print("Test samples:", len(y_true))

print()
print("TEST RESULTS")
print("-" * 50)

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

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
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


metrics = {
    "model": "ghostnet",
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "test_samples": int(len(y_true)),
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


np.savez(
    OUTPUT_DIR / "test_predictions.npz",
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
)


print()
print("Saved:")
print(
    OUTPUT_DIR / "test_metrics.json"
)
print(
    OUTPUT_DIR / "test_predictions.npz"
)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)