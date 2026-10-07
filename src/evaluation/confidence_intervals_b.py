import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

N_BOOTSTRAP = 5000
SEED = 42

pred_file = "results/classification/resnet18_groupaware_b/test_predictions.npz"

data = np.load(pred_file)
y_true = data["y_true"]
y_pred = data["y_pred"]
y_prob = data["y_prob"]

rng = np.random.default_rng(SEED)
n = len(y_true)

def metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }

original = metrics(y_true, y_pred, y_prob)

boot = {k: [] for k in original}

for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)

    y_b = y_true[idx]
    pred_b = y_pred[idx]
    prob_b = y_prob[idx]

    # ROC-AUC requires both classes in the bootstrap sample
    if len(np.unique(y_b)) < 2:
        continue

    m = metrics(y_b, pred_b, prob_b)

    for k in boot:
        boot[k].append(m[k])

print()
print("=" * 75)
print("TEST SET PERFORMANCE WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 75)
print(f"Test samples: {n}")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")
print()

for name, value in original.items():
    values = np.array(boot[name])

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    print(
        f"{name:15s}: "
        f"{value:.4f} "
        f"(95% CI: {lower:.4f} - {upper:.4f})"
    )

print()
print("=" * 75)
