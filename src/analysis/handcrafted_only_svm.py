import json
from pathlib import Path

import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


FEATURE_DIR = Path(
    "results/features/handcrafted"
)

OUTPUT_DIR = Path(
    "results/analysis/handcrafted_only"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

K_BEST = 30


def load_split(name):

    data = np.load(
        FEATURE_DIR /
        f"{name}_features.npz"
    )

    return data["X"], data["y"]


X_train, y_train = load_split(
    "group_train"
)

X_val, y_val = load_split(
    "group_val"
)

X_test, y_test = load_split(
    "group_test"
)


print("=" * 70)
print("HANDCRAFTED-ONLY SVM")
print("=" * 70)

print()
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# SELECT TOP 30 HANDCRAFTED FEATURES
# ============================================================

selector = SelectKBest(
    score_func=f_classif,
    k=K_BEST
)

X_train_sel = selector.fit_transform(
    X_train,
    y_train
)

X_val_sel = selector.transform(
    X_val
)

X_test_sel = selector.transform(
    X_test
)


print()
print("Selected features:", K_BEST)


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_sel
)

X_val_scaled = scaler.transform(
    X_val_sel
)

X_test_scaled = scaler.transform(
    X_test_sel
)


# ============================================================
# SVM
# ============================================================

svm = SVC(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42,
)

svm.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X, y):

    pred = svm.predict(X)

    prob = svm.predict_proba(X)[:, 1]

    acc = accuracy_score(
        y,
        pred
    )

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        pred
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    auc = roc_auc_score(
        y,
        prob
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        classification_report(
            y,
            pred,
            target_names=[
                "colon_n",
                "colon_aca"
            ],
            digits=4,
            zero_division=0
        )
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


val_metrics = evaluate(
    "VALIDATION RESULTS",
    X_val_scaled,
    y_val
)

test_metrics = evaluate(
    "TEST RESULTS",
    X_test_scaled,
    y_test
)


# ============================================================
# SELECTED FEATURES
# ============================================================

selected_indices = selector.get_support(
    indices=True
)

selected_scores = selector.scores_[
    selected_indices
]

selected_pvalues = selector.pvalues_[
    selected_indices
]


print()
print("=" * 70)
print("SELECTED HANDCRAFTED FEATURES")
print("=" * 70)

for rank, (
    idx,
    score,
    pvalue
) in enumerate(
    zip(
        selected_indices,
        selected_scores,
        selected_pvalues
    ),
    start=1
):

    print(
        f"{rank:2d}. "
        f"feature_{idx:03d} "
        f"F={score:.4f} "
        f"p={pvalue:.4e}"
    )


# ============================================================
# SAVE
# ============================================================

results = {
    "model": "handcrafted_only",
    "input_features": 104,
    "selected_features": 30,
    "classifier": "RBF-SVM",
    "C": 1.0,
    "gamma": "scale",
    "validation": val_metrics,
    "test": test_metrics,
    "selected_indices": selected_indices.tolist(),
}


with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print("=" * 70)
print("HANDCRAFTED-ONLY ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_DIR / "metrics.json"
)