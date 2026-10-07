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


# ============================================================
# CONFIG
# ============================================================

DEEP_DIR = Path(
    "results/features/deep/ghostnet"
)

HAND_DIR = Path(
    "results/features/handcrafted"
)

OUTPUT_DIR = Path(
    "results/fusion/ghostnet"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

K_BEST = 30

SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"


# ============================================================
# LOAD + FUSE
# ============================================================

def load_split(split):

    deep = np.load(
        DEEP_DIR / f"{split}_features.npz"
    )

    hand = np.load(
        HAND_DIR / f"{split}_features.npz"
    )

    X_deep = deep["X"]
    X_hand = hand["X"]

    y_deep = deep["y"]
    y_hand = hand["y"]

    if not np.array_equal(y_deep, y_hand):
        raise RuntimeError(
            f"Label mismatch in {split}"
        )

    X = np.concatenate(
        [X_deep, X_hand],
        axis=1
    )

    return X, y_deep


print("=" * 70)
print("GHOSTNET + HANDCRAFTED FEATURE FUSION")
print("=" * 70)


X_train, y_train = load_split(
    "group_train"
)

X_val, y_val = load_split(
    "group_val"
)

X_test, y_test = load_split(
    "group_test"
)


print()
print("Fused feature shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# SANITY CHECK
# ============================================================

for name, X in [
    ("Train", X_train),
    ("Validation", X_val),
    ("Test", X_test),
]:

    if np.isnan(X).any():
        raise RuntimeError(
            f"{name} contains NaN"
        )

    if np.isinf(X).any():
        raise RuntimeError(
            f"{name} contains Inf"
        )


# ============================================================
# SELECTKBEST
# ============================================================

print()
print("-" * 70)
print("SELECTKBEST")
print("-" * 70)

selector = SelectKBest(
    score_func=f_classif,
    k=K_BEST
)

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

X_val_selected = selector.transform(
    X_val
)

X_test_selected = selector.transform(
    X_test
)

print(
    "Selected features:",
    K_BEST
)

print(
    "Train selected shape:",
    X_train_selected.shape
)

print(
    "Validation selected shape:",
    X_val_selected.shape
)

print(
    "Test selected shape:",
    X_test_selected.shape
)


# ============================================================
# STANDARD SCALER
# ============================================================

print()
print("-" * 70)
print("STANDARD SCALER")
print("-" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_selected
)

X_val_scaled = scaler.transform(
    X_val_selected
)

X_test_scaled = scaler.transform(
    X_test_selected
)

print(
    "Scaler fitted on training data only."
)


# ============================================================
# RBF SVM
# ============================================================

print()
print("-" * 70)
print("RBF SVM")
print("-" * 70)

svm = SVC(
    C=SVM_C,
    kernel=SVM_KERNEL,
    gamma=SVM_GAMMA,
    probability=True,
    random_state=42,
)

svm.fit(
    X_train_scaled,
    y_train
)

print(
    "SVM training complete."
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X, y):

    predictions = svm.predict(X)

    probabilities = svm.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"ROC-AUC:      {roc_auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "colon_n",
                "colon_aca",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# VALIDATION
# ============================================================

val_metrics = evaluate(
    "VALIDATION RESULTS",
    X_val_scaled,
    y_val,
)


# ============================================================
# TEST
# ============================================================

test_metrics = evaluate(
    "TEST RESULTS",
    X_test_scaled,
    y_test,
)


# ============================================================
# FEATURE SELECTION DETAILS
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


selected_info = []

for rank, (
    index,
    score,
    pvalue
) in enumerate(
    zip(
        selected_indices,
        selected_scores,
        selected_pvalues,
    ),
    start=1
):

    if index < 1280:

        feature_type = "deep"

        feature_name = (
            f"ghostnet_feature_{index}"
        )

    else:

        feature_type = "handcrafted"

        handcrafted_index = (
            index - 1280
        )

        feature_name = (
            f"handcrafted_feature_{handcrafted_index}"
        )

    selected_info.append(
        {
            "rank": rank,
            "original_index": int(index),
            "feature_type": feature_type,
            "feature_name": feature_name,
            "anova_f_score": float(score),
            "anova_p_value": float(pvalue),
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "ghostnet",
    "deep_features": 1280,
    "handcrafted_features": 104,
    "fused_features": 1384,
    "selected_features": K_BEST,
    "feature_selection": "SelectKBest",
    "score_function": "f_classif",
    "scaler": "StandardScaler",
    "classifier": "SVC",
    "kernel": SVM_KERNEL,
    "C": SVM_C,
    "gamma": SVM_GAMMA,
    "validation": val_metrics,
    "test": test_metrics,
    "selected_feature_details": selected_info,
}


with open(
    OUTPUT_DIR / "fusion_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


np.savez(
    OUTPUT_DIR / "selected_features.npz",
    selected_indices=selected_indices,
    selected_scores=selected_scores,
    selected_pvalues=selected_pvalues,
)


print()
print("=" * 70)
print("GHOSTNET FUSION COMPLETE")
print("=" * 70)

print(
    "Results:",
    OUTPUT_DIR / "fusion_metrics.json"
)

print(
    "Selected features:",
    OUTPUT_DIR / "selected_features.npz"
)