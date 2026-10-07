import json
import glob
import os
import csv
import math


# ============================================================
# LOAD METRICS
# ============================================================

def get_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Fusion results have metrics inside "test"
    if "test" in data:
        return data["test"]

    # Classification and handcrafted-only results
    return data


# ============================================================
# MODEL NAME
# ============================================================

def get_name(path):
    parent = os.path.basename(os.path.dirname(path))

    # Handcrafted result
    if "handcrafted_only" in path:
        return "Handcrafted-only SVM"

    # Fusion results
    if os.path.normpath("results/fusion") in os.path.normpath(path):
        return parent + " + Handcrafted"

    # Split B
    if parent == "resnet18_groupaware_b":
        return "ResNet18 (Split B)"

    # Classification models
    mapping = {
        "resnet18_groupaware": "ResNet18",
        "densenet121_groupaware": "DenseNet121",
        "efficientnet_b0_groupaware": "EfficientNet-B0",
        "vgg16_groupaware": "VGG16",
        "ghostnet_groupaware": "GhostNet",
        "inception_v3_groupaware": "InceptionV3",
        "residual_multiscale_cnn_groupaware":
            "Residual Multi-Scale CNN",
    }

    return mapping.get(parent, parent)


# ============================================================
# GET VALUE FROM JSON
# ============================================================

def get_value(data, *keys, default=float("nan")):
    """
    Return the first available value from the supplied keys.
    """

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


# ============================================================
# CONVERT NaN / INF TO JSON-SAFE VALUES
# ============================================================

def clean_for_json(value):
    """
    Convert NaN and infinity to None so that the resulting
    JSON is fully standards-compliant.
    """

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            key: clean_for_json(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            clean_for_json(item)
            for item in value
        ]

    return value


# ============================================================
# FIND RESULT FILES
# ============================================================

classification_files = glob.glob(
    "results/classification/*/test_metrics.json"
)

fusion_files = glob.glob(
    "results/fusion/*/fusion_metrics.json"
)

handcrafted_file = [
    "results/analysis/handcrafted_only/metrics.json"
]

files = (
    classification_files
    + fusion_files
    + handcrafted_file
)


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 125)

print(
    f"{'MODEL':40}"
    f"{'ACC':>10}"
    f"{'PREC':>10}"
    f"{'SENS':>10}"
    f"{'SPEC':>10}"
    f"{'F1':>10}"
    f"{'AUC':>10}"
)

print("=" * 125)


# ============================================================
# PROCESS EACH RESULT FILE
# ============================================================

for path in files:

    data = get_metrics(path)

    name = get_name(path)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = get_value(
        data,
        "accuracy"
    )

    precision = get_value(
        data,
        "precision"
    )

    sensitivity = get_value(
        data,
        "sensitivity",
        "recall"
    )

    specificity = get_value(
        data,
        "specificity"
    )

    # Classification files use:
    #     f1_score
    #
    # Fusion files may use:
    #     f1
    f1 = get_value(
        data,
        "f1_score",
        "f1"
    )

    auc = get_value(
        data,
        "roc_auc",
        "auc"
    )

    # --------------------------------------------------------
    # Confusion matrix values
    # --------------------------------------------------------

    tn = get_value(
        data,
        "true_negatives",
        "tn"
    )

    fp = get_value(
        data,
        "false_positives",
        "fp"
    )

    fn = get_value(
        data,
        "false_negatives",
        "fn"
    )

    tp = get_value(
        data,
        "true_positives",
        "tp"
    )

    total = get_value(
        data,
        "test_samples",
        "total"
    )

    # --------------------------------------------------------
    # Calculate number of correct predictions
    # --------------------------------------------------------

    if (
        not math.isnan(tn)
        and not math.isnan(tp)
    ):
        correct = tn + tp

    else:
        correct = float("nan")

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        f"{name:40}"
        f"{accuracy * 100:10.2f}"
        f"{precision * 100:10.2f}"
        f"{sensitivity * 100:10.2f}"
        f"{specificity * 100:10.2f}"
        f"{f1 * 100:10.2f}"
        f"{auc * 100:10.2f}"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "model": name,

        "accuracy": accuracy,

        "precision": precision,

        "sensitivity": sensitivity,

        "specificity": specificity,

        "f1": f1,

        "roc_auc": auc,

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp,

        "total": total,

        "correct": correct
    })


# ============================================================
# PROCESSING SUMMARY
# ============================================================

print("=" * 125)

print(
    f"Total files processed: {len(files)}"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = "results/analysis/final"

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# SAVE FINAL CSV
# ============================================================

csv_path = os.path.join(
    output_dir,
    "final_statistics.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "total",
            "correct"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# CLEAN RESULTS FOR JSON
# ============================================================

clean_results = clean_for_json(results)


# ============================================================
# SAVE FINAL JSON
# ============================================================

json_path = os.path.join(
    output_dir,
    "final_statistics.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        clean_results,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print("=" * 125)

print(
    "FINAL COMPARISON SAVED"
)

print("=" * 125)

print(
    f"CSV  : {csv_path}"
)

print(
    f"JSON : {json_path}"
)

print("=" * 125)