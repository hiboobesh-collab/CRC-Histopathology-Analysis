import json
import glob
import csv
import os
from math import comb


# ---------------------------------------------------------
# Exact binomial 95% CI using Clopper-Pearson
# ---------------------------------------------------------

def beta_cdf_inverse(alpha, a, b):
    from scipy.stats import beta
    return beta.ppf(alpha, a, b)


def exact_binomial_ci(correct, total, confidence=0.95):
    alpha = 1 - confidence

    if correct == 0:
        lower = 0.0
    else:
        lower = beta_cdf_inverse(alpha / 2, correct, total - correct + 1)

    if correct == total:
        upper = 1.0
    else:
        upper = beta_cdf_inverse(
            1 - alpha / 2,
            correct + 1,
            total - correct
        )

    return lower, upper


# ---------------------------------------------------------
# Load metrics
# ---------------------------------------------------------

def load_result(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "test" in data:
        metrics = data["test"]
    else:
        metrics = data

    return data, metrics


# ---------------------------------------------------------
# Name results
# ---------------------------------------------------------

def result_name(path):
    path_norm = path.replace("\\", "/")

    if "/fusion/" in path_norm:
        model = os.path.basename(os.path.dirname(path))
        return model + " + Handcrafted"

    if "/handcrafted_only/" in path_norm:
        return "Handcrafted-only SVM"

    folder = os.path.basename(os.path.dirname(path))

    mapping = {
        "resnet18_groupaware": "ResNet18",
        "resnet18_groupaware_b": "ResNet18 (Split B)",
        "densenet121_groupaware": "DenseNet121",
        "vgg16_groupaware": "VGG16",
        "ghostnet_groupaware": "GhostNet",
        "inception_v3_groupaware": "InceptionV3",
    }

    return mapping.get(folder, folder)


# ---------------------------------------------------------
# Collect files
# ---------------------------------------------------------

files = (
    glob.glob("results/classification/*/test_metrics.json")
    + glob.glob("results/fusion/*/fusion_metrics.json")
    + ["results/analysis/handcrafted_only/metrics.json"]
)


records = []


# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

for path in files:

    data, metrics = load_result(path)

    name = result_name(path)

    cm = metrics.get("confusion_matrix")

    if cm:
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        total = tn + fp + fn + tp
        correct = tn + tp

        ci_low, ci_high = exact_binomial_ci(correct, total)

        # Calculate F1 directly from precision and sensitivity
        precision = metrics.get("precision", 0)
        sensitivity = metrics.get("sensitivity", 0)

        if precision + sensitivity > 0:
            f1 = (
                2 * precision * sensitivity
                / (precision + sensitivity)
            )
        else:
            f1 = 0.0

    else:
        tn = fp = fn = tp = None
        total = None
        correct = None
        ci_low = ci_high = None

        precision = metrics.get("precision", 0)
        sensitivity = metrics.get("sensitivity", 0)

        if precision + sensitivity > 0:
            f1 = (
                2 * precision * sensitivity
                / (precision + sensitivity)
            )
        else:
            f1 = 0.0

    record = {
        "model": name,
        "accuracy": metrics.get("accuracy"),
        "accuracy_ci_lower": ci_low,
        "accuracy_ci_upper": ci_high,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": metrics.get("specificity"),
        "f1": f1,
        "roc_auc": metrics.get("roc_auc"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "correct": correct,
    }

    records.append(record)


# ---------------------------------------------------------
# Print final table
# ---------------------------------------------------------

print()
print("=" * 150)
print("FINAL STATISTICAL ANALYSIS")
print("=" * 150)

print(
    f"{'MODEL':40}"
    f"{'ACC':>9}"
    f"{'95% CI':>22}"
    f"{'PREC':>9}"
    f"{'SENS':>9}"
    f"{'SPEC':>9}"
    f"{'F1':>9}"
    f"{'AUC':>9}"
)

print("-" * 150)

for r in records:

    ci = (
        f"{r['accuracy_ci_lower']*100:.2f}-"
        f"{r['accuracy_ci_upper']*100:.2f}"
        if r["accuracy_ci_lower"] is not None
        else "-"
    )

    print(
        f"{r['model']:40}"
        f"{r['accuracy']*100:9.2f}"
        f"{ci:>22}"
        f"{r['precision']*100:9.2f}"
        f"{r['sensitivity']*100:9.2f}"
        f"{r['specificity']*100:9.2f}"
        f"{r['f1']*100:9.2f}"
        f"{r['roc_auc']*100:9.2f}"
    )

print("=" * 150)


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

os.makedirs("results/analysis/final", exist_ok=True)

json_path = "results/analysis/final/final_statistics.json"

with open(json_path, "w") as f:
    json.dump(records, f, indent=2)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

csv_path = "results/analysis/final/final_statistics.csv"

fieldnames = list(records[0].keys())

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


print()
print("Saved:")
print(json_path)
print(csv_path)
print()