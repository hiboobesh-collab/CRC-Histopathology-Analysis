import json
from pathlib import Path

import numpy as np


BASE_DIR = Path("results/fusion")

MODELS = [
    "resnet18",
    "densenet121",
    "vgg16",
    "ghostnet",
]


print("=" * 80)
print("FUSION FEATURE ANALYSIS")
print("=" * 80)


summary = {}


for model_name in MODELS:

    print()
    print("-" * 80)
    print(model_name.upper())
    print("-" * 80)

    feature_file = (
        BASE_DIR /
        model_name /
        "selected_features.npz"
    )

    metrics_file = (
        BASE_DIR /
        model_name /
        "fusion_metrics.json"
    )

    data = np.load(
        feature_file
    )

    selected_indices = data[
        "selected_indices"
    ]

    selected_scores = data[
        "selected_scores"
    ]

    selected_pvalues = data[
        "selected_pvalues"
    ]

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    details = metrics[
        "selected_feature_details"
    ]

    deep_features = [
        x for x in details
        if x["feature_type"] == "deep"
    ]

    handcrafted_features = [
        x for x in details
        if x["feature_type"] == "handcrafted"
    ]

    print(
        "Total selected:",
        len(details)
    )

    print(
        "Deep selected:",
        len(deep_features)
    )

    print(
        "Handcrafted selected:",
        len(handcrafted_features)
    )

    print()
    print("Selected features:")

    for item in details:

        print(
            f'{item["rank"]:2d}. '
            f'{item["feature_type"]:13s} '
            f'{item["feature_name"]:35s} '
            f'F={item["anova_f_score"]:.4f}'
        )

    summary[model_name] = {
        "total_selected": len(details),
        "deep_selected": len(deep_features),
        "handcrafted_selected": len(
            handcrafted_features
        ),
        "deep_percentage": (
            len(deep_features) /
            len(details) *
            100
        ),
        "handcrafted_percentage": (
            len(handcrafted_features) /
            len(details) *
            100
        ),
    }


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for model_name, values in summary.items():

    print()
    print(model_name)

    print(
        "Deep:",
        values["deep_selected"],
        f'({values["deep_percentage"]:.1f}%)'
    )

    print(
        "Handcrafted:",
        values["handcrafted_selected"],
        f'({values["handcrafted_percentage"]:.1f}%)'
    )


output = (
    Path("results/analysis")
)

output.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    output / "fusion_feature_analysis.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "Saved:",
    output / "fusion_feature_analysis.json"
)