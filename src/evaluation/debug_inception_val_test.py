import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))

from src.models.classifier import build_model
from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform


DEVICE = torch.device("cpu")

VAL_CSV = "data/splits/group_val.csv"
TEST_CSV = "data/splits/group_test.csv"

CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

BATCH_SIZE = 16
NUM_CLASSES = 2
IMAGE_SIZE = 299


print("=" * 70)
print("INCEPTION V3 VALIDATION VS TEST DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading checkpoint...")

model = build_model(
    "inception_v3",
    NUM_CLASSES,
    False,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.to(DEVICE)
model.eval()

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch')}"
)

print(
    f"Checkpoint val loss: "
    f"{checkpoint.get('val_loss')}"
)

print(
    f"Checkpoint val accuracy: "
    f"{checkpoint.get('val_accuracy')}"
)


# ============================================================
# DATASET LOADER
# ============================================================

def load_csv(csv_path):

    print(f"\nLoading: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df)}")

    print("\nCSV label counts:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    transform = get_val_test_transform(
        IMAGE_SIZE
    )

    # IMPORTANT:
    # ColonHistopathologyDataset accepts csv_path,
    # NOT dataframe.
    dataset = ColonHistopathologyDataset(
        csv_path=csv_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return df, dataset, loader


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, csv_path):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df, dataset, loader = load_csv(
        csv_path
    )

    all_true = []
    all_pred = []
    all_prob = []
    all_logits = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            if hasattr(
                outputs,
                "logits"
            ):
                logits = outputs.logits
            else:
                logits = outputs

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_true.extend(
                labels.cpu().numpy()
            )

            all_pred.extend(
                predictions.cpu().numpy()
            )

            all_prob.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            all_logits.extend(
                logits.cpu()
                .numpy()
            )

    y_true = np.array(
        all_true
    )

    y_pred = np.array(
        all_pred
    )

    p1 = np.array(
        all_prob
    )

    logits = np.array(
        all_logits
    )

    accuracy = np.mean(
        y_true == y_pred
    )

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        f"\nSamples: {len(y_true)}"
    )

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print("\nTrue counts:")

    print(
        f"  colon_n   (0): "
        f"{np.sum(y_true == 0)}"
    )

    print(
        f"  colon_aca (1): "
        f"{np.sum(y_true == 1)}"
    )

    print("\nPredicted counts:")

    print(
        f"  colon_n   (0): "
        f"{np.sum(y_pred == 0)}"
    )

    print(
        f"  colon_aca (1): "
        f"{np.sum(y_pred == 1)}"
    )

    print(
        "\nProbability statistics "
        "P(colon_aca):"
    )

    print(
        f"  mean:   {p1.mean():.6f}"
    )

    print(
        f"  median: {np.median(p1):.6f}"
    )

    print(
        f"  min:    {p1.min():.6f}"
    )

    print(
        f"  max:    {p1.max():.6f}"
    )

    print(
        "\nProbability by TRUE class:"
    )

    for label, class_name in [
        (0, "colon_n"),
        (1, "colon_aca"),
    ]:

        mask = y_true == label

        class_probs = p1[mask]

        print(
            f"  {class_name}: "
            f"count={len(class_probs)}, "
            f"mean={class_probs.mean():.6f}, "
            f"median={np.median(class_probs):.6f}, "
            f"min={class_probs.min():.6f}, "
            f"max={class_probs.max():.6f}"
        )

    wrong = np.where(
        y_true != y_pred
    )[0]

    print(
        f"\nIncorrect predictions: "
        f"{len(wrong)}"
    )

    if len(wrong) > 0:

        print(
            "\nFirst 20 incorrect predictions:"
        )

        for i in wrong[:20]:

            print(
                f"{i:4d} | "
                f"{df.iloc[i]['filename']:25s} | "
                f"true={y_true[i]} | "
                f"pred={y_pred[i]} | "
                f"P1={p1[i]:.6f} | "
                f"logits=["
                f"{logits[i,0]:.4f}, "
                f"{logits[i,1]:.4f}]"
            )

    return {
        "df": df,
        "y_true": y_true,
        "y_pred": y_pred,
        "p1": p1,
        "logits": logits,
    }


# ============================================================
# VALIDATION
# ============================================================

val_result = evaluate(
    "VALIDATION SET",
    VAL_CSV,
)


# ============================================================
# TEST
# ============================================================

test_result = evaluate(
    "TEST SET",
    TEST_CSV,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION VS TEST SUMMARY")
print("=" * 70)

val_accuracy = np.mean(
    val_result["y_true"]
    == val_result["y_pred"]
)

test_accuracy = np.mean(
    test_result["y_true"]
    == test_result["y_pred"]
)

print(
    f"\nValidation accuracy: "
    f"{val_accuracy:.6f}"
)

print(
    f"Test accuracy:       "
    f"{test_accuracy:.6f}"
)

print(
    f"\nValidation mean P(colon_aca): "
    f"{val_result['p1'].mean():.6f}"
)

print(
    f"Test mean P(colon_aca):       "
    f"{test_result['p1'].mean():.6f}"
)

print("\nClass-wise comparison:")

for label, class_name in [
    (0, "colon_n"),
    (1, "colon_aca"),
]:

    val_mask = (
        val_result["y_true"] == label
    )

    test_mask = (
        test_result["y_true"] == label
    )

    print(
        f"\n{class_name}:"
    )

    print(
        f"  Validation mean P1: "
        f"{val_result['p1'][val_mask].mean():.6f}"
    )

    print(
        f"  Test mean P1:       "
        f"{test_result['p1'][test_mask].mean():.6f}"
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
