from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "classification"


def load_split(name):
    path = SPLIT_DIR / f"group_{name}.csv"
    df = pd.read_csv(path)

    print(f"\n{name.upper()} SPLIT")
    print("-" * 60)
    print("Rows:", len(df))
    print("Groups:", df["group_id"].nunique())
    print("Classes:")
    print(df["label"].value_counts().to_dict())

    return df


def check_split_integrity(train, val, test):
    print("\n" + "=" * 70)
    print("SPLIT INTEGRITY")
    print("=" * 70)

    train_paths = set(train["filepath"].astype(str))
    val_paths = set(val["filepath"].astype(str))
    test_paths = set(test["filepath"].astype(str))

    train_groups = set(train["group_id"])
    val_groups = set(val["group_id"])
    test_groups = set(test["group_id"])

    print("\nImage overlap:")
    print("Train ∩ Val :", len(train_paths & val_paths))
    print("Train ∩ Test:", len(train_paths & test_paths))
    print("Val ∩ Test  :", len(val_paths & test_paths))

    print("\nGroup overlap:")
    print("Train ∩ Val :", len(train_groups & val_groups))
    print("Train ∩ Test:", len(train_groups & test_groups))
    print("Val ∩ Test  :", len(val_groups & test_groups))

    all_paths = train_paths | val_paths | test_paths

    print("\nUnique image paths:", len(all_paths))
    print("Expected images :", len(train) + len(val) + len(test))

    missing = 0

    for path in all_paths:
        if not Path(path).exists():
            missing += 1

    print("Missing files:", missing)


def inspect_predictions(model_name, test):
    result_dir = RESULTS_DIR / model_name
    npz_path = result_dir / "test_predictions.npz"

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} PREDICTION AUDIT")
    print("=" * 70)

    if not npz_path.exists():
        print("Prediction file NOT FOUND:")
        print(npz_path)
        return

    data = np.load(npz_path)

    print("\nStored arrays:")
    for key in data.files:
        print(f"  {key}: {data[key].shape}")

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    print("\nPrediction count:")
    print("CSV test samples :", len(test))
    print("Predictions      :", len(y_pred))
    print("True labels      :", len(y_true))

    print("\nCorrect predictions:", int((y_true == y_pred).sum()))
    print("Wrong predictions  :", int((y_true != y_pred).sum()))

    accuracy = np.mean(y_true == y_pred)

    print(f"\nAccuracy: {accuracy:.6f} ({accuracy * 100:.4f}%)")

    if "probabilities" in data.files:
        probs = data["probabilities"]

        print("\nProbability audit:")
        print("Shape:", probs.shape)
        print("Finite:", np.isfinite(probs).all())
        print(
            "Probability sums approximately 1:",
            np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        )

        confidence = probs.max(axis=1)

        print("\nConfidence:")
        print("Min :", float(confidence.min()))
        print("Max :", float(confidence.max()))
        print("Mean:", float(confidence.mean()))
        print("Median:", float(np.median(confidence)))

        print("\nConfidence percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"{p:>3}% :", float(np.percentile(confidence, p)))

    csv_labels = test["label"].to_numpy()

    print("\nCSV vs prediction y_true:")
    print("Same length:", len(csv_labels) == len(y_true))

    if len(csv_labels) == len(y_true):
        print("Same labels :", np.array_equal(csv_labels, y_true))

        if not np.array_equal(csv_labels, y_true):
            mismatches = np.where(csv_labels != y_true)[0]

            print("Number of mismatches:", len(mismatches))

            print("First mismatches:")
            for idx in mismatches[:10]:
                print(
                    idx,
                    test.iloc[idx]["filename"],
                    "CSV:",
                    csv_labels[idx],
                    "NPZ:",
                    y_true[idx],
                )


def inspect_training_history(model_name):
    path = RESULTS_DIR / f"{model_name}_training_history.json"

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} TRAINING HISTORY")
    print("=" * 70)

    if not path.exists():
        print("Training history NOT FOUND:")
        print(path)
        return

    with open(path, "r") as f:
        history = json.load(f)

    print("\nHistory type:", type(history).__name__)

    if isinstance(history, dict):
        print("Keys:", list(history.keys()))

        epochs = history.get("epochs")

        if epochs is not None:
            print("Epoch records:", len(epochs))

            print("\nEpoch-by-epoch:")
            for e in epochs:
                print(e)

            return

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_accuracy", [])
        val_acc = history.get("val_accuracy", [])

        if train_loss:
            print("\nEpoch | Train Loss | Val Loss | Train Acc | Val Acc")
            print("-" * 65)

            n = max(
                len(train_loss),
                len(val_loss),
                len(train_acc),
                len(val_acc),
            )

            for i in range(n):
                tl = train_loss[i] if i < len(train_loss) else None
                vl = val_loss[i] if i < len(val_loss) else None
                ta = train_acc[i] if i < len(train_acc) else None
                va = val_acc[i] if i < len(val_acc) else None

                print(
                    f"{i+1:5d} | "
                    f"{tl!s:11} | "
                    f"{vl!s:8} | "
                    f"{ta!s:9} | "
                    f"{va!s:7}"
                )


def main():
    print("=" * 70)
    print("CRC HISTOPATHOLOGY GENERALIZATION AUDIT")
    print("=" * 70)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    check_split_integrity(train, val, test)

    inspect_predictions(
        "resnet18_groupaware",
        test,
    )

    inspect_predictions(
        "densenet121_groupaware",
        test,
    )

    inspect_training_history(
        "resnet18_groupaware"
    )

    inspect_training_history(
        "densenet121_groupaware"
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()