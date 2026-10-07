from pathlib import Path
import numpy as np


FEATURE_DIR = Path(
    "results/features/handcrafted"
)


FILES = [
    "group_train_features.npz",
    "group_val_features.npz",
    "group_test_features.npz",
]


print("=" * 70)
print("HANDCRAFTED FEATURE VERIFICATION")
print("=" * 70)


for filename in FILES:

    path = FEATURE_DIR / filename

    print()
    print("-" * 70)
    print(filename)
    print("-" * 70)

    if not path.exists():

        print("ERROR: File not found")

        continue

    data = np.load(
        path,
        allow_pickle=False
    )

    X = data["X"]
    y = data["y"]
    groups = data["group_id"]
    filenames = data["filename"]
    filepaths = data["filepath"]
    feature_names = data["feature_names"]

    print(f"X shape:             {X.shape}")
    print(f"y shape:             {y.shape}")
    print(f"group_id shape:      {groups.shape}")
    print(f"filename shape:      {filenames.shape}")
    print(f"filepath shape:      {filepaths.shape}")
    print(f"feature_names:       {len(feature_names)}")
    print(
        f"unique groups:       "
        f"{len(np.unique(groups))}"
    )

    print()
    print("Class distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        if label == 0:
            name = "colon_n"
        elif label == 1:
            name = "colon_aca"
        else:
            name = "UNKNOWN"

        print(
            f"  {name}: {count}"
        )

    print()
    print(
        f"NaN: {np.isnan(X).any()}"
    )

    print(
        f"Inf: {np.isinf(X).any()}"
    )

    print(
        f"Feature dimensions correct: "
        f"{X.shape[1] == 104}"
    )


print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)