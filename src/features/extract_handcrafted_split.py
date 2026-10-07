"""
Batch extraction of handcrafted features for CRC histopathology.

Extracts 104 handcrafted color + texture features for every image
listed in the group-aware split CSV files.

No scaling, feature selection, or model fitting is performed here.

Label mapping:
    colon_n   -> 0
    colon_aca -> 1
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.features.handcrafted_features import (
    extract_handcrafted_features,
    get_feature_names,
)


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = Path("data/splits")

OUTPUT_DIR = Path(
    "results/features/handcrafted"
)

SPLITS = [
    "group_train",
    "group_val",
    "group_test",
]


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "colon_n": 0,
    "colon_aca": 1,
}


# ============================================================
# EXTRACT ONE SPLIT
# ============================================================

def extract_split(split_name):

    csv_path = SPLIT_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "filepath",
        "label",
        "class_name",
        "filename",
        "group_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing}"
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    unknown_labels = (
        set(df["label"].unique())
        - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels in {csv_path}: "
            f"{unknown_labels}"
        )

    feature_names = get_feature_names()

    print()
    print("=" * 70)
    print(f"EXTRACTING: {split_name}")
    print("=" * 70)

    print(f"Images: {len(df)}")
    print(
        f"Expected features per image: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    features = []
    labels = []
    group_ids = []
    filenames = []
    filepaths = []

    start_time = time.time()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, row in df.iterrows():

        image_path = Path(
            str(row["filepath"])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Extract handcrafted features
        vector = extract_handcrafted_features(
            image_path
        )

        # Validate feature count
        if len(vector) != len(feature_names):
            raise ValueError(
                f"Unexpected feature count for "
                f"{image_path}: "
                f"{len(vector)} "
                f"(expected {len(feature_names)})"
            )

        # Validate numerical values
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"NaN/Inf detected in: "
                f"{image_path}"
            )

        # Convert label
        label_name = str(
            row["label"]
        )

        if label_name not in LABEL_MAP:
            raise ValueError(
                f"Unknown label: {label_name}"
            )

        label = LABEL_MAP[label_name]

        # Store
        features.append(vector)

        labels.append(label)

        group_ids.append(
            int(row["group_id"])
        )

        filenames.append(
            str(row["filename"])
        )

        filepaths.append(
            str(row["filepath"])
        )

        # Progress
        processed = index + 1

        if (
            processed % 250 == 0
            or processed == len(df)
        ):

            elapsed = time.time() - start_time

            if elapsed > 0:
                rate = processed / elapsed
            else:
                rate = 0.0

            print(
                f"[{processed:5d}/{len(df)}] "
                f"{rate:.2f} images/sec"
            )

    # ========================================================
    # Convert to NumPy arrays
    # ========================================================

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    groups = np.asarray(
        group_ids,
        dtype=np.int64
    )

    filenames = np.asarray(
        filenames,
        dtype=str
    )

    filepaths = np.asarray(
        filepaths,
        dtype=str
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    expected_shape = (
        len(df),
        len(feature_names)
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected feature matrix shape: "
            f"{X.shape}; "
            f"expected {expected_shape}"
        )

    if y.shape[0] != len(df):
        raise ValueError(
            "Label count does not match "
            "number of images."
        )

    if groups.shape[0] != len(df):
        raise ValueError(
            "Group ID count does not match "
            "number of images."
        )

    if filenames.shape[0] != len(df):
        raise ValueError(
            "Filename count does not match "
            "number of images."
        )

    if filepaths.shape[0] != len(df):
        raise ValueError(
            "Filepath count does not match "
            "number of images."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Feature matrix contains "
            "NaN or Inf values."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}_features.npz"
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        group_id=groups,
        filename=filenames,
        filepath=filepaths,
        feature_names=np.asarray(
            feature_names,
            dtype=str
        ),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

    print()
    print("-" * 70)
    print(f"COMPLETED: {split_name}")
    print("-" * 70)

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Labels shape:         {y.shape}"
    )

    print(
        f"Groups shape:         {groups.shape}"
    )

    print(
        f"Filenames shape:      {filenames.shape}"
    )

    print(
        f"Filepaths shape:      {filepaths.shape}"
    )

    print(
        f"Output:               {output_path}"
    )

    print(
        f"Time:                 "
        f"{elapsed / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("Class distribution:")

    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique_labels,
        counts
    ):

        if label == 0:
            class_name = "colon_n"
        elif label == 1:
            class_name = "colon_aca"
        else:
            class_name = "UNKNOWN"

        print(
            f"  {class_name:10s}: {count}"
        )

    # --------------------------------------------------------
    # Group information
    # --------------------------------------------------------

    print()
    print(
        f"Unique groups: "
        f"{len(np.unique(groups))}"
    )

    print()
    print("Validation: PASSED")

    return X, y, groups


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CRC HISTOPATHOLOGY")
    print("HANDCRAFTED FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print(
        f"Feature dimensions: "
        f"{len(get_feature_names())}"
    )

    print()
    print("Label mapping:")
    print("  colon_n   -> 0")
    print("  colon_aca -> 1")

    print()

    for split_name in SPLITS:

        extract_split(
            split_name
        )

    print()
    print("=" * 70)
    print("ALL SPLITS COMPLETED")
    print("=" * 70)