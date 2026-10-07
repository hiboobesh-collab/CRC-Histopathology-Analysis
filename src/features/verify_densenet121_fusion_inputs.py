import numpy as np
from pathlib import Path


DEEP_DIR = Path(
    "results/features/deep/densenet121"
)

HAND_DIR = Path(
    "results/features/handcrafted"
)

SPLITS = [
    "group_train",
    "group_val",
    "group_test",
]


print("=" * 70)
print("DENSENET121 FUSION INPUT VERIFICATION")
print("=" * 70)


all_passed = True


for split in SPLITS:

    print()
    print("-" * 70)
    print(split)
    print("-" * 70)

    deep = np.load(
        DEEP_DIR /
        f"{split}_features.npz"
    )

    hand = np.load(
        HAND_DIR /
        f"{split}_features.npz"
    )

    X_deep = deep["X"]
    X_hand = hand["X"]

    y_deep = deep["y"]
    y_hand = hand["y"]

    sample_count_match = (
        X_deep.shape[0]
        == X_hand.shape[0]
    )

    labels_identical = np.array_equal(
        y_deep,
        y_hand
    )

    deep_nan = np.isnan(
        X_deep
    ).any()

    deep_inf = np.isinf(
        X_deep
    ).any()

    hand_nan = np.isnan(
        X_hand
    ).any()

    hand_inf = np.isinf(
        X_hand
    ).any()

    print(
        "Deep shape:",
        X_deep.shape
    )

    print(
        "Handcrafted shape:",
        X_hand.shape
    )

    print(
        "Deep labels:",
        y_deep.shape
    )

    print(
        "Handcrafted labels:",
        y_hand.shape
    )

    print(
        "Sample count match:",
        sample_count_match
    )

    print(
        "Labels identical:",
        labels_identical
    )

    print(
        "Deep NaN:",
        deep_nan
    )

    print(
        "Deep Inf:",
        deep_inf
    )

    print(
        "Handcrafted NaN:",
        hand_nan
    )

    print(
        "Handcrafted Inf:",
        hand_inf
    )

    if not (
        sample_count_match
        and labels_identical
        and not deep_nan
        and not deep_inf
        and not hand_nan
        and not hand_inf
    ):
        all_passed = False


print()
print("=" * 70)

if all_passed:
    print("ALL DENSENET121 FUSION INPUT CHECKS PASSED")
else:
    print("DENSENET121 FUSION INPUT CHECK FAILED")

print("=" * 70)

if not all_passed:
    raise SystemExit(1)