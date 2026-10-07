import numpy as np
from pathlib import Path


DEEP_DIR = Path(
    "results/features/deep/resnet18"
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
print("RESNET18 FUSION INPUT VERIFICATION")
print("=" * 70)


for split in SPLITS:

    print()
    print("-" * 70)
    print(split)
    print("-" * 70)

    deep_path = (
        DEEP_DIR /
        f"{split}_features.npz"
    )

    hand_path = (
        HAND_DIR /
        f"{split}_features.npz"
    )

    deep = np.load(
        deep_path,
        allow_pickle=True
    )

    hand = np.load(
        hand_path,
        allow_pickle=True
    )

    X_deep = deep["X"]
    y_deep = deep["y"]

    X_hand = hand["X"]
    y_hand = hand["y"]

    print("Deep shape:", X_deep.shape)
    print("Handcrafted shape:", X_hand.shape)

    print("Deep labels:", y_deep.shape)
    print("Handcrafted labels:", y_hand.shape)

    print(
        "Sample count match:",
        len(y_deep) == len(y_hand)
    )

    print(
        "Labels identical:",
        np.array_equal(
            y_deep,
            y_hand
        )
    )

    print(
        "Deep NaN:",
        np.isnan(X_deep).any()
    )

    print(
        "Deep Inf:",
        np.isinf(X_deep).any()
    )

    print(
        "Handcrafted NaN:",
        np.isnan(X_hand).any()
    )

    print(
        "Handcrafted Inf:",
        np.isinf(X_hand).any()
    )

    if len(y_deep) != len(y_hand):
        raise RuntimeError(
            f"{split}: sample counts do not match"
        )

    if not np.array_equal(
        y_deep,
        y_hand
    ):
        raise RuntimeError(
            f"{split}: labels do not match"
        )


print()
print("=" * 70)
print("ALL FUSION INPUT CHECKS PASSED")
print("=" * 70)