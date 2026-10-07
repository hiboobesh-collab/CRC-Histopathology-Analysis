import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.classifier import build_model
from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 16

CHECKPOINT = Path(
    "models/classification/"
    "resnet18_groupaware/best/"
    "resnet18_best.pth"
)

SPLITS = {
    "group_train": Path(
        "data/splits/group_train.csv"
    ),
    "group_val": Path(
        "data/splits/group_val.csv"
    ),
    "group_test": Path(
        "data/splits/group_test.csv"
    ),
}

OUTPUT_DIR = Path(
    "results/features/deep/resnet18"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("RESNET18 DEEP FEATURE EXTRACTION")
print("=" * 70)

print("Checkpoint:", CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model = build_model(
    "resnet18",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# REMOVE FINAL CLASSIFIER
# ============================================================

# ResNet18:
# backbone -> avgpool -> flatten -> fc
#
# Replace fc with Identity so the output is the
# 512-dimensional deep feature vector.

model.fc = nn.Identity()

model.eval()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

transform = get_val_test_transform(
    IMAGE_SIZE
)


for split_name, csv_path in SPLITS.items():

    print()
    print("-" * 70)
    print("Processing:", split_name)
    print("CSV:", csv_path)
    print("-" * 70)

    dataset = ColonHistopathologyDataset(
        csv_path=csv_path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    all_features = []
    all_labels = []

    print(
        "Images:",
        len(dataset)
    )

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)

            features = model(images)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            if (
                batch_idx + 1
            ) % 50 == 0:

                print(
                    f"Processed "
                    f"{(batch_idx + 1) * BATCH_SIZE} "
                    f"/ {len(dataset)}"
                )

    X = np.concatenate(
        all_features,
        axis=0
    )

    y = np.concatenate(
        all_labels,
        axis=0
    )

    output_path = (
        OUTPUT_DIR /
        f"{split_name}_features.npz"
    )

    np.savez(
        output_path,
        X=X,
        y=y
    )

    print()
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print(
        "NaN:",
        np.isnan(X).any()
    )
    print(
        "Inf:",
        np.isinf(X).any()
    )

    print(
        "Saved:",
        output_path
    )


print()
print("=" * 70)
print("DEEP FEATURE EXTRACTION COMPLETE")
print("=" * 70)