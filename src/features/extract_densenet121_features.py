import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform
from src.models.classifier import build_model


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

CHECKPOINT = Path(
    "models/classification/densenet121_groupaware/best/densenet121_best.pth"
)

OUTPUT_DIR = Path(
    "results/features/deep/densenet121"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPLITS = {
    "group_train": "data/splits/group_train.csv",
    "group_val": "data/splits/group_val.csv",
    "group_test": "data/splits/group_test.csv",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("DENSENET121 DEEP FEATURE EXTRACTION")
print("=" * 70)

print()
print("Checkpoint:")
print(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

print()
print("Checkpoint epoch:")
print(checkpoint.get("epoch"))

print(
    "Recorded validation accuracy:",
    checkpoint.get("val_acc")
)

print(
    "Recorded validation loss:",
    checkpoint.get("val_loss")
)


model = build_model(
    "densenet121",
    num_classes=2,
    pretrained=False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# REPLACE CLASSIFIER WITH IDENTITY
# ============================================================

# DenseNet121:
# classifier input = 1024
#
# Replacing the final classifier with Identity gives
# the 1024-dimensional deep representation.

model.classifier = nn.Identity()

model.eval()


print()
print("DenseNet121 feature extractor loaded.")
print("Feature dimension: 1024")


# ============================================================
# EXTRACT FEATURES
# ============================================================

@torch.no_grad()
def extract_features(csv_path):

    dataset = ColonHistopathologyDataset(
        csv_path=csv_path,
        transform=get_val_test_transform(
            image_size=224
        )
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    features = []
    labels = []

    total = len(dataset)

    print()
    print(
        f"Processing {csv_path}"
    )

    processed = 0

    for images, batch_labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        features.append(
            outputs.cpu().numpy()
        )

        labels.append(
            batch_labels.numpy()
        )

        processed += len(images)

        if (
            processed % 1000 == 0
            or processed == total
        ):
            print(
                f"Processed {processed}/{total}"
            )

    X = np.concatenate(
        features,
        axis=0
    )

    y = np.concatenate(
        labels,
        axis=0
    )

    return X, y


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

for split_name, csv_path in SPLITS.items():

    X, y = extract_features(
        csv_path
    )

    print()
    print(
        f"{split_name} feature shape:",
        X.shape
    )

    print(
        f"{split_name} label shape:",
        y.shape
    )

    print(
        "NaN:",
        np.isnan(X).any()
    )

    print(
        "Inf:",
        np.isinf(X).any()
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

    print(
        "Saved:",
        output_path
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("DENSENET121 FEATURE EXTRACTION COMPLETE")
print("=" * 70)