from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# IMAGE PATHS
# ============================================================

RESNET18_IMAGE = Path(
    "data/lung_colon_image_set/"
    "colon_image_sets/colon_aca/"
    "colonca1907.jpeg"
)

RESIDUAL_CNN_IMAGE = Path(
    "data/lung_colon_image_set/"
    "colon_image_sets/colon_n/"
    "colonn472.jpeg"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    "results/classification/"
    "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGES
# ============================================================

resnet_image = Image.open(
    RESNET18_IMAGE
).convert("RGB")

residual_image = Image.open(
    RESIDUAL_CNN_IMAGE
).convert("RGB")


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)


# ============================================================
# RESNET18 ERROR
# ============================================================

axes[0].imshow(
    resnet_image
)

axes[0].set_title(
    "ResNet18 — Misclassified Sample",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel(
    "True: colon_aca\n"
    "Predicted: colon_n\n"
    "P(colon_aca) = 26.37%\n"
    "Group ID = 171",
    fontsize=11
)

axes[0].axis("off")


# ============================================================
# RESIDUAL MULTI-SCALE CNN ERROR
# ============================================================

axes[1].imshow(
    residual_image
)

axes[1].set_title(
    "Residual Multi-Scale CNN — Misclassified Sample",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel(
    "True: colon_n\n"
    "Predicted: colon_aca\n"
    "P(colon_aca) = 64.59%\n"
    "Group ID = 290",
    fontsize=11
)

axes[1].axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Misclassification Analysis — Group-Aware Test Set",
    fontsize=17,
    fontweight="bold"
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR /
    "misclassification_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("MISCLASSIFICATION VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"Saved to: {output_path}"
)