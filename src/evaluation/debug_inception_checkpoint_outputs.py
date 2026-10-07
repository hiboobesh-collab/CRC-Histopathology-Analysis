import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.append(os.path.abspath("."))

from src.models.classifier import build_model
from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cpu")

VAL_CSV = "data/splits/group_val.csv"

CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

BATCH_SIZE = 16
NUM_CLASSES = 2
IMAGE_SIZE = 299


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INCEPTION V3 CHECKPOINT OUTPUT DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

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

print(
    f"\nCheckpoint epoch: "
    f"{checkpoint.get('epoch')}"
)

print(
    f"Checkpoint recorded val accuracy: "
    f"{checkpoint.get('val_accuracy')}"
)

print(
    f"Checkpoint recorded val loss: "
    f"{checkpoint.get('val_loss')}"
)


# ============================================================
# DATASET
# ============================================================

transform = get_val_test_transform(
    IMAGE_SIZE
)

dataset = ColonHistopathologyDataset(
    csv_path=VAL_CSV,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# OUTPUT EXTRACTION
# ============================================================

def extract_outputs(outputs):

    """
    Handle both possible InceptionV3 output formats.

    Depending on how the model is configured/wrapped,
    outputs can be:

        1. Tensor
        2. InceptionOutputs with:
              outputs.logits
              outputs.aux_logits
    """

    # Plain Tensor
    if isinstance(outputs, torch.Tensor):

        main_logits = outputs
        aux_logits = None

        return main_logits, aux_logits

    # InceptionOutputs / named tuple
    if hasattr(outputs, "logits"):

        main_logits = outputs.logits

        if hasattr(outputs, "aux_logits"):
            aux_logits = outputs.aux_logits
        else:
            aux_logits = None

        return main_logits, aux_logits

    # Generic tuple/list fallback
    if isinstance(outputs, (tuple, list)):

        main_logits = outputs[0]

        if len(outputs) > 1:
            aux_logits = outputs[1]
        else:
            aux_logits = None

        return main_logits, aux_logits

    raise TypeError(
        f"Unsupported model output type: "
        f"{type(outputs)}"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_mode(mode):

    if mode == "eval":

        model.eval()

    elif mode == "train":

        model.train()

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    main_true = []
    main_pred = []
    main_prob = []

    aux_true = []
    aux_pred = []
    aux_prob = []

    first_output_type = None

    print(
        f"\nRunning model in "
        f"{mode.upper()} mode..."
    )

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Record output type
            # ------------------------------------------------

            if first_output_type is None:

                first_output_type = type(
                    outputs
                )

                print(
                    f"Output type: "
                    f"{first_output_type}"
                )

            # ------------------------------------------------
            # Extract logits
            # ------------------------------------------------

            main_logits, aux_logits = (
                extract_outputs(outputs)
            )

            # ------------------------------------------------
            # MAIN CLASSIFIER
            # ------------------------------------------------

            main_probs = torch.softmax(
                main_logits,
                dim=1
            )

            main_predictions = torch.argmax(
                main_probs,
                dim=1
            )

            main_true.extend(
                labels.cpu().numpy()
            )

            main_pred.extend(
                main_predictions
                .cpu()
                .numpy()
            )

            main_prob.extend(
                main_probs[:, 1]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # AUXILIARY CLASSIFIER
            # ------------------------------------------------

            if (
                aux_logits is not None
                and isinstance(
                    aux_logits,
                    torch.Tensor
                )
            ):

                aux_probs = torch.softmax(
                    aux_logits,
                    dim=1
                )

                aux_predictions = (
                    torch.argmax(
                        aux_probs,
                        dim=1
                    )
                )

                aux_true.extend(
                    labels.cpu().numpy()
                )

                aux_pred.extend(
                    aux_predictions
                    .cpu()
                    .numpy()
                )

                aux_prob.extend(
                    aux_probs[:, 1]
                    .cpu()
                    .numpy()
                )

    # ========================================================
    # MAIN RESULTS
    # ========================================================

    main_true = np.array(
        main_true
    )

    main_pred = np.array(
        main_pred
    )

    main_prob = np.array(
        main_prob
    )

    main_accuracy = np.mean(
        main_true == main_pred
    )

    print(
        "\nMAIN CLASSIFIER:"
    )

    print(
        f"  Accuracy: "
        f"{main_accuracy:.6f}"
    )

    print(
        f"  Predictions: "
        f"["
        f"{np.sum(main_pred == 0)}, "
        f"{np.sum(main_pred == 1)}"
        f"]"
    )

    print(
        f"  Mean P(class 1): "
        f"{main_prob.mean():.6f}"
    )

    print(
        f"  Min P(class 1): "
        f"{main_prob.min():.6f}"
    )

    print(
        f"  Max P(class 1): "
        f"{main_prob.max():.6f}"
    )

    # ========================================================
    # AUX RESULTS
    # ========================================================

    if len(aux_true) > 0:

        aux_true = np.array(
            aux_true
        )

        aux_pred = np.array(
            aux_pred
        )

        aux_prob = np.array(
            aux_prob
        )

        aux_accuracy = np.mean(
            aux_true == aux_pred
        )

        print(
            "\nAUXILIARY CLASSIFIER:"
        )

        print(
            f"  Accuracy: "
            f"{aux_accuracy:.6f}"
        )

        print(
            f"  Predictions: "
            f"["
            f"{np.sum(aux_pred == 0)}, "
            f"{np.sum(aux_pred == 1)}"
            f"]"
        )

        print(
            f"  Mean P(class 1): "
            f"{aux_prob.mean():.6f}"
        )

        print(
            f"  Min P(class 1): "
            f"{aux_prob.min():.6f}"
        )

        print(
            f"  Max P(class 1): "
            f"{aux_prob.max():.6f}"
        )

    else:

        print(
            "\nAUXILIARY CLASSIFIER:"
        )

        print(
            "  Not available."
        )


# ============================================================
# EVAL MODE
# ============================================================

evaluate_mode(
    "eval"
)


# ============================================================
# TRAIN MODE
# ============================================================

evaluate_mode(
    "train"
)


# ============================================================
# FINISH
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DIAGNOSTIC COMPLETE"
)

print(
    "=" * 70
)