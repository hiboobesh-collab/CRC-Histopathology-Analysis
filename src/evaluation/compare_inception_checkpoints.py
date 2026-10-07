import torch
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.models.classifier import build_model
from src.data.dataset import ColonHistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")

VAL_CSV = "data/splits/group_val.csv"
TRANSFORM = get_val_test_transform(299)

dataset = ColonHistopathologyDataset(
    csv_path=VAL_CSV,
    transform=TRANSFORM
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

checkpoints = [
    (
        "BEST_EPOCH_8",
        "models/classification/inception_v3_groupaware/best/inception_v3_best.pth"
    ),
    (
        "LATEST_EPOCH_10",
        "models/classification/inception_v3_groupaware/checkpoints/latest.pth"
    ),
    (
        "FINAL",
        "models/classification/inception_v3_groupaware/final/inception_v3_final.pth"
    ),
]

print("=" * 80)
print("INCEPTION V3 CHECKPOINT COMPARISON")
print("=" * 80)
print(f"Validation samples: {len(dataset)}")

for name, checkpoint_path in checkpoints:

    print("\n" + "-" * 80)
    print(name)
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(
        "inception_v3",
        2,
        False
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0
    predictions = []
    probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs[:, 1].cpu().tolist())

    predictions = torch.tensor(predictions)
    probabilities = torch.tensor(probabilities)

    print("Epoch metadata:", checkpoint.get("epoch", "N/A"))
    print("Recorded val accuracy:", checkpoint.get("val_accuracy", "N/A"))
    print("Recorded val loss:", checkpoint.get("val_loss", "N/A"))

    print("ACTUAL validation accuracy:", correct / total)
    print(
        "Predicted counts:",
        torch.bincount(predictions, minlength=2).tolist()
    )
    print(
        "Mean P(class 1):",
        probabilities.mean().item()
    )
    print(
        "Min P(class 1):",
        probabilities.min().item()
    )
    print(
        "Max P(class 1):",
        probabilities.max().item()
    )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
