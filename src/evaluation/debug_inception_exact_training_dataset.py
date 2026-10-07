import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.training.train_inception_v3_groupaware import HistopathologyDataset
from src.data.comparison_transforms import get_val_test_transform
from src.models.classifier import build_model

DEVICE = torch.device("cpu")

VAL_CSV = Path("data/splits/group_val.csv")
IMAGE_SIZE = 299

print("=" * 80)
print("EXACT TRAINING DATASET REPRODUCTION TEST")
print("=" * 80)

transform = get_val_test_transform(IMAGE_SIZE)

dataset = HistopathologyDataset(
    VAL_CSV,
    transform=transform
)

print("Dataset length:", len(dataset))
print("First 5 labels:", [dataset[i][1] for i in range(5)])

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

checkpoint = torch.load(
    "models/classification/inception_v3_groupaware/best/inception_v3_best.pth",
    map_location=DEVICE,
    weights_only=False
)

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()

correct = 0
total = 0
predictions = []
labels_all = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        predictions.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

print()
print("Checkpoint epoch:", checkpoint["epoch"])
print("Recorded validation accuracy:", checkpoint["val_accuracy"])
print()
print("ACTUAL accuracy using EXACT training dataset class:", correct / total)
print(
    "Predicted counts:",
    torch.bincount(torch.tensor(predictions), minlength=2).tolist()
)
print(
    "True counts:",
    torch.bincount(torch.tensor(labels_all), minlength=2).tolist()
)

print("=" * 80)
print("DONE")
print("=" * 80)
