import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform

DEVICE = torch.device("cpu")


class HistopathologyDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {
            "colon_n": 0,
            "colon_aca": 1,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        label = self.label_map[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


dataset = HistopathologyDataset(
    "data/splits/group_val.csv",
    get_val_test_transform(299)
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


checkpoint = torch.load(
    "models/classification/inception_v3_groupaware/best/inception_v3_best.pth",
    map_location="cpu",
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


def evaluate(mode):

    if mode == "eval":
        model.eval()
    else:
        model.train()

    correct = 0
    total = 0
    predictions = []

    first = True

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            preds = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            predictions.extend(
                preds.tolist()
            )

            if first:

                print()
                print("=" * 70)
                print(mode.upper(), "MODE - FIRST BATCH")
                print("=" * 70)

                print("Output type:", type(outputs))

                print(
                    "Labels:",
                    labels.tolist()
                )

                print(
                    "Predictions:",
                    preds.tolist()
                )

                print("Logits:")
                print(logits[:5])

                print("Probabilities:")
                print(
                    torch.softmax(
                        logits[:5],
                        dim=1
                    )
                )

                first = False

    print()
    print(mode.upper(), "MODE RESULT")
    print("Accuracy:", correct / total)

    print(
        "Predicted counts:",
        torch.bincount(
            torch.tensor(predictions),
            minlength=2
        ).tolist()
    )


print("=" * 70)
print("INCEPTION V3 EVAL VS TRAIN MODE")
print("=" * 70)

evaluate("eval")
evaluate("train")

print("=" * 70)
print("DONE")
print("=" * 70)
