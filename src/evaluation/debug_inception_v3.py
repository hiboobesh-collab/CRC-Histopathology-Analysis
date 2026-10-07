import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


MODEL_NAME = "inception_v3"
IMAGE_SIZE = 299
BATCH_SIZE = 16

CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

LABEL_MAP = {
    "colon_n": 0,
    "colon_aca": 1,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class DatasetCheck(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.transform = get_val_test_transform(
            IMAGE_SIZE
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        image = self.transform(image)

        label = LABEL_MAP[row["label"]]

        return (
            image,
            label,
            row["filename"],
        )


def evaluate(csv_path, name):

    dataset = DatasetCheck(csv_path)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        MODEL_NAME,
        2,
        False,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []
    probs = []

    with torch.no_grad():

        for images, labels, filenames in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            p = torch.softmax(
                logits,
                dim=1,
            )

            pred = torch.argmax(
                p,
                dim=1,
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )

            probs.extend(
                p.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.array(probs)

    accuracy = (
        y_true == y_pred
    ).mean()

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Samples:", len(y_true))
    print("Accuracy:", accuracy)

    print(
        "Predicted class counts:",
        np.bincount(
            y_pred,
            minlength=2
        )
    )

    print(
        "True class counts:",
        np.bincount(
            y_true,
            minlength=2
        )
    )

    print(
        "Mean P(colon_n):",
        probs[:, 0].mean()
    )

    print(
        "Mean P(colon_aca):",
        probs[:, 1].mean()
    )

    print(
        "Min P(colon_aca):",
        probs[:, 1].min()
    )

    print(
        "Max P(colon_aca):",
        probs[:, 1].max()
    )

    print("\nFirst 10 predictions:")

    for i in range(
        min(10, len(y_true))
    ):

        print(
            f"{dataset.df.iloc[i]['filename']} | "
            f"true={y_true[i]} | "
            f"pred={y_pred[i]} | "
            f"P0={probs[i,0]:.6f} | "
            f"P1={probs[i,1]:.6f}"
        )


print("\nLoading checkpoint:")
print(CHECKPOINT)

evaluate(
    "data/splits/group_train.csv",
    "TRAIN SET"
)

evaluate(
    "data/splits/group_val.csv",
    "VALIDATION SET"
)

evaluate(
    "data/splits/group_test.csv",
    "TEST SET"
)