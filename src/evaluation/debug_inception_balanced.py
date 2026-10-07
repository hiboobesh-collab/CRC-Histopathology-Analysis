import torch
import pandas as pd
from PIL import Image
from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

CSV = "data/splits/group_val.csv"

device = torch.device("cpu")

df = pd.read_csv(CSV)

normal = df[df["label"] == "colon_n"].head(10)
cancer = df[df["label"] == "colon_aca"].head(10)

samples = pd.concat([normal, cancer], ignore_index=True)

transform = get_val_test_transform(299)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model = build_model(
    "inception_v3",
    2,
    False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 70)
print("BALANCED INCEPTION V3 DEBUG")
print("=" * 70)

correct = 0

with torch.no_grad():

    for _, row in samples.iterrows():

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        x = transform(image).unsqueeze(0)

        output = model(x)

        logits = (
            output.logits
            if hasattr(output, "logits")
            else output
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        pred = torch.argmax(probs).item()

        true = 0 if row["label"] == "colon_n" else 1

        if pred == true:
            correct += 1

        print(
            f"{row['filename']:20s} "
            f"true={true} "
            f"pred={pred} "
            f"P0={probs[0]:.4f} "
            f"P1={probs[1]:.4f}"
        )

print()
print("=" * 70)
print(f"Correct: {correct}/20")
print(f"Accuracy: {correct / 20:.2%}")
print("=" * 70)