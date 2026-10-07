import torch
from PIL import Image

from src.models.classifier import build_model
from src.data.comparison_transforms import get_val_test_transform


CHECKPOINT = (
    "models/classification/inception_v3_groupaware/"
    "best/inception_v3_best.pth"
)

IMAGE_PATH = (
    "data/lung_colon_image_set/colon_image_sets/"
    "colon_aca/colonca1.jpeg"
)

device = torch.device("cpu")


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

model = model.to(device)


transform = get_val_test_transform(299)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

x = transform(image).unsqueeze(0).to(device)


def predict(mode):

    if mode == "eval":
        model.eval()
    else:
        model.train()

    # Inception auxiliary branch cannot handle batch size 1
    # reliably in training mode, so duplicate the image.
    xb = x.repeat(2, 1, 1, 1)

    with torch.no_grad():

        output = model(xb)

        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        probs = torch.softmax(
            logits,
            dim=1
        )

    print()
    print("=" * 60)
    print(mode.upper(), "MODE")
    print("=" * 60)

    print("Logits:", logits[0])
    print("Probabilities:", probs[0])
    print(
        "Prediction:",
        torch.argmax(probs[0]).item()
    )


predict("eval")
predict("train")