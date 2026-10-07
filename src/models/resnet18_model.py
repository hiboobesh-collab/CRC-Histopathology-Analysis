import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=2, pretrained=True):
    """
    ResNet18 model for binary colorectal histopathology classification.

    Architecture:
        Input image (224 x 224 x 3)
                ?
            ResNet18
                ?
        Feature representation
                ?
        Fully Connected Layer
                ?
        2-class output

    Classes:
        0 = colon_n
        1 = colon_aca
    """

    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    # Replace ImageNet's 1000-class classifier
    # with our 2-class CRC classifier.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
