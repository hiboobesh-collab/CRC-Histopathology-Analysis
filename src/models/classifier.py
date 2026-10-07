
import torch.nn as nn
from torchvision import models
import timm

from src.models.residual_multiscale_cnn import (
    build_residual_multiscale_cnn
)


NUM_CLASSES = 2


def build_model(
    name="resnet18",
    num_classes=NUM_CLASSES,
    pretrained=True
):
    """
    Build an image classification model for 2-class
    colorectal cancer histopathology classification.

    Classes:
        0 -> colon_n
        1 -> colon_aca

    Supported models:
        - resnet18
        - resnet50
        - densenet121
        - vgg16
        - inception_v3
        - ghostnet
        - efficientnet_b0
        - custom_cnn
        - residual_multiscale_cnn
    """

    name = name.lower()

    # ============================================================
    # ResNet18
    # ============================================================
    if name == "resnet18":

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet18(
            weights=weights
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    # ============================================================
    # ResNet50
    # ============================================================
    elif name == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    # ============================================================
    # DenseNet121
    # ============================================================
    elif name == "densenet121":

        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.densenet121(
            weights=weights
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    # ============================================================
    # VGG16
    # ============================================================
    elif name == "vgg16":

        weights = (
            models.VGG16_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.vgg16(
            weights=weights
        )

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            num_classes
        )

    # ============================================================
    # Inception V3
    # ============================================================
    elif name == "inception_v3":

        weights = (
            models.Inception_V3_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.inception_v3(
            weights=weights,
            aux_logits=True
        )

        # Main classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        # Auxiliary classifier
        if model.AuxLogits is not None:

            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features,
                num_classes
            )

    # ============================================================
    # GhostNet
    # ============================================================
    elif name == "ghostnet":

        model = timm.create_model(
            "ghostnet_100",
            pretrained=pretrained,
            num_classes=num_classes
        )

    # ============================================================
    # EfficientNet-B0
    # ============================================================
    elif name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    # ============================================================
    # Custom CNN
    # ============================================================
    elif name == "custom_cnn":

        model = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            ),
        )

    # ============================================================
    # Residual Multi-Scale CNN
    # ============================================================
    elif name == "residual_multiscale_cnn":

        model = build_residual_multiscale_cnn(
            num_classes=num_classes
        )

    # ============================================================
    # Invalid model
    # ============================================================
    else:

        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose from: "
            "resnet18, "
            "resnet50, "
            "densenet121, "
            "vgg16, "
            "inception_v3, "
            "ghostnet, "
            "efficientnet_b0, "
            "custom_cnn, "
            "residual_multiscale_cnn"
        )

    return model
