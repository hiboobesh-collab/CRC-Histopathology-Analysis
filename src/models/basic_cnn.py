import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    Basic CNN baseline for colorectal histopathology classification.

    Input:
        3 x 224 x 224

    Output:
        2 classes

    Architecture:
        Conv 3x3: 3 -> 32
        BatchNorm + ReLU
        MaxPool

        Conv 3x3: 32 -> 64
        BatchNorm + ReLU
        MaxPool

        Conv 3x3: 64 -> 128
        BatchNorm + ReLU
        MaxPool

        Global Average Pooling
        Dropout
        Fully Connected Classifier
    """

    def __init__(
        self,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        # ========================================================
        # Convolutional Feature Extraction
        # ========================================================

        self.features = nn.Sequential(

            # ----------------------------------------------------
            # Layer 1
            # ----------------------------------------------------
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # ----------------------------------------------------
            # Layer 2
            # ----------------------------------------------------
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # ----------------------------------------------------
            # Layer 3
            # ----------------------------------------------------
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # ========================================================
        # Global Feature Representation
        # ========================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # ========================================================
        # Classification Head
        # ========================================================

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        # Convolutional features
        x = self.features(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.classifier(x)

        return x


def build_basic_cnn(
    num_classes=2,
    dropout=0.3
):
    """
    Factory function for Basic CNN.
    """

    return BasicCNN(
        num_classes=num_classes,
        dropout=dropout
    )


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":

    model = build_basic_cnn(
        num_classes=2
    )

    # Dummy histopathology batch
    x = torch.randn(
        2,
        3,
        224,
        224
    )

    # Forward pass
    y = model(x)

    # Parameter count
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 70)
    print("BASIC CNN")
    print("=" * 70)

    print(
        f"Input shape       : {x.shape}"
    )

    print(
        f"Output shape      : {y.shape}"
    )

    print(
        f"Total parameters  : {total_params:,}"
    )

    print(
        f"Trainable params  : {trainable_params:,}"
    )

    print("=" * 70)