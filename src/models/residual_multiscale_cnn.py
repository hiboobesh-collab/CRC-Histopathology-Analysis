import torch
import torch.nn as nn


class MultiScaleResidualBlock(nn.Module):
    """
    Multi-Scale Residual Block

    Three parallel convolution branches:
        1. Standard 3x3 convolution
        2. Standard 5x5 convolution
        3. Dilated 3x3 convolution

    Their outputs are concatenated and fused using a 1x1 convolution.
    A residual/shortcut connection is then added.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # --------------------------------------------------------
        # Branch 1: Local features
        # --------------------------------------------------------
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------------
        # Branch 2: Medium-scale features
        # --------------------------------------------------------
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------------
        # Branch 3: Larger receptive field
        # Dilated 3x3 convolution
        # --------------------------------------------------------
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------------
        # Fuse the three scales
        # 3 * out_channels -> out_channels
        # --------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # --------------------------------------------------------
        # Residual shortcut
        # --------------------------------------------------------
        if (
            in_channels != out_channels
            or stride != 1
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        # Multi-scale feature extraction
        branch1 = self.branch_3x3(x)
        branch2 = self.branch_5x5(x)
        branch3 = self.branch_dilated(x)

        # Concatenate multi-scale features
        multi_scale = torch.cat(
            [
                branch1,
                branch2,
                branch3
            ],
            dim=1
        )

        # Fuse features
        fused = self.fusion(
            multi_scale
        )

        # Residual connection
        residual = self.shortcut(x)

        output = fused + residual

        output = self.activation(
            output
        )

        return output


class ResidualMultiScaleCNN(nn.Module):
    """
    Residual Multi-Scale CNN for
    colorectal histopathology classification.

    Input:
        3 x 224 x 224

    Output:
        2 classes

    Architecture:
        Initial convolution
        ↓
        Multi-Scale Residual Block 1
        ↓
        Multi-Scale Residual Block 2
        ↓
        Multi-Scale Residual Block 3
        ↓
        Global Average Pooling
        ↓
        Dropout
        ↓
        Fully Connected Classifier
    """

    def __init__(
        self,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        # ========================================================
        # Initial feature extraction
        # ========================================================

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ========================================================
        # Multi-Scale Residual Block 1
        # 32 -> 64
        # ========================================================

        self.block1 = MultiScaleResidualBlock(
            in_channels=32,
            out_channels=64,
            stride=2
        )

        # ========================================================
        # Multi-Scale Residual Block 2
        # 64 -> 128
        # ========================================================

        self.block2 = MultiScaleResidualBlock(
            in_channels=64,
            out_channels=128,
            stride=2
        )

        # ========================================================
        # Multi-Scale Residual Block 3
        # 128 -> 256
        # ========================================================

        self.block3 = MultiScaleResidualBlock(
            in_channels=128,
            out_channels=256,
            stride=2
        )

        # ========================================================
        # Global feature representation
        # ========================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # ========================================================
        # Classification head
        # ========================================================

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        # Initial features
        x = self.stem(x)

        # Residual multi-scale stages
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

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


def build_residual_multiscale_cnn(
    num_classes=2,
    dropout=0.3
):
    """
    Factory function for Residual Multi-Scale CNN.
    """

    return ResidualMultiScaleCNN(
        num_classes=num_classes,
        dropout=dropout
    )


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":

    model = build_residual_multiscale_cnn(
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
    print("RESIDUAL MULTI-SCALE CNN")
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