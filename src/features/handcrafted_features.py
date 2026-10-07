"""
Handcrafted feature extraction for CRC histopathology images.

Inspired by the feature-fusion methodology used in the reference paper,
adapted for LC25000 colon histopathology images.

Feature groups:
1. Color:
   - RGB
   - HSV
   - Lab
   - YCbCr
   - RGB histograms

2. Texture:
   - GLCM
   - Local Binary Pattern (LBP)

This module does NOT perform classification.
It only converts an image into a fixed-length numerical feature vector.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================
# Configuration
# ============================================================

RGB_HIST_BINS = 16

# GLCM configuration
GLCM_LEVELS = 8
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [
    0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
]

# LBP configuration
LBP_POINTS = 24
LBP_RADIUS = 3


# ============================================================
# Utility functions
# ============================================================

def load_image(image_path):
    """
    Load image and return RGB uint8 numpy array.
    """
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def channel_statistics(image):
    """
    Calculate mean and standard deviation for each channel.

    Returns:
        2 values per channel:
        mean, std
    """

    features = []

    for channel in range(image.shape[2]):
        data = image[:, :, channel].astype(np.float32)

        features.append(float(np.mean(data)))
        features.append(float(np.std(data)))

    return features


# ============================================================
# Color features
# ============================================================

def extract_color_features(image):
    """
    Extract color features from:

    - RGB
    - HSV
    - Lab
    - YCbCr

    For each channel:
        mean
        standard deviation

    Additionally:
        RGB histogram with 16 bins per channel.
    """

    features = []

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = image

    features.extend(channel_statistics(rgb))

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    features.extend(channel_statistics(hsv))

    # --------------------------------------------------------
    # Lab
    # --------------------------------------------------------

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    features.extend(channel_statistics(lab))

    # --------------------------------------------------------
    # YCbCr
    #
    # OpenCV returns YCrCb, so reorder channels:
    # Y, Cr, Cb -> Y, Cb, Cr
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    ycbcr = ycrcb[:, :, [0, 2, 1]]

    features.extend(channel_statistics(ycbcr))

    # --------------------------------------------------------
    # RGB histograms
    # --------------------------------------------------------

    for channel in range(3):

        histogram, _ = np.histogram(
            rgb[:, :, channel],
            bins=RGB_HIST_BINS,
            range=(0, 256),
            density=True,
        )

        features.extend(histogram.tolist())

    return features


# ============================================================
# Texture features
# ============================================================

def convert_to_quantized_grayscale(image):
    """
    Convert RGB image to grayscale and quantize it into
    a small number of intensity levels for GLCM.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert 0-255 grayscale into 0-(GLCM_LEVELS-1)
    quantized = (
        gray.astype(np.uint16) * GLCM_LEVELS // 256
    ).astype(np.uint8)

    return quantized


def extract_glcm_features(image):
    """
    Extract GLCM texture features.

    Properties:
        contrast
        dissimilarity
        homogeneity
        energy
        correlation
        ASM

    Computed over:
        distances = [1, 2]
        angles = [0, 45, 90, 135 degrees]

    The values are averaged over all distance/angle combinations.
    """

    gray = convert_to_quantized_grayscale(image)

    glcm = graycomatrix(
        gray,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    features = []

    for prop in properties:

        values = graycoprops(glcm, prop)

        # Average over distances and angles
        features.append(float(np.mean(values)))

    return features


def extract_lbp_features(image):
    """
    Extract Local Binary Pattern histogram.

    Uses:
        P = 24 neighboring points
        R = 3 pixel radius
        method = uniform

    For uniform LBP:
        number of histogram bins = P + 2
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )

    n_bins = LBP_POINTS + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )

    return histogram.tolist()


def extract_texture_features(image):
    """
    Combine GLCM and LBP features.
    """

    features = []

    features.extend(extract_glcm_features(image))
    features.extend(extract_lbp_features(image))

    return features


# ============================================================
# Complete handcrafted feature extractor
# ============================================================

def extract_handcrafted_features(image_path):
    """
    Extract the complete handcrafted feature vector.

    Feature groups:
        Color + Texture

    Returns:
        numpy.ndarray
    """

    image = load_image(image_path)

    features = []

    # Color branch
    features.extend(extract_color_features(image))

    # Texture branch
    features.extend(extract_texture_features(image))

    return np.asarray(features, dtype=np.float32)


# ============================================================
# Feature names
# ============================================================

def get_feature_names():
    """
    Return names corresponding to every feature dimension.
    """

    names = []

    # --------------------------------------------------------
    # Color statistics
    # --------------------------------------------------------

    color_spaces = {
        "rgb": ["R", "G", "B"],
        "hsv": ["H", "S", "V"],
        "lab": ["L", "a", "b"],
        "ycbcr": ["Y", "Cb", "Cr"],
    }

    for space, channels in color_spaces.items():

        for channel in channels:

            names.append(f"{space}_{channel}_mean")
            names.append(f"{space}_{channel}_std")

    # --------------------------------------------------------
    # RGB histograms
    # --------------------------------------------------------

    for channel in ["R", "G", "B"]:

        for bin_index in range(RGB_HIST_BINS):

            names.append(
                f"rgb_{channel}_hist_bin_{bin_index}"
            )

    # --------------------------------------------------------
    # GLCM
    # --------------------------------------------------------

    glcm_properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    for prop in glcm_properties:

        names.append(f"glcm_{prop}")

    # --------------------------------------------------------
    # LBP
    # --------------------------------------------------------

    for bin_index in range(LBP_POINTS + 2):

        names.append(
            f"lbp_hist_bin_{bin_index}"
        )

    return names


# ============================================================
# Main test
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python -m src.features.handcrafted_features "
            "<image_path>"
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    features = extract_handcrafted_features(
        image_path
    )

    names = get_feature_names()

    print("=" * 70)
    print("HANDCRAFTED FEATURE EXTRACTION TEST")
    print("=" * 70)

    print(f"Image: {image_path}")
    print(f"Feature count: {len(features)}")
    print(f"Feature name count: {len(names)}")

    print()
    print("First 10 features:")

    for name, value in zip(
        names[:10],
        features[:10],
    ):
        print(f"{name:35s}: {value:.6f}")

    print()
    print("Last 10 features:")

    for name, value in zip(
        names[-10:],
        features[-10:],
    ):
        print(f"{name:35s}: {value:.6f}")

    print()
    print("Feature extraction: PASSED")