import time
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.models.basic_cnn import build_basic_cnn
from src.models.residual_multiscale_cnn import build_residual_multiscale_cnn

from src.models.classifier import build_model as build_classifier_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Same input size used for the CRC classification experiments
INPUT_SIZE = (1, 3, 224, 224)

# Inference timing
WARMUP_RUNS = 10
TIMING_RUNS = 50

# Output directory
OUTPUT_DIR = Path(
    "results/analysis/computational_complexity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAC / FLOP CALCULATION
# ============================================================

def calculate_macs(model, input_tensor):
    """
    Calculate MACs for Conv2D and Linear layers.

    MAC = Multiply-Accumulate.

    FLOPs are reported using:
        FLOPs = 2 x MACs

    This is because one MAC is counted as:
        1 multiplication + 1 addition
    """

    macs = 0

    hooks = []

    # --------------------------------------------------------
    # Conv2D hook
    # --------------------------------------------------------

    def conv_hook(module, inputs, output):

        nonlocal macs

        x = inputs[0]

        batch_size = x.shape[0]

        out_channels = module.out_channels

        kernel_h, kernel_w = module.kernel_size

        in_channels = module.in_channels

        groups = module.groups

        out_h = output.shape[2]
        out_w = output.shape[3]

        kernel_operations = (
            kernel_h
            * kernel_w
            * (in_channels // groups)
        )

        layer_macs = (
            batch_size
            * out_h
            * out_w
            * out_channels
            * kernel_operations
        )

        macs += layer_macs

    # --------------------------------------------------------
    # Linear hook
    # --------------------------------------------------------

    def linear_hook(module, inputs, output):

        nonlocal macs

        x = inputs[0]

        batch_size = x.shape[0]

        layer_macs = (
            batch_size
            * module.in_features
            * module.out_features
        )

        macs += layer_macs

    # --------------------------------------------------------
    # Register hooks
    # --------------------------------------------------------

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            hooks.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

        elif isinstance(module, nn.Linear):

            hooks.append(
                module.register_forward_hook(
                    linear_hook
                )
            )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        model(input_tensor)

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    for hook in hooks:

        hook.remove()

    return macs


# ============================================================
# PARAMETER STATISTICS
# ============================================================

def parameter_statistics(model):
    """
    Calculate total, trainable and non-trainable parameters.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    non_trainable_parameters = (
        total_parameters
        - trainable_parameters
    )

    # FP32 = 4 bytes per parameter
    parameter_memory_bytes = (
        total_parameters * 4
    )

    parameter_memory_mb = (
        parameter_memory_bytes
        / (1024 ** 2)
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": non_trainable_parameters,
        "parameter_memory_MB_FP32": parameter_memory_mb,
    }


# ============================================================
# INFERENCE TIMING
# ============================================================

def measure_inference_time(
    model,
    input_tensor
):
    """
    Measure single-image inference time.

    Batch size = 1.

    Returns:
        average inference time in ms
        throughput in images/sec
    """

    model.eval()

    with torch.no_grad():

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        for _ in range(WARMUP_RUNS):

            _ = model(input_tensor)

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        # ----------------------------------------------------
        # Timed runs
        # ----------------------------------------------------

        start_time = time.perf_counter()

        for _ in range(TIMING_RUNS):

            _ = model(input_tensor)

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        end_time = time.perf_counter()

    total_time = (
        end_time
        - start_time
    )

    average_time_ms = (
        total_time
        / TIMING_RUNS
    ) * 1000

    images_per_second = (
        1000
        / average_time_ms
    )

    return {
        "average_inference_time_ms":
            average_time_ms,

        "images_per_second":
            images_per_second,
    }


# ============================================================
# BUILD MODELS
# ============================================================

def get_models():

    models = {}

    # ========================================================
    # 1. BASIC CNN
    # ========================================================

    models["Basic CNN"] = build_basic_cnn(
        num_classes=2
    )

    # ========================================================
    # 2. RESIDUAL MULTI-SCALE CNN
    # ========================================================

    models["Residual Multi-Scale CNN"] = (
        build_residual_multiscale_cnn(
            num_classes=2
        )
    )

    # ========================================================
    # 3. RESNET18
    # ========================================================

    models["ResNet18"] = build_classifier_model(
        name="resnet18",
        num_classes=2,
        pretrained=False
    )

    return models


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MODEL COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 80)

    print()
    print(f"Device      : {DEVICE}")
    print(f"Input       : {INPUT_SIZE}")
    print(f"Warmup runs : {WARMUP_RUNS}")
    print(f"Timing runs : {TIMING_RUNS}")

    print()
    print("Evaluation configuration")
    print("-" * 80)
    print("Input size  : 224 x 224")
    print("Batch size  : 1")
    print("Precision   : FP32")
    print("FLOPs rule  : 1 MAC = 2 FLOPs")
    print()

    # ========================================================
    # INPUT TENSOR
    # ========================================================

    input_tensor = torch.randn(
        *INPUT_SIZE,
        device=DEVICE
    )

    # ========================================================
    # BUILD MODELS
    # ========================================================

    models = get_models()

    results = []

    # ========================================================
    # ANALYZE MODELS
    # ========================================================

    for model_name, model in models.items():

        print()
        print("=" * 80)
        print(model_name)
        print("=" * 80)

        model = model.to(DEVICE)

        model.eval()

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        stats = parameter_statistics(
            model
        )

        # ----------------------------------------------------
        # MACs
        # ----------------------------------------------------

        macs = calculate_macs(
            model,
            input_tensor
        )

        # ----------------------------------------------------
        # FLOPs
        # ----------------------------------------------------

        flops = macs * 2

        stats["MACs"] = macs

        stats["MACs_G"] = (
            macs / 1e9
        )

        stats["FLOPs"] = flops

        stats["FLOPs_G"] = (
            flops / 1e9
        )

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        timing = measure_inference_time(
            model,
            input_tensor
        )

        stats.update(
            timing
        )

        # ----------------------------------------------------
        # Model name
        # ----------------------------------------------------

        stats["model"] = model_name

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            stats
        )

        # ====================================================
        # PRINT MODEL RESULTS
        # ====================================================

        print()

        print(
            f"Total parameters : "
            f"{stats['total_parameters']:,}"
        )

        print(
            f"Trainable        : "
            f"{stats['trainable_parameters']:,}"
        )

        print(
            f"Non-trainable    : "
            f"{stats['non_trainable_parameters']:,}"
        )

        print(
            f"Parameter size   : "
            f"{stats['parameter_memory_MB_FP32']:.3f} MB"
        )

        print(
            f"MACs             : "
            f"{stats['MACs_G']:.4f} G"
        )

        print(
            f"FLOPs            : "
            f"{stats['FLOPs_G']:.4f} G"
        )

        print(
            f"Inference        : "
            f"{stats['average_inference_time_ms']:.3f} "
            f"ms/image"
        )

        print(
            f"Throughput       : "
            f"{stats['images_per_second']:.2f} "
            f"images/sec"
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "model_computational_complexity.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "model_computational_complexity.csv"
    )

    fieldnames = [
        "model",
        "total_parameters",
        "trainable_parameters",
        "non_trainable_parameters",
        "parameter_memory_MB_FP32",
        "MACs",
        "MACs_G",
        "FLOPs",
        "FLOPs_G",
        "average_inference_time_ms",
        "images_per_second",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'Model':30}"
        f"{'Parameters':>15}"
        f"{'FLOPs(G)':>15}"
        f"{'Time(ms)':>15}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['model']:30}"
            f"{result['total_parameters']:>15,}"
            f"{result['FLOPs_G']:>15.4f}"
            f"{result['average_inference_time_ms']:>15.3f}"
        )

    print()

    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"CSV : {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()