"""
Model exporter — ONNX and TorchScript export with quantization and profiling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str | Path,
    opset_version: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
) -> Path:
    """Export model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        Model to export (in eval mode).
    dummy_input : torch.Tensor
        Dummy input tensor for tracing.
    output_path : str or Path
        Output file path.
    opset_version : int
        ONNX opset version.
    input_names : list[str], optional
    output_names : list[str], optional
    dynamic_axes : dict, optional
        Dynamic axis specification.

    Returns
    -------
    Path
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    if input_names is None:
        input_names = ["eeg"]
    if output_names is None:
        output_names = ["logits"]

    logger.info(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {
            "eeg": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
    except ImportError:
        logger.warning("onnx package not installed — skipping verification")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model saved: {file_size_mb:.2f} MB")

    return output_path


def export_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str | Path,
    method: str = "trace",
) -> Path:
    """Export model to TorchScript.

    Parameters
    ----------
    model : nn.Module
        Model to export.
    dummy_input : torch.Tensor
        Dummy input for tracing.
    output_path : str or Path
        Output file path.
    method : str
        'trace' or 'script'.

    Returns
    -------
    Path
        Path to exported TorchScript file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if method == "trace":
        scripted = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown export method: {method}")

    scripted.save(str(output_path))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"TorchScript model saved: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply dynamic quantization (post-training, INT8).

    Parameters
    ----------
    model : nn.Module
        Model to quantize.
    dtype : torch.dtype
        Quantization data type.

    Returns
    -------
    nn.Module
        Quantized model.
    """
    logger.info("Applying dynamic INT8 quantization")

    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
    )

    # Report size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(
        p.numel() * p.element_size()
        for p in quantized.parameters()
    )

    logger.info(
        f"Quantization: {original_size / 1e6:.2f} MB → {quantized_size / 1e6:.2f} MB "
        f"({quantized_size / original_size * 100:.1f}%)"
    )

    return quantized


def profile_inference(
    model: nn.Module,
    dummy_input: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    """Profile inference latency and memory.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    dummy_input : torch.Tensor
        Input tensor.
    n_warmup : int
        Number of warmup iterations.
    n_runs : int
        Number of timed iterations.
    device : str
        Device to profile on.

    Returns
    -------
    dict[str, float]
        Profiling results: latency percentiles, peak memory, etc.
    """
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Synchronize CUDA
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)  # ms

    latencies = sorted(latencies)

    results = {
        "device": device,
        "n_runs": n_runs,
        "latency_mean_ms": float(sum(latencies) / len(latencies)),
        "latency_p50_ms": float(latencies[len(latencies) // 2]),
        "latency_p95_ms": float(latencies[int(len(latencies) * 0.95)]),
        "latency_p99_ms": float(latencies[int(len(latencies) * 0.99)]),
        "latency_min_ms": float(latencies[0]),
        "latency_max_ms": float(latencies[-1]),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    if device == "cuda":
        results["peak_memory_MB"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Real-time factor (assume 2s window)
    window_ms = 2000.0
    results["real_time_factor"] = results["latency_mean_ms"] / window_ms
    results["meets_realtime"] = results["latency_p95_ms"] < window_ms

    logger.info(
        f"Profiling ({device}): "
        f"p50={results['latency_p50_ms']:.1f}ms, "
        f"p95={results['latency_p95_ms']:.1f}ms, "
        f"RTF={results['real_time_factor']:.3f}"
    )

    return results
