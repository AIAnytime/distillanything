"""Device detection and memory accounting across CUDA, Apple Silicon (MPS), and CPU."""

from __future__ import annotations

import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed-precision dtype for a device, or None to train in fp32.

    MPS autocast support is still uneven for training, so we stay in fp32 there.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        # MPS has no peak counter; current allocation is the best available signal.
        return torch.mps.current_allocated_memory()
    return None


def device_summary() -> str:
    device = best_device()
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
