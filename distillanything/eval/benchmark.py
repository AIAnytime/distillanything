"""Deployment-facing benchmarks: latency percentiles, throughput, memory, footprint, cost.

Accuracy metrics live in the trainer/judge; this module answers the question customers
actually ask — "is the student 10x cheaper without losing much quality?"
"""

from __future__ import annotations

import statistics
import time
from typing import Optional

import torch

from distillanything.hardware import best_device, peak_memory_bytes, reset_peak_memory


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; robust for the small n we run on laptops."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def cost_per_1k_tokens(tokens_per_s: float, hardware_cost_per_hour: float) -> float:
    """Serving cost of 1K generated tokens at a given $/hour for the hardware."""
    if tokens_per_s <= 0:
        return float("inf")
    return round(hardware_cost_per_hour / 3600.0 / tokens_per_s * 1000.0, 6)


@torch.no_grad()
def benchmark_model(
    model,
    tokenizer,
    prompt: str = "Explain what knowledge distillation is in two sentences.",
    max_new_tokens: int = 128,
    n_runs: int = 5,
    warmup: int = 1,
    device: torch.device | None = None,
    hardware_cost_per_hour: Optional[float] = None,
) -> dict:
    device = device or best_device()
    model = model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=pad_id)

    reset_peak_memory(device)
    latencies: list[float] = []
    total_tokens = 0
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
        )
        latencies.append(time.perf_counter() - start)
        total_tokens += int(output.shape[1] - inputs["input_ids"].shape[1])

    latencies.sort()
    total_time = sum(latencies)
    tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0
    params = sum(p.numel() for p in model.parameters())
    memory = peak_memory_bytes(device)

    metrics = {
        "device": device.type,
        "parameters_m": round(params / 1e6, 1),
        "n_runs": len(latencies),
        "generated_tokens": total_tokens,
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(_percentile(latencies, 0.95), 3),
        "tokens_per_s": round(tokens_per_s, 1),
        "memory_mb": round(memory / 1e6, 1) if memory is not None else None,
        "disk_size_mb": round(params * 4 / 1e6, 1),  # fp32 checkpoint estimate
    }
    if hardware_cost_per_hour is not None:
        metrics["hardware_cost_per_hour"] = hardware_cost_per_hour
        metrics["cost_per_1k_tokens_usd"] = cost_per_1k_tokens(tokens_per_s, hardware_cost_per_hour)
    return metrics
