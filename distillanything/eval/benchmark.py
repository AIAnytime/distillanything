"""Deployment-facing benchmarks: latency, throughput, memory, footprint.

Accuracy metrics live in the trainer's evaluate(); this module answers the question
customers actually ask — "is the student 10x cheaper without losing much quality?"
"""

from __future__ import annotations

import time

import torch

from distillanything.hardware import best_device, peak_memory_bytes, reset_peak_memory


@torch.no_grad()
def benchmark_model(
    model,
    tokenizer,
    prompt: str = "Explain what knowledge distillation is in two sentences.",
    max_new_tokens: int = 128,
    warmup: int = 1,
    device: torch.device | None = None,
) -> dict:
    device = device or best_device()
    model = model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=pad_id)

    reset_peak_memory(device)
    start = time.perf_counter()
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
    )
    elapsed = time.perf_counter() - start

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    params = sum(p.numel() for p in model.parameters())
    memory = peak_memory_bytes(device)

    return {
        "device": device.type,
        "parameters_m": round(params / 1e6, 1),
        "generated_tokens": int(new_tokens),
        "latency_s": round(elapsed, 3),
        "tokens_per_s": round(new_tokens / elapsed, 1) if elapsed > 0 else None,
        "memory_mb": round(memory / 1e6, 1) if memory is not None else None,
        "disk_size_mb": round(params * 4 / 1e6, 1),  # fp32 checkpoint estimate
    }
