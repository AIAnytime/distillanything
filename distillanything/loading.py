"""Model loading and LoRA/QLoRA adapter utilities.

LoRA lets the student be a 1-3B model on a 16GB laptop: the base weights stay
frozen (no optimizer state, no gradients) and only small low-rank adapters train.
QLoRA additionally loads the frozen base in 4-bit NF4 — CUDA only, since
bitsandbytes has no Apple Silicon / CPU backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rich.console import Console

if TYPE_CHECKING:
    from distillanything.config import LoraSettings

console = Console()


def _require_peft():
    try:
        import peft
    except ImportError as e:
        raise ImportError(
            'LoRA/QLoRA support requires peft: pip install "distill-anything[lora]"'
        ) from e
    return peft


def _quantization_config_4bit():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def apply_lora(model, settings: "LoraSettings"):
    """Wrap a causal LM with trainable LoRA adapters (base weights frozen).

    ``target_modules=None`` lets peft pick the right projection layers for known
    architectures (Llama, Qwen, Mistral, ...). For exotic models, set it
    explicitly, e.g. ``["q_proj", "v_proj"]``.
    """
    peft = _require_peft()
    if settings.qlora:
        model = peft.prepare_model_for_kbit_training(model)
    lora_config = peft.LoraConfig(
        task_type="CAUSAL_LM",
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=settings.target_modules,
    )
    model = peft.get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(
        f"LoRA r={settings.r}: [green]{trainable / 1e6:.1f}M trainable[/] "
        f"of {total / 1e6:.1f}M params ({trainable / total:.2%})"
    )
    return model


def load_student_model(model_name: str, lora: "LoraSettings | None" = None, trust_remote_code: bool = False):
    """Load a base model for training, optionally 4-bit quantized + LoRA-wrapped."""
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if lora is not None and lora.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA needs CUDA (bitsandbytes has no MPS/CPU backend). "
                "On Apple Silicon use plain LoRA: set student.lora.qlora: false."
            )
        kwargs["quantization_config"] = _quantization_config_4bit()
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if lora is not None:
        model = apply_lora(model, lora)
    return model


def load_model_and_tokenizer(path_or_repo: str, trust_remote_code: bool = False):
    """Load a plain HF checkpoint OR a LoRA adapter directory.

    Adapter dirs (they contain ``adapter_config.json``) are merged into their base
    model on load, so downstream code (generation, benchmark, report) never has to
    care how the model was trained.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (Path(path_or_repo) / "adapter_config.json").exists():
        peft = _require_peft()
        model = peft.AutoPeftModelForCausalLM.from_pretrained(
            path_or_repo, trust_remote_code=trust_remote_code
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path_or_repo, trust_remote_code=trust_remote_code)
    return model, tokenizer
