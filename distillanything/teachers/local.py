"""Local Hugging Face teacher — white-box (logits available for KD)."""

from __future__ import annotations

from typing import Optional

import torch

from distillanything.hardware import best_device
from distillanything.teachers.base import Teacher


class HFTeacher(Teacher):
    white_box = True

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name_or_path
        self.device = device or best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

    def _render(self, prompt: str, system: Optional[str]) -> str:
        if self.tokenizer.chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prefix = f"{system}\n\n" if system else ""
        return f"{prefix}{prompt}\n"

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
    ) -> list[str]:
        outputs: list[str] = []
        for prompt in prompts:
            text = self._render(prompt, system)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            completion = generated[0][inputs["input_ids"].shape[1]:]
            outputs.append(self.tokenizer.decode(completion, skip_special_tokens=True).strip())
        return outputs

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        return out.logits
