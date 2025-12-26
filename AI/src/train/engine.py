# src/train/engine.py
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        labels = batch["labels"]  # CPU 텐서
        mask = labels != -1
        if mask.sum().item() == 0:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1).detach().cpu()

        y_true.extend(labels[mask].tolist())
        y_pred.extend(preds[mask].tolist())

    return compute_metrics(y_true, y_pred)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    log_every: int = 20,
) -> None:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"]
        mask = labels != -1
        if mask.sum().item() == 0:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(loss.item())

        if step % log_every == 0 or step == len(loader):
            avg_loss = running_loss / step
            print(f"epoch {epoch} step {step}/{len(loader)} loss {avg_loss:.4f}")


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def save_best(model, tokenizer, out_dir: str, id2label: Dict[int, str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for i in range(len(id2label)):
            f.write(f"{i}\t{id2label[i]}\n")
