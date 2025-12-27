# train.py
from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.columns import infer_text_and_label_columns
from src.data.dataset import TextClsDataset, Collator, build_label_mapping, make_label_dicts
from src.train.engine import train_one_epoch, evaluate, build_scheduler, save_best
from src.train.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="src/data/processed/train.csv")
    parser.add_argument("--valid_csv", default="src/data/processed/valid.csv")
    parser.add_argument("--model_name", default="klue/roberta-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", default="models/emotion_cls_sample")
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print("device:", device)

    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)

    text_col, label_col = infer_text_and_label_columns(train_df)
    print(f"[INFO] inferred columns: text='{text_col}', label='{label_col}'")

    label2id = build_label_mapping(train_df[label_col])
    id2label, label2id = make_label_dicts(label2id)

    print("[INFO] num_labels:", len(label2id))
    print("[INFO] labels:", label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label={i: id2label[i] for i in range(len(label2id))},
        label2id={id2label[i]: i for i in range(len(label2id))},
    ).to(device)

    train_ds = TextClsDataset(train_df, text_col, label_col, label2id)
    valid_ds = TextClsDataset(valid_df, text_col, label_col, label2id)
    collate = Collator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            log_every=args.log_every,
        )

        metrics = evaluate(model, valid_loader, device)
        print(f"[EVAL] epoch {epoch} acc={metrics['acc']:.4f} f1_macro={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            print(f"[SAVE] best f1_macro={best_f1:.4f} -> {args.out_dir}")
            save_best(model, tokenizer, args.out_dir, id2label)

    print("[DONE] best_f1_macro:", best_f1)


if __name__ == "__main__":
    main()
