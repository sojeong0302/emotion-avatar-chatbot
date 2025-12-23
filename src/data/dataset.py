# src/data/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def build_label_mapping(train_labels: pd.Series) -> Dict[str, int]:
    """
    train set 라벨 기준으로 label2id 생성.
    숫자 라벨도 안전하게 처리.
    """
    if pd.api.types.is_numeric_dtype(train_labels):
        uniq = sorted(train_labels.dropna().unique().tolist())
        return {str(int(v)): int(v) for v in uniq}
    uniq = sorted(train_labels.dropna().astype(str).unique().tolist())
    return {lab: i for i, lab in enumerate(uniq)}


def normalize_label_value(v) -> str:
    """CSV 라벨 값(float 0.0 등)도 안정적으로 문자열 키로 변환."""
    if pd.isna(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, label2id: Dict[str, int]):
        self.texts = df[text_col].astype(str).fillna("").tolist()

        self.labels: List[int] = []
        for v in df[label_col].tolist():
            if pd.isna(v):
                self.labels.append(-1)  # 결측 라벨은 학습/평가에서 제외
                continue
            key = normalize_label_value(v)
            if key not in label2id:
                raise ValueError(f"라벨 '{key}'가 train label set에 없습니다. 매핑을 확인하세요.")
            self.labels.append(label2id[key])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx], "label": self.labels[idx]}


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc


def make_label_dicts(label2id: Dict[str, int]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {v: k for k, v in label2id.items()}
    return id2label, label2id
