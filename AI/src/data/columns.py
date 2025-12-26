# src/data/columns.py
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def infer_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    train.csv에서 텍스트 컬럼/라벨 컬럼을 최대한 자동으로 추정.
    - 텍스트: 흔한 후보 컬럼명 우선 -> 없으면 object dtype 중 평균 길이 가장 큰 컬럼
    - 라벨: 흔한 후보 컬럼명 우선 -> 없으면 text 제외 컬럼 중 고유값 수가 가장 작은 컬럼
    """
    text_candidates = ["text", "sentence", "utterance", "content", "message", "dialogue", "prompt", "input"]
    label_candidates = ["label", "emotion", "sentiment", "class", "target", "y"]

    text_col = _pick_column(df, text_candidates)
    label_col = _pick_column(df, label_candidates)

    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError("텍스트로 보이는(object) 컬럼을 찾지 못했습니다. CSV 컬럼명을 확인하세요.")
        lens = []
        for c in obj_cols:
            s = df[c].astype(str).fillna("")
            lens.append((c, float(s.map(len).mean())))
        lens.sort(key=lambda x: x[1], reverse=True)
        text_col = lens[0][0]

    if label_col is None:
        candidates = [c for c in df.columns if c != text_col]
        if not candidates:
            raise ValueError("라벨 컬럼을 찾지 못했습니다. CSV 컬럼명을 확인하세요.")
        uniq = [(c, df[c].nunique(dropna=True)) for c in candidates]
        uniq.sort(key=lambda x: x[1])
        label_col = uniq[0][0]

    return text_col, label_col
