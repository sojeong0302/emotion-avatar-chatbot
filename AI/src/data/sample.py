from pathlib import Path
import pandas as pd

# 너 프로젝트 구조 기준: src/data/sample.py -> 상위 2단계가 프로젝트 루트
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IN = PROCESSED / "train_mapped.csv"
VALID_IN = PROCESSED / "valid_mapped.csv"

TRAIN_OUT = OUT_DIR / "train_sample.csv"
VALID_OUT = OUT_DIR / "valid_sample.csv"

# 처음은 이 정도가 적당함 (빠르게 학습 성공 목적)
N_TRAIN = 400_000
N_VALID = 80_000
RANDOM_SEED = 42

def stratified_sample(df: pd.DataFrame, n_total: int, label_col: str = "emotion") -> pd.DataFrame:
    """라벨 비율 유지(층화)하면서 n_total개 샘플링"""
    if n_total >= len(df):
        return df

    label_counts = df[label_col].value_counts()
    # 각 라벨별로 전체 비율만큼 할당(최소 1개)
    target_per_label = (label_counts / len(df) * n_total).round().astype(int).clip(lower=1)

    pieces = []
    for label, n in target_per_label.items():
        sub = df[df[label_col] == label]
        # 라벨 데이터가 적으면 가능한 만큼만
        take = min(n, len(sub))
        pieces.append(sub.sample(n=take, random_state=RANDOM_SEED))

    out = pd.concat(pieces, ignore_index=True)

    # 반올림 때문에 목표 개수보다 많아질 수 있음 → 다시 정확히 맞추기
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=RANDOM_SEED).reset_index(drop=True)
    elif len(out) < n_total:
        # 부족하면 전체에서 추가 랜덤 샘플
        extra = df.drop(out.index, errors="ignore")
        need = n_total - len(out)
        if need > 0 and len(extra) > 0:
            add = extra.sample(n=min(need, len(extra)), random_state=RANDOM_SEED)
            out = pd.concat([out, add], ignore_index=True)

    return out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)  # 셔플

def main():
    print("[LOAD]", TRAIN_IN)
    train = pd.read_csv(TRAIN_IN)
    print(" train rows:", len(train))

    print("[LOAD]", VALID_IN)
    valid = pd.read_csv(VALID_IN)
    print(" valid rows:", len(valid))

    train_s = stratified_sample(train, N_TRAIN, "emotion")
    valid_s = stratified_sample(valid, N_VALID, "emotion")

    train_s.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    valid_s.to_csv(VALID_OUT, index=False, encoding="utf-8-sig")

    print("[SAVE]", TRAIN_OUT, "rows:", len(train_s))
    print(train_s["emotion"].value_counts())
    print("[SAVE]", VALID_OUT, "rows:", len(valid_s))
    print(valid_s["emotion"].value_counts())

if __name__ == "__main__":
    main()
