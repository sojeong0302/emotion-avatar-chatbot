import pandas as pd
import os

TRAIN_CSV = r"C:\Users\01duq\Downloads\train.csv"
VALID_CSV = r"C:\Users\01duq\Downloads\valid.csv"
OUT_DIR   = r"C:\Users\01duq\Downloads\processed"
os.makedirs(OUT_DIR, exist_ok=True)

# 표정용 라벨로 통합 (필요하면 너 스타일에 맞게 더 수정 가능)
MAP = {
    "기쁨": "HAPPY",
    "행복": "HAPPY",
    "만족": "HAPPY",
    "흥분": "HAPPY",

    "슬픔": "SAD",
    "우울": "SAD",
    "상처": "SAD",

    "화남": "ANGRY",
    "분노": "ANGRY",
    "짜증": "ANGRY",
    "혐오": "ANGRY",

    "놀라움": "SURPRISE",
    "당황": "SURPRISE",

    "두려움": "FEAR",
    "공포": "FEAR",
    "불안": "FEAR",

    "없음": "NEUTRAL",
    "중립": "NEUTRAL",
}

def normalize_level(x: str) -> str:
    x = str(x).strip()
    if x in ["약함", "약", "낮음"]:
        return "weak"
    if x in ["보통", "중간", "중"]:
        return "normal"
    if x in ["강함", "강", "높음"]:
        return "strong"
    return "normal"

def map_emotion(target: str, category: str) -> str:
    t = str(target).strip()
    c = str(category).strip()
    if t in MAP:
        return MAP[t]
    # target이 비었거나 낯선 값이면 category 기반으로 fallback
    if c == "긍정":
        return "HAPPY"
    if c == "부정":
        # 부정인데 target 미정이면 기본은 SAD로 두는 게 자연스러움(원하면 ANGRY로 바꿔도 됨)
        return "SAD"
    return "NEUTRAL"

def process(in_csv: str, out_csv: str):
    # 용량이 크니까 chunk로 처리
    chunks = []
    for chunk in pd.read_csv(in_csv, chunksize=200_000):
        chunk["emotion"] = chunk.apply(
            lambda r: map_emotion(r.get("VerifyEmotionTarget", ""), r.get("VerifyEmotionCategory", "")),
            axis=1
        )
        chunk["intensity"] = chunk["VerifyEmotionLevel"].apply(normalize_level)
        # 최종적으로 모델에 필요한 컬럼만 남김
        chunk = chunk[["text", "emotion", "intensity"]]
        chunks.append(chunk)

    out = pd.concat(chunks, ignore_index=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", out_csv, "rows:", len(out))

process(TRAIN_CSV, os.path.join(OUT_DIR, "train_mapped.csv"))
process(VALID_CSV, os.path.join(OUT_DIR, "valid_mapped.csv"))
