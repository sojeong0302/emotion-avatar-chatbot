import os
import zipfile
import json
import csv
from glob import glob

# ====== 너 환경에 맞게 여기만 수정 ======
DATA_DIR = r"C:\Users\01duq\OneDrive\문서\GitHub\emotion-avatar-chatbot\AI\src\data\raw"         # ZIP들이 있는 폴더
OUT_DIR  = r"C:\Users\01duq\Downloads"          # 결과 저장 폴더
# =======================================

os.makedirs(OUT_DIR, exist_ok=True)

def unzip_all(zip_paths, extract_root):
    os.makedirs(extract_root, exist_ok=True)
    for zp in zip_paths:
        name = os.path.splitext(os.path.basename(zp))[0]
        out = os.path.join(extract_root, name)
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(zp, 'r') as z:
            z.extractall(out)
        print(f"[UNZIP] {zp} -> {out}")

def iter_json_files(root):
    # ZIP 풀린 폴더 안에서 json 전부 찾기
    return glob(os.path.join(root, "**", "*.json"), recursive=True)

def extract_rows(json_path):
    rows = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conv = data.get("Conversation", [])
        for utt in conv:
            text = (utt.get("Text") or "").strip()
            if not text:
                continue
            row = {
                "text": text,
                "VerifyEmotionCategory": utt.get("VerifyEmotionCategory", ""),
                "VerifyEmotionTarget": utt.get("VerifyEmotionTarget", ""),
                "VerifyEmotionLevel": utt.get("VerifyEmotionLevel", ""),
                "source_file": os.path.basename(json_path),
            }
            rows.append(row)

    except Exception as e:
        print(f"[SKIP] {json_path} ({e})")

    return rows

def build_csv(split_name, extract_root, out_csv):
    json_files = iter_json_files(extract_root)
    print(f"[{split_name}] json files found:", len(json_files))

    all_rows = []
    for jp in json_files:
        all_rows.extend(extract_rows(jp))

    print(f"[{split_name}] utterances extracted:", len(all_rows))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "VerifyEmotionCategory", "VerifyEmotionTarget", "VerifyEmotionLevel", "source_file"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[SAVE] {out_csv}")

def main():
    # ZIP 분리: Training(TL) / Validation(VL)
    tl_zips = sorted(glob(os.path.join(DATA_DIR, "TL_*.zip")))
    vl_zips = sorted(glob(os.path.join(DATA_DIR, "VL_*.zip")))

    if not tl_zips and not vl_zips:
        raise RuntimeError("TL_*.zip / VL_*.zip 를 DATA_DIR에서 못 찾았어. 파일명/경로 확인해줘.")

    train_root = os.path.join(OUT_DIR, "train_extracted")
    val_root   = os.path.join(OUT_DIR, "val_extracted")

    # 1) ZIP 풀기
    unzip_all(tl_zips, train_root)
    unzip_all(vl_zips, val_root)

    # 2) CSV 만들기 (4개 필드만)
    build_csv("TRAIN", train_root, os.path.join(OUT_DIR, "train.csv"))
    build_csv("VALID", val_root, os.path.join(OUT_DIR, "valid.csv"))

if __name__ == "__main__":
    main()
