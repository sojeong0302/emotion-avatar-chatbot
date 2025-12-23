## 📁 Project Structure

```text
emotion-avatar-chatbot/
├─ .venv/
├─ data/
│  ├─ raw/
│  └─ processed/
│     ├─ sample/
│     ├─ train.csv
│     ├─ valid.csv
│     ├─ train_mapped.csv
│     └─ valid_mapped.csv
├─ models/
│  └─ emotion_cls_sample/
├─ src/
│  ├─ data/
│  ├─ train/
│  └─ config.py
├─ train.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 📂 Directory Details

#### `.venv/`
- Python 가상환경 디렉토리
- 프로젝트별 패키지 의존성 격리
- torch, transformers 등 설치

#### `data/raw/`
- 전처리 전 원본 데이터 (수정 금지)
- 학습/검증 데이터가 압축 파일 형태로 저장됨

#### `data/processed/`
- 전처리 및 학습에 사용되는 데이터
- `sample/` : 빠른 실험용 소규모 데이터
- `train.csv`, `valid.csv` : 전체 학습/검증 데이터
- `*_mapped.csv` : 감정 라벨이 정규화된 데이터

#### `models/emotion_cls_sample/`
- 학습된 감정 분류 모델 저장 디렉토리
- 모델 가중치, 토크나이저, 라벨 매핑 정보 포함

#### `src/data/`
- 데이터 전처리 및 Dataset 관련 코드

#### `src/train/`
- 모델 학습, 평가, 지표 계산 로직

#### `train.py`
- 학습 엔트리포인트
- 데이터 로딩 → 모델 생성 → 학습/평가 → 모델 저장
