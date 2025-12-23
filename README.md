emotion-avatar-chatbot/
├─ data/
│  ├─ raw/            # 원본 (zip)
│  ├─ processed/      # 학습/추론에 실제 사용
│  └─ interim/        # (선택) 필요 시만
│
├─ src/
│  ├─ data/           # 데이터 처리 파이프라인
│  ├─ train/          # 학습 코드 (← 다음에 만들 것)
│  └─ infer/          # 추론/API (← 그 다음)
│
├─ models/             # 학습 결과 (자동 생성)
├─ config.py
├─ requirements.txt
└─ README.md
