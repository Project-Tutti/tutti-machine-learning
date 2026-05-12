# Qwen MIDI 생성 - 프로덕션 파이프라인 (v2)

**Qwen2.5-0.5B를 사용한 MIDI 음악 생성용 프로덕션 준비 분산 학습 & 추론 시스템**

## 📋 개요

이 폴더는 다음을 포함하는 MIDI 생성 시스템의 **최적화된 모듈식 구현**을 포함합니다:

- **분산 학습**: 다중 GPU 학습을 위한 DDP (Distributed Data Parallel) 지원
- **고급 데이터 파이프라인**: 가중 샘플링, memmap 기반 효율적 로딩, 장르 균형
- **개선된 토큰화**: 향상된 의미론적 표현을 가진 V6 어휘
- **프로덕션 추론**: 구성 가능한 매개변수를 사용한 빠르고 안정적인 MIDI 생성
- **모듈식 아키텍처**: 데이터셋, 모델, 학습 및 추론에 대한 관심 분리

## 🗂️ 폴더 구조

```
projects_v2/
├── docs/
│   └── v6_pipeline_hybrid_upgrade_guide.md
├── data/
│   ├── dataset.py              # 가중 샘플링이 있는 Memmap 기반 데이터셋 로더
│   ├── vocab_v6.py             # 향상된 V6 어휘 빌더
│   ├── preprocess_lmd.py        # LMD 데이터셋 전처리 유틸리티
│   └── jsonl_to_memmap.py       # JSONL → 바이너리 memmap 형식 변환
├── models/
│   └── train.py                # 메인 학습 스크립트 (DDP 준비)
├── inference/
│   └── inference_v6.py          # 고급 추론 파이프라인
└── README.md
```

## 📝 파일 설명

### `models/train.py`

**메인 분산 학습 스크립트**

**기능:**
- ✅ DDP (Distributed Data Parallel) - 다중 GPU 지원
- ✅ 단일 GPU 폴백
- ✅ 가중 장르 손실 (Rock 곡: 더 나은 다양성을 위해 2.5배 가중치)
- ✅ 최고 모델 체크포인트를 사용한 조기 중지
- ✅ 효율성을 위한 FlashAttention-2 통합
- ✅ 그래디언트 누적 & 혼합 정밀도 (bfloat16)

**사용법:**

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=1 python3 models/train.py

# 다중 GPU (DDP, 3 GPUs)
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 models/train.py

# Custom config (edit constants in script)
# - DATA_DIR: path to processed MIDI dataset
# - CKPT_DIR: checkpoint output directory
# - GENRE_LOSS_WEIGHTS: adjust per-genre training emphasis
```

**설정:**
```python
VOCAB_SIZE = 645
SEQ_LEN = 8192  # v1보다 긴 시퀀스
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BATCH_SIZE = 24 (디바이스당)
LEARNING_RATE = 1e-3 # 전이 학습을 위해서 lr을 낮춰서 수렴하도록 함
```

### `data/dataset.py`

**장르 인식 샘플링을 사용한 Memmap 기반 PyTorch 데이터셋**

**기능:**
- ✅ 효율적인 memmap 로딩 (RAM에 전체 데이터셋 없음)
- ✅ 가중 무작위 샘플링 (Rock과 같은 소수 장르 오버샘플)
- ✅ 메타데이터 기반 필터링 (장르, 지속 시간, 품질)
- ✅ FIM (Fill-In-Middle) 증강 지원
- ✅ 배치 준비용 Collate 함수

**클래스: `MemmapMidiDataset`**
```python
dataset = MemmapMidiDataset(
    bin_path="data/train.bin",
    meta_path="data/train_meta.json",
    vocab=vocab,
    max_length=8192
)
```

**메타데이터 형식 (JSON):**
```json
{
  "samples": [
    {
      "id": "piece_0",
      "path": "lmd/A/Alicia_Keys/FallingSlowly.mid",
      "genre": "pop",
      "duration_sec": 245,
      "inst_count": 5,
      "quality": "high"
    }
  ]
}
```

### `data/vocab_v6.py`

**645개 토큰으로 향상된 어휘**

**V5 대비 개선 사항:**
- 더 세밀한 시간 양자화 (TIME: 0-95)
- 더 나은 장르 표현
- 악기 계열 그룹화
- 추가 아티큘레이션 마커
- 최적화된 토큰 분포

**사용법:**
```python
from data.vocab_v6 import build_v6_vocab
vocab = build_v6_vocab()
print(f"어휘 크기: {len(vocab)}")  # 645
```

### `inference/inference_v6.py`

**고급 MIDI 생성 추론**

**기능:**
- ✅ 컨텍스트 인식 생성 (기존 바 읽기)
- ✅ 대상 악기 지정
- ✅ 다중 악기 다성부 생성
- ✅ Temperature & top-k 샘플링 제어
- ✅ 출력 MIDI 파일 생성

**사용법:**

```bash
# 기본 사용
python inference/inference_v6.py --song input.mid --output output.mid

# 특정 악기 생성
python inference/inference_v6.py --song input.mid --target violin --output output.mid

# 다성부 생성
python inference/inference_v6.py --song input.mid --polyphonic --num_bars 16

# 고급 샘플링
python inference/inference_v6.py --song input.mid --temperature 0.8 --top_k 50
```

**매개변수:**
| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `--song` | 입력 MIDI 파일 | 필수 |
| `--target` | 대상 악기 (예: 피아노, 바이올린) | 모두 |
| `--num_bars` | 생성할 바 수 | 8 |
| `--temperature` | 샘플링 온도 (높을수록 더 창의적) | 0.9 |
| `--top_k` | Top-K 샘플링 (0 = 비활성화) | 50 |
| `--polyphonic` | 다중 악기 생성 | False |
| `--output` | 출력 MIDI 파일 경로 | `output.mid` |

### `data/preprocess_lmd.py`

**LMD 데이터셋 전처리 유틸리티**

다음을 처리합니다:
- 중첩된 디렉토리에서 MIDI 추출
- 품질 검증 (지속 시간, 트랙 수 등)
- 메타데이터 추출
- 학습 데이터셋 필터링

**사용법:**
```bash
python data/preprocess_lmd.py --input /path/to/lmd --output /path/to/processed
```

### `data/jsonl_to_memmap.py`

**토큰화된 JSONL → 바이너리 memmap 형식 변환**

**memmap을 사용하는 이유?**
- ✅ 일정한 메모리 풋프린트 (RAM에 전체 데이터셋 없음)
- ✅ 빠른 무작위 액세스
- ✅ DDP의 프로세스 간 공유

**사용법:**
```bash
python data/jsonl_to_memmap.py --input train.jsonl --output train.bin --seq_len 8192
```

**입력 형식 (JSONL):**
```json
{"tokens": [1, 42, 128, 7, 64, ...]}
{"tokens": [1, 55, 100, 3, 72, ...]}
```

**출력 형식 (바이너리):**
```
uint16 배열: [n_samples, seq_len] → .bin 파일로 저장
메타데이터: 샘플 인덱스 & 메타데이터가 있는 .json 파일
```

## 🔧 요구 사항

```
torch>=2.5.0
transformers>=4.51.0
datasets>=3.0.0
accelerate>=1.2.0
numpy
pretty_midi
tqdm
```

**선택 사항 (효율성을 위해):**
```
flash-attn>=2.8.0  # 더 빠른 학습을 위한 FlashAttention-2
peft>=0.4.0        # 매개변수 효율적 미세 조정용
```

## 📊 데이터 파이프라인

```
LMD 데이터셋 (원본 MIDI)
  ↓ [data/preprocess_lmd.py]
필터링 & 추출된 MIDI
  ↓ [토큰화]
토큰 시퀀스 (JSONL)
  ↓ [data/jsonl_to_memmap.py]
바이너리 Memmap 형식 (.bin)
  ↓ [models/train.py]
분산 학습 (DDP)
  ↓
최고 체크포인트
  ↓ [inference/inference_v6.py]
생성된 MIDI
```

## 🚀 빠른 시작

### 1. 데이터 준비

```bash
# LMD 추출 및 전처리
python data/preprocess_lmd.py \
  --input /path/to/lmd_full \
  --output /path/to/processed_data

# 토큰을 memmap으로 변환 (더 빠른 로딩용)
python data/jsonl_to_memmap.py \
  --input /path/to/processed_data/train.jsonl \
  --output /path/to/data/train.bin \
  --seq_len 8192
```

### 2. 모델 학습 (단일 GPU)

```bash
python models/train.py
```

### 3. 모델 학습 (다중 GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 models/train.py
```

### 4. 음악 생성

```bash
python inference/inference_v6.py \
  --song input_song.mid \
  --target violin \
  --num_bars 16 \
  --temperature 0.8
```

## ⚡ 학습 설정

| 설정 | 값 |
|------|-----|
| 모델 | Qwen2.5-0.5B |
| 어휘 크기 | 645 토큰 |
| 시퀀스 길이 | 8192 토큰 |
| 배치 크기 (GPU당) | 16 |
| 그래디언트 누적 | 2 스텝 |
| 유효 배치 | 64 (4 GPUs) |
| 학습률 | 5e-5 (코사인) |
| 워밍업 | 스텝의 5% |
| 최적화 | AdamW + bfloat16 |
| 어텐션 | FlashAttention-2 |
| 조기 중지 | 10 에포크 인내 |

## 📈 어휘 V6 사양

**토큰 카테고리 (총 645개):**

| 카테고리 | 수 | 목적 |
|---------|-----|------|
| 특수 | 15 | 제어 흐름 (BOS, EOS, PHRASE_END 등) |
| 장르 | 7 | 음악 장르 분류 |
| 조성 | 25 | 음조 중심 (C maj/min, C# maj/min, ...) |
| 박자표 | 6 | 박자표 (4/4, 3/4 등) |
| 밀도 | 5 | 음표 밀도 지표 |
| 악기 | 129 | MIDI 프로그램 + 드럼 |
| 피치 | 128 | 음표 피치 (0-127) |
| 지속 시간 | 192 | 양자화된 음표 지속 시간 |
| 벨로시티 | 32 | MIDI 벨로시티 (0-127 → 32 수준) |
| 타이밍 | 96 | 바 내 타이밍 |
| 표현 | 32 | 표현 CC (모드 휠 등) |
| 아티큘레이션 | 4 | 스타카토, 레가토, 비브라토, 일반 |
| 텍스트 마커 | 11 | 의미 키워드 |

## 💾 출력 구조

```
checkpoints/
├── train_logs.jsonl          # 학습 메트릭
├── best_model/               # 최고 검증 체크포인트
│   ├── pytorch_model.bin
│   ├── config.json
│   └── generation_config.json
└── checkpoint-*/             # 중간 체크포인트
    └── ...
```

## 🎯 기능 & 장점 (v2 vs v1)

| 기능 | v1 | v2 |
|------|----|----|
| **학습** | 단일 GPU Colab | 다중 GPU DDP |
| **시퀀스 길이** | 2048 | 8192 |
| **어휘** | 682 (V5) | 645 (V6, 최적화) |
| **데이터 로딩** | JSONL + Tensor | Memmap (메모리 효율적) |
| **장르 균형** | 수동 분할 | 가중 샘플러 |
| **추론** | 기본 생성 | 고급 전략 |
| **코드 구조** | 단일 노트북 | 모듈식 스크립트 |
| **프로덕션 준비** | ⚠️ 실험용 | ✅ 프로덕션 |

## 🔍 모니터링 & 디버깅

**학습 상태 확인:**
```bash
# GPU 사용 모니터링
watch -n 1 nvidia-smi

# 학습 손실 확인
tail -f train_logs.jsonl | grep loss

# 체크포인트에서 재개
# (train.py를 수정하여 resume_from_checkpoint 설정)
```

## 📚 참고 자료

- **토큰화 전략**: [Music Transformer](https://arxiv.org/abs/1809.04281)에서 영감을 받은 이벤트 기반 MIDI 표현
- **LMD 데이터셋**: [Lakh MIDI Dataset](https://www.karolpiczak.com/lmd/)
- **분산 학습**: Hugging Face [Transformers + Accelerate](https://huggingface.co/docs/transformers/training)
- **FlashAttention**: [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)

## 📚 Documentation
* [V6 Pipeline.20260512](./v6_pipeline_hybrid_upgrade_guide.md): 이번 업데이트에서 개선된 전처리 및 학습 파이프라인 가이드입니다.

## 📄 라이선스

연구/교육용

## 👤 저자

졸업 음악 프로젝트 - Qwen MIDI 생성 팀

---

**상태**: 프로덕션 준비 완료 (v2) ✅

실험 코드 & 전처리 세부 사항은 `projects_v1/` 참고
