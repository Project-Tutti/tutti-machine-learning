# Qwen MIDI GPT2 - 전처리 & 초기 학습 (v1)

**Qwen2.5-0.5B 모델을 사용한 토큰 수준의 음악 표현 기반 LLM MIDI 음악 생성 시스템**

## 📋 개요

이 폴더는 언어 모델을 사용하여 MIDI 음악을 생성하기 위한 **초기 전처리 파이프라인과 실험 학습 설정**을 포함합니다. 다음에 중점을 둡니다:

- **데이터 준비**: LMD(Lakh MIDI Dataset)에서 MIDI 추출 및 토큰 시퀀스로 변환
- **어휘 정의**: 피치, 지속 시간, 벨로시티, 아티큘레이션, 장르, 조성, 박자표 등으로 구성된 포괄적인 음악 토큰 어휘 생성
- **토큰화**: MIDI 이벤트를 언어 모델 학습에 최적화된 구조화된 토큰 형식으로 변환
- **사전학습**: 음악 구조 보존을 위한 가중 손실을 사용하여 Qwen2.5-0.5B를 MIDI 토큰에 미세 조정

## 🗂️ 폴더 구조

```
projects_v1/
├── notebooks/
│   └── Final_Graduation_Project (50).ipynb   # 메인 전처리 & 사전학습 노트북
├── models/
│   └── inference_full_v4.py                  # 추론 파이프라인 (음악 생성)
└── README.md
```

## 📝 파일 설명

### `notebooks/Final_Graduation_Project (50).ipynb`

**전체 파이프라인을 포함하는 메인 Jupyter 노트북:**

| 섹션 | 목적 |
|------|------|
| Drive 마운트 & 환경 설정 | Google Drive 통합, PyTorch/Transformers 설정 |
| 어휘 정의 | 682개 토큰으로 V5 어휘 구축 (장르, 조성, 박자표, 악기, 다이나믹스 등) |
| 토크나이저 | MIDI → 토큰 시퀀스 변환 (템포, 아티큘레이션, 표현 처리) |
| 실행 | 병렬 MIDI 처리 및 2048 토큰 시퀀스로 청킹 |
| 검증 | 토큰 분포 분석, 데이터 품질 확인 |
| 사전학습 | Qwen2.5-0.5B 학습 (KEY/GENRE 토큰에 더 높은 가중치) |
| 미세조정 | (하위 작업 학습용 플레이스홀더) |

**주요 기능:**
- ✅ 템포 인식 지속 시간 계산
- ✅ 박자표 & 조성 추적
- ✅ 표현, 아티큘레이션 및 다이나믹스 인코딩
- ✅ 텍스트 캡션 지원 (MidiCaps)
- ✅ 학습 중 FIM (Fill-In-Middle) 증강
- ✅ 조기 중지 및 최고 모델 체크포인트

### `models/inference_full_v4.py`

**음악 생성 추론 스크립트**

사용법:
```bash
python models/inference_full_v4.py --song KissTheRain.mid --target flute
```

주어진 MIDI 파일에 대해 선택적 대상 악기 지정으로 새로운 악기 파트를 생성합니다.

## 🔧 요구 사항

```
torch==2.5.1+cu121
transformers==4.51.3
pretty_midi
tqdm
scikit-learn
accelerate==1.2.1
flash-attn==2.8.3  # 효율적인 어텐션용
```

## 📊 데이터 파이프라인

```
LMD 데이터셋 (.tar)
  ↓
추출 & 정렬
  ↓
MIDI 파싱 (pretty_midi)
  ↓
이벤트 토큰화 (어휘 V5)
  ↓
청킹 & 분할 (학습/검증 95/5)
  ↓
바이너리 형식 (.bin / .jsonl)
  ↓
학습
```

## 📈 어휘 사양 (V5)

**토큰 카테고리:**
- **특수 토큰**: PAD, BOS, EOS, SEP, PIECE_START/END, BAR_START/END, PHRASE_END
- **컨텍스트 마커**: PRE (접두사), SUF (접미사), MID (중간) — FIM용
- **구조**: GENRE (7), KEY (25), METER (6), DENSITY (5)
- **음표 이벤트**: INST (129), PITCH (128), DUR (192), VEL (32), TIME (96)
- **표현**: ART (4개 아티큘레이션), EXPR (32 표현 수준)
- **텍스트**: TEXT (11개 의미 키워드)

**총 어휘 크기: 682 토큰**

## 🚀 빠른 시작

1. **Google Drive 마운트** (Colab용):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **데이터셋 준비** (셀 3 실행):
   - Drive → 로컬 SSD에서 tar 복사
   - MIDI 파일 추출
   - 추출 확인

3. **전처리 실행** (셀 10 실행):
   - 모든 MIDI 파일을 병렬로 토큰화
   - 학습/검증 JSONL 파일 생성
   - 토큰 분포 검증

4. **모델 학습** (셀 23 실행):
   - 사전학습된 Qwen2.5-0.5B 로드
   - MIDI 토큰에 미세 조정
   - 최고 체크포인트 저장

5. **음악 생성** (학습 후 셀 실행):
   - 최고 체크포인트 로드
   - `inference_full_v4.py` 호출하여 생성

## 💾 출력 파일

| 파일 | 목적 |
|------|------|
| `train_pretrain.jsonl` | 학습 토큰 (데이터셋의 95%) |
| `val_pretrain.jsonl` | 검증 토큰 (데이터셋의 5%) |
| `train_pretrain.bin` / `val_pretrain.bin` | 바이너리 memmap 형식 (더 빠른 로딩) |
| `checkpoints/pretrain_0.5b/` | 저장된 학습 체크포인트 |
| `checkpoints/pretrain_0.5b/best/` | 최고 검증 손실 모델 |

## ⚡ 학습 설정

```
모델: Qwen2.5-0.5B
배치 크기: 16 (디바이스당) × 2 (그래디언트 누적) = 32 유효
학습률: 5e-5 (코사인 스케줄러, 5% 워밍업)
시퀀스 길이: 2048 토큰
에포크: 3
평가 빈도: 2000 스텝마다
조기 중지: 10 에포크 인내
손실: 가중 CrossEntropyLoss
  - KEY 토큰: 3.0×
  - GENRE 토큰: 2.0×
  - 기타: 1.0×
```

## 🎯 다음 단계 (v2)

`projects_v2/`에서 보기:
- ✅ 프로덕션 준비 코드 구조
- ✅ 분산 학습 (DDP, 다중 GPU)
- ✅ V6 어휘 개선
- ✅ 가중 샘플링을 사용한 더 나은 데이터 파이프라인
- ✅ 고급 추론 전략

## 📚 참고 자료

- **LMD 데이터셋**: [Lakh MIDI Dataset](https://comet.dfci.harvard.edu/index.php/Lakh_MIDI_Dataset)
- **MidiCaps**: [MidiCaps 데이터셋](https://github.com/google-research/midi-ddm)
- **Qwen 모델**: [Hugging Face의 Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **음악 표현**: 시간 인식이 있는 MIDI 이벤트 토큰화

## 📄 라이선스

연구/교육용

## 👤 저자

졸업 음악 프로젝트 - Qwen MIDI 생성 팀

---

**상태**: 실험용 (v1) - 프로덕션 배포는 v2 사용
