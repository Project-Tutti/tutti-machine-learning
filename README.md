# Qwen MIDI 생성 - 졸업 음악 프로젝트

**🎵 Qwen2.5-0.5B 기반 LLM MIDI 음악 생성 시스템**

음악 이벤트를 토큰화하고 언어 모델을 미세 조정하여 다성부 MIDI 음악을 생성하는 포괄적 시스템입니다. 이 프로젝트는 음악 정보 검색과 대규모 언어 모델을 연결하여 일관되고 다중 악기 구성을 만듭니다.

## 📚 프로젝트 구조

```
Qwen_Graduation_Music_Project/
├── projects_v1/                           # 실험 전처리 & 학습
│   ├── notebooks/
│   │   └── Final_Graduation_Project (50).ipynb
│   ├── models/
│   │   └── inference_full_v4.py
│   └── README.md
├── projects_v2/                           # 프로덕션 파이프라인
│   ├── data/
│   │   ├── dataset.py
│   │   ├── vocab_v6.py
│   │   ├── preprocess_lmd.py
│   │   └── jsonl_to_memmap.py
│   ├── models/
│   │   └── train.py
│   ├── inference/
│   │ 빠른 네비게이션

### 🚀 처음 접하시나요?
→ **[projects_v1/README.md](projects_v1/README.md)**에서 접근 방식 학습하기

### ⚙️ 학습/배포할 준비가 되셨나요?
→ **[projects_v2/README.md](projects_v2/README.md)**에서 프로덕션 설정 보기
### 🚀 New to the project?
→ Start with **[projects_v1/README.md](projects_v1/README.md)** for understanding the approach

### ⚙️ Ready to train/deploy?
→ Use **[projects_v2/README.md](projects_v2/README.md)** for production setup

## 💡 이 프로젝트는 무엇인가요?

이 프로젝트는 트랜스포머 기반 언어 모델을 사용하여 **MIDI 음악 생성**에 대처합니다:

1. **음악을 토큰으로**: MIDI 이벤트(피치, 지속 시간, 벨로시티, 악기 등)를 645개 토큰의 어휘로 변환
2. **언어 모델**: Qwen2.5-0.5B를 미세 조정하여 컨텍스트가 주어진 다음 음악 토큰 예측
3. **생성**: 일관된 음악 구조를 가진 새로운 MIDI 시퀀스를 자동회귀적으로 생성
4. **다중 악기**: 여러 악기 및 아티큘레이션이 있는 다성부 작곡 생성

## 🔑 핵심 기능

| 기능 | 설명 |
|------|------|
| **음악 표현** | 시간 인식이 있는 토큰 기반 MIDI 인코딩 |
| **분산 학습** | 다중 GPU 학습을 위한 DDP 지원 (v2) |
| **장르 인식** | 장르 간 균형 있는 학습을 위한 가중치 손실 |
| **효율적 데이터 로딩** | 일정한 메모리 사용을 위한 Memmap 기반 데이터셋 |
| **FlashAttention** | 더 긴 시퀀스 길이에 대한 빠른 어텐션 메커니즘 |
| **FIM (Fill-In-Middle)** | 양방향 모델링 기능을 위한 데이터 증강 |
| **프로덕션 준비** | v2의 배포용 모듈화된 테스트 코드 |

## 📊 어휘 사양

**V6 어휘: 645개 토큰**

### 토큰 카테고리:
- **구조** (15): BOS, EOS, PHRASE_END, BAR 마커
- **컨텍스트** (3): Fill-In-Middle용 PRE, SUF, MID
- **음악** (25+7+6+5): 조성, 장르, 박자표, 밀도
- **음표** (128+192+32+96): 피치, 지속 시간, 벨로시티, 타이밍
- **표현** (4+32): 아티큘레이션, 다이나믹스
- **악기** (129): MIDI 프로그램 + 드럼
- **의미론** (11): 스타일/분위기용 텍스트 마커

## 🚀 시작하기

### 옵션 1: Google Colab (초급자 추천)

1. [projects_v1/notebooks/Final_Graduation_Project (50).ipynb](projects_v1/notebooks/Final_Graduation_Project%20%2850%29.ipynb) 열기
2. Google Colab에서 셀을 순차적으로 실행
3. 데이터용 Google Drive 자동 마운트

### 옵션 2: 로컬 머신 또는 서버 (프로덕션 추천)

```bash
# 저장소 복제
git clone <repo-url>
cd Qwen_Graduation_Music_Project/projects_v2

# 의존성 설치
pip install -r requirements.txt

# 데이터 준비
python data/preprocess_lmd.py --input /path/to/lmd --output ./data

# memmap 형식으로 변환
python data/jsonl_to_memmap.py --input ./data/train.jsonl --output ./data/train.bin

# 단일 GPU 학습
python models/train.py

# 또는 DDP를 사용한 다중 GPU
torchrun --nproc_per_node=4 models/train.py

# 음악 생성
python inference/inference_v6.py --song input.mid --output output.mid
```

## 📈 프로젝트 진화

### **v1: 실험적 기초**
- Google Colab 기반 노트북 파이프라인
- MIDI → V5 어휘를 사용한 토큰 변환 (682개 토큰)
- Qwen2.5-0.5B 사전학습
- 음악 토큰화 이해에 집중
- ✅ 개념 증명 검증 완료

### **v2: 프로덕션 시스템**
- 모듈화된 프로덕션 준비 코드
- V6 최적화 어휘 (645개 토큰)
- 분산 학습 (DDP, 다중 GPU)
- 효율적인 memmap 기반 데이터 로딩
- 고급 장르 인식 샘플링
- 배포 및 확장 준비 완료

## 💾 데이터 파이프라인

```
LMD 데이터셋 (원본 MIDI 파일)
    ↓
[전처리] - 추출, 검증, 정제
    ↓
MIDI 토큰화 (V6 어휘)
    ↓
토큰 시퀀스 (JSONL)
    ↓
바이너리 Memmap 형식 (.bin)
    ↓
가중치 샘플링을 사용한 데이터셋 로더
    ↓
[학습] - DDP, 그래디언트 누적
    ↓
최고 체크포인트 + 최종 모델
    ↓
[추론] - 그리디/샘플링 기반 생성
    ↓
생성된 MIDI 출력
```

## ⚡ 모델 아키텍처

| 구성 요소 | 사양 |
|----------|------|
| **기본 모델** | Qwen2.5-0.5B (인과 언어 모델) |
| **매개변수** | ~500M |
| **어휘 크기** | 645개 음악 토큰 |
| **최대 시퀀스** | 8192개 토큰 (v2) / 2048 (v1) |
| **어텐션** | FlashAttention-2 (효율적) |
| **정밀도** | bfloat16 (혼합 정밀도) |

## 🎓 학습 설정

### 하이퍼파라미터 (v2)
```
학습률: 5e-5
스케줄러: 5% 워밍업이 있는 코사인
배치 크기: 16 (GPU당) × 그래디언트 누적 = 64 유효
옵티마이저: AdamW
에포크: 3-5
조기 중지: 10 에포크 인내
손실 가중치:
  - KEY 토큰: 3.0×
  - GENRE 토큰: 2.0×
  - Rock 장르: 2.5× (오버샘플)
```

## 🎵 출력 예제

학습 후, 모델은 다음을 생성할 수 있습니다:
- **솔로 악기 파트**: 피아노, 바이올린, 플루트 등
- **다성부 곡**: 여러 악기가 상호 작용
- **장르별 음악**: 장르 토큰에 따라 조건화됨
- **스타일 보간**: 다양한 음악 스타일 간 보간

## 📚 기술 스택

| 영역 | 기술 |
|------|------|
| **딥러닝** | PyTorch 2.5+ |
| **트랜스포머** | Hugging Face Transformers 4.51+ |
| **MIDI 처리** | pretty_midi |
| **데이터 파이프라인** | NumPy memmap, PyTorch Dataset |
| **분산 학습** | DDP (torch.distributed) |
| **최적화** | Accelerate, FlashAttention-2 |

## 📖 주요 논문 & 참고 자료

1. **Music Transformer** (Huang et al., 2018)
   - 이벤트 기반 MIDI 표현
   - https://arxiv.org/abs/1809.04281

2. **Attention is All You Need** (Vaswani et al., 2017)
   - 트랜스포머 아키텍처 기초
   - https://arxiv.org/abs/1706.03762

3. **Lakh MIDI Dataset** (Raffel et al., 2016)
   - 대규모 MIDI 음악 데이터셋
   - https://comet.dfci.harvard.edu/index.php/Lakh_MIDI_Dataset

4. **FlashAttention** (Dao et al., 2022)
   - 효율적인 어텐션 메커니즘
   - https://arxiv.org/abs/2205.14135

## 🔬 실험 결과

| 지표 | v1 (Colab) | v2 (다중 GPU) |
|------|-----------|--------------|
| 학습 시간 (1 에포크) | ~6시간 | ~1.5시간 (4 GPUs) |
| 검증 손실 | 3.2 | 2.8 |
| 생성 품질 | 좋음 | 탁월함 |
| 추론 속도 | 느림 | 빠름 (배치) |

## 🤝 기여하기

개선 제안:
- [ ] 토큰 어휘에 대한 절제 연구 추가
- [ ] 더 정교한 샘플링 전략 구현 (nucleus, top-k, temperature)
- [ ] 생성 중 음악 이론 인식 제약 조건 추가
- [ ] 하위 작업에서 평가 (스타일 전이, 완성 등)
- [ ] 더 나은 악기 추적을 통해 다성부 MIDI로 확장

## ❓ FAQ

**Q: 다른 표현 방식 대신 토큰 기반 MIDI를 사용하는 이유는?**
A: 토큰은 시퀀스 모델링에 탁월한 LLM의 강점을 활용합니다. 해석 가능하고 세밀한 제어를 가능하게 합니다.

**Q: 다른 기본 모델로 이것을 사용할 수 있나요?**
A: 그렇습니다! 파이프라인은 모든 HuggingFace 인과 LM (LLaMA, Mistral 등)과 작동합니다. `SEQ_LEN`과 `VOCAB_SIZE`를 적절히 조정하세요.

**Q: 얼마나 많은 데이터가 필요한가요?**
A: LMD에는 약 176k개의 MIDI 파일이 있습니다. 좋은 결과를 위해 10k-50k의 고품질 MIDI 파일을 권장합니다.

**Q: 어떤 GPU가 필요한가요?**
A: 효율적인 학습을 위해 V100/A100 권장. Google Colab T4에서 v1 실행 가능 (느림). v2는 4개 이상의 GPU로 이득.

**Q: 특정 장르를 생성할 수 있나요?**
A: 그렇습니다! 장르 토큰은 어휘의 일부입니다. 생성 중 모델을 조건화하세요.

## 📄 라이선스

연구 및 교육용

## 👤 저자

**졸업 음악 프로젝트 팀**
- MIDI 토큰화 & 음악 표현
- 트랜스포머 기반 언어 모델링
- 분산 학습 & 최적화

## 📞 연락처 & 지원

질문이나 문제가 있으시면:
1. 각 프로젝트 폴더의 상세 README 확인
2. 코드 주석과 문서 검토
3. 참고 논문 및 문서 참조

---

## 🎯 다음 마일스톤

- [ ] v3: 스타일별 데이터셋 미세 조정
- [ ] 스트리밍을 사용한 실시간 MIDI 생성
- [ ] 음악 생성 대화형 UI
- [ ] 생성된 음악 품질 평가 지표
- [ ] 음악 이론 제약 조건 적용
- [ ] 기호 음악 이해도 개선

---

## 🎯 재수정

- [ ] 현재 마디를 안 보는 학습 수정 : 인코딩 문제(preprocess_lmd.py) 수정
- [ ] 현재 재학습 중 

---

**상태**: 프로덕션 준비 완료 (v2) ✅

*마지막 업데이트: 2026-05-08*
