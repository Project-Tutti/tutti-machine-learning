#!/usr/bin/env python3
"""
Tutti V6 Training Script (Qwen2.5-0.5B)

사용법 (DDP, GPU 1/2/3):
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train.py

사용법 (Single GPU):
    CUDA_VISIBLE_DEVICES=1 python3 train.py
"""
import os
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, ConcatDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from vocab_v6 import build_v6_vocab
from dataset import MemmapMidiDataset

# ──────────────────────────────────────────
# 상수
# ──────────────────────────────────────────
VOCAB_SIZE = 645
SEQ_LEN    = 8192
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

DATA_DIR   = "/data/tutti/Gemma4_Dataset/"
CKPT_DIR   = "/data2/tutti/Qwen_Checkpoints/"

# 장르별 loss 가중치 (rock 토큰에 더 높은 가중치)
GENRE_LOSS_WEIGHTS = {
    "pop": 1.0,
    "electronic": 1.0,
    "classical": 1.0,
    "rock": 2.5,
    "other": 1.0,
}


# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────
def load_model(vocab):
    """
    Qwen2.5-0.5B의 Transformer 가중치(Attention, FFN)는 유지하고,
    Embedding과 LM Head만 우리 음악 vocab(645)에 맞게 교체합니다.
    """
    pad_id = vocab["PAD"]

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.vocab_size              = VOCAB_SIZE
    config.pad_token_id            = pad_id
    config.max_position_embeddings = SEQ_LEN
    config.sliding_window          = None

    # Pretrained Transformer 블록 로드
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
            attn_implementation="flash_attention_2",
        )
        print("✅ FlashAttention-2 활성화")
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=config,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
                attn_implementation="sdpa",
            )
            print("⚡ FlashAttention-2 미지원, PyTorch 고속 SDPA(Scaled Dot-Product Attention) 활성화")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=config,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
            )
            print("⚠️ FlashAttention-2 및 SDPA 미지원, 기본 attention 사용")

    model.config.use_cache = False

    # Embedding / LM Head 교체 (음악 vocab 645개용)
    model.model.embed_tokens = nn.Embedding(
        VOCAB_SIZE, config.hidden_size
    ).to(torch.bfloat16)
    model.lm_head = nn.Linear(
        config.hidden_size, VOCAB_SIZE, bias=False
    ).to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 모델 로드 완료: {total_params:.1f}M params")

    return model


# ──────────────────────────────────────────
# 학습 실행
# ──────────────────────────────────────────
def main():
    vocab = build_v6_vocab()
    assert len(vocab) == VOCAB_SIZE

    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. 모델
    model = load_model(vocab)

    # 2. 데이터셋
    train_dataset = MemmapMidiDataset(
        bin_path=os.path.join(DATA_DIR, "train.bin"),
        meta_path=os.path.join(DATA_DIR, "train_meta.json"),
        vocab=vocab,
        max_length=SEQ_LEN,
    )
    val_dataset = MemmapMidiDataset(
        bin_path=os.path.join(DATA_DIR, "val.bin"),
        meta_path=os.path.join(DATA_DIR, "val_meta.json"),
        vocab=vocab,
        max_length=SEQ_LEN,
    )
    
    # 평가 시간 단축을 위해 Val 데이터가 너무 많으면 1만 개로 제한
    if len(val_dataset) > 10000:
        unused_val = Subset(val_dataset, range(10000, len(val_dataset)))
        val_dataset = Subset(val_dataset, range(10000))
        # 버려지는 10만여 개의 데이터가 아깝지 않도록 Train 데이터셋에 병합!
        train_dataset = ConcatDataset([train_dataset, unused_val])
        
    print(f"✅ 데이터셋 로드: Train {len(train_dataset):,}개 / Val {len(val_dataset):,}개")

    # 3. TrainingArguments
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,

        # 배치 (효율 batch = 24 × 3 GPU × 4 = 288)
        per_device_train_batch_size=24,
        gradient_accumulation_steps=4,

        # 옵티마이저
        learning_rate=1e-4,  # 상향 조정 (2e-5 -> 1e-4)
        num_train_epochs=10,
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,

        # 평가 & 저장
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # 메모리
        gradient_checkpointing=True,

        # 로깅
        report_to="none",
        logging_steps=50,
        logging_first_step=True,

        # 데이터 로더
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # 가중치 학습용 토큰 추출
    KEY_IDS = {v for k, v in vocab.items() if k.startswith("KEY_")}
    GENRE_IDS = {v for k, v in vocab.items() if k.startswith("GENRE_")}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1)
            )

            weights = torch.ones_like(shift_labels, dtype=torch.float)
            for kid in KEY_IDS:
                weights[shift_labels == kid] = 3.0
            for gid in GENRE_IDS:
                weights[shift_labels == gid] = 2.0

            w = weights.view(-1)
            w[shift_labels.view(-1) == -100] = 0.0
            weighted_loss = (loss * w).sum() / max(w.nonzero().numel(), 1)
            return (weighted_loss, outputs) if return_outputs else weighted_loss

    # 4. Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=8),
        ],
    )

    # 5. 학습 시작 (체크포인트 자동 이어하기)
    last_ckpt = None
    if os.path.isdir(CKPT_DIR):
        ckpts = [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            last_ckpt = os.path.join(CKPT_DIR, sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1])
            print(f"📌 체크포인트에서 이어서 학습: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # 6. Best 모델 저장
    best_dir = os.path.join(CKPT_DIR, "best_model")
    trainer.save_model(best_dir)
    print(f"✅ Best 모델 저장 완료: {best_dir}")


if __name__ == "__main__":
    main()
