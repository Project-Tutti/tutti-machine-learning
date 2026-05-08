"""
pretrain.py
Qwen2.5-0.5B 새 매핑 기준 Pretrain 스크립트

서버 환경: Ubuntu, 4× RTX 4090
jaehyeon이 GPU 0 사용 중 → GPU 1,2,3 사용

실행:
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 pretrain.py
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 pretrain.py --resume
"""

import os
import json
import random
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    TrainerCallback, EarlyStoppingCallback,
)
from tqdm.auto import tqdm as tqdm_auto

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
DATA_DIR   = "/data/tutti/lmd_memmap_remapped"
TRAIN_BIN  = os.path.join(DATA_DIR, "train_pretrain.bin")
VAL_BIN    = os.path.join(DATA_DIR, "val_pretrain.bin")
META_PATH  = os.path.join(DATA_DIR, "dataset_meta.json")
CKPT_DIR   = "/data2/tutti/pretrain_new_ckpt"
BEST_DIR   = os.path.join(CKPT_DIR, "best")
FINAL_DIR  = os.path.join(CKPT_DIR, "final")

# ──────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────
SEQ_LEN    = 2048
VOCAB_SIZE = 679
PAD_ID     = 0
FIM_RATE   = 0.5
PATIENCE   = 10

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="마지막 체크포인트에서 이어서 학습")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
def build_v5_vocab():
    vocab = {}
    def add(prefix, r):
        for i in r: vocab[f"{prefix}{i}"] = len(vocab)
    for t in ["PAD","BOS","EOS","SEP","PIECE_START","PIECE_END",
              "BAR_START","BAR_END","PHRASE_END","<PRE>","<SUF>","<MID>"]:
        vocab[t] = len(vocab)
    for g in ["CLASSICAL","JAZZ","POP","ROCK","ELECTRONIC","FOLK","UNKNOWN"]:
        vocab[f"GENRE_{g}"] = len(vocab)
    roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    for r in roots:
        for m in [":maj",":min"]: vocab[f"KEY_{r}{m}"] = len(vocab)
    vocab["KEY_NONE"] = len(vocab)
    for m in ["4:4","3:4","2:4","6:8","12:8","OTHER"]:
        vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1,6))
    add("INST=",    range(129))
    for a in ["ART_NORMAL","ART_LEGATO","ART_VIBRATO","ART_STACCATO"]:
        vocab[a] = len(vocab)
    add("EXPR_",  range(32))
    add("TIME=",  range(96))
    add("PITCH=", range(128))
    add("DUR=",   range(1,193))
    add("VEL=",   range(32))
    for w in ["melodic","epic","calm","fast","slow","sad","happy",
              "piano","strings","orchestra","cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


# ──────────────────────────────────────────────
# FIM (Fill-In-the-Middle) 온라인 적용
# ──────────────────────────────────────────────
def apply_fim_online(tokens, vocab, seq_len):
    if random.random() > FIM_RATE:
        return tokens

    BAR_START_ID = vocab["BAR_START"]
    PRE_ID       = vocab["<PRE>"]
    SUF_ID       = vocab["<SUF>"]
    MID_ID       = vocab["<MID>"]

    try:
        first_bar = tokens.index(BAR_START_ID)
    except ValueError:
        return tokens

    header      = tokens[:first_bar]
    body_tokens = tokens[first_bar:]
    bar_positions = [i for i, t in enumerate(body_tokens) if t == BAR_START_ID]

    if len(bar_positions) < 3:
        return tokens

    mid_idx   = random.randint(1, len(bar_positions) - 2)
    mid_start = bar_positions[mid_idx]
    mid_end   = (bar_positions[mid_idx + 1]
                 if mid_idx + 1 < len(bar_positions)
                 else len(body_tokens))

    prefix = body_tokens[:mid_start]
    middle = body_tokens[mid_start:mid_end]
    suffix = body_tokens[mid_end:]

    fim = [PRE_ID] + header + prefix + [SUF_ID] + suffix + [MID_ID] + middle
    return fim[:seq_len]


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class MidiBinaryDataset(Dataset):
    def __init__(self, bin_path, n_samples, seq_len, vocab):
        self.seq_len  = seq_len
        self.n        = n_samples
        self.vocab    = vocab
        self.arr      = np.memmap(bin_path, dtype="uint16", mode="r",
                                  shape=(n_samples, seq_len))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        toks = self.arr[idx].tolist()
        toks = apply_fim_online(toks, self.vocab, self.seq_len)
        toks = (toks + [PAD_ID] * self.seq_len)[:self.seq_len]
        return torch.tensor(toks, dtype=torch.long)


def midi_collate_fn(batch):
    ids    = torch.stack(batch)
    labels = ids.clone()
    labels[labels == PAD_ID] = -100
    mask   = (ids != PAD_ID).long()
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ──────────────────────────────────────────────
# WeightedTrainer
# ──────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, *args, key_ids, genre_weights, inst_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_ids       = key_ids
        self.genre_weights = genre_weights  # {token_id: weight}
        self.inst_weights  = inst_weights   # {token_id: weight}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fct     = CrossEntropyLoss(reduction="none", ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, VOCAB_SIZE),
            shift_labels.view(-1))

        weights = torch.ones_like(shift_labels, dtype=torch.float)
        for kid in self.key_ids:
            weights[shift_labels == kid] = 3.0
        for gid, w_val in self.genre_weights.items():
            weights[shift_labels == gid] = w_val
        for iid, w_val in self.inst_weights.items():
            weights[shift_labels == iid] = w_val

        w = weights.view(-1)
        w[shift_labels.view(-1) == -100] = 0.0
        weighted_loss = (loss * w).sum() / max(w.nonzero().numel(), 1)
        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ──────────────────────────────────────────────
# 콜백
# ──────────────────────────────────────────────
class TqdmProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps  = total_steps
        self.pbar         = None
        self.cur_loss     = 0.0
        self.cur_lr       = 0.0
        self.patience     = 0
        self.max_patience = PATIENCE

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.pbar = tqdm_auto(
                total        = self.total_steps,
                desc         = "Pretrain",
                dynamic_ncols= True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not self.pbar:
            return
        if logs:
            self.cur_loss = logs.get("loss", self.cur_loss)
            self.cur_lr   = logs.get("learning_rate", self.cur_lr)
        self.pbar.set_postfix_str(
            f"loss={self.cur_loss:.4f}, "
            f"lr={self.cur_lr:.2e}, "
            f"patience={self.patience}/{self.max_patience}"
        )
        self.pbar.n = state.global_step
        self.pbar.refresh()

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


class SaveBestCallback(TrainerCallback):
    def __init__(self, best_dir, output_dir, tqdm_cb=None):
        self.best_dir      = best_dir
        self.output_dir    = output_dir
        self.best_loss     = float("inf")
        self.best_step     = 0
        self.patience_cnt  = 0
        self._tqdm_cb      = tqdm_cb
        self._pending_save = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_local_process_zero or metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float("inf"))
        step      = state.global_step

        if eval_loss < self.best_loss:
            self.best_loss     = eval_loss
            self.best_step     = step
            self.patience_cnt  = 0
            self._pending_save = True
            tqdm_auto.write(f"\n  Step {step}: val_loss={eval_loss:.4f} (best 갱신)")
        else:
            self.patience_cnt += 1
            tqdm_auto.write(
                f"\n  Step {step}: val_loss={eval_loss:.4f} "
                f"(best={self.best_loss:.4f}, "
                f"patience={self.patience_cnt}/{PATIENCE})"
            )

        if self._tqdm_cb:
            self._tqdm_cb.patience = self.patience_cnt

    def on_save(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return
        step = state.global_step
        if self._pending_save:
            ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}")
            if os.path.exists(ckpt_path):
                os.makedirs(os.path.dirname(self.best_dir), exist_ok=True)
                if os.path.exists(self.best_dir):
                    shutil.rmtree(self.best_dir)
                shutil.copytree(ckpt_path, self.best_dir)
                tqdm_auto.write(
                    f"  ✓ Best model saved → {self.best_dir} "
                    f"(val_loss={self.best_loss:.4f})"
                )
                self._pending_save = False

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            tqdm_auto.write(
                f"\nPretrain 완료! Best val_loss: {self.best_loss:.4f} "
                f"(step {self.best_step})"
            )


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    vocab      = build_v5_vocab()
    key_ids    = {v for k, v in vocab.items()
                  if k.startswith("KEY_") and k != "KEY_NONE"}
    # 장르별 loss 가중치 (분포 역수 기반)
    # ELECTRONIC 44%, POP 33%, CLASSICAL 11%, ROCK 9%, JAZZ 2%, FOLK 0.5%
    genre_weights = {
        vocab["GENRE_CLASSICAL"]:  4.0,
        vocab["GENRE_ROCK"]:       4.0,
        vocab["GENRE_POP"]:        2.0,
        vocab["GENRE_JAZZ"]:       6.0,
        vocab["GENRE_FOLK"]:       8.0,
        vocab["GENRE_ELECTRONIC"]: 1.0,
        vocab["GENRE_UNKNOWN"]:    1.0,
    }

    # 악기별 loss 가중치 (분포 역수 기반)
    # Drum 29%, Guitar 19%, Keyboard 19%, Ensemble 8.6%, Bass 8.4%,
    # DistGuitar 3.9%, Organ 2.8%, Woodwind 2.2%, Brass/Synth 1.7%,
    # Mallet/Sax 1.3~1.4%, SoloString 0.9% (가장 적음 → 최고 가중치)
    inst_weights = {
        vocab["INST=128"]: 1.0,   # Drum       29%
        vocab["INST=25"]:  1.0,   # Guitar     19%
        vocab["INST=0"]:   1.0,   # Keyboard   19%
        vocab["INST=48"]:  1.5,   # Ensemble    8.6%
        vocab["INST=33"]:  1.5,   # Bass        8.4%
        vocab["INST=30"]:  2.0,   # DistGuitar  3.9%
        vocab["INST=16"]:  3.0,   # Organ       2.8%
        vocab["INST=73"]:  3.0,   # Woodwind    2.2%
        vocab["INST=56"]:  4.0,   # Brass       1.7%
        vocab["INST=81"]:  4.0,   # Synth       1.7%
        vocab["INST=12"]:  4.0,   # Mallet      1.4%
        vocab["INST=65"]:  4.0,   # Saxophone   1.3%
        vocab["INST=40"]:  6.0,   # SoloString  0.9% ← 최고
    }

    # 데이터셋 로드
    with open(META_PATH) as f:
        meta = json.load(f)

    train_n = meta["train_chunks"]
    val_n   = meta["val_chunks"]

    print(f"[INFO] train: {train_n:,} chunks / val: {val_n:,} chunks")

    train_ds = MidiBinaryDataset(TRAIN_BIN, train_n, SEQ_LEN, vocab)
    val_ds   = MidiBinaryDataset(VAL_BIN,   val_n,   SEQ_LEN, vocab)

    # 모델 로드
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.vocab_size              = VOCAB_SIZE
    config.pad_token_id            = PAD_ID
    config.max_position_embeddings = SEQ_LEN
    config.sliding_window          = None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config                  = config,
        torch_dtype             = torch.bfloat16,
        ignore_mismatched_sizes = True,
        attn_implementation     = "sdpa",
    )
    model.config.use_cache   = False
    model.model.embed_tokens = nn.Embedding(VOCAB_SIZE, config.hidden_size).to(torch.bfloat16)
    model.lm_head            = nn.Linear(config.hidden_size, VOCAB_SIZE, bias=False).to(torch.bfloat16)

    print(f"[INFO] 파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # resume 체크포인트 찾기
    resume_from = None
    if args.resume:
        if os.path.exists(CKPT_DIR):
            ckpts = sorted(
                [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1])
            )
            if ckpts:
                resume_from = os.path.join(CKPT_DIR, ckpts[-1])
                print(f"[INFO] Resume from: {resume_from}")

    # TrainingArguments
    PER_DEVICE_BATCH = 4
    GRAD_ACCUM       = 8   # effective batch = 4 * 8 * 3 GPU = 96
    N_EPOCHS         = 3
    total_steps      = (train_n // (PER_DEVICE_BATCH * GRAD_ACCUM)) * N_EPOCHS

    training_args = TrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = N_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = 5e-5,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = True,
        eval_strategy               = "steps",
        eval_steps                  = 2000,
        save_strategy               = "steps",
        save_steps                  = 2000,
        save_total_limit            = 2,
        logging_steps               = 200,
        dataloader_num_workers      = 4,
        dataloader_drop_last        = True,
        report_to                   = "none",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        disable_tqdm                = True,
        ignore_data_skip            = True,
        remove_unused_columns       = False,
        ddp_find_unused_parameters  = False,
    )

    # 콜백
    tqdm_cb      = TqdmProgressCallback(total_steps=total_steps)
    save_best_cb = SaveBestCallback(
        best_dir   = BEST_DIR,
        output_dir = CKPT_DIR,
        tqdm_cb    = tqdm_cb,
    )
    tqdm_cb.max_patience = PATIENCE

    trainer = WeightedTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = midi_collate_fn,
        key_ids       = key_ids,
        genre_weights = genre_weights,
        inst_weights  = inst_weights,
        callbacks     = [
            EarlyStoppingCallback(early_stopping_patience=PATIENCE),
            save_best_cb,
            tqdm_cb,
        ],
    )

    print(f"[INFO] 🚀 Pretrain 시작")
    print(f"       모델: Qwen2.5-0.5B")
    print(f"       effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM * 3}")
    print(f"       steps/epoch: {train_n // (PER_DEVICE_BATCH * GRAD_ACCUM):,}")
    print(f"       총 steps: {total_steps:,}")
    print(f"       eval/save: 매 2000 steps")
    print(f"       early stopping patience: {PATIENCE}")
    print(f"       best → {BEST_DIR}")
    print(f"       resume: {resume_from or '없음 (처음부터)'}")

    trainer.train(resume_from_checkpoint=resume_from)

    # best → final 복사 (rank 0만)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        if os.path.exists(BEST_DIR):
            if os.path.exists(FINAL_DIR):
                shutil.rmtree(FINAL_DIR)
            shutil.copytree(BEST_DIR, FINAL_DIR)
            print(f"✅ Final 저장: {FINAL_DIR}")

        if trainer.state.global_step < total_steps:
            print(f"🛑 Early Stopping! {PATIENCE}번 연속 개선 없음")

    print("✅ Pretrain 완료")


if __name__ == "__main__":
    main()