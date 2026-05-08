"""
Tutti V6 Dataset — Memmap 기반 PyTorch Dataset + 장르 가중치 샘플러
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from vocab_v6 import build_v6_vocab

# 장르별 샘플링 가중치 (rock 2.5배 오버샘플링)
GENRE_WEIGHTS = {
    "pop": 1.0,
    "electronic": 1.0,
    "classical": 1.0,
    "rock": 2.5,
    "other": 1.0,
}


class MemmapMidiDataset(Dataset):
    """
    uint16 memmap .bin 파일에서 데이터를 로드하는 Dataset.
    __getitem__에서 input_ids와 labels를 반환하며,
    labels는 <MID> 마커 이전 토큰과 PAD 토큰을 -100으로 마스킹합니다.
    """

    def __init__(self, bin_path, meta_path, vocab, max_length=8192):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.n_samples = meta["num_samples"]
        self.max_length = meta["max_length"]
        self.data = np.memmap(
            bin_path, dtype=np.uint16, mode='r',
            shape=(self.n_samples, self.max_length)
        )
        self.vocab = vocab
        self.pad_id = vocab["PAD"]
        self.mid_id = vocab["<MID>"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        tokens = np.array(self.data[idx], dtype=np.int64)
        input_ids = torch.tensor(tokens, dtype=torch.long)

        labels = input_ids.clone()

        # <MID> 토큰 위치 찾기 → 이전 토큰 전부 loss 무시
        mid_positions = (input_ids == self.mid_id).nonzero(as_tuple=True)[0]
        if len(mid_positions) > 0:
            mid_pos = mid_positions[0].item()
            labels[:mid_pos + 1] = -100

        # PAD 토큰 loss 무시
        labels[input_ids == self.pad_id] = -100

        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_sample_weights(jsonl_path):
    """
    JSONL 파일에서 각 샘플의 장르를 읽어 가중치 리스트를 반환합니다.
    WeightedRandomSampler에 사용됩니다.
    """
    weights = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            genre = data.get("genre", "other")
            weights.append(GENRE_WEIGHTS.get(genre, 1.0))
    return weights


def get_weighted_sampler(jsonl_path):
    """장르 가중치 기반 WeightedRandomSampler 생성"""
    weights = build_sample_weights(jsonl_path)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


if __name__ == "__main__":
    # 간단한 검증
    vocab = build_v6_vocab()
    DATA_DIR = "/data/tutti/Gemma4_Dataset/"

    val_bin = os.path.join(DATA_DIR, "val.bin")
    val_meta = os.path.join(DATA_DIR, "val_meta.json")

    if os.path.exists(val_bin):
        ds = MemmapMidiDataset(val_bin, val_meta, vocab)
        print(f"Val dataset size: {len(ds)}")
        sample = ds[0]
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  labels shape:    {sample['labels'].shape}")
        n_masked = (sample['labels'] == -100).sum().item()
        n_total = sample['labels'].shape[0]
        print(f"  masked tokens:   {n_masked}/{n_total} ({n_masked/n_total:.1%})")
    else:
        print(f"Val binary not found at {val_bin}. Run jsonl_to_memmap.py first.")
