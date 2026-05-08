import os
import json
import numpy as np
from vocab_v6 import build_v6_vocab

def jsonl_to_memmap(jsonl_path, bin_path, meta_path, vocab, max_length=8192):
    """JSONL → uint16 memmap 변환."""
    print(f"Processing {jsonl_path}...")
    
    # 1차 패스: 샘플 수 카운트
    n_samples = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                n_samples += 1
                
    if n_samples == 0:
        print(f"No samples found in {jsonl_path}.")
        return
        
    print(f"Total samples: {n_samples}. Creating memmap...")
    
    # memmap 생성
    fp = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=(n_samples, max_length))
    
    pad_id = vocab["PAD"]
    
    # 2차 패스: 데이터 채우기
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            tokens = data["tokens"]
            
            if len(tokens) < max_length:
                tokens += [pad_id] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
                
            fp[i] = np.array(tokens, dtype=np.uint16)
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{n_samples} samples.")
                
    fp.flush()
    
    # 메타데이터 저장
    with open(meta_path, 'w') as f:
        json.dump({
            "num_samples": n_samples,
            "max_length": max_length,
            "dtype": "uint16"
        }, f)
        
    print(f"Saved binary to {bin_path} and meta to {meta_path}.")

if __name__ == "__main__":
    vocab = build_v6_vocab()
    
    DATA_DIR = "/data/tutti/Gemma4_Dataset/" # as per spec
    
    # Train set
    train_jsonl = os.path.join(DATA_DIR, "train.jsonl")
    train_bin = os.path.join(DATA_DIR, "train.bin")
    train_meta = os.path.join(DATA_DIR, "train_meta.json")
    
    if os.path.exists(train_jsonl):
        jsonl_to_memmap(train_jsonl, train_bin, train_meta, vocab)
        
    # Val set
    val_jsonl = os.path.join(DATA_DIR, "val.jsonl")
    val_bin = os.path.join(DATA_DIR, "val.bin")
    val_meta = os.path.join(DATA_DIR, "val_meta.json")
    
    if os.path.exists(val_jsonl):
        jsonl_to_memmap(val_jsonl, val_bin, val_meta, vocab)
