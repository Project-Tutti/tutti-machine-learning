import sys
import os
import argparse
import json
import torch
import shutil
import random
import bisect
import copy
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, LogitsProcessorList

# Qwen2.5_0.5B 모듈 경로 추가
QWEN_DIR = "/home/globaltutti/Qwen2.5_0.5B"
sys.path.append(QWEN_DIR)

# 웹 백엔드 모듈 경로 추가 (매핑 및 후처리용)
BACKEND_DIR = "/home/globaltutti/tutti-backend_ai/tutti-backend-ai"
sys.path.append(BACKEND_DIR)

from vocab_v6 import build_v6_vocab
from preprocess_lmd import generate_bars
from app.schemas.request import Mapping
from app.services.midi_processor import remap_original_tracks
import pretty_midi

# inference_v6.py (학습 구조 완전 일치 버전) 기준으로 로직 가져오기
from inference_v6 import (
    TARGET_CONFIG, 
    V6GrammarProcessor, 
    parse_midi_like_training, 
    generate_sliding_window, 
    postprocess, 
    save_midi
)

def parse_args():
    parser = argparse.ArgumentParser(description="Tutti V6 (Qwen2.5) Web Backend Inference Wrapper - V6 Structure")
    parser.add_argument("--song", required=True, help="입력 MIDI 파일 경로")
    parser.add_argument("--output", required=True, help="출력 MIDI 파일 경로 (생성 완료된 파일)")
    parser.add_argument("--ckpt", default="/data2/tutti/Qwen_Checkpoints/checkpoint-45000", help="모델 체크포인트 경로")
    parser.add_argument("--target", default="Cello", help="타겟 악기 이름 (예: Violin, Cello 등)")
    parser.add_argument("--genre", default="pop", help="장르 (pop, electronic, classical, rock, other)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--pitch_min", type=int, default=None)
    parser.add_argument("--pitch_max", type=int, default=None)
    parser.add_argument("--mappings", default="[]", help="JSON 형태의 매핑")
    parser.add_argument("--context_bars", type=int, default=8)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--future_bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 시드 고정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("-" * 50)
    print(f"▶ Tutti V6 추론 시작 (V6 일치 모드, Checkpoint: {args.ckpt})")
    print("-" * 50)
    
    # 1. 악기 설정 확인
    target_key = next((k for k in TARGET_CONFIG.keys() if k.lower() == args.target.lower()), None)
    if not target_key:
        print(f"❌ 오류: 지원하지 않는 타겟 악기입니다. 지원 목록: {list(TARGET_CONFIG.keys())}")
        return
        
    cfg = TARGET_CONFIG[target_key]
    target_prog = cfg["program"]
    pitch_min = args.pitch_min if args.pitch_min is not None else cfg["pitch_min"]
    pitch_max = args.pitch_max if args.pitch_max is not None else cfg["pitch_max"]
    monophonic = cfg["monophonic"]
    
    # 2. 모델 및 Vocab 로드
    print("[*] Vocab 및 모델 로딩 중...")
    VOCAB = build_v6_vocab()
    VOCAB_R = {v: k for k, v in VOCAB.items()}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.eval()
    
    # 3. 매핑 데이터 파싱 및 파일 재매핑
    mappings_data = json.loads(args.mappings)
    mappings = [Mapping(**m) for m in mappings_data]
    
    inference_input_path = args.song
    if mappings:
        print(f"▶ 원본 트랙 재매핑 진행 중...")
        mapped_midi_path = Path(args.song).with_name(Path(args.song).stem + "_mapped.mid")
        shutil.copy2(args.song, mapped_midi_path)
        remap_original_tracks(str(mapped_midi_path), mappings)
        inference_input_path = str(mapped_midi_path)
    
    # 4. MIDI 파싱 (inference_v6 스타일: timeline key = inst_idx)
    print(f"📄 MIDI 파싱: {inference_input_path}")
    (timeline, pm, key_changes, key_times,
     ts_changes, ts_times, inst_idx_map) = parse_midi_like_training(inference_input_path)
    
    # 타겟 악기의 inst_idx 찾기 (generate_bars의 exclude 용)
    target_inst_idx = inst_idx_map.get(target_prog, -999)
    print(f"  target_prog={target_prog}, target_inst_idx={target_inst_idx}")
    
    genre_tok = VOCAB.get(f"GENRE_{args.genre.upper()}", VOCAB["GENRE_OTHER"])
    
    # 5. 슬라이딩 윈도우 생성 실행
    print(f"\n🎵 생성 시작 (Target: {target_key}, Genre: {args.genre})")
    all_notes = generate_sliding_window(
        model, pm, timeline, target_prog, target_inst_idx,
        pitch_min, pitch_max, monophonic,
        args.window_bars, args.context_bars, args.future_bars,
        args.temperature, args.top_p,
        VOCAB, VOCAB_R, device,
        key_changes, key_times, ts_changes, ts_times,
        genre_tok
    )
    
    print(f"\n   전체 디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, target_key, monophonic=monophonic)
    print(f"   후처리 후 노트:   {len(all_notes)}")
    
    # 6. 저장
    if not all_notes:
        print("❌ 결과가 없습니다. 설정을 조정해 보세요.")
    else:
        save_midi(all_notes, pm, args.output, target_prog, target_key)
        print("-" * 50)
        print(f"✅ 완료! 저장 경로: {args.output}")

if __name__ == "__main__":
    main()
