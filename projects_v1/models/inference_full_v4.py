"""
inference.py
Qwen2.5-0.5B 기반 악기 파트 생성 추론 파이프라인 (확장판)
v2 : context bar와 window bars and future bars 적용
future_Bars 학습을 안 해봐서 생성을 더 못하는 걸 지도
이 코드가 기준으로 좋은 미디 찾아야 함.

사용법:
    python inference.py --song KissTheRain.mid
    python inference.py --song KissTheRain.mid --target flute
    python inference.py --song KissTheRain.mid --target trumpet
    python inference.py --song KissTheRain.mid --target piano --polyphonic
"""

import os
import argparse
import bisect
import random
import copy

import torch
import torch.nn as nn
import pretty_midi
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM

# ──────────────────────────────────────────────
# 기본 경로 설정
# ──────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "/home/globaltutti/INPUT"
DEFAULT_OUTPUT_DIR = "/home/globaltutti/OUTPUT/Pretrain"
DEFAULT_CKPT       = "/data2/tutti/pretrain_ckpt"

# ──────────────────────────────────────────────
# 악기 설정 테이블
# monophonic=True  → 단선율 강제 (violin/flute/oboe/trumpet/clarinet 등)
# monophonic=False → 폴리포닉 허용 (piano/guitar/harp 등)
# ──────────────────────────────────────────────
TARGET_CONFIG = {
    # ── Piano (0-7) ──
    "Acoustic Grand Piano":    {"program": 0,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Bright Acoustic Piano":   {"program": 1,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Electric Grand Piano":    {"program": 2,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Honky-tonk Piano":        {"program": 3,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Electric Piano 1":        {"program": 4,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Electric Piano 2":        {"program": 5,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Harpsichord":             {"program": 6,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "Clavi":                   {"program": 7,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},

    # ── Chromatic Perc (8-15) ──
    "Celesta":                 {"program": 8,   "pitch_min": 60,  "pitch_max": 108, "monophonic": False},
    "Glockenspiel":            {"program": 9,   "pitch_min": 72,  "pitch_max": 108, "monophonic": True},
    "Music Box":               {"program": 10,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Vibraphone":              {"program": 11,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": False},
    "Marimba":                 {"program": 12,  "pitch_min": 45,  "pitch_max": 96,  "monophonic": False},
    "Xylophone":               {"program": 13,  "pitch_min": 65,  "pitch_max": 108, "monophonic": False},
    "Tubular Bells":           {"program": 14,  "pitch_min": 60,  "pitch_max": 77,  "monophonic": True},
    "Dulcimer":                {"program": 15,  "pitch_min": 48,  "pitch_max": 96,  "monophonic": False},

    # ── Organ (16-23) ──
    "Drawbar Organ":           {"program": 16,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Percussive Organ":        {"program": 17,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Rock Organ":              {"program": 18,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Church Organ":            {"program": 19,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Reed Organ":              {"program": 20,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Accordion":               {"program": 21,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": True},
    "Harmonica":               {"program": 22,  "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "Tango Accordion":         {"program": 23,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": True},

    # ── Guitar (24-31) ──
    "Acoustic Guitar (nylon)": {"program": 24,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Acoustic Guitar (steel)": {"program": 25,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Electric Guitar (jazz)":  {"program": 26,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Electric Guitar (clean)": {"program": 27,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Electric Guitar (muted)": {"program": 28,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Overdriven Guitar":       {"program": 29,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Distortion Guitar":       {"program": 30,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Guitar harmonics":        {"program": 31,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": True},

    # ── Bass (32-39) ──
    "Acoustic Bass":           {"program": 32,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Electric Bass (finger)":  {"program": 33,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Electric Bass (pick)":    {"program": 34,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Fretless Bass":           {"program": 35,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Slap Bass 1":             {"program": 36,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Slap Bass 2":             {"program": 37,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Synth Bass 1":            {"program": 38,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Synth Bass 2":            {"program": 39,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},

    # ── Strings (40-47) ──
    "Violin":                  {"program": 40,  "pitch_min": 55,  "pitch_max": 100, "monophonic": True},
    "Viola":                   {"program": 41,  "pitch_min": 48,  "pitch_max": 91,  "monophonic": True},
    "Cello":                   {"program": 42,  "pitch_min": 36,  "pitch_max": 81,  "monophonic": True},
    "Contrabass":              {"program": 43,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "Tremolo Strings":         {"program": 44,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Pizzicato Strings":       {"program": 45,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "Orchestral Harp":         {"program": 46,  "pitch_min": 23,  "pitch_max": 103, "monophonic": False},
    "Timpani":                 {"program": 47,  "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},

    # ── Ensemble (48-55) ──
    "String Ensemble 1":       {"program": 48,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "String Ensemble 2":       {"program": 49,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "SynthStrings 1":          {"program": 50,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "SynthStrings 2":          {"program": 51,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "Choir Aahs":              {"program": 52,  "pitch_min": 48,  "pitch_max": 79,  "monophonic": False},
    "Voice Oohs":              {"program": 53,  "pitch_min": 48,  "pitch_max": 79,  "monophonic": True},
    "Synth Voice":             {"program": 54,  "pitch_min": 48,  "pitch_max": 84,  "monophonic": True},
    "Orchestra Hit":           {"program": 55,  "pitch_min": 48,  "pitch_max": 72,  "monophonic": False},

    # ── Brass (56-63) ──
    "Trumpet":                 {"program": 56,  "pitch_min": 52,  "pitch_max": 82,  "monophonic": True},
    "Trombone":                {"program": 57,  "pitch_min": 40,  "pitch_max": 72,  "monophonic": True},
    "Tuba":                    {"program": 58,  "pitch_min": 28,  "pitch_max": 58,  "monophonic": True},
    "Muted Trumpet":           {"program": 59,  "pitch_min": 52,  "pitch_max": 82,  "monophonic": True},
    "French Horn":             {"program": 60,  "pitch_min": 34,  "pitch_max": 77,  "monophonic": True},
    "Brass Section":           {"program": 61,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "SynthBrass 1":            {"program": 62,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "SynthBrass 2":            {"program": 63,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},

    # ── Reed (64-71) ──
    "Soprano Sax":             {"program": 64,  "pitch_min": 54,  "pitch_max": 84,  "monophonic": True},
    "Alto Sax":                {"program": 65,  "pitch_min": 49,  "pitch_max": 80,  "monophonic": True},
    "Tenor Sax":               {"program": 66,  "pitch_min": 44,  "pitch_max": 75,  "monophonic": True},
    "Baritone Sax":            {"program": 67,  "pitch_min": 36,  "pitch_max": 69,  "monophonic": True},
    "Oboe":                    {"program": 68,  "pitch_min": 58,  "pitch_max": 91,  "monophonic": True},
    "English Horn":            {"program": 69,  "pitch_min": 52,  "pitch_max": 81,  "monophonic": True},
    "Bassoon":                 {"program": 70,  "pitch_min": 34,  "pitch_max": 75,  "monophonic": True},
    "Clarinet":                {"program": 71,  "pitch_min": 50,  "pitch_max": 94,  "monophonic": True},

    # ── Pipe (72-79) ──
    "Piccolo":                 {"program": 72,  "pitch_min": 74,  "pitch_max": 108, "monophonic": True},
    "Flute":                   {"program": 73,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Recorder":                {"program": 74,  "pitch_min": 60,  "pitch_max": 91,  "monophonic": True},
    "Pan Flute":               {"program": 75,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Blown Bottle":            {"program": 76,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Shakuhachi":              {"program": 77,  "pitch_min": 55,  "pitch_max": 84,  "monophonic": True},
    "Whistle":                 {"program": 78,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Ocarina":                 {"program": 79,  "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},

    # ── Synth Lead (80-87) ──
    "Lead 1 (square)":         {"program": 80,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "Lead 2 (sawtooth)":       {"program": 81,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "Lead 3 (calliope)":       {"program": 82,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "Lead 4 (chiff)":          {"program": 83,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "Lead 5 (charang)":        {"program": 84,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "Lead 6 (voice)":          {"program": 85,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "Lead 7 (fifths)":         {"program": 86,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "Lead 8 (bass+lead)":      {"program": 87,  "pitch_min": 28,  "pitch_max": 72,  "monophonic": False},

    # ── Synth Pad (88-95) ──
    "Pad 1 (new age)":         {"program": 88,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 2 (warm)":            {"program": 89,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 3 (polysynth)":       {"program": 90,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 4 (choir)":           {"program": 91,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 5 (bowed)":           {"program": 92,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 6 (metallic)":        {"program": 93,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 7 (halo)":            {"program": 94,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "Pad 8 (sweep)":           {"program": 95,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},

    # ── Synth Effects (96-103) ──
    "FX 1 (rain)":             {"program": 96,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 2 (soundtrack)":       {"program": 97,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 3 (crystal)":          {"program": 98,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 4 (atmosphere)":       {"program": 99,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 5 (brightness)":       {"program": 100, "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 6 (goblins)":          {"program": 101, "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 7 (echoes)":           {"program": 102, "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "FX 8 (sci-fi)":           {"program": 103, "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},

    # ── Ethnic (104-111) ──
    "Sitar":                   {"program": 104, "pitch_min": 48,  "pitch_max": 77,  "monophonic": True},
    "Banjo":                   {"program": 105, "pitch_min": 48,  "pitch_max": 84,  "monophonic": False},
    "Shamisen":                {"program": 106, "pitch_min": 50,  "pitch_max": 79,  "monophonic": True},
    "Koto":                    {"program": 107, "pitch_min": 55,  "pitch_max": 84,  "monophonic": True},
    "Kalimba":                 {"program": 108, "pitch_min": 60,  "pitch_max": 91,  "monophonic": True},
    "Bagpipe":                 {"program": 109, "pitch_min": 55,  "pitch_max": 79,  "monophonic": True},
    "Fiddle":                  {"program": 110, "pitch_min": 55,  "pitch_max": 96,  "monophonic": True},
    "Shanai":                  {"program": 111, "pitch_min": 48,  "pitch_max": 79,  "monophonic": True},

    # ── Percussive (112-119) ──
    "Tinkle Bell":             {"program": 112, "pitch_min": 72,  "pitch_max": 108, "monophonic": True},
    "Agogo":                   {"program": 113, "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "Steel Drums":             {"program": 114, "pitch_min": 52,  "pitch_max": 84,  "monophonic": False},
    "Woodblock":               {"program": 115, "pitch_min": 60,  "pitch_max": 72,  "monophonic": True},
    "Taiko Drum":              {"program": 116, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "Melodic Tom":             {"program": 117, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "Synth Drum":              {"program": 118, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "Reverse Cymbal":          {"program": 119, "pitch_min": 36,  "pitch_max": 57,  "monophonic": False},

    # ── Sound Effects (120-127) ──
    "Guitar Fret Noise":       {"program": 120, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "Breath Noise":            {"program": 121, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "Seashore":                {"program": 122, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "Bird Tweet":              {"program": 123, "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "Telephone Ring":          {"program": 124, "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "Helicopter":              {"program": 125, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "Applause":                {"program": 126, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "Gunshot":                 {"program": 127, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
}

NOTE_TOKEN_LEN = 7
VEL_OFFSET     = 6


# ──────────────────────────────────────────────
# 설정 (CLI 인수)
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen 기반 악기 파트 생성 (확장판)")
    parser.add_argument("--song",         required=True,
                        help="입력 MIDI 파일명 (예: KissTheRain.mid)")
    parser.add_argument("--input_dir",    default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir",   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt",         default=DEFAULT_CKPT)
    parser.add_argument("--target",       default="violin",
                        choices=list(TARGET_CONFIG.keys()))
    parser.add_argument("--genre",        default="CLASSICAL",
                        choices=["CLASSICAL","JAZZ","POP","ROCK","ELECTRONIC","FOLK","UNKNOWN"])
    parser.add_argument("--pitch_min", type=int, default=None,
                        help="최저 pitch 오버라이드 (기본값: TARGET_CONFIG 값 사용)")
    parser.add_argument("--pitch_max", type=int, default=None,
                        help="최고 pitch 오버라이드 (기본값: TARGET_CONFIG 값 사용)")
    parser.add_argument("--context_bars", type=int, default=8)
    parser.add_argument("--window_bars",  type=int,   default=8)
    parser.add_argument("--future_bars", type=int,    default=0)
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--top_p",        type=float, default=0.95)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--polyphonic",   action="store_true",
                        help="단선율 강제 해제 (piano/guitar 등에 사용)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
def build_v5_vocab():
    vocab = {}

    def add(prefix, r):
        for i in r: vocab[f"{prefix}{i}"] = len(vocab)

    for t in ["PAD", "BOS", "EOS", "SEP", "PIECE_START", "PIECE_END",
              "BAR_START", "BAR_END", "PHRASE_END", "<PRE>", "<SUF>", "<MID>"]:
        vocab[t] = len(vocab)
    for g in ["CLASSICAL", "JAZZ", "POP", "ROCK", "ELECTRONIC", "FOLK", "UNKNOWN"]:
        vocab[f"GENRE_{g}"] = len(vocab)
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for r in roots:
        for m in [":maj", ":min"]: vocab[f"KEY_{r}{m}"] = len(vocab)
    vocab["KEY_NONE"] = len(vocab)
    for p in [40, 68, 73]: vocab[f"TARGET_{p}"] = len(vocab)
    for m in ["4:4", "3:4", "2:4", "6:8", "12:8", "OTHER"]:
        vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1, 6))
    add("INST=", range(129))
    for a in ["ART_NORMAL", "ART_LEGATO", "ART_VIBRATO", "ART_STACCATO"]:
        vocab[a] = len(vocab)
    add("EXPR_", range(32))
    add("TIME=", range(96))
    add("PITCH=", range(128))
    add("DUR=", range(1, 193))
    add("VEL=", range(32))
    for w in ["melodic", "epic", "calm", "fast", "slow", "sad", "happy",
              "piano", "strings", "orchestra", "cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B"
}

NOTE_TOKEN_LEN = 7
VEL_OFFSET = 6

# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(ckpt_path, vocab_size, vocab):
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.vocab_size              = vocab_size
    config.pad_token_id            = vocab["PAD"]
    config.max_position_embeddings = 8192
    config.sliding_window          = None

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)
    model.model.embed_tokens = nn.Embedding(vocab_size, config.hidden_size).to(torch.bfloat16)
    model.lm_head            = nn.Linear(config.hidden_size, vocab_size, bias=False).to(torch.bfloat16)

    sf   = os.path.join(ckpt_path, "model.safetensors")
    bin_ = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        state = load_file(sf, device="cpu")
        model.load_state_dict(state, strict=True)
        print(f"✅ 체크포인트 로드 (safetensors): {sf}")
    elif os.path.exists(bin_):
        state = torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"✅ 체크포인트 로드 (.bin): {bin_}")
    else:
        raise FileNotFoundError(f"❌ 체크포인트 없음: {ckpt_path}")

    # RTX 4090 / T4 TF32 가속 (웹 서버 환경과 동일하게 맞춤)
    torch.set_float32_matmul_precision("high")

    model.config.use_cache = False
    model.eval()
    return model


# ──────────────────────────────────────────────
# MIDI → 마디 토큰
# ──────────────────────────────────────────────
def midi_to_bar_tokens(midi_path, genre, VOCAB):
    pm          = pretty_midi.PrettyMIDI(midi_path)
    res         = pm.resolution
    timeline    = defaultdict(lambda: defaultdict(list))
    key_changes = sorted(pm.key_signature_changes,  key=lambda x: x.time)
    key_times   = [k.time for k in key_changes]
    ts_changes  = sorted(pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = pm.get_tempo_changes()

    for inst in pm.instruments:
        p     = 128 if inst.is_drum else inst.program
        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        last_end_tick = -1

        for i, n in enumerate(notes):
            note_dur   = n.end - n.start
            tempo_idx  = max(0, bisect.bisect_right(tempo_times, n.start) - 1)
            bpm        = tempos[tempo_idx] if len(tempos) > 0 else 120.0
            s_per_beat = 60.0 / bpm
            dur_tick   = max(1, min(192, round((note_dur / s_per_beat) * 24)))

            ts_idx = max(0, bisect.bisect_right(ts_times, n.start) - 1)
            if ts_idx < len(ts_changes):
                ts            = ts_changes[ts_idx]
                mkey          = f"{ts.numerator}:{ts.denominator}"
                meter_tok     = VOCAB.get(f"METER_{mkey}", VOCAB["METER_OTHER"])
                beats_per_bar = ts.numerator
            else:
                meter_tok     = VOCAB["METER_OTHER"]
                beats_per_bar = 4

            k_idx = bisect.bisect_right(key_times, n.start) - 1
            if 0 <= k_idx < len(key_changes):
                ks    = key_changes[k_idx]
                root  = ks.key_number % 12
                mode  = "maj" if ks.key_number < 12 else "min"
                roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                rname = FLAT_TO_SHARP.get(roots[root], roots[root])
                key_tok = VOCAB.get(f"KEY_{rname}:{mode}", VOCAB["KEY_NONE"])
            else:
                key_tok = VOCAB["KEY_NONE"]

            bar_idx        = int(pm.time_to_tick(n.start) // (res * beats_per_bar))
            bar_start_tick = bar_idx * res * beats_per_bar
            rel_tick       = pm.time_to_tick(n.start) - bar_start_tick
            time_tok       = VOCAB[f"TIME={min(95, rel_tick * 96 // (res * beats_per_bar))}"]

            n_start_tick  = pm.time_to_tick(n.start)
            is_phrase_end = (last_end_tick > 0 and
                             (n_start_tick - last_end_tick) >= res)

            nxt_start    = notes[i+1].start if i+1 < len(notes) else n.end + 1
            legato_ratio = note_dur / max(nxt_start - n.start, 1e-6)
            if dur_tick <= 2:          art_tok = VOCAB["ART_STACCATO"]
            elif nxt_start == n.start: art_tok = VOCAB["ART_NORMAL"]
            elif legato_ratio > 0.95:  art_tok = VOCAB["ART_LEGATO"]
            else:                      art_tok = VOCAB["ART_NORMAL"]

            expr_tok = VOCAB[f"EXPR_{min(31, n.velocity * 32 // 128)}"]
            vel_tok  = VOCAB[f"VEL={min(31, n.velocity * 32 // 128)}"]
            inst_tok = VOCAB[f"INST={p}"]

            timeline[bar_idx][p].append(
                (time_tok, inst_tok, art_tok, expr_tok,
                 n.pitch, dur_tick, vel_tok,
                 meter_tok, key_tok, is_phrase_end))
            last_end_tick = pm.time_to_tick(n.end)

    header      = [VOCAB["PIECE_START"], VOCAB[f"GENRE_{genre}"]]
    final_beats = ts_changes[-1].numerator if ts_changes else 4
    max_bar     = int(pm.time_to_tick(pm.get_end_time()) // (res * final_beats))

    bar_tokens        = {}
    accumulated_ticks = 0
    for bar_idx in range(max_bar + 1):
        bar_time  = pm.tick_to_time(accumulated_ticks)
        ts_idx    = max(0, bisect.bisect_right(ts_times, bar_time) - 1)
        beats     = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
        mkey      = (f"{ts_changes[ts_idx].numerator}:{ts_changes[ts_idx].denominator}"
                     if ts_idx < len(ts_changes) else "OTHER")
        meter_tok = VOCAB.get(f"METER_{mkey}", VOCAB["METER_OTHER"])

        k_idx = bisect.bisect_right(key_times, bar_time) - 1
        if 0 <= k_idx < len(key_changes):
            ks    = key_changes[k_idx]
            root  = ks.key_number % 12
            mode  = "maj" if ks.key_number < 12 else "min"
            roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            rname = FLAT_TO_SHARP.get(roots[root], roots[root])
            key_tok = VOCAB.get(f"KEY_{rname}:{mode}", VOCAB["KEY_NONE"])
        else:
            key_tok = VOCAB["KEY_NONE"]

        if bar_idx in timeline:
            bar_data    = timeline[bar_idx]
            total_notes = sum(len(v) for v in bar_data.values())
            density     = min(5, max(1, total_notes // 4))
            btoks = [VOCAB["BAR_START"], key_tok, meter_tok, VOCAB[f"DENSITY_{density}"]]
            all_notes = []
            for p in bar_data.keys():
                for (tt, it, at, et, pitch, dt, vt, _, _, phrase_end) in bar_data[p]:
                    all_notes.append((tt - VOCAB["TIME=0"], it, phrase_end,
                                      it, at, et, tt, pitch, dt, vt))
            all_notes.sort(key=lambda x: (x[0], x[1]))
            for (_, _, phrase_end, it, at, et, tt, pitch, dt, vt) in all_notes:
                if phrase_end: btoks.append(VOCAB["PHRASE_END"])
                btoks += [it, at, et, tt, VOCAB[f"PITCH={pitch}"],
                          VOCAB[f"DUR={dt}"], vt]
            btoks.append(VOCAB["BAR_END"])
        else:
            btoks = [VOCAB["BAR_START"], key_tok, meter_tok,
                     VOCAB["DENSITY_1"], VOCAB["BAR_END"]]

        bar_tokens[bar_idx]    = btoks
        accumulated_ticks     += res * beats

    return header, bar_tokens, max_bar, pm


# ──────────────────────────────────────────────
# 컨텍스트 트리밍 (마디 단위, 비율 보존)
# ──────────────────────────────────────────────
def trim_bars_preserving_ratio(past_bars, future_bars,
                               header_len, current_len, max_ctx,
                               target_past_ratio):
    """마디 단위로 제거하며 목표 past:future 비율을 최대한 유지.

    Args:
        past_bars:   [(bar_idx, [tokens]), ...] 오래된→최근 순
        future_bars: [(bar_idx, [tokens]), ...] 가까운→먼 순
        header_len:  헤더 토큰 수 (고정)
        current_len: 현재 윈도우 토큰 수 (고정)
        max_ctx:     최대 컨텍스트 토큰 수
        target_past_ratio: 목표 비율 = context_bars/(context_bars+future_bars)

    Returns:
        (trimmed_past_bars, trimmed_future_bars)
    """
    p_list = list(past_bars)
    f_list = list(future_bars)

    def total():
        return (header_len + current_len
                + sum(len(t) for _, t in p_list)
                + sum(len(t) for _, t in f_list))

    while total() > max_ctx and (p_list or f_list):
        p_count = len(p_list)
        f_count = len(f_list)

        # 한쪽만 남은 경우
        if not p_list:
            removed = f_list.pop()  # 가장 먼 미래 제거
            print(f"    ✂️  Future Bar {removed[0]} 제거 ({len(removed[1])} tok)")
            continue
        if not f_list:
            removed = p_list.pop(0)  # 가장 오래된 과거 제거
            print(f"    ✂️  Past Bar {removed[0]} 제거 ({len(removed[1])} tok)")
            continue

        # 양쪽 다 있을 때: 목표 비율에 더 가까워지는 쪽을 제거
        ratio_if_remove_past = (p_count - 1) / (p_count - 1 + f_count) if (p_count - 1 + f_count) > 0 else 0
        ratio_if_remove_future = p_count / (p_count + f_count - 1) if (p_count + f_count - 1) > 0 else 1

        diff_past   = abs(ratio_if_remove_past   - target_past_ratio)
        diff_future = abs(ratio_if_remove_future - target_past_ratio)

        if diff_future <= diff_past:
            # Future 먼 미래 제거가 비율에 더 유리하거나 동일
            removed = f_list.pop()
            print(f"    ✂️  Future Bar {removed[0]} 제거 ({len(removed[1])} tok)")
        else:
            # Past 오래된 과거 제거가 비율에 더 유리
            removed = p_list.pop(0)
            print(f"    ✂️  Past Bar {removed[0]} 제거 ({len(removed[1])} tok)")

    return p_list, f_list


# ──────────────────────────────────────────────
# 마디 토큰 파싱 / 인터리빙 병합
# ──────────────────────────────────────────────
def parse_bar_notes(bar_toks, VOCAB_R):
    """마디 토큰을 header와 노트 리스트로 분리.
    Returns: (header_tokens, [(time_val, inst_val, note_token_list), ...])
    """
    header = []
    body   = []
    in_header = True
    for tok in bar_toks:
        name = VOCAB_R.get(tok, "")
        if in_header:
            if name.startswith("INST=") or name == "PHRASE_END":
                in_header = False
                body.append(tok)
            elif name == "BAR_END":
                break
            else:
                header.append(tok)
        else:
            if name != "BAR_END":
                body.append(tok)

    notes = []
    i = 0
    while i < len(body):
        name = VOCAB_R.get(body[i], "")
        prefix = []
        if name == "PHRASE_END":
            prefix = [body[i]]
            i += 1
            if i >= len(body):
                break
            name = VOCAB_R.get(body[i], "")
        if name.startswith("INST=") and i + 7 <= len(body):
            note_toks = prefix + body[i:i+7]
            time_name = VOCAB_R.get(body[i+3], "")
            time_val  = int(time_name.split("=")[1]) if time_name.startswith("TIME=") else 0
            inst_val  = int(name.split("=")[1])
            notes.append((time_val, inst_val, note_toks))
            i += 7
        else:
            i += 1
    return header, notes


def merge_bars(source_bar_toks, gen_bar_toks, VOCAB, VOCAB_R):
    """원곡 마디와 생성 마디를 학습 포맷대로 인터리빙 병합."""
    if not gen_bar_toks:
        return list(source_bar_toks)
    if not source_bar_toks:
        return list(gen_bar_toks)

    src_header, src_notes = parse_bar_notes(source_bar_toks, VOCAB_R)
    _,          gen_notes = parse_bar_notes(gen_bar_toks,    VOCAB_R)

    all_notes = src_notes + gen_notes
    all_notes.sort(key=lambda x: (x[0], x[1]))

    # density 업데이트
    density = min(5, max(1, len(all_notes) // 4))
    updated_header = []
    for tok in src_header:
        name = VOCAB_R.get(tok, "")
        if name.startswith("DENSITY_"):
            updated_header.append(VOCAB[f"DENSITY_{density}"])
        else:
            updated_header.append(tok)

    merged = updated_header
    for _, _, toks in all_notes:
        merged.extend(toks)
    merged.append(VOCAB["BAR_END"])
    return merged


# ──────────────────────────────────────────────
# 슬라이딩 윈도우 생성
# monophonic=True  → INST 재발음을 VEL 완결 전까지 억제 (단선율)
# monophonic=False → 억제 없음 (폴리포닉)
# ──────────────────────────────────────────────
def generate_sliding_window(model, header, bar_tokens, max_bar,
                            target_prog, pitch_min, pitch_max,
                            window_bars, context_bars, future_bars,
                            temperature, top_p,
                            VOCAB, VOCAB_R, source_pm, device,
                            monophonic=True):
    SEQ_LEN = 8192
    MAX_CTX = SEQ_LEN - 300
    all_notes = []
    gen_bar_tokens = {}
    
    # ─── 이 부분이 반드시 함수 내부에 있어야 합니다 ───
    INST_TARGET_ID = VOCAB[f"INST={target_prog}"]
    VEL_IDS = {VOCAB[f"VEL={i}"] for i in range(32)}
    # ─────────────────────────────────────────────

    # 목표 Past 비율 (함수 파라미터 int 값으로 미리 계산)
    target_past_ratio = context_bars / (context_bars + future_bars) if (context_bars + future_bars) > 0 else 0.5

    total_windows = (max_bar // window_bars) + 1

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end = min(win_start + window_bars - 1, max_bar)
        if win_start > max_bar: break

        ctx_start = max(0, win_start - context_bars)
        fut_end = min(max_bar, win_end + future_bars)

        # ─── 각 영역별 마디 단위 토큰 리스트 생성 ───

        # 1. Header (고정)
        header_toks = list(header)

        # 2. Past (과거 원본 + 과거 생성 결과) — 마디별 리스트
        past_bar_list = []
        for b in range(ctx_start, win_start):
            bar_toks = list(bar_tokens.get(b, []))
            if b in gen_bar_tokens:
                bar_toks = merge_bars(bar_toks, gen_bar_tokens[b], VOCAB, VOCAB_R)
            if bar_toks:
                past_bar_list.append((b, bar_toks))

        # 3. Current Window (현재 작곡해야 할 구간의 원본/반주)
        current_toks = []
        for b in range(win_start, win_end + 1):
            current_toks += bar_tokens.get(b, [])

        # 4. Future (미래 가이드라인 원본) — 마디별 리스트
        future_bar_list = []
        for b in range(win_end + 1, fut_end + 1):
            bar_toks = list(bar_tokens.get(b, []))
            if bar_toks:
                future_bar_list.append((b, bar_toks))

        # ─── [실시간 토큰 통계 출력] ───
        h_len = len(header_toks)
        p_len = sum(len(t) for _, t in past_bar_list)
        c_len = len(current_toks)
        f_len = sum(len(t) for _, t in future_bar_list)
        total_len = h_len + p_len + c_len + f_len

        print(f"\n" + "─" * 50)
        print(f"WINDOW #{win_idx} (Bar {win_start}~{win_end}) Token Statistics:")
        print(f"  ● Header  : {h_len:>4} tokens")
        print(f"  ● Past    : {p_len:>4} tokens ({len(past_bar_list)} bars, {ctx_start}~{win_start - 1})")
        print(f"  ● Current : {c_len:>4} tokens (Bars {win_start}~{win_end})")
        print(f"  ● Future  : {f_len:>4} tokens ({len(future_bar_list)} bars, {win_end + 1}~{fut_end})")
        print(f"  --------------------------")
        print(f"  ● TOTAL   : {total_len:>4} / {SEQ_LEN} (Used: {total_len / SEQ_LEN * 100:.1f}%)")

        # ─── 오버플로우 발생 시 마디 단위 비율 보존 트리밍 ───
        if total_len > MAX_CTX:
            overflow = total_len - MAX_CTX
            print(f"  ⚠️  OVERFLOW! {overflow} tokens over limit.")
            print(f"  📐 Target Past ratio: {target_past_ratio:.3f} ({context_bars}:{future_bars})")

            past_bar_list, future_bar_list = trim_bars_preserving_ratio(
                past_bar_list, future_bar_list,
                h_len, c_len, MAX_CTX, target_past_ratio)

            new_p = sum(len(t) for _, t in past_bar_list)
            new_f = sum(len(t) for _, t in future_bar_list)
            print(f"  → Past  : {p_len:>4} → {new_p:>4} (-{p_len - new_p}), {len(past_bar_list)} bars")
            print(f"  → Future: {f_len:>4} → {new_f:>4} (-{f_len - new_f}), {len(future_bar_list)} bars")

        # 마디별 리스트 → flat 토큰 리스트로 병합
        past_toks = []
        for _, toks in past_bar_list:
            past_toks += toks
        future_toks = []
        for _, toks in future_bar_list:
            future_toks += toks

        context = header_toks + past_toks + current_toks + future_toks
        print(f"  → Final Context: {len(context)} tokens")
        print("─" * 50)

        # ─── 모델 추론 시작 (input_ids 생성) ───
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        gen_toks = []

        with torch.no_grad():
            # KV 캐시 활용을 위한 초기 인코딩
            out = model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values

            # 마디의 시작과 타겟 악기 토큰 강제 주입
            for tok in [VOCAB["BAR_START"], INST_TARGET_ID]:
                t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
                out = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                gen_toks.append(tok)

            bar_count = 1
            target_playing = True
            cur_in = torch.tensor([[gen_toks[-1]]], dtype=torch.long, device=device)

            for step in range(1024):
                out = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                logits = out.logits[0, -1, :].float()

                # 피치 마스킹
                for pitch in range(128):
                    if pitch < pitch_min or pitch > pitch_max:
                        logits[VOCAB[f"PITCH={pitch}"]] = -1e9

                if monophonic and target_playing:
                    logits[INST_TARGET_ID] = -1e9

                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                # Top-p Sampling
                s_probs, s_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(s_probs, dim=0)
                mask = cumsum - s_probs > top_p
                s_probs[mask] = 0
                s_probs /= s_probs.sum()

                next_tok = s_idx[torch.multinomial(s_probs, 1)].item()
                gen_toks.append(next_tok)

                # 상태 업데이트 및 종료 조건
                if next_tok == INST_TARGET_ID:
                    target_playing = True
                elif monophonic and target_playing and next_tok in VEL_IDS:
                    target_playing = False

                if next_tok == VOCAB["BAR_START"]:
                    bar_count += 1
                    target_playing = False
                    if bar_count > window_bars: break

                if next_tok in (VOCAB["PIECE_END"], VOCAB["EOS"]): break
                cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)

        # ─── 결과 수집 및 마디 분리 ───
        # (생성된 gen_toks를 gen_bar_tokens 딕셔너리에 저장하는 로직 - 기존과 동일)
        cur_bar_toks = []
        cur_bar_num = win_start
        for tok in gen_toks:
            if VOCAB_R.get(tok, "") == "BAR_START":
                if cur_bar_toks:
                    gen_bar_tokens[cur_bar_num] = cur_bar_toks
                    cur_bar_num += 1
                cur_bar_toks = [tok]
            else:
                cur_bar_toks.append(tok)
        if cur_bar_toks:
            gen_bar_tokens[cur_bar_num] = cur_bar_toks

        win_notes = decode_tokens(gen_toks, source_pm, target_prog,
                                  bar_offset=win_start, win_start=win_start, win_end=win_end,
                                  VOCAB=VOCAB, VOCAB_R=VOCAB_R,
                                  pitch_min=pitch_min, pitch_max=pitch_max)
        all_notes.extend(win_notes)
        print(f"   -> Result: Generated {len(win_notes)} notes for this window.")

    return all_notes

# ──────────────────────────────────────────────
# 토큰 디코딩
# ──────────────────────────────────────────────
def decode_tokens(tokens, source_pm, target_prog,
                  bar_offset=0, win_start=0, win_end=9999,
                  VOCAB=None, VOCAB_R=None,
                  pitch_min=0, pitch_max=127):   # ← 추가
    res         = source_pm.resolution
    ts_changes  = sorted(source_pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = source_pm.get_tempo_changes()

    bar_tick_map = {}
    acc = 0
    for b in range(2000):
        bar_tick_map[b] = acc
        bt  = source_pm.tick_to_time(acc)
        idx = max(0, bisect.bisect_right(ts_times, bt) - 1)
        bpb = ts_changes[idx].numerator if idx < len(ts_changes) else 4
        acc += res * bpb

    notes_out = []
    bar_idx   = bar_offset - 1
    cur_inst  = cur_time_tok = cur_pitch = cur_dur = cur_vel = None

    for tok in tokens:
        name = VOCAB_R.get(tok, "?")
        if   name == "BAR_START":       bar_idx += 1
        elif name.startswith("INST="):  cur_inst     = int(name.split("=")[1])
        elif name.startswith("TIME="):  cur_time_tok = int(name.split("=")[1])
        elif name.startswith("PITCH="): cur_pitch    = int(name.split("=")[1])
        elif name.startswith("DUR="):   cur_dur      = int(name.split("=")[1])
        elif name.startswith("VEL="):
            cur_vel = int(name.split("=")[1])
            if (cur_inst == target_prog and
                    cur_pitch    is not None and
                    cur_dur      is not None and
                    cur_time_tok is not None and
                    win_start <= bar_idx <= win_end):

                # ── 디버그 추가 ──
                print(f"  [BAR {bar_idx}] INST={cur_inst} PITCH={cur_pitch} "
                      f"DUR={cur_dur} VEL={cur_vel} "
                      f"{'⚠️ OUT OF RANGE' if cur_pitch < pitch_min or cur_pitch > pitch_max else '✅'}")

                b_tick    = bar_tick_map.get(bar_idx, 0)
                b_time    = source_pm.tick_to_time(b_tick)
                ts_idx    = max(0, bisect.bisect_right(ts_times, b_time) - 1)
                bpb       = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
                bar_ticks = res * bpb
                abs_tick  = b_tick + cur_time_tok * bar_ticks // 96
                start_sec = source_pm.tick_to_time(abs_tick)

                t_idx   = max(0, bisect.bisect_right(tempo_times, start_sec) - 1)
                bpm     = tempos[t_idx] if len(tempos) > 0 else 120.0
                dur_sec = (cur_dur / 24.0) * (60.0 / bpm)

                notes_out.append({
                    "start":    start_sec,
                    "end":      start_sec + dur_sec,
                    "pitch":    cur_pitch,
                    "velocity": max(1, min(127, (cur_vel + 1) * 4)),
                })
            cur_pitch = cur_dur = cur_vel = None

    return notes_out


# ──────────────────────────────────────────────
# 후처리
# monophonic=True  → 겹침 제거 + 도약 완화 적용
# monophonic=False → 겹침 제거만 생략 (폴리포닉 허용)
# ──────────────────────────────────────────────
def postprocess(notes, target_name, monophonic=True):
    cfg = TARGET_CONFIG[target_name]

    # 1. 음역 클리핑
    notes = [n for n in notes
             if cfg["pitch_min"] <= n["pitch"] <= cfg["pitch_max"]]

    # 2. 비정상적으로 긴 음표 클리핑
    MAX_DUR = 4.0
    for n in notes:
        if n["end"] - n["start"] > MAX_DUR:
            n["end"] = n["start"] + MAX_DUR

    # 3. 너무 짧은 음표 제거
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    # 4. 단선율 악기만: 폴리포닉 제거
    if monophonic:
        notes = sorted(notes, key=lambda x: x["start"])
        mono  = []
        for n in notes:
            if mono and n["start"] < mono[-1]["end"]:
                mono[-1]["end"] = n["start"]
                if mono[-1]["end"] - mono[-1]["start"] < 0.05:
                    mono.pop()
            mono.append(n)
        notes = mono

    # 5. 슬라이딩 윈도우 경계 연결 보정
    LEGATO_GAP = 0.03
    notes = sorted(notes, key=lambda x: x["start"])
    for i in range(len(notes) - 1):
        gap = notes[i+1]["start"] - notes[i]["end"]
        if 0 < gap < LEGATO_GAP:
            notes[i]["end"] = notes[i+1]["start"]

    # 6. 단선율 악기만: 큰 도약 완화
    if monophonic:
        MAX_INTERVAL = 12
        notes = sorted(notes, key=lambda x: x["start"])
        for i in range(1, len(notes)):
            interval = notes[i]["pitch"] - notes[i-1]["pitch"]
            if abs(interval) > MAX_INTERVAL:
                if interval > 0: notes[i]["pitch"] -= 12
                else:            notes[i]["pitch"] += 12
                if not (cfg["pitch_min"] <= notes[i]["pitch"] <= cfg["pitch_max"]):
                    if interval > 0: notes[i]["pitch"] += 12
                    else:            notes[i]["pitch"] -= 12

    # 7. 최종 음역 재확인
    notes = [n for n in notes
             if cfg["pitch_min"] <= n["pitch"] <= cfg["pitch_max"]]

    # 8. 최종 길이 재확인
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    return notes


# ──────────────────────────────────────────────
# MIDI 저장
# ──────────────────────────────────────────────
def save_midi(notes, source_pm, output_path, target_prog, target_name):
    out_pm   = copy.deepcopy(source_pm)
    new_inst = pretty_midi.Instrument(
        program=target_prog, is_drum=False, name=target_name)
    for n in notes:
        new_inst.notes.append(pretty_midi.Note(
            velocity=n["velocity"], pitch=n["pitch"],
            start=n["start"],       end=n["end"]))
    out_pm.instruments.append(new_inst)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_pm.write(output_path)
    print(f"✅ 저장 완료: {output_path}  ({len(notes)} 노트)")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    VOCAB      = build_v5_vocab()
    VOCAB_R    = {v: k for k, v in VOCAB.items()}
    VOCAB_SIZE = len(VOCAB)
    print(f"Vocab 크기: {VOCAB_SIZE}")

    cfg         = TARGET_CONFIG[args.target]
    target_prog = cfg["program"]
    pitch_min = args.pitch_min if args.pitch_min is not None else cfg["pitch_min"]
    pitch_max = args.pitch_max if args.pitch_max is not None else cfg["pitch_max"]
    # CLI --polyphonic 플래그가 있으면 강제 override, 없으면 테이블 기본값 사용
    monophonic  = cfg["monophonic"] and not args.polyphonic

    input_path  = os.path.join(args.input_dir, args.song)
    song_stem   = os.path.splitext(args.song)[0]
    output_path = os.path.join(args.output_dir,
                               f"{song_stem}_{args.target}_final.mid")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 입력 파일 없음: {input_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📄 입력 MIDI 토큰화: {input_path}")
    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(
        input_path, args.genre, VOCAB)
    print(f"   총 마디 수: {max_bar + 1}")

    print(f"🤖 모델 로드: {args.ckpt}")
    model = load_model(args.ckpt, VOCAB_SIZE, VOCAB)
    model.to(DEVICE)

    print(f"\n🎵 슬라이딩 윈도우 생성 시작 "
          f"(target={args.target} [prog={target_prog}], "
          f"window={args.window_bars}마디, "
          f"context={args.context_bars}마디, "
          f"{'단선율' if monophonic else '폴리포닉'})")

    all_notes = generate_sliding_window(
        model, header, bar_tokens, max_bar,
        target_prog, pitch_min, pitch_max,
        args.window_bars, args.context_bars, args.future_bars,
        args.temperature, args.top_p,
        VOCAB, VOCAB_R, source_pm, DEVICE,
        monophonic=monophonic,
    )

    print(f"\n   전체 디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, args.target, monophonic=monophonic)
    print(f"   후처리 후 노트:   {len(all_notes)}")

    if not all_notes:
        print("❌ 노트 없음 — --temperature를 높이거나 --window_bars를 늘려보세요")
    else:
        save_midi(all_notes, source_pm, output_path, target_prog, args.target)


if __name__ == "__main__":
    main()