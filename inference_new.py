"""
inference.py
Qwen2.5-0.5B 기반 악기 파트 생성 추론 파이프라인

새 매핑 기준 13개 악기 그룹 지원.
입력 MIDI 전체를 컨텍스트로 받아 새 악기 파트를 추가 생성.

사용법:
    python inference_new.py --song KissTheRain.mid --target violin
    python inference_new.py --song KissTheRain.mid --target woodwind
    아직은 violin만 가능(woodwind 토큰 자체가 없음)
    python inference_new.py --song KissTheRain.mid --target violin --pitch_min 60 --pitch_max 96
"""

import os
import argparse
import bisect
import random
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import pretty_midi
from transformers import AutoConfig, AutoModelForCausalLM

# ──────────────────────────────────────────────
# 1. 악기 그룹 정의 (13개 리매핑 기준 완벽 유지)
# ──────────────────────────────────────────────
INSTRUMENT_GROUPS = {
    "drum": {"representative": 128, "is_drum": True, "pitch_min": 35, "pitch_max": 81},
    "keyboard": {"representative": 0, "is_drum": False, "pitch_min": 21, "pitch_max": 108},
    "organ": {"representative": 16, "is_drum": False, "pitch_min": 36, "pitch_max": 96},
    "mallet": {"representative": 12, "is_drum": False, "pitch_min": 48, "pitch_max": 96},
    "guitar": {"representative": 25, "is_drum": False, "pitch_min": 40, "pitch_max": 88},
    "dist_guitar": {"representative": 30, "is_drum": False, "pitch_min": 40, "pitch_max": 88},
    "bass": {"representative": 33, "is_drum": False, "pitch_min": 28, "pitch_max": 67},
    "violin": {"representative": 40, "is_drum": False, "pitch_min": 55, "pitch_max": 103},
    "woodwind": {"representative": 73, "is_drum": False, "pitch_min": 60, "pitch_max": 96},
    "saxophone": {"representative": 65, "is_drum": False, "pitch_min": 49, "pitch_max": 80},
    "synth": {"representative": 81, "is_drum": False, "pitch_min": 36, "pitch_max": 96},
    "brass": {"representative": 56, "is_drum": False, "pitch_min": 52, "pitch_max": 82},
    "ensemble": {"representative": 48, "is_drum": False, "pitch_min": 36, "pitch_max": 96},
}

_GROUPING_PROGRAMS = {
    128: [128],
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    16: [16, 17, 18, 19, 20, 21, 22, 23],
    12: [8, 9, 10, 11, 12, 13, 14, 15, 112, 114],
    25: [24, 25, 26, 27, 28, 31, 45, 46, 104, 105, 106, 107, 108, 110],
    30: [29, 30],
    33: [32, 33, 34, 35, 36, 37, 38, 39],
    40: [40, 41, 42, 43],
    73: [68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 111],
    65: [64, 65, 66, 67],
    81: [80, 81, 82, 83, 84, 85, 86, 87],
    56: [56, 57, 58, 59, 60],
    48: [44, 48, 49, 50, 51, 52, 53, 54, 61, 62, 63, 76, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
         103],
}

PROGRAM_TO_REP = {p: rep for rep, programs in _GROUPING_PROGRAMS.items() for p in programs}
DROP_SET = {47, 55, 109, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}


# ──────────────────────────────────────────────
# 2. CLI 인자 설정 (에러 났던 seed, pitch_min/max 포함)
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen 기반 초고속 악기 파트 생성")
    parser.add_argument("--song", required=True, help="입력 MIDI 파일명")
    parser.add_argument("--input_dir", default="/data/tutti/inference/INPUT")
    parser.add_argument("--output_dir", default="/home/globaltutti/OUTPUT")
    parser.add_argument("--ckpt", default="/data2/tutti/pretrain_ckpt")
    parser.add_argument("--target", default="violin", choices=list(INSTRUMENT_GROUPS.keys()))
    parser.add_argument("--pitch_min", type=int, default=None)
    parser.add_argument("--pitch_max", type=int, default=None)
    parser.add_argument("--genre", default="CLASSICAL")
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--context_bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


# ──────────────────────────────────────────────
# 3. Vocabulary & Model Loader
# ──────────────────────────────────────────────
def build_v5_vocab():
    vocab = {}

    def add(prefix, r):
        for i in r: vocab[f"{prefix}{i}"] = len(vocab)

    tokens = ["PAD", "BOS", "EOS", "SEP", "PIECE_START", "PIECE_END", "BAR_START", "BAR_END", "PHRASE_END", "<PRE>",
              "<SUF>", "<MID>"]
    for t in tokens: vocab[t] = len(vocab)
    for g in ["CLASSICAL", "JAZZ", "POP", "ROCK", "ELECTRONIC", "FOLK", "UNKNOWN"]: vocab[f"GENRE_{g}"] = len(vocab)
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for r in roots:
        for m in [":maj", ":min"]: vocab[f"KEY_{r}{m}"] = len(vocab)
    vocab["KEY_NONE"] = len(vocab)
    for p in [40, 68, 73]: vocab[f"TARGET_{p}"] = len(vocab)
    for m in ["4:4", "3:4", "2:4", "6:8", "12:8", "OTHER"]: vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1, 6));
    add("INST=", range(129))
    for a in ["ART_NORMAL", "ART_LEGATO", "ART_VIBRATO", "ART_STACCATO"]: vocab[a] = len(vocab)
    add("EXPR_", range(32));
    add("TIME=", range(96));
    add("PITCH=", range(128))
    add("DUR=", range(1, 193));
    add("VEL=", range(32))
    for w in ["melodic", "epic", "calm", "fast", "slow", "sad", "happy", "piano", "strings", "orchestra", "cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


def load_model(ckpt_path, vocab_size, pad_id, device):
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    config.vocab_size, config.pad_token_id, config.max_position_embeddings = vocab_size, pad_id, 2048
    config.sliding_window = None
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model.model.embed_tokens = nn.Embedding(vocab_size, config.hidden_size).to(torch.bfloat16)
    model.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False).to(torch.bfloat16)
    sf = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(sf, device="cpu"), strict=True)
    model.eval().to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("⚡ torch.compile 활성화")
    return model


# ──────────────────────────────────────────────
# 4. 최적화된 추론 엔진 (Vectorized)
# ──────────────────────────────────────────────
@torch.no_grad()
def generate_optimized(model, header, bar_tokens, max_bar, target_name, pitch_min, pitch_max,
                       window_bars, context_bars, temp, top_p, max_new_tokens, vocab, device, source_pm):
    TIME_IDS = torch.tensor([vocab[f"TIME={i}"] for i in range(96)], device=device)
    PITCH_IDS = torch.tensor([vocab[f"PITCH={i}"] for i in range(128)], device=device)
    VEL_IDS = torch.tensor([vocab[f"VEL={i}"] for i in range(32)], device=device)
    INST_TARGET_ID = vocab[f"INST={INSTRUMENT_GROUPS[target_name]['representative']}"]
    BAR_START_ID, PIECE_END_ID, EOS_ID = vocab["BAR_START"], vocab["PIECE_END"], vocab["EOS"]

    pitch_mask = torch.full((len(vocab),), -1e9, device=device)
    pitch_mask[PITCH_IDS[pitch_min: pitch_max + 1]] = 0.0

    all_notes, gen_bar_tokens, VOCAB_R = [], {}, {v: k for k, v in vocab.items()}
    total_windows = (max_bar // window_bars) + 1

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end = min(win_start + window_bars - 1, max_bar)
        if win_start > max_bar: break

        context = list(header)
        for b in range(max(0, win_start - context_bars), win_start):
            context += bar_tokens.get(b, []) + gen_bar_tokens.get(b, [])
        for b in range(win_start, win_end + 1):
            context += bar_tokens.get(b, [])

        input_ids = torch.tensor([context[-1748:]], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, use_cache=True)
        pkv, gen_toks = out.past_key_values, [BAR_START_ID, INST_TARGET_ID]

        for tok in gen_toks:
            out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values

        bar_count, target_playing, last_time_val = 1, True, -1
        cur_in = torch.tensor([[gen_toks[-1]]], device=device)

        for _ in range(max_new_tokens):
            out = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
            pkv, logits = out.past_key_values, out.logits[0, -1, :].float()

            logits[PITCH_IDS] += pitch_mask[PITCH_IDS]
            if last_time_val >= 0: logits[TIME_IDS[:last_time_val + 1]] = -1e9
            if target_playing: logits[INST_TARGET_ID] = -1e9

            probs = torch.softmax(logits / max(temp, 1e-8), dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            gen_toks.append(next_tok)

            if next_tok == INST_TARGET_ID:
                target_playing, last_time_val = True, -1
            elif target_playing and (next_tok in VEL_IDS):
                target_playing = False
            elif next_tok == BAR_START_ID:
                bar_count += 1
                last_time_val, target_playing = -1, False
                if bar_count > window_bars: break
            elif next_tok in [PIECE_END_ID, EOS_ID]:
                break
            if (next_tok >= TIME_IDS[0]) and (next_tok <= TIME_IDS[-1]): last_time_val = next_tok - TIME_IDS[0]
            cur_in = torch.tensor([[next_tok]], device=device)

        win_notes = decode_tokens(gen_toks, source_pm, INSTRUMENT_GROUPS[target_name]['representative'], win_start,
                                  win_start, win_end, vocab, VOCAB_R)
        all_notes.extend(win_notes)
        print(f"   Window {win_idx + 1}/{total_windows} 완료")

    return all_notes


# ──────────────────────────────────────────────
# 5. MIDI Utils
# ──────────────────────────────────────────────
def midi_to_bar_tokens(midi_path, genre, VOCAB):
    pm = pretty_midi.PrettyMIDI(midi_path)
    res = pm.resolution
    timeline = defaultdict(lambda: defaultdict(list))
    ts_changes = sorted(pm.time_signature_changes, key=lambda x: x.time)
    tempo_times, tempos = pm.get_tempo_changes()

    for inst in pm.instruments:
        rep = 128 if inst.is_drum else PROGRAM_TO_REP.get(inst.program, None)
        if rep is None or inst.program in DROP_SET: continue
        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        for n in notes:
            tempo_idx = max(0, bisect.bisect_right(tempo_times, n.start) - 1)
            bpm = tempos[tempo_idx] if len(tempos) > 0 else 120.0
            dur_tick = max(1, min(192, round(((n.end - n.start) / (60.0 / bpm)) * 24)))
            bpb = ts_changes[
                max(0, bisect.bisect_right([t.time for t in ts_changes], n.start) - 1)].numerator if ts_changes else 4
            bar_idx = int(pm.time_to_tick(n.start) // (res * bpb))
            rel_tick = pm.time_to_tick(n.start) - (bar_idx * res * bpb)
            time_tok = VOCAB[f"TIME={min(95, rel_tick * 96 // (res * bpb))}"]
            timeline[bar_idx][rep].append(
                (time_tok, VOCAB[f"INST={rep}"], n.pitch, dur_tick, VOCAB[f"VEL={min(31, n.velocity * 32 // 128)}"]))

    max_bar = int(pm.time_to_tick(pm.get_end_time()) // (res * 4))
    bar_tokens = {b: [VOCAB["BAR_START"], VOCAB["KEY_NONE"], VOCAB["METER_4:4"], VOCAB["DENSITY_1"]] +
                     [t for r_notes in timeline.get(b, {}).values() for n in r_notes for t in
                      [n[1], VOCAB["ART_NORMAL"], VOCAB["EXPR_0"], n[0], VOCAB[f"PITCH={n[2]}"], VOCAB[f"DUR={n[3]}"],
                       n[4]]] +
                     [VOCAB["BAR_END"]] for b in range(max_bar + 1)}
    return [VOCAB["PIECE_START"], VOCAB[f"GENRE_{genre}"]], bar_tokens, max_bar, pm


def decode_tokens(tokens, source_pm, target_prog, bar_offset, win_start, win_end, VOCAB, VOCAB_R):
    res, (tempo_times, tempos) = source_pm.resolution, source_pm.get_tempo_changes()
    notes, bar_idx, cur_time, cur_pitch, cur_dur = [], bar_offset - 1, None, None, None
    for tok in tokens:
        name = VOCAB_R.get(tok, "?")
        if name == "BAR_START":
            bar_idx += 1
        elif name.startswith("TIME="):
            cur_time = int(name.split("=")[1])
        elif name.startswith("PITCH="):
            cur_pitch = int(name.split("=")[1])
        elif name.startswith("DUR="):
            cur_dur = int(name.split("=")[1])
        elif name.startswith("VEL="):
            if cur_pitch and win_start <= bar_idx <= win_end:
                start_sec = source_pm.tick_to_time(bar_idx * res * 4 + cur_time * (res * 4) // 96)
                bpm = tempos[max(0, bisect.bisect_right(tempo_times, start_sec) - 1)] if len(tempos) > 0 else 120.0
                notes.append(
                    {"start": start_sec, "end": start_sec + (cur_dur / 24.0) * (60.0 / bpm), "pitch": cur_pitch,
                     "velocity": (int(name.split("=")[1]) + 1) * 4})
            cur_pitch = cur_dur = None
    return notes


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device} (GPU {args.gpu})")

    random.seed(args.seed);
    torch.manual_seed(args.seed)
    vocab = build_v5_vocab()

    cfg = INSTRUMENT_GROUPS[args.target]
    p_min = args.pitch_min if args.pitch_min is not None else cfg["pitch_min"]
    p_max = args.pitch_max if args.pitch_max is not None else cfg["pitch_max"]

    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(os.path.join(args.input_dir, args.song), "CLASSICAL",
                                                                vocab)
    model = load_model(args.ckpt, len(vocab), vocab["PAD"], device)

    all_notes = generate_optimized(model, header, bar_tokens, max_bar, args.target, p_min, p_max, args.window_bars,
                                   args.context_bars, args.temperature, args.top_p, args.max_new_tokens, vocab, device,
                                   source_pm)

    out_pm = copy.deepcopy(source_pm)
    inst = pretty_midi.Instrument(program=cfg['representative'] if not cfg['is_drum'] else 0, is_drum=cfg['is_drum'],
                                  name=args.target)
    for n in all_notes: inst.notes.append(
        pretty_midi.Note(velocity=n["velocity"], pitch=n["pitch"], start=n["start"], end=n["end"]))
    out_pm.instruments.append(inst)
    out_pm.write(os.path.join(args.output_dir, f"{os.path.splitext(args.song)[0]}_{args.target}_opt.mid"))
    print(f"✅ 완료: {args.output_dir}")


if __name__ == "__main__": main()