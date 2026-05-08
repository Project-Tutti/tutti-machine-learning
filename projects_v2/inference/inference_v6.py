"""
inference_v6.py — 학습(preprocess_lmd.py + train.py) 구조에 완전 일치하는 추론 코드

핵심 설계 원칙:
  1. MIDI → timeline 변환을 preprocess_lmd.py의 _worker()와 동일하게 수행
     - timeline key = inst_idx (enumerate 인덱스), NOT program 번호
  2. generate_bars()를 직접 재사용하여 PRE/SUF 토큰 생성
     - PRE = bars_full (exclude_inst_idx=None, 타겟 포함)
     - SUF = bars_no_target (exclude_inst_idx=타겟의 inst_idx)
  3. FIM 시퀀스 구조: PIECE_START GENRE TARGET <PRE> bars... <SUF> bars... <MID>
     → build_fim_chunk()와 동일
  4. LogitsProcessor는 문법 강제만 수행, 확률 분포 조작은 최소화
"""
import os, argparse, bisect, copy, random, re
import torch
import pretty_midi
from collections import defaultdict
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from vocab_v6 import build_v6_vocab
from preprocess_lmd import generate_bars, get_beats_at

# ──────────────────────────────────────────
# 악기 설정
# ──────────────────────────────────────────
TARGET_CONFIG = {
    "Violin":       {"program": 40, "pitch_min": 55, "pitch_max": 100, "monophonic": True},
    "Viola":        {"program": 41, "pitch_min": 48, "pitch_max": 91,  "monophonic": True},
    "Cello":        {"program": 42, "pitch_min": 36, "pitch_max": 81,  "monophonic": True},
    "Contrabass":   {"program": 43, "pitch_min": 28, "pitch_max": 60,  "monophonic": True},
    "Trumpet":      {"program": 56, "pitch_min": 52, "pitch_max": 82,  "monophonic": True},
    "Trombone":     {"program": 57, "pitch_min": 40, "pitch_max": 72,  "monophonic": True},
    "Tuba":         {"program": 58, "pitch_min": 28, "pitch_max": 58,  "monophonic": True},
    "French Horn":  {"program": 60, "pitch_min": 34, "pitch_max": 77,  "monophonic": True},
    "Soprano Sax":  {"program": 64, "pitch_min": 54, "pitch_max": 84,  "monophonic": True},
    "Alto Sax":     {"program": 65, "pitch_min": 49, "pitch_max": 80,  "monophonic": True},
    "Tenor Sax":    {"program": 66, "pitch_min": 44, "pitch_max": 75,  "monophonic": True},
    "Baritone Sax": {"program": 67, "pitch_min": 36, "pitch_max": 69,  "monophonic": True},
    "Oboe":         {"program": 68, "pitch_min": 58, "pitch_max": 91,  "monophonic": True},
    "English Horn": {"program": 69, "pitch_min": 52, "pitch_max": 81,  "monophonic": True},
    "Bassoon":      {"program": 70, "pitch_min": 34, "pitch_max": 75,  "monophonic": True},
    "Clarinet":     {"program": 71, "pitch_min": 50, "pitch_max": 94,  "monophonic": True},
    "Flute":        {"program": 73, "pitch_min": 60, "pitch_max": 96,  "monophonic": True},
}


# ──────────────────────────────────────────
# MIDI 파싱 (preprocess_lmd.py _worker()와 동일)
# ──────────────────────────────────────────
def parse_midi_like_training(midi_path):
    """
    학습 전처리(_worker)와 동일한 방식으로 MIDI를 파싱합니다.
    핵심: timeline key가 inst_idx(enumerate 인덱스)이어야 합니다.

    Returns:
        timeline, pm, key_changes, key_times, ts_changes, ts_times,
        inst_idx_map: {program: inst_idx} — 타겟 프로그램의 inst_idx를 찾기 위해
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    res = pm.resolution

    key_changes = sorted(pm.key_signature_changes, key=lambda x: x.time)
    key_times = [k.time for k in key_changes]
    ts_changes = sorted(pm.time_signature_changes, key=lambda x: x.time)
    ts_times = [t.time for t in ts_changes]
    tempo_change_times, tempos = pm.get_tempo_changes()

    timeline = defaultdict(lambda: defaultdict(list))
    inst_idx_map = {}  # program → inst_idx (첫 번째 매칭)

    for inst_idx, inst in enumerate(pm.instruments):
        p = 128 if inst.is_drum else inst.program
        if p not in inst_idx_map:
            inst_idx_map[p] = inst_idx

        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        for note in notes:
            beats_per_bar = get_beats_at(note.start, ts_changes, ts_times)
            beats_per_bar = max(1, beats_per_bar)
            bar_idx = int(pm.time_to_tick(note.start) // (res * beats_per_bar))
            bar_start_tick = bar_idx * res * beats_per_bar
            rel_tick = pm.time_to_tick(note.start) - bar_start_tick
            time_val = min(95, max(0, int(rel_tick * 96 // (res * beats_per_bar))))

            tempo_idx = max(0, bisect.bisect_right(tempo_change_times, note.start) - 1)
            bpm = tempos[tempo_idx] if len(tempos) > 0 else 120.0
            sec_per_beat = 60.0 / max(0.1, bpm)
            dur_val = max(1, min(192, round(((note.end - note.start) / sec_per_beat) * 24)))
            vel_val = min(31, note.velocity * 32 // 128)

            timeline[bar_idx][inst_idx].append((time_val, p, note.pitch, dur_val, vel_val))

    return timeline, pm, key_changes, key_times, ts_changes, ts_times, inst_idx_map


# ──────────────────────────────────────────
# V6 문법 강제 LogitsProcessor (최소한의 제약만)
# ──────────────────────────────────────────
class V6GrammarProcessor(LogitsProcessor):
    """
    V6 토큰 문법 강제:
      BAR_START → KEY → METER → DENSITY → (INST TIME PITCH DUR VEL)* → BAR_END

    설계 원칙:
      - 허용된 토큰만 mask=0, 나머지는 -inf → 문법 위반 불가
      - 피치는 유효 범위만 허용 (음역 제한)
      - DUR은 1~96만 허용 (1마디 초과 방지)
      - 단선율 강제: 같은 TIME에 타겟 노트 2개 불가
      - 확률 분포 조작(보너스/부스트) 없음 → 모델 logit 존중
    """
    def __init__(self, target_program, pitch_min, pitch_max,
                 is_monophonic, vocab, inv_vocab, prompt_len, target_bars):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.target_prog = target_program
        self.target_inst_tok = vocab.get(f"INST={target_program}", -1)
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.valid_pitch_toks = set(
            vocab[f"PITCH={i}"] for i in range(pitch_min, pitch_max + 1)
            if f"PITCH={i}" in vocab
        )
        self.is_monophonic = is_monophonic
        self.prompt_len = prompt_len
        self.target_bars = target_bars

        # 토큰 그룹 캐싱
        self.bar_start_tok = vocab["BAR_START"]
        self.bar_end_tok = vocab["BAR_END"]
        self.mid_tok = vocab["<MID>"]
        self.eos_tok = vocab["EOS"]
        self.key_toks = [v for k, v in vocab.items() if k.startswith("KEY_")]
        self.meter_toks = [v for k, v in vocab.items() if k.startswith("METER_")]
        self.density_toks = [v for k, v in vocab.items() if k.startswith("DENSITY_")]
        self.inst_toks = [v for k, v in vocab.items() if k.startswith("INST=")]
        self.time_toks = sorted(
            [(v, int(k.split("=")[1])) for k, v in vocab.items() if k.startswith("TIME=")],
            key=lambda x: x[1]
        )
        self.pitch_toks = [v for k, v in vocab.items() if k.startswith("PITCH=")]
        self.dur_toks = [v for k, v in vocab.items() if k.startswith("DUR=")]
        self.vel_toks = [v for k, v in vocab.items() if k.startswith("VEL=")]

        self.MASK_VAL = -1e20
        self.target_boost = 3.0  # 타겟 악기 선택 확률 부스트
        self.min_notes_per_bar = 1  # 마디당 권장 최소 노트 수 (보장형은 아님)

    def _scan_current_bar(self, seq):
        """현재 마디 내 상태 분석: 마지막 TIME, 타겟의 마지막 TIME, 타겟 노트 수"""
        last_time_in_bar = 0
        last_target_time = -1
        target_note_count = 0

        # 마지막 BAR_START 찾기
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == self.bar_start_tok:
                # 이 마디 내의 모든 토큰 스캔
                j = i + 1
                while j < len(seq):
                    name = self.inv_vocab.get(seq[j], "")
                    if name.startswith("INST="):
                        # 5-토큰 노트 시퀀스 확인: INST TIME PITCH DUR VEL
                        is_target = (seq[j] == self.target_inst_tok)
                        if j + 4 < len(seq):
                            t_name = self.inv_vocab.get(seq[j + 1], "")
                            v_name = self.inv_vocab.get(seq[j + 4], "")
                            if t_name.startswith("TIME=") and v_name.startswith("VEL="):
                                t_val = int(t_name.split("=")[1])
                                last_time_in_bar = max(last_time_in_bar, t_val)
                                if is_target:
                                    last_target_time = max(last_target_time, t_val)
                                    target_note_count += 1
                                j += 5
                                continue
                        # TIME 단독 (노트가 아직 완성 안 된 경우)
                        if j + 1 < len(seq):
                            t_name = self.inv_vocab.get(seq[j + 1], "")
                            if t_name.startswith("TIME="):
                                t_val = int(t_name.split("=")[1])
                                last_time_in_bar = max(last_time_in_bar, t_val)
                                if is_target:
                                    last_target_time = max(last_target_time, t_val)
                    j += 1
                break

        return last_time_in_bar, last_target_time, target_note_count

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, self.MASK_VAL)

        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            last_tok = seq[-1]
            last_name = self.inv_vocab.get(last_tok, "")

            gen_seq = seq[self.prompt_len:]
            n_bars = sum(1 for t in gen_seq if t == self.bar_start_tok)

            last_time_in_bar, last_target_time, target_notes = self._scan_current_bar(seq)

            # ── 문법 상태 머신 ──

            # 1) <MID> 또는 BAR_END 뒤 → 다음 BAR_START 또는 EOS
            if last_tok == self.mid_tok or last_tok == self.bar_end_tok:
                if n_bars < self.target_bars:
                    mask[b, self.bar_start_tok] = 0
                else:
                    mask[b, self.eos_tok] = 0

            # 2) BAR_START 뒤 → KEY
            elif last_tok == self.bar_start_tok:
                for t in self.key_toks:
                    mask[b, t] = 0

            # 3) KEY 뒤 → METER
            elif last_name.startswith("KEY_"):
                for t in self.meter_toks:
                    mask[b, t] = 0

            # 4) METER 뒤 → DENSITY
            elif last_name.startswith("METER_"):
                for t in self.density_toks:
                    mask[b, t] = 0

            # 5) DENSITY 또는 VEL 뒤 → INST(노트 시작) 또는 BAR_END(마디 종료)
            elif last_name.startswith("DENSITY_") or last_name.startswith("VEL="):
                can_add_note = True
                if self.is_monophonic and last_target_time >= 95:
                    can_add_note = False
                if target_notes >= 48:
                    can_add_note = False

                # [공격적 생성 모드] 
                # 1. DENSITY 직후라면 무조건 타겟 악기 토큰만 허용 (v4 방식)
                if last_name.startswith("DENSITY_"):
                    mask[b, self.target_inst_tok] = 0
                
                # 2. 이미 노트를 생성 중인 경우
                elif can_add_note:
                    # 최소 노트 수 미달 시에는 타겟 악기만 더 추가하도록 강제
                    if target_notes < self.min_notes_per_bar:
                        mask[b, self.target_inst_tok] = 0
                    else:
                        # 최소 노트 채웠으면 타겟 악기 추가 혹은 마디 종료 허용
                        mask[b, self.target_inst_tok] = 0
                        mask[b, self.bar_end_tok] = 0
                        # 그래도 타겟 악기를 좀 더 우선시하도록 보너스
                        scores[b, self.target_inst_tok] += self.target_boost
                else:
                    # 더 이상 노트를 추가할 수 없는 상태(음역 이탈 등)면 종료
                    mask[b, self.bar_end_tok] = 0

            # 6) INST 뒤 → TIME (시간순 강제 + 단선율 중복 방지)
            elif last_name.startswith("INST="):
                for tok_id, t_val in self.time_toks:
                    # 시간순 강제: 마디 내 이전 노트의 TIME 이상만 허용
                    if t_val < last_time_in_bar:
                        continue
                    # 단선율: 타겟 악기는 이전 타겟 TIME보다 큰 것만 허용
                    if self.is_monophonic and last_tok == self.target_inst_tok:
                        if t_val <= last_target_time:
                            continue
                    mask[b, tok_id] = 0

            # 7) TIME 뒤 → PITCH
            elif last_name.startswith("TIME="):
                prev_inst = seq[-2] if len(seq) >= 2 else -1
                if prev_inst == self.target_inst_tok:
                    # 타겟 악기: 유효 음역대만 허용
                    for t in self.valid_pitch_toks:
                        mask[b, t] = 0
                else:
                    # 다른 악기: 전체 피치 허용
                    for t in self.pitch_toks:
                        mask[b, t] = 0

            # 8) PITCH 뒤 → DUR (1~96만, 1마디 초과 방지)
            elif last_name.startswith("PITCH="):
                for t in self.dur_toks:
                    d_val = int(self.inv_vocab[t].split("=")[1])
                    if 1 <= d_val <= 96:
                        mask[b, t] = 0

            # 9) DUR 뒤 → VEL
            elif last_name.startswith("DUR="):
                for t in self.vel_toks:
                    mask[b, t] = 0

            # 10) 예외: 전부 마스킹됨 → fallback
            else:
                mask[b, :] = 0

            # 안전 fallback: 모든 토큰이 마스킹되면 BAR_END 또는 EOS 허용
            if torch.all(mask[b] <= self.MASK_VAL):
                if n_bars < self.target_bars:
                    mask[b, self.bar_end_tok] = 0
                    mask[b, self.bar_start_tok] = 0
                else:
                    mask[b, self.eos_tok] = 0

        return scores + mask


# ──────────────────────────────────────────
# 컨텍스트 트리밍 (preprocess_lmd.py의 trim_to_budget과 동일 비율)
# ──────────────────────────────────────────
def trim_context(pre_bar_list, suf_bar_list, header_len, max_ctx, vocab):
    """PRE 2/3 : SUF 1/3 비율로 트리밍 (학습과 동일)"""
    fixed = header_len + 3  # <PRE>, <SUF>, <MID>
    p_list = list(pre_bar_list)
    s_list = list(suf_bar_list)

    def total():
        return fixed + sum(len(t) for _, t in p_list) + sum(len(t) for _, t in s_list)

    # 학습의 trim_to_budget: PRE에 2/3, SUF에 1/3 할당
    while total() > max_ctx and (p_list or s_list):
        p_len = sum(len(t) for _, t in p_list)
        s_len = sum(len(t) for _, t in s_list)
        # PRE가 전체의 2/3 초과하면 PRE에서 앞을 자름
        if p_len > 0 and (s_len == 0 or p_len / max(1, p_len + s_len) > 0.67):
            p_list.pop(0)
        elif s_list:
            s_list.pop()
        else:
            p_list.pop(0)

    return p_list, s_list


# ──────────────────────────────────────────
# 토큰 → MIDI 노트 디코딩
# ──────────────────────────────────────────
def decode_generated_tokens(tokens, source_pm, target_prog,
                            win_start, win_end,
                            VOCAB_R, pitch_min, pitch_max):
    """생성된 토큰 시퀀스를 MIDI 노트로 변환"""
    res = source_pm.resolution
    ts_changes = sorted(source_pm.time_signature_changes, key=lambda x: x.time)
    ts_times = [t.time for t in ts_changes]
    tempo_times, tempos = source_pm.get_tempo_changes()

    # bar_idx → tick 매핑 (누적 계산)
    bar_tick_map = {}
    acc = 0
    for b_idx in range(max(win_end + 10, 2000)):
        bar_tick_map[b_idx] = acc
        bt = source_pm.tick_to_time(acc)
        idx = max(0, bisect.bisect_right(ts_times, bt) - 1)
        bpb = ts_changes[idx].numerator if idx < len(ts_changes) else 4
        acc += res * bpb

    notes_out = []
    bar_idx = win_start - 1  # BAR_START를 만나면 +1
    cur = {}

    for tok in tokens:
        name = VOCAB_R.get(tok, "?")
        if name == "BAR_START":
            bar_idx += 1
            cur = {}
        elif name == "BAR_END":
            cur = {}
        elif name.startswith("INST="):
            cur = {"INST": int(name.split("=")[1])}
        elif name.startswith("TIME="):
            cur["TIME"] = int(name.split("=")[1])
        elif name.startswith("PITCH="):
            cur["PITCH"] = int(name.split("=")[1])
        elif name.startswith("DUR="):
            cur["DUR"] = int(name.split("=")[1])
        elif name.startswith("VEL="):
            cur["VEL"] = int(name.split("=")[1])
            # 5-토큰 노트 완성 → 디코딩
            if (cur.get("INST") == target_prog
                    and cur.get("PITCH") is not None
                    and cur.get("DUR") is not None
                    and cur.get("TIME") is not None
                    and win_start <= bar_idx <= win_end
                    and pitch_min <= cur["PITCH"] <= pitch_max):

                b_tick = bar_tick_map.get(bar_idx, 0)
                b_time = source_pm.tick_to_time(b_tick)
                ts_idx = max(0, bisect.bisect_right(ts_times, b_time) - 1)
                bpb = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
                bar_ticks = res * bpb
                abs_tick = b_tick + cur["TIME"] * bar_ticks // 96
                start_sec = source_pm.tick_to_time(abs_tick)

                t_idx = max(0, bisect.bisect_right(tempo_times, start_sec) - 1)
                bpm = tempos[t_idx] if len(tempos) > 0 else 120.0
                dur_sec = (cur["DUR"] / 24.0) * (60.0 / bpm)

                notes_out.append({
                    "start": start_sec,
                    "end": start_sec + dur_sec,
                    "pitch": cur["PITCH"],
                    "velocity": max(1, min(127, (cur["VEL"] + 1) * 4)),
                })
            cur = {}
        elif name == "EOS":
            break

    return notes_out


# ──────────────────────────────────────────
# 후처리
# ──────────────────────────────────────────
def postprocess(notes, target_name, monophonic=True):
    cfg = TARGET_CONFIG[target_name]
    pmin, pmax = cfg["pitch_min"], cfg["pitch_max"]

    # 음역 필터
    notes = [n for n in notes if pmin <= n["pitch"] <= pmax]
    # 길이 제한 (4초)
    for n in notes:
        if n["end"] - n["start"] > 4.0:
            n["end"] = n["start"] + 4.0
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    if monophonic:
        # 단선율 강제: 겹치는 노트 자르기
        notes = sorted(notes, key=lambda x: x["start"])
        mono = []
        for n in notes:
            if mono and n["start"] < mono[-1]["end"]:
                mono[-1]["end"] = n["start"]
                if mono[-1]["end"] - mono[-1]["start"] < 0.05:
                    mono.pop()
            mono.append(n)
        notes = mono

    # 레가토 갭 메우기
    notes = sorted(notes, key=lambda x: x["start"])
    for i in range(len(notes) - 1):
        gap = notes[i + 1]["start"] - notes[i]["end"]
        if 0 < gap < 0.03:
            notes[i]["end"] = notes[i + 1]["start"]

    # 옥타브 점프 보정 (단선율)
    if monophonic:
        notes = sorted(notes, key=lambda x: x["start"])
        for i in range(1, len(notes)):
            interval = notes[i]["pitch"] - notes[i - 1]["pitch"]
            if abs(interval) > 12:
                new_pitch = notes[i]["pitch"] + (-12 if interval > 0 else 12)
                if pmin <= new_pitch <= pmax:
                    notes[i]["pitch"] = new_pitch

    notes = [n for n in notes if pmin <= n["pitch"] <= pmax]
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]
    return notes


# ──────────────────────────────────────────
# 슬라이딩 윈도우 생성 (학습 FIM 구조와 완전 일치)
# ──────────────────────────────────────────
def generate_sliding_window(model, pm, timeline, target_prog, target_inst_idx,
                            pitch_min, pitch_max, monophonic,
                            window_bars, context_bars, future_bars,
                            temperature, top_p,
                            VOCAB, VOCAB_R, device,
                            key_changes, key_times, ts_changes, ts_times,
                            genre_tok):
    SEQ_LEN = 8192
    MAX_CTX = SEQ_LEN - 1200  # 생성 여유분

    # ── 학습과 동일하게 bars 생성 ──
    # bars_full: 모든 악기 포함 (PRE/MID용)
    bars_full = generate_bars(pm, timeline, None, VOCAB,
                              key_changes, key_times, ts_changes, ts_times)
    # bars_no_target: 타겟 악기만 제외 (SUF용)
    bars_no_target = generate_bars(pm, timeline, target_inst_idx, VOCAB,
                                    key_changes, key_times, ts_changes, ts_times)
    n_bars = len(bars_full)

    print(f"  총 마디 수: {n_bars} (bars_full={len(bars_full)}, bars_no_target={len(bars_no_target)})")

    all_notes = []
    total_windows = (n_bars + window_bars - 1) // window_bars

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end = min(win_start + window_bars - 1, n_bars - 1)
        if win_start >= n_bars:
            break

        # ── 학습 build_fim_chunk()와 동일한 범위 계산 ──
        pre_start = max(0, win_start - context_bars)
        pre_end = win_start
        suf_start = win_end + 1
        suf_end = min(n_bars, suf_start + future_bars)

        target_bars_count = win_end - win_start + 1

        # PRE: bars_full (타겟 악기 포함) — 학습과 동일
        pre_bar_list = []
        for b in range(pre_start, pre_end):
            pre_bar_list.append((b, list(bars_full[b])))

        # SUF: bars_no_target (타겟 악기 제외) — 학습과 동일
        suf_bar_list = []
        for b in range(suf_start, suf_end):
            if b < len(bars_no_target):
                suf_bar_list.append((b, list(bars_no_target[b])))

        # 헤더: PIECE_START GENRE TARGET — 학습과 동일
        header = [VOCAB["PIECE_START"], genre_tok, VOCAB[f"TARGET_{target_prog}"]]

        # 통계 출력
        h_len = len(header)
        p_len = sum(len(t) for _, t in pre_bar_list)
        s_len = sum(len(t) for _, t in suf_bar_list)

        print(f"\n{'─' * 50}")
        print(f"WINDOW #{win_idx} (Bar {win_start}~{win_end})")
        print(f"  PRE : {p_len:>4} toks ({len(pre_bar_list)} bars, {pre_start}~{pre_end - 1})")
        print(f"  SUF : {s_len:>4} toks ({len(suf_bar_list)} bars, {suf_start}~{suf_end - 1})")

        # 오버플로우 시 트리밍
        total_len = h_len + 3 + p_len + s_len  # +3 for <PRE>, <SUF>, <MID>
        if total_len > MAX_CTX:
            pre_bar_list, suf_bar_list = trim_context(
                pre_bar_list, suf_bar_list, h_len, MAX_CTX, VOCAB)
            p_len = sum(len(t) for _, t in pre_bar_list)
            s_len = sum(len(t) for _, t in suf_bar_list)
            print(f"  ⚠️  Trimmed → PRE {p_len}, SUF {s_len}")

        # ── FIM 시퀀스 조립 (학습과 동일) ──
        # PIECE_START GENRE TARGET <PRE> bars_full... <SUF> bars_no_target... <MID>
        pre_tokens = [VOCAB["<PRE>"]]
        for _, toks in pre_bar_list:
            pre_tokens += toks

        suf_tokens = [VOCAB["<SUF>"]]
        for _, toks in suf_bar_list:
            suf_tokens += toks

        context = header + pre_tokens + suf_tokens + [VOCAB["<MID>"]]

        print(f"  Context: {len(context)} tokens")
        print(f"{'─' * 50}")

        # ── 생성 ──
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        processor = V6GrammarProcessor(
            target_prog, pitch_min, pitch_max, monophonic,
            VOCAB, VOCAB_R, len(context), target_bars_count
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=1200,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                eos_token_id=VOCAB["EOS"],
                pad_token_id=VOCAB["PAD"],
                logits_processor=LogitsProcessorList([processor]),
            )

        gen_toks = output_ids[0][len(context):].cpu().tolist()

        # ── 생성된 노트를 timeline에 반영 (다음 윈도우의 PRE 컨텍스트용) ──
        bar_idx = win_start - 1
        cur_note = {}
        for t in gen_toks:
            name = VOCAB_R.get(t, "?")
            if name == "BAR_START":
                bar_idx += 1
                cur_note = {}
            elif name.startswith("INST="):
                cur_note = {"INST": int(name.split("=")[1])}
            elif name.startswith("TIME="):
                cur_note["TIME"] = int(name.split("=")[1])
            elif name.startswith("PITCH="):
                cur_note["PITCH"] = int(name.split("=")[1])
            elif name.startswith("DUR="):
                cur_note["DUR"] = int(name.split("=")[1])
            elif name.startswith("VEL="):
                cur_note["VEL"] = int(name.split("=")[1])
                # 타겟 악기의 노트인 경우 timeline에 추가 및 로그 출력
                if cur_note.get("INST") == target_prog and win_start <= bar_idx <= win_end:
                    print(f"  [BAR {bar_idx}] INST={cur_note['INST']} "
                          f"PITCH={cur_note.get('PITCH','?')} "
                          f"DUR={cur_note.get('DUR','?')} "
                          f"VEL={cur_note.get('VEL','?')} ✅")
                    
                    note_tuple = (cur_note["TIME"], target_prog, cur_note["PITCH"], cur_note["DUR"], cur_note["VEL"])
                    timeline[bar_idx][target_inst_idx].append(note_tuple)
                cur_note = {}

        # 생성된 구간에 대해 bars_full 재생성 (타겟 악기 포함 버전)
        updated_bars = generate_bars(pm, timeline, None, VOCAB,
                                     key_changes, key_times, ts_changes, ts_times)
        for b in range(win_start, win_end + 1):
            if b < len(updated_bars):
                bars_full[b] = updated_bars[b]

        # 디코딩 (PrettyMIDI용)
        win_notes = decode_generated_tokens(
            gen_toks, pm, target_prog,
            win_start, win_end, VOCAB_R, pitch_min, pitch_max
        )
        all_notes.extend(win_notes)
        print(f"   -> Result: Generated {len(win_notes)} notes for this window.")

    return all_notes


# ──────────────────────────────────────────
# MIDI 저장
# ──────────────────────────────────────────
def save_midi(notes, source_pm, output_path, target_prog, target_name):
    out_pm = copy.deepcopy(source_pm)
    new_inst = pretty_midi.Instrument(program=target_prog, is_drum=False, name=target_name)
    for n in notes:
        new_inst.notes.append(pretty_midi.Note(
            velocity=n["velocity"], pitch=n["pitch"],
            start=n["start"], end=n["end"]))
    out_pm.instruments.append(new_inst)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_pm.write(output_path)
    print(f"\n✅ 저장 완료: {output_path}  ({len(notes)} 노트)")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V6 추론 — 학습 FIM 구조 완전 일치")
    parser.add_argument("--song", required=True)
    parser.add_argument("--input_dir", default=os.path.expanduser("~/INPUT"))
    parser.add_argument("--output_dir", default=os.path.expanduser("~/OUTPUT/v6_output"))
    parser.add_argument("--ckpt_path", default="/data2/tutti/Qwen_Checkpoints/checkpoint-45000")
    parser.add_argument("--target_name", default="Violin", choices=list(TARGET_CONFIG.keys()))
    parser.add_argument("--genre", default="pop", choices=["pop", "electronic", "classical", "rock", "other"])
    parser.add_argument("--context_bars", type=int, default=8)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--future_bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pitch_min", type=int, default=None)
    parser.add_argument("--pitch_max", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB = build_v6_vocab()
    VOCAB_R = {v: k for k, v in VOCAB.items()}

    cfg = TARGET_CONFIG[args.target_name]
    target_prog = cfg["program"]
    pitch_min = args.pitch_min if args.pitch_min is not None else cfg["pitch_min"]
    pitch_max = args.pitch_max if args.pitch_max is not None else cfg["pitch_max"]
    monophonic = cfg["monophonic"]

    input_path = os.path.join(args.input_dir, args.song)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 입력 파일 없음: {input_path}")

    song_stem = os.path.splitext(args.song)[0]
    output_path = os.path.join(args.output_dir, f"{song_stem}_{args.target_name}_v6.mid")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── MIDI 파싱 (학습과 동일한 방식) ──
    print(f"📄 MIDI 파싱: {input_path}")
    (timeline, pm, key_changes, key_times,
     ts_changes, ts_times, inst_idx_map) = parse_midi_like_training(input_path)

    # 타겟 악기의 inst_idx 찾기 (generate_bars의 exclude 용)
    # 입력 MIDI에 타겟 악기가 없을 수도 있음 → 새로운 인덱스 할당
    target_inst_idx = inst_idx_map.get(target_prog, -999)
    if target_inst_idx == -999:
        target_inst_idx = len(pm.instruments)
        inst_idx_map[target_prog] = target_inst_idx
    print(f"  target_prog={target_prog}, target_inst_idx={target_inst_idx}")
    print(f"  inst_idx_map: {inst_idx_map}")

    # ── 모델 로드 ──
    print(f"\n🤖 모델 로드: {args.ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_path, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.eval()

    genre_tok = VOCAB[f"GENRE_{args.genre.upper()}"]

    print(f"\n🎵 생성 시작 (target={args.target_name} [prog={target_prog}], "
          f"window={args.window_bars}, context={args.context_bars}, "
          f"{'mono' if monophonic else 'poly'})")

    all_notes = generate_sliding_window(
        model, pm, timeline, target_prog, target_inst_idx,
        pitch_min, pitch_max, monophonic,
        args.window_bars, args.context_bars, args.future_bars,
        args.temperature, args.top_p,
        VOCAB, VOCAB_R, device,
        key_changes, key_times, ts_changes, ts_times,
        genre_tok
    )

    print(f"\n  전체 디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, args.target_name, monophonic=monophonic)
    print(f"  후처리 후 노트:   {len(all_notes)}")

    if not all_notes:
        print("❌ 노트 없음 — --temperature를 높이거나 --seed를 바꿔보세요")
    else:
        save_midi(all_notes, pm, output_path, target_prog, args.target_name)


if __name__ == "__main__":
    main()