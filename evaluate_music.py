"""
evaluate_music.py
생성된 MIDI의 음악적 품질 평가

논문 기반 지표:
1. Chord Accuracy       — 생성 파트가 소스의 화음에 얼마나 맞는지 (NeurIPS 2024 Structured Arrangement)
2. Pitch Class Histogram Similarity (PCH) — 소스와 생성 파트의 조성 분포 유사도 (AccoMontage)
3. DOA (Degree of Arrangement) — 트랙 간 음고 다양성 (창의성 지표) (NeurIPS 2024)
4. Dissonance Rate      — 동시 발음 음정의 불협화 비율 (AccoMontage2)

사용법:
    python3 evaluate_music.py --source INPUT/KissTheRain.mid --generated OUTPUT/KissTheRain_violin.mid
    python3 evaluate_music.py --source INPUT/KissTheRain.mid --generated OUTPUT/KissTheRain_violin.mid --verbose
"""

import argparse
import numpy as np
import pretty_midi
from collections import defaultdict

# ──────────────────────────────────────────────
# 협화음 테이블 (음정 → 협화도)
# 반음 간격 기준, 0=완전 협화 / 1=불협화
# 참고: AccoMontage2 dissonance 정의
# ──────────────────────────────────────────────
DISSONANCE_TABLE = {
    0:  0.0,   # 동음 (완전협화)
    1:  1.0,   # 단2도 (불협화)
    2:  0.5,   # 장2도 (약협화)
    3:  0.2,   # 단3도 (협화)
    4:  0.2,   # 장3도 (협화)
    5:  0.1,   # 완전4도 (협화)
    6:  1.0,   # 삼전음 (증4도, 강불협화)
    7:  0.0,   # 완전5도 (완전협화)
    8:  0.2,   # 단6도 (협화)
    9:  0.2,   # 장6도 (협화)
    10: 0.5,   # 단7도 (약불협화)
    11: 0.8,   # 장7도 (불협화)
}

# 코드 구성음 테이블 (루트 기준 반음 간격)
CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
}


def parse_args():
    parser = argparse.ArgumentParser(description="음악적 평가 지표 계산")
    parser.add_argument("--source",    required=True, help="원본 소스 MIDI (밴드 트랙)")
    parser.add_argument("--generated", required=True, help="생성된 타겟 악기 MIDI")
    parser.add_argument("--target_program", type=int, default=None,
                        help="타겟 악기 program 번호 (없으면 자동 감지)")
    parser.add_argument("--verbose",   action="store_true", help="상세 출력")
    return parser.parse_args()


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def get_active_pitches_at(pm, time, exclude_programs=None):
    """특정 시간에 울리는 음들의 pitch 집합 반환"""
    pitches = set()
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if exclude_programs and inst.program in exclude_programs:
            continue
        for note in inst.notes:
            if note.start <= time < note.end:
                pitches.add(note.pitch % 12)
    return pitches


def get_pitch_class_histogram(pm, programs=None):
    """
    Pitch Class Histogram (12차원 벡터) 계산
    AccoMontage / NeurIPS 2024에서 사용
    programs: None이면 전체, 리스트면 해당 program만
    """
    pch = np.zeros(12)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if programs is not None and inst.program not in programs:
            continue
        for note in inst.notes:
            duration = note.end - note.start
            pch[note.pitch % 12] += duration
    total = pch.sum()
    if total > 0:
        pch /= total
    return pch


def detect_chord_at(pitches):
    """
    현재 울리는 음들로 가장 유사한 코드 추정
    반환: (root, chord_type, chord_tones) 또는 None
    """
    if len(pitches) < 2:
        return None

    best_match = None
    best_score = -1

    for root in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            chord_tones = {(root + i) % 12 for i in intervals}
            # 현재 음들이 코드 구성음과 얼마나 겹치는지
            overlap = len(pitches & chord_tones)
            score   = overlap / max(len(chord_tones), len(pitches))
            if score > best_score:
                best_score  = score
                best_match  = (root, chord_type, chord_tones)

    return best_match if best_score > 0.5 else None


# ──────────────────────────────────────────────
# 지표 1: Chord Accuracy
# NeurIPS 2024 Structured Arrangement
# 생성 파트의 음들이 소스의 화음 구성음에 얼마나 맞는지
# ──────────────────────────────────────────────
def compute_chord_accuracy(source_pm, generated_pm, target_programs,
                            resolution=0.125, verbose=False):
    """
    매 resolution 초마다 소스의 코드를 추정하고,
    생성 파트의 음이 그 코드 구성음에 포함되는 비율 계산
    """
    end_time   = max(source_pm.get_end_time(), generated_pm.get_end_time())
    times      = np.arange(0, end_time, resolution)

    total      = 0
    in_chord   = 0

    for t in times:
        # 소스 코드 추정 (타겟 악기 제외)
        source_pitches = get_active_pitches_at(source_pm, t, exclude_programs=target_programs)
        chord_info     = detect_chord_at(source_pitches)
        if chord_info is None:
            continue

        _, _, chord_tones = chord_info

        # 생성 파트에서 울리는 음
        gen_pitches = set()
        for inst in generated_pm.instruments:
            if inst.is_drum:
                continue
            if inst.program in target_programs:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        gen_pitches.add(note.pitch % 12)

        if not gen_pitches:
            continue

        # 코드 구성음과 겹치는 비율
        for p in gen_pitches:
            total += 1
            if p in chord_tones:
                in_chord += 1

    chord_acc = in_chord / total if total > 0 else 0.0
    if verbose:
        print(f"  Chord Accuracy: {chord_acc:.4f} ({in_chord}/{total} 음정 일치)")
    return chord_acc


# ──────────────────────────────────────────────
# 지표 2: Pitch Class Histogram Similarity (PCH)
# AccoMontage / NeurIPS 2024 Faithfulness
# 소스와 생성 파트의 조성 분포 코사인 유사도
# ──────────────────────────────────────────────
def compute_pch_similarity(source_pm, generated_pm, target_programs, verbose=False):
    """
    소스 PCH vs 생성 파트 PCH 코사인 유사도
    높을수록 조성적으로 일관됨
    """
    source_pch = get_pitch_class_histogram(source_pm)
    gen_pch    = get_pitch_class_histogram(generated_pm, programs=target_programs)

    # 코사인 유사도
    dot    = np.dot(source_pch, gen_pch)
    norm_s = np.linalg.norm(source_pch)
    norm_g = np.linalg.norm(gen_pch)

    if norm_s == 0 or norm_g == 0:
        similarity = 0.0
    else:
        similarity = dot / (norm_s * norm_g)

    if verbose:
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        print(f"  PCH Similarity (cosine): {similarity:.4f}")
        print(f"    Source PCH:    {dict(zip(note_names, [f'{v:.3f}' for v in source_pch]))}")
        print(f"    Generated PCH: {dict(zip(note_names, [f'{v:.3f}' for v in gen_pch]))}")

    return similarity


# ──────────────────────────────────────────────
# 지표 3: DOA (Degree of Arrangement)
# NeurIPS 2024 Structured Arrangement
# 트랙 간 음고 다양성 — 생성 파트가 소스와 얼마나 다른 음역을 커버하는지
# DOA = 1 - PCH 코사인 유사도 (차별성)
# ──────────────────────────────────────────────
def compute_doa(source_pm, generated_pm, target_programs, verbose=False):
    """
    DOA = 생성 파트가 소스와 얼마나 다른 음고 분포를 갖는지
    높을수록 창의적/다양한 편곡
    낮을수록 소스를 그대로 따라감
    """
    similarity = compute_pch_similarity(source_pm, generated_pm,
                                        target_programs, verbose=False)
    doa = 1.0 - similarity

    if verbose:
        print(f"  DOA (Degree of Arrangement): {doa:.4f}")
        print(f"    (1 - PCH similarity: {similarity:.4f})")

    return doa


# ──────────────────────────────────────────────
# 지표 4: Dissonance Rate
# AccoMontage2
# 생성 파트와 소스가 동시에 울릴 때 불협화 비율
# ──────────────────────────────────────────────
def compute_dissonance_rate(source_pm, generated_pm, target_programs,
                             resolution=0.125, verbose=False):
    """
    매 resolution 초마다 소스-생성 파트 간 음정 불협화도 계산
    0에 가까울수록 협화, 1에 가까울수록 불협화
    """
    end_time = max(source_pm.get_end_time(), generated_pm.get_end_time())
    times    = np.arange(0, end_time, resolution)

    total_dissonance = 0.0
    count            = 0

    for t in times:
        source_pitches = get_active_pitches_at(source_pm, t,
                                               exclude_programs=target_programs)
        gen_pitches    = set()
        for inst in generated_pm.instruments:
            if inst.is_drum:
                continue
            if inst.program in target_programs:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        gen_pitches.add(note.pitch % 12)

        if not source_pitches or not gen_pitches:
            continue

        # 모든 소스-생성 음정 쌍의 불협화도 평균
        pair_dissonances = []
        for sp in source_pitches:
            for gp in gen_pitches:
                interval = abs(sp - gp) % 12
                pair_dissonances.append(DISSONANCE_TABLE[interval])

        total_dissonance += np.mean(pair_dissonances)
        count            += 1

    dissonance_rate = total_dissonance / count if count > 0 else 0.0

    if verbose:
        print(f"  Dissonance Rate: {dissonance_rate:.4f}")
        print(f"    (0=완전협화, 1=완전불협화, 측정 프레임: {count})")

    return dissonance_rate


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def evaluate(source_path, generated_path, target_program=None, verbose=False):
    source_pm    = pretty_midi.PrettyMIDI(source_path)
    generated_pm = pretty_midi.PrettyMIDI(generated_path)

    # 타겟 프로그램 자동 감지
    if target_program is None:
        source_progs = {inst.program for inst in source_pm.instruments
                        if not inst.is_drum}
        gen_progs    = {inst.program for inst in generated_pm.instruments
                        if not inst.is_drum}
        target_programs = gen_progs - source_progs
        if not target_programs:
            target_programs = gen_progs
    else:
        target_programs = {target_program}

    print(f"\n📊 음악적 평가 지표")
    print(f"   소스:    {source_path}")
    print(f"   생성:    {generated_path}")
    print(f"   타겟 악기 프로그램: {target_programs}")
    print(f"   소스 길이: {source_pm.get_end_time():.1f}초")
    print()

    # 지표 계산
    chord_acc   = compute_chord_accuracy(source_pm, generated_pm,
                                         target_programs, verbose=verbose)
    pch_sim     = compute_pch_similarity(source_pm, generated_pm,
                                         target_programs, verbose=verbose)
    doa         = compute_doa(source_pm, generated_pm,
                              target_programs, verbose=verbose)
    dissonance  = compute_dissonance_rate(source_pm, generated_pm,
                                          target_programs, verbose=verbose)

    print(f"┌─────────────────────────────────────────────────┐")
    print(f"│  지표                          점수   (범위)    │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│  Chord Accuracy          {chord_acc:>8.4f}   (0~1, ↑좋음) │")
    print(f"│  PCH Similarity          {pch_sim:>8.4f}   (0~1, ↑좋음) │")
    print(f"│  DOA (창의성)            {doa:>8.4f}   (0~1, ↑다양) │")
    print(f"│  Dissonance Rate         {dissonance:>8.4f}   (0~1, ↓좋음) │")
    print(f"└─────────────────────────────────────────────────┘")

    return {
        "chord_accuracy":   chord_acc,
        "pch_similarity":   pch_sim,
        "doa":              doa,
        "dissonance_rate":  dissonance,
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        source_path    = args.source,
        generated_path = args.generated,
        target_program = args.target_program,
        verbose        = args.verbose,
    )