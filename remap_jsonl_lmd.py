"""
remap_jsonl.py
기존 JSONL의 INST= 토큰을 새 매핑 기준 representative로 교체.

- drop_list에 해당하는 INST= 노트 세트(7토큰) 통째로 제거
- 나머지는 representative INST= 토큰으로 교체
- BAR/KEY/METER 등 구조 토큰은 그대로 유지

Usage:
    python remap_jsonl.py \
        --input_dir  /data/tutti/lmd_jsonl/processed_data_final_pretrain_inter \
        --output_dir /data/tutti/lmd_jsonl/remapped
"""

import os
import json
import argparse
from tqdm import tqdm

# ──────────────────────────────────────────────
# 새 매핑
# ──────────────────────────────────────────────
DROP_SET = {47, 55, 109, 113, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127}

_GROUPING = {
    128: [128],
    0:   [0,1,2,3,4,5,6,7],
    16:  [16,17,18,19,20,21,22,23],
    12:  [8,9,10,11,12,13,14,15,112,114],
    25:  [24,25,26,27,28,31,45,46,104,105,106,107,108,110],
    30:  [29,30],
    33:  [32,33,34,35,36,37,38,39],
    40:  [40,41,42,43],
    73:  [68,69,70,71,72,73,74,75,77,78,79,111],
    65:  [64,65,66,67],
    81:  [80,81,82,83,84,85,86,87],
    56:  [56,57,58,59,60],
    48:  [44,48,49,50,51,52,53,54,61,62,63,76,88,89,90,
          91,92,93,94,95,96,97,98,99,100,101,102,103],
}
PROGRAM_TO_REP = {}
for rep, programs in _GROUPING.items():
    for p in programs:
        PROGRAM_TO_REP[p] = rep


# ──────────────────────────────────────────────
# Vocab
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
# 리매핑 핵심 함수
# 노트 구조: INST(0) ART(1) EXPR(2) TIME(3) PITCH(4) DUR(5) VEL(6)
# ──────────────────────────────────────────────
def remap_tokens(tokens, inst0_id, inst128_id, inst_id_to_rep, drop_prog_set):
    NOTE_LEN = 7
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if inst0_id <= tok <= inst128_id:
            prog = tok - inst0_id

            if i + NOTE_LEN <= len(tokens):
                if prog in drop_prog_set:
                    # drop → 7토큰 통째로 건너뜀
                    i += NOTE_LEN
                    continue
                else:
                    # representative로 INST= 교체, 나머지 6토큰 그대로
                    rep_id = inst_id_to_rep.get(tok, tok)
                    result.append(rep_id)
                    result.extend(tokens[i+1 : i+NOTE_LEN])
                    i += NOTE_LEN
                    continue
            else:
                # 끝부분 잘린 노트 → drop
                i += 1
                continue

        result.append(tok)
        i += 1

    return result


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="/data/tutti/lmd_jsonl/processed_data_final_pretrain_inter")
    parser.add_argument("--output_dir", default="/data/tutti/lmd_jsonl/remapped")
    parser.add_argument("--splits",     nargs="+", default=["train_pretrain","val_pretrain"])
    parser.add_argument("--min_tokens", type=int,  default=100,
                        help="리매핑 후 이 값 미만 청크는 버림 (기본 100)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vocab      = build_v5_vocab()
    inst0_id   = vocab["INST=0"]
    inst128_id = vocab["INST=128"]

    inst_id_to_rep = {
        vocab[f"INST={p}"]: vocab[f"INST={rep}"]
        for p, rep in PROGRAM_TO_REP.items()
    }

    print(f"INST=0 token ID : {inst0_id}")
    print(f"drop programs   : {sorted(DROP_SET)}")
    print(f"remap entries   : {len(inst_id_to_rep)}")

    for split in args.splits:
        in_path  = os.path.join(args.input_dir,  f"{split}.jsonl")
        out_path = os.path.join(args.output_dir, f"{split}.jsonl")

        if not os.path.exists(in_path):
            print(f"[WARN] {in_path} 없음, 스킵")
            continue

        with open(in_path, "rb") as f:
            total = sum(1 for _ in f)

        kept = dropped = 0
        with open(in_path) as fin, open(out_path, "w") as fout:
            for line in tqdm(fin, total=total, desc=split):
                line = line.strip()
                if not line:
                    continue
                obj        = json.loads(line)
                new_tokens = remap_tokens(
                    obj["tokens"], inst0_id, inst128_id,
                    inst_id_to_rep, DROP_SET)

                if len(new_tokens) < args.min_tokens:
                    dropped += 1
                    continue

                fout.write(json.dumps({"tokens": new_tokens}) + "\n")
                kept += 1

        size_gb = os.path.getsize(out_path) / 1e9
        print(f"[{split}] kept={kept:,} dropped={dropped:,} → {size_gb:.2f} GB")

    print("\n✅ 완료")


if __name__ == "__main__":
    main()