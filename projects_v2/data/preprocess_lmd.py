import os
import json
import bisect
import multiprocessing
import pretty_midi
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from vocab_v6 import build_v6_vocab

GENRE_MAP = {
    "pop": "pop", "dance": "pop", "world": "pop", "latin": "pop",
    "country": "pop", "popfolk": "pop",
    "electronic": "electronic", "techno": "electronic", "trance": "electronic",
    "house": "electronic", "synthpop": "electronic",
    "classical": "classical", "soundtrack": "classical", 
    "orchestral": "classical", "newage": "classical",
    "rock": "rock", "instrumentalrock": "rock", "alternative": "rock",
    "poprock": "rock", "punkrock": "rock", "metal": "rock",
}

def map_genre(genre_list):
    for g in genre_list:
        g_lower = g.lower().strip()
        if g_lower in GENRE_MAP:
            return GENRE_MAP[g_lower]
    return "other"

def get_key_tok_at(bar_time, key_changes, key_times, vocab):
    k_idx = bisect.bisect_right(key_times, bar_time) - 1
    if 0 <= k_idx < len(key_changes):
        ks = key_changes[k_idx]
        root = ks.key_number % 12
        mode = "maj" if ks.key_number < 12 else "min"
        roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        FLAT_TO_SHARP = {"Db":"C#", "Eb":"D#", "Fb":"E", "Gb":"F#", "Ab":"G#", "Bb":"A#", "Cb":"B"}
        rname = FLAT_TO_SHARP.get(roots[root], roots[root])
        return vocab.get(f"KEY_{rname}:{mode}", vocab["KEY_NONE"])
    return vocab["KEY_NONE"]

def get_meter_tok_at(bar_time, ts_changes, ts_times, vocab):
    ts_idx = max(0, bisect.bisect_right(ts_times, bar_time) - 1)
    if ts_idx < len(ts_changes):
        ts = ts_changes[ts_idx]
        mkey = f"{ts.numerator}:{ts.denominator}"
        return vocab.get(f"METER_{mkey}", vocab["METER_OTHER"])
    return vocab["METER_OTHER"]

def get_beats_at(bar_time, ts_changes, ts_times):
    ts_idx = max(0, bisect.bisect_right(ts_times, bar_time) - 1)
    if ts_idx < len(ts_changes):
        return ts_changes[ts_idx].numerator
    return 4

def generate_bars(pm, timeline, exclude_inst_idx, vocab, key_changes, key_times, ts_changes, ts_times):
    res = pm.resolution
    final_beats = ts_changes[-1].numerator if ts_changes else 4
    max_tick = pm.time_to_tick(pm.get_end_time())
    max_bar = int(max_tick // (res * final_beats))
    
    accumulated_ticks = 0
    bars = []
    
    for bar_idx in range(max_bar + 1):
        bar_time = pm.tick_to_time(accumulated_ticks)
        beats = get_beats_at(bar_time, ts_changes, ts_times)
        key_tok = get_key_tok_at(bar_time, key_changes, key_times, vocab)
        meter_tok = get_meter_tok_at(bar_time, ts_changes, ts_times, vocab)
        
        bar_toks = [vocab["BAR_START"]]
        
        if bar_idx in timeline:
            bar_data = timeline[bar_idx]
            all_notes = []
            for inst_idx, notes in bar_data.items():
                if inst_idx == exclude_inst_idx:
                    continue
                for (time_val, p, pitch_val, dur_val, vel_val) in notes:
                    all_notes.append((time_val, p, pitch_val, dur_val, vel_val))
            
            density = min(5, max(1, len(all_notes) // 4))
            bar_toks += [key_tok, meter_tok, vocab[f"DENSITY_{density}"]]
            
            all_notes.sort(key=lambda x: (x[0], x[1]))
            
            for (time_val, p, pitch_val, dur_val, vel_val) in all_notes:
                bar_toks += [
                    vocab[f"INST={p}"],
                    vocab[f"TIME={time_val}"],
                    vocab[f"PITCH={pitch_val}"],
                    vocab[f"DUR={dur_val}"],
                    vocab[f"VEL={vel_val}"]
                ]
        else:
            bar_toks += [key_tok, meter_tok, vocab["DENSITY_1"]]
            
        bar_toks.append(vocab["BAR_END"])
        bars.append(bar_toks)
        accumulated_ticks += res * beats
        
    return bars

def trim_bars_from_start(tokens, budget, bar_start_id):
    if len(tokens) <= budget: return tokens
    target_start = len(tokens) - budget
    for i in range(target_start, len(tokens)):
        if tokens[i] == bar_start_id:
            return tokens[i:]
    return tokens[-budget:]

def trim_bars_from_end(tokens, budget, bar_start_id):
    if len(tokens) <= budget: return tokens
    for i in range(budget, -1, -1):
        if tokens[i] == bar_start_id:
            return tokens[:i]
    return tokens[:budget]

def trim_to_budget(full_seq, header, pre_tokens, suf_tokens, mid_tokens, max_len, vocab):
    if len(full_seq) <= max_len:
        return full_seq
    
    fixed_size = len(header) + 1 + 1 + len(mid_tokens)
    available = max_len - fixed_size
    if available <= 0:
        return None
    
    pre_budget = int(available * 2 / 3)
    suf_budget = available - pre_budget
    
    bar_start_id = vocab["BAR_START"]
    trimmed_pre = trim_bars_from_start(pre_tokens[1:], pre_budget, bar_start_id)
    trimmed_pre = [vocab["<PRE>"]] + trimmed_pre
    
    trimmed_suf = trim_bars_from_end(suf_tokens[1:], suf_budget, bar_start_id)
    trimmed_suf = [vocab["<SUF>"]] + trimmed_suf
    
    return header + trimmed_pre + trimmed_suf + mid_tokens

def build_fim_chunk(bars_full, bars_no_target, target_program, chunk_start, n_bars, genre_tok, vocab):
    mid_start = chunk_start
    mid_end = min(chunk_start + 8, n_bars)
    pre_start = max(0, mid_start - 8)
    pre_end = mid_start
    suf_start = mid_end
    suf_end = min(mid_end + 8, n_bars)
    
    header = [vocab["PIECE_START"], genre_tok, vocab[f"TARGET_{target_program}"]]
    
    pre_tokens = [vocab["<PRE>"]]
    for b in range(pre_start, pre_end): pre_tokens += bars_full[b]
        
    suf_tokens = [vocab["<SUF>"]]
    for b in range(suf_start, suf_end): suf_tokens += bars_no_target[b]
        
    mid_tokens = [vocab["<MID>"]]
    for b in range(mid_start, mid_end): mid_tokens += bars_full[b]
        
    mid_tokens.append(vocab["EOS"])
    full_seq = header + pre_tokens + suf_tokens + mid_tokens
    
    return trim_to_budget(full_seq, header, pre_tokens, suf_tokens, mid_tokens, 8192, vocab)

_global_vocab = None
def _init_worker(v):
    global _global_vocab
    _global_vocab = v

def _worker(entry):
    try:
        midi_path = entry["path"]
        genre_str = map_genre(entry.get("genre", []))
        genre_tok = _global_vocab[f"GENRE_{genre_str.upper()}"]
        
        pm = pretty_midi.PrettyMIDI(midi_path)
        res = pm.resolution
        
        key_changes = sorted(pm.key_signature_changes, key=lambda x: x.time)
        key_times   = [k.time for k in key_changes]
        ts_changes  = sorted(pm.time_signature_changes, key=lambda x: x.time)
        ts_times    = [t.time for t in ts_changes]
        tempo_change_times, tempos = pm.get_tempo_changes()
        
        timeline = defaultdict(lambda: defaultdict(list))
        for inst_idx, inst in enumerate(pm.instruments):
            p = 128 if inst.is_drum else inst.program
            notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
            
            for note in notes:
                beats_per_bar = get_beats_at(note.start, ts_changes, ts_times)
                # Safeguard against 0 beats_per_bar
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
                
        bars_full = generate_bars(pm, timeline, None, _global_vocab, key_changes, key_times, ts_changes, ts_times)
        n_bars = len(bars_full)
        target_programs = {40,41,42,43, 56,57,58,60, 64,65,66,67, 68,69,70,71,73}
        
        samples = []
        for inst_idx, inst in enumerate(pm.instruments):
            if inst.is_drum or inst.program not in target_programs:
                continue
                
            target_program = inst.program
            bars_no_target = generate_bars(pm, timeline, inst_idx, _global_vocab, key_changes, key_times, ts_changes, ts_times)
            
            for chunk_start in range(0, n_bars - 7, 8):
                sample_tokens = build_fim_chunk(bars_full, bars_no_target, target_program, chunk_start, n_bars, genre_tok, _global_vocab)
                if sample_tokens:
                    samples.append({
                        "tokens": [int(t) for t in sample_tokens],
                        "target_program": int(target_program),
                        "genre": str(genre_str),
                        "md5": str(entry.get("md5", "")),
                        "chunk_start": int(chunk_start)
                    })
        return samples
    except Exception:
        return []

def run_preprocess(jsonl_in, save_dir, vocab):
    os.makedirs(save_dir, exist_ok=True)
    
    entries = []
    with open(jsonl_in, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
                
    train_entries, val_entries = train_test_split(entries, test_size=0.05, random_state=42)
    print(f"Total entries: {len(entries)} (Train: {len(train_entries)}, Val: {len(val_entries)})")
    
    def process_split(split_entries, split_name):
        out_path = os.path.join(save_dir, f"{split_name}.jsonl")
        total_chunks = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=_init_worker, initargs=(vocab,)) as pool:
            with open(out_path, 'w', encoding='utf-8') as f_out:
                for results in tqdm(pool.imap_unordered(_worker, split_entries), total=len(split_entries), desc=split_name):
                    for r in results:
                        f_out.write(json.dumps(r) + '\n')
                        total_chunks += 1
        print(f"Saved {total_chunks} chunks to {out_path}")
        
    # process_split(val_entries, "val") # Already completed
    process_split(train_entries, "train")

if __name__ == "__main__":
    vocab = build_v6_vocab()
    INPUT_JSONL = "/data/tutti/lmd_v4/filtered_17inst/filtered_17inst.jsonl"
    OUTPUT_DIR = "/data/tutti/Gemma4_Dataset/" # using path from spec
    
    print("Starting preprocessing...")
    run_preprocess(INPUT_JSONL, OUTPUT_DIR, vocab)
