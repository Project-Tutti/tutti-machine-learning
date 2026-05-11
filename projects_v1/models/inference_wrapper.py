import argparse
import importlib.util
import os
import pathlib
import sys


def load_tutti_inference_module():
    root = pathlib.Path(__file__).resolve().parent.parent
    inference_path = root / "tutti-backend-ai" / "app" / "services" / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"Cannot find inference module at {inference_path}")

    spec = importlib.util.spec_from_file_location("tutti_backend_inference", str(inference_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_target_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_target(target: str, config: dict):
    if target is None:
        return None

    target = target.strip()
    if target.isdigit():
        return int(target)

    norm = normalize_target_name(target)
    for key, value in config.items():
        if normalize_target_name(key) == norm:
            return value["program"]

    raise ValueError(f"Unknown target instrument: {target}")


def build_reverse_vocab(vocab: dict) -> dict:
    return {v: k for k, v in vocab.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wrapper for tutti-backend-ai inference.run_arrangement"
    )
    parser.add_argument("--song", required=True, help="Input MIDI file path")
    parser.add_argument("--target", required=True,
                        help="Target instrument name or MIDI program number")
    parser.add_argument("--genre", default="UNKNOWN",
                        help="Genre tag used for tokenization (default: UNKNOWN)")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint directory containing model weights")
    parser.add_argument("--output", default=None,
                        help="Output MIDI file path. Defaults to <song>_ai.mid")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                        help="Torch device to use (default: cuda if available else cpu)")
    parser.add_argument("--pitch-min", type=int, default=None,
                        help="Optional minimum pitch for generation")
    parser.add_argument("--pitch-max", type=int, default=None,
                        help="Optional maximum pitch for generation")
    parser.add_argument("--original-song", default=None,
                        help="Original MIDI file path for MIDI append preservation. If omitted, uses --song")
    parser.add_argument("--instrument-name", default=None,
                        help="Optional instrument name used in output MIDI track metadata")
    parser.add_argument("--midi-program", type=int, default=None,
                        help="Optional program number to write into the output MIDI track")
    return parser.parse_args()


def main():
    args = parse_args()
    service = load_tutti_inference_module()

    if args.original_song is None:
        args.original_song = args.song

    target_prog = resolve_target(args.target, service.TARGET_CONFIG)
    vocab = service.build_v5_vocab()
    vocab_r = build_reverse_vocab(vocab)

    model = service.load_model(args.ckpt, len(vocab), vocab, args.device)

    output_path = args.output
    if output_path is None:
        base = pathlib.Path(args.song).with_suffix("")
        output_path = str(base) + f"_ai_{target_prog}.mid"

    print(f"Loading model from: {args.ckpt}")
    print(f"Generating instrument program: {target_prog}")
    print(f"Input MIDI: {args.song}")
    print(f"Output MIDI: {output_path}")

    result_path = service.run_arrangement(
        song_path=args.song,
        target_prog=target_prog,
        genre=args.genre,
        temperature=1.0,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        output_path=output_path,
        model=model,
        vocab=vocab,
        vocab_r=vocab_r,
        device=args.device,
        original_song_path=args.original_song,
        actual_instrument_name=args.instrument_name,
        actual_midi_program=args.midi_program,
    )

    print(f"Saved generated MIDI to: {result_path}")


if __name__ == "__main__":
    main()
