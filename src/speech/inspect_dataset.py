"""Read-only inspection of the RAVDESS speech audio dataset.

Scans datasets/speech recursively for .wav files, parses the 7-field file
names, detects duplicate copies, and reads audio headers. Nothing is
modified, moved or deleted.

File name format (from the dataset documentation, NOT stored in the audio):
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
The code-to-name tables below are therefore ASSUMPTIONS that must be checked
against the dataset's official description before training.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "datasets" / "speech"

FIELD_NAMES = (
    "modality",
    "vocal_channel",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
)
ACTOR_FOLDER_PATTERN = re.compile(r"^Actor_(\d+)$", re.IGNORECASE)
MAX_EXAMPLES_SHOWN = 5

ASSUMED_MODALITY = {1: "full-AV", 2: "video-only", 3: "audio-only"}
ASSUMED_CHANNEL = {1: "speech", 2: "song"}
ASSUMED_EMOTION = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
ASSUMED_INTENSITY = {1: "normal", 2: "strong"}


def parse_name(path: Path) -> dict[str, int] | None:
    """Parse a RAVDESS file name into its 7 integer fields, or None if invalid."""
    parts = path.stem.split("-")
    if len(parts) != len(FIELD_NAMES):
        return None
    try:
        values = [int(part) for part in parts]
    except ValueError:
        return None
    return dict(zip(FIELD_NAMES, values))


def top_level_folder(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "(root)"


def describe(code: int, names: dict[int, str]) -> str:
    return f"{code} {names.get(code, 'UNKNOWN')}"


def report_files_and_duplicates(root: Path) -> dict[str, Path]:
    """Print file/duplicate statistics; return one path per unique file name."""
    wav_paths = sorted(p for p in root.rglob("*.wav") if p.is_file())
    print(f"Folder searched      : {root}")
    print(f"Total .wav files     : {len(wav_paths)}")

    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in wav_paths:
        by_name[path.name].append(path)
    print(f"Unique file names    : {len(by_name)}")

    top_counts = Counter(top_level_folder(p, root) for p in wav_paths)
    actor_folder_files = sum(
        count for name, count in top_counts.items() if ACTOR_FOLDER_PATTERN.match(name)
    )
    print(f"Files under Actor_NN : {actor_folder_files}")
    for name, count in sorted(top_counts.items()):
        if not ACTOR_FOLDER_PATTERN.match(name):
            print(f"Files under '{name}' : {count}")

    duplicated = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    print(f"\nFile names that appear more than once: {len(duplicated)}")
    if duplicated:
        same_size = sum(
            1 for paths in duplicated.values() if len({p.stat().st_size for p in paths}) == 1
        )
        print(f"  of these, copies with identical byte size: {same_size}")
        print("  Examples:")
        for name, paths in list(duplicated.items())[:MAX_EXAMPLES_SHOWN]:
            print(f"    {name}")
            for path in paths:
                print(f"      {path.relative_to(root)}  ({path.stat().st_size} bytes)")
        print("  NOTE: statistics below use ONE copy per file name.")

    return {name: paths[0] for name, paths in by_name.items()}


def report_labels(unique_files: dict[str, Path], root: Path) -> None:
    parsed: dict[str, dict[str, int]] = {}
    unparsable: list[str] = []
    for name, path in unique_files.items():
        fields = parse_name(path)
        if fields is None:
            unparsable.append(name)
        else:
            parsed[name] = fields

    print(f"\n--- File name parsing ---")
    print(f"Parsed OK            : {len(parsed)}")
    print(f"Could not parse      : {len(unparsable)}")
    for name in unparsable[:MAX_EXAMPLES_SHOWN]:
        print(f"  {name}")

    def counter_for(field: str) -> Counter:
        return Counter(fields[field] for fields in parsed.values())

    print("\n--- Modality (names ASSUMED) ---")
    for code, count in sorted(counter_for("modality").items()):
        print(f"  {describe(code, ASSUMED_MODALITY):<16}{count:>6}")
    print("\n--- Vocal channel (names ASSUMED) ---")
    for code, count in sorted(counter_for("vocal_channel").items()):
        print(f"  {describe(code, ASSUMED_CHANNEL):<16}{count:>6}")

    print("\n--- Emotion (names ASSUMED, verify against dataset documentation) ---")
    emotion_counts = counter_for("emotion")
    for code, count in sorted(emotion_counts.items()):
        print(f"  {describe(code, ASSUMED_EMOTION):<16}{count:>6}")

    print("\n--- Emotion x Intensity ---")
    intensities = sorted(counter_for("intensity"))
    print(f"  {'emotion':<16}" + "".join(f"{describe(i, ASSUMED_INTENSITY):>12}" for i in intensities))
    cross = Counter((f["emotion"], f["intensity"]) for f in parsed.values())
    for code in sorted(emotion_counts):
        cells = "".join(f"{cross[(code, i)]:>12}" for i in intensities)
        print(f"  {describe(code, ASSUMED_EMOTION):<16}{cells}")

    print("\n--- Files per actor ---")
    actor_counts = counter_for("actor")
    line = ""
    for actor, count in sorted(actor_counts.items()):
        line += f"  A{actor:02d}:{count:>3}"
        if len(line) > 70:
            print(line)
            line = ""
    if line:
        print(line)
    print(f"  Distinct actors: {len(actor_counts)}")

    mismatches = []
    for name, fields in parsed.items():
        match = ACTOR_FOLDER_PATTERN.match(unique_files[name].parent.name)
        if match and int(match.group(1)) != fields["actor"]:
            mismatches.append(name)
    print(f"\nActor folder vs file-name actor mismatches: {len(mismatches)}")
    for name in mismatches[:MAX_EXAMPLES_SHOWN]:
        print(f"  {name} in {unique_files[name].parent.name}")


def report_audio_headers(unique_files: dict[str, Path]) -> None:
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    durations: list[float] = []
    failures: list[tuple[str, str]] = []

    for name, path in unique_files.items():
        try:
            info = sf.info(str(path))
        except Exception as exc:  # report every unreadable file instead of hiding it
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        sample_rates[info.samplerate] += 1
        channels[info.channels] += 1
        subtypes[info.subtype] += 1
        durations.append(info.frames / info.samplerate)

    print("\n--- Audio headers ---")
    print(f"Readable files       : {len(durations)}")
    print(f"Unreadable files     : {len(failures)}")
    for name, reason in failures[:MAX_EXAMPLES_SHOWN]:
        print(f"  {name}: {reason}")
    print(f"Sample rates (Hz)    : {dict(sample_rates)}")
    print(f"Channels             : {dict(channels)}")
    print(f"Sample format        : {dict(subtypes)}")
    if durations:
        mean = sum(durations) / len(durations)
        print(f"Duration (s)         : min={min(durations):.2f} "
              f"mean={mean:.2f} max={max(durations):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the RAVDESS speech dataset.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"ERROR: folder not found: {args.dir}", file=sys.stderr)
        return 1
    unique_files = report_files_and_duplicates(args.dir)
    if not unique_files:
        print("ERROR: no .wav files found.", file=sys.stderr)
        return 1
    report_labels(unique_files, args.dir)
    report_audio_headers(unique_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())