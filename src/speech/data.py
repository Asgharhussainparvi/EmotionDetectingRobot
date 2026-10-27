"""Build actor-independent train/validation/test file lists for RAVDESS.

Only the top-level Actor_NN folders are read. Any other folder (for example a
second, duplicated copy of the dataset) is ignored on purpose, so the same
recording can never appear in two splits.

Files whose emotion is excluded (see RAVDESS_EXCLUDED in
src/common/emotions.py) are counted and skipped, never relabelled.

This module does not load audio; it only prepares file paths and labels.

Split policy (by actor, so no speaker appears in more than one split):
    train      Actor_01 .. Actor_16
    validation Actor_17 .. Actor_20
    test       Actor_21 .. Actor_24
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.common.emotions import CANONICAL_EMOTIONS, ravdess_code_to_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "datasets" / "speech"

ACTOR_FOLDER_PATTERN = re.compile(r"^Actor_(\d{2})$")
NAME_FIELD_COUNT = 7
AUDIO_ONLY_MODALITY = 3
SPEECH_CHANNEL = 1

TRAIN_ACTORS = tuple(range(1, 17))
VALIDATION_ACTORS = tuple(range(17, 21))
TEST_ACTORS = tuple(range(21, 25))
EXPECTED_ACTOR_COUNT = 24


@dataclass(frozen=True)
class SpeechSample:
    """One audio file with its canonical label."""

    path: Path
    label: int  # canonical class index (see CANONICAL_EMOTIONS)
    actor: int
    intensity: int  # 1 = normal, 2 = strong (as documented for RAVDESS)


@dataclass(frozen=True)
class SpeechSplits:
    """Actor-independent splits plus bookkeeping counts."""

    train: list[SpeechSample]
    validation: list[SpeechSample]
    test: list[SpeechSample]
    excluded_files: int
    ignored_files_outside_actor_folders: int


def parse_file_name(path: Path) -> list[int]:
    """Parse a RAVDESS file name into its 7 integer fields."""
    parts = path.stem.split("-")
    if len(parts) != NAME_FIELD_COUNT:
        raise ValueError(f"Unexpected file name (need 7 fields): {path.name}")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Non-numeric field in file name: {path.name}") from exc


def _check_split_definition() -> None:
    train, validation, test = set(TRAIN_ACTORS), set(VALIDATION_ACTORS), set(TEST_ACTORS)
    if train & validation or train & test or validation & test:
        raise ValueError("An actor appears in more than one split")
    if len(train | validation | test) != EXPECTED_ACTOR_COUNT:
        raise ValueError("The splits do not cover exactly 24 actors")


def load_ravdess(root: Path = DEFAULT_DIR) -> SpeechSplits:
    """Scan the Actor_NN folders and return actor-independent splits."""
    if not root.is_dir():
        raise FileNotFoundError(f"Speech dataset folder not found: {root}")
    _check_split_definition()

    split_of_actor: dict[int, str] = {}
    for actor in TRAIN_ACTORS:
        split_of_actor[actor] = "train"
    for actor in VALIDATION_ACTORS:
        split_of_actor[actor] = "validation"
    for actor in TEST_ACTORS:
        split_of_actor[actor] = "test"

    splits: dict[str, list[SpeechSample]] = {"train": [], "validation": [], "test": []}
    excluded_files = 0
    actors_found: set[int] = set()
    seen_names: set[str] = set()

    for folder in sorted(root.iterdir()):
        match = ACTOR_FOLDER_PATTERN.match(folder.name)
        if not folder.is_dir() or match is None:
            continue
        actor_from_folder = int(match.group(1))
        if actor_from_folder not in split_of_actor:
            raise ValueError(f"Unexpected actor folder: {folder.name}")
        actors_found.add(actor_from_folder)

        for wav_path in sorted(folder.glob("*.wav")):
            modality, channel, emotion, intensity, _stmt, _rep, actor = parse_file_name(wav_path)
            if modality != AUDIO_ONLY_MODALITY or channel != SPEECH_CHANNEL:
                raise ValueError(f"Not audio-only speech: {wav_path.name}")
            if actor != actor_from_folder:
                raise ValueError(f"Actor mismatch between folder and file name: {wav_path}")
            if wav_path.name in seen_names:
                raise ValueError(f"Duplicate file name inside Actor folders: {wav_path.name}")
            seen_names.add(wav_path.name)

            label = ravdess_code_to_index(emotion)
            if label is None:
                excluded_files += 1
                continue
            sample = SpeechSample(path=wav_path, label=label, actor=actor, intensity=intensity)
            splits[split_of_actor[actor]].append(sample)

    if len(actors_found) != EXPECTED_ACTOR_COUNT:
        raise ValueError(f"Found {len(actors_found)} actor folders, expected {EXPECTED_ACTOR_COUNT}")

    ignored = sum(
        1
        for folder in root.iterdir()
        if folder.is_dir() and ACTOR_FOLDER_PATTERN.match(folder.name) is None
        for _ in folder.rglob("*.wav")
    )
    return SpeechSplits(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        excluded_files=excluded_files,
        ignored_files_outside_actor_folders=ignored,
    )


def main() -> int:
    try:
        splits = load_ravdess()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Excluded files (Calm/Disgust/Surprised): {splits.excluded_files}")
    print(f"Ignored .wav files outside Actor_NN folders: "
          f"{splits.ignored_files_outside_actor_folders}")
    for name, samples, actors in (
        ("train", splits.train, TRAIN_ACTORS),
        ("validation", splits.validation, VALIDATION_ACTORS),
        ("test", splits.test, TEST_ACTORS),
    ):
        print(f"\n[{name}] files={len(samples)} actors={actors[0]}..{actors[-1]} "
              f"({len(actors)} actors)")
        counts = Counter(sample.label for sample in samples)
        for index, emotion in enumerate(CANONICAL_EMOTIONS):
            print(f"    {index} {emotion:<8} {counts[index]:>4}")

    train_actors = {s.actor for s in splits.train}
    validation_actors = {s.actor for s in splits.validation}
    test_actors = {s.actor for s in splits.test}
    overlap = (train_actors & validation_actors) | (train_actors & test_actors) | (
        validation_actors & test_actors
    )
    print(f"\nActors shared between splits: {sorted(overlap) if overlap else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())