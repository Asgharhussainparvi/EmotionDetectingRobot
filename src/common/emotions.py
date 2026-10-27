"""Canonical emotion labels shared by facial, speech and fusion modules.

Every model must output probabilities in the order of CANONICAL_EMOTIONS.
Dataset-specific label ids are mapped to this list explicitly; classes that
are not mapped are excluded on purpose and the reason is recorded here.
"""

from __future__ import annotations

CANONICAL_EMOTIONS: tuple[str, ...] = ("Angry", "Fear", "Happy", "Sad", "Neutral")
EMOTION_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(CANONICAL_EMOTIONS)
}

# ---------------------------------------------------------------------------
# FER2013 (facial)
# ---------------------------------------------------------------------------
# FER2013 stores numeric labels only. This naming is the commonly documented
# order and was checked by eye on sample images (results/facial/label_preview.png).
FER2013_LABEL_NAMES: dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

FER2013_TO_CANONICAL: dict[int, str] = {
    0: "Angry",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    6: "Neutral",
}

# Excluded on purpose (never silently merged into another emotion).
FER2013_EXCLUDED: dict[int, str] = {
    1: "Disgust: only 547 of 35887 rows (1.5%), too few to learn reliably.",
    5: "Surprise: not one of the project's five target emotions; merging it "
    "into another class would mix semantically different emotions.",
}

# ---------------------------------------------------------------------------
# RAVDESS (speech)
# ---------------------------------------------------------------------------
# The emotion code is the 3rd field of the file name. The names below come
# from the dataset documentation, not from the audio files, so they should be
# checked against the RAVDESS download page.
RAVDESS_LABEL_NAMES: dict[int, str] = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised",
}

RAVDESS_TO_CANONICAL: dict[int, str] = {
    1: "Neutral",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fear",
}

# Excluded on purpose (never silently merged into another emotion).
RAVDESS_EXCLUDED: dict[int, str] = {
    2: "Calm: a distinct low-arousal emotion; merging it into Neutral would "
    "mix semantically different labels.",
    7: "Disgust: not one of the project's five target emotions.",
    8: "Surprised: not one of the project's five target emotions.",
}


def fer2013_label_to_index(label: int) -> int | None:
    """Return the canonical class index for a FER2013 label.

    Returns None when the label is excluded on purpose.
    Raises ValueError for labels that are not part of FER2013.
    """
    if label in FER2013_TO_CANONICAL:
        return EMOTION_TO_INDEX[FER2013_TO_CANONICAL[label]]
    if label in FER2013_EXCLUDED:
        return None
    raise ValueError(f"Unknown FER2013 label: {label}")


def ravdess_code_to_index(code: int) -> int | None:
    """Return the canonical class index for a RAVDESS emotion code.

    Returns None when the code is excluded on purpose.
    Raises ValueError for codes that are not part of RAVDESS.
    """
    if code in RAVDESS_TO_CANONICAL:
        return EMOTION_TO_INDEX[RAVDESS_TO_CANONICAL[code]]
    if code in RAVDESS_EXCLUDED:
        return None
    raise ValueError(f"Unknown RAVDESS emotion code: {code}")


def _check_mapping(
    dataset: str,
    all_labels: set[int],
    mapped: dict[int, str],
    excluded: dict[int, str],
) -> None:
    """Verify that one dataset's mapping tables are consistent."""
    assert set(mapped).isdisjoint(excluded), f"{dataset}: a label is both mapped and excluded"
    assert set(mapped) | set(excluded) == all_labels, f"{dataset}: a label is unaccounted for"
    for label, name in mapped.items():
        assert name in EMOTION_TO_INDEX, f"{dataset}: {name} is not a canonical emotion"
    assert set(mapped.values()) == set(CANONICAL_EMOTIONS), (
        f"{dataset}: a canonical emotion has no source label"
    )


def _self_check() -> None:
    """Verify that all mapping tables are consistent with each other."""
    _check_mapping(
        "FER2013", set(FER2013_LABEL_NAMES), FER2013_TO_CANONICAL, FER2013_EXCLUDED
    )
    for label, name in FER2013_TO_CANONICAL.items():
        assert FER2013_LABEL_NAMES[label] == name, f"FER2013 name mismatch for label {label}"
    _check_mapping(
        "RAVDESS", set(RAVDESS_LABEL_NAMES), RAVDESS_TO_CANONICAL, RAVDESS_EXCLUDED
    )


def _print_mapping(
    title: str,
    label_names: dict[int, str],
    to_index,
    excluded: dict[int, str],
) -> None:
    print(f"\n{title}:")
    for label in sorted(label_names):
        target = to_index(label)
        status = f"class {target}" if target is not None else "EXCLUDED"
        print(f"  {label} {label_names[label]:<10} -> {status}")
    print("  Exclusion reasons:")
    for label, reason in excluded.items():
        print(f"    {label}: {reason}")


def main() -> None:
    _self_check()
    print("Canonical emotions (model output order):")
    for index, name in enumerate(CANONICAL_EMOTIONS):
        print(f"  {index} -> {name}")
    _print_mapping("FER2013 mapping", FER2013_LABEL_NAMES, fer2013_label_to_index, FER2013_EXCLUDED)
    _print_mapping("RAVDESS mapping", RAVDESS_LABEL_NAMES, ravdess_code_to_index, RAVDESS_EXCLUDED)
    print("\nConsistency check passed.")


if __name__ == "__main__":
    main()