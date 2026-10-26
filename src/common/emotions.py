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


def _self_check() -> None:
    """Verify that the mapping tables are consistent with each other."""
    all_labels = set(FER2013_LABEL_NAMES)
    mapped = set(FER2013_TO_CANONICAL)
    excluded = set(FER2013_EXCLUDED)
    assert mapped.isdisjoint(excluded), "A label is both mapped and excluded"
    assert mapped | excluded == all_labels, "Some FER2013 label is unaccounted for"
    for label, name in FER2013_TO_CANONICAL.items():
        assert name in EMOTION_TO_INDEX, f"{name} is not a canonical emotion"
        assert FER2013_LABEL_NAMES[label] == name, f"Name mismatch for label {label}"
    covered = set(FER2013_TO_CANONICAL.values())
    assert covered == set(CANONICAL_EMOTIONS), "A canonical emotion has no FER2013 source"


def main() -> None:
    _self_check()
    print("Canonical emotions (model output order):")
    for index, name in enumerate(CANONICAL_EMOTIONS):
        print(f"  {index} -> {name}")
    print("\nFER2013 mapping:")
    for label in sorted(FER2013_LABEL_NAMES):
        target = fer2013_label_to_index(label)
        status = f"class {target}" if target is not None else "EXCLUDED"
        print(f"  {label} {FER2013_LABEL_NAMES[label]:<9} -> {status}")
    print("\nExclusion reasons:")
    for label, reason in FER2013_EXCLUDED.items():
        print(f"  {label}: {reason}")
    print("\nConsistency check passed.")


if __name__ == "__main__":
    main()