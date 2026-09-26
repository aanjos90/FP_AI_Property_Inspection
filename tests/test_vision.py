import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.vision import (  # noqa: E402
    CONDITION_LABELS,
    CONFIDENCE_FLAG_THRESHOLD,
    reshape_to_damage_analysis,
)

FLOOR = CONDITION_LABELS["floor"]


# ---------------------------------------------------- HELPERS


def _floor_reading(condition: str, confidence: float) -> dict:
    """The reading for one photo of a floor"""
    labels = FLOOR
    # the other three conditions share the rest of the score
    scores = {}
    for label in labels:
        scores[label] = (1.0 - confidence) / 3
    scores[condition] = confidence
    return {
        "component": "floor",
        "component_confidence": 0.95,
        "condition": condition,
        "condition_confidence": confidence,
        "component_scores": {"floor": 0.95},
        "condition_scores": scores,
    }


# ---------------------------------------------------- TESTS


def test_damage_analysis_shape():
    raw = {"a.jpg": _floor_reading(FLOOR[1], 0.82)}

    out = reshape_to_damage_analysis(raw, "living_room", "8705")

    assert set(out) == {"room", "property_id", "components"}
    assert set(out["components"][0]) == {
        "component",
        "condition",
        "condition_confidence",
        "condition_distribution",
        "flagged_for_review",
        "n_observations",
        "level_counts",
        "source_images",
    }


def test_disagreeing_photos_flagged():
    raw = {
        "1.jpg": _floor_reading(FLOOR[3], 0.41),
        "2.jpg": _floor_reading(FLOOR[1], 0.37),
        "3.jpg": _floor_reading(FLOOR[0], 0.38),
        "4.jpg": _floor_reading(FLOOR[0], 0.59),
        "5.jpg": _floor_reading(FLOOR[0], 0.45),
        "6.jpg": _floor_reading(FLOOR[0], 0.54),
    }

    out = reshape_to_damage_analysis(raw, "living_room", "8705")

    comp = out["components"][0]
    assert comp["flagged_for_review"] is True
    assert comp["condition"] == FLOOR[3]
    assert comp["condition_confidence"] == 0.41
    assert comp["n_observations"] == 6
    # the distribution kept is the deciding photo one
    assert comp["condition_distribution"][FLOOR[3]] == 0.41


def test_weak_photo_flagged():
    weak = CONFIDENCE_FLAG_THRESHOLD - 0.05
    raw = {"a.jpg": _floor_reading(FLOOR[1], weak)}

    out = reshape_to_damage_analysis(raw, "living_room", "8705")

    comp = out["components"][0]
    assert comp["flagged_for_review"] is True
    assert comp["n_observations"] == 1


def test_level_counts():
    # 3 photos at level 0, 1 at level 1 and 2 at level 3
    raw = {
        "1.jpg": _floor_reading(FLOOR[0], 0.6),
        "2.jpg": _floor_reading(FLOOR[0], 0.6),
        "3.jpg": _floor_reading(FLOOR[0], 0.6),
        "4.jpg": _floor_reading(FLOOR[1], 0.6),
        "5.jpg": _floor_reading(FLOOR[3], 0.5),
        "6.jpg": _floor_reading(FLOOR[3], 0.4),
    }
    comp = reshape_to_damage_analysis(raw, "living_room", "8705")["components"][0]
    assert comp["level_counts"] == [3, 1, 0, 2]
    assert sum(comp["level_counts"]) == comp["n_observations"]
    # the summary still reports the worst
    assert comp["condition"] == FLOOR[3]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
