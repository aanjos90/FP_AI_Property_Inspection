import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.followups import (  # noqa: E402
    MAX_QUESTIONS,
    REASON_LOW_CONFIDENCE,
    apply_answers,
    generate_followups,
)
from src.pipeline.schema import SECTIONS  # noqa: E402

ROOM = "livingroom"


# ---------------------------------------------------- HELPERS


def _slot(observed=False, description_value=None, condition_value=None, flagged=False, level_counts=None):
    """One section"""
    condition = {
        "value": condition_value,
        "source": "vision" if condition_value is not None else "none",
        "confidence": 0.5 if condition_value is not None else None,
        "flagged_for_review": flagged,
    }
    if level_counts is not None:
        condition["level_counts"] = level_counts
    return {
        "observed": observed,
        "description": {
            "value": description_value,
            "source": "transcript" if description_value is not None else "none",
        },
        "condition": condition,
    }


def _room_sections(overrides, room=ROOM):
    """All the sections of a room"""
    sections = {section: _slot() for section in SECTIONS}
    sections.update(overrides)
    return {"schema_version": 1, "property_id": "8705", "room": room, "confirmed": False, "sections": sections}


def _values(followups, key):
    """One field from every question"""
    return [followup[key] for followup in followups]


def _questions_and_sections():
    """Sections with a flagged floor and an undescribed door"""
    room_sections = _room_sections(
        {
            "floor": _slot(observed=True, condition_value="damaged flooring, chips stains or cracks", flagged=True),
            "door": _slot(observed=True, condition_value="good condition with minor wear"),
        }
    )
    return room_sections, generate_followups(room_sections)


def _flagged_question(level_counts):
    """The question for a flagged window with the given photo counts"""
    room_sections = _room_sections(
        {
            "window": _slot(
                observed=True,
                description_value="a window",
                condition_value="worn window",
                flagged=True,
                level_counts=level_counts,
            )
        }
    )
    return generate_followups(room_sections)[0]["question"]


# ---------------------------------------------------- TESTS


def test_flagged_come_first():
    # "other" comes after "door" in SECTIONS but flagged goes first
    room_sections = _room_sections(
        {
            "door": _slot(observed=True, condition_value="new or like-new condition"),
            "other": _slot(
                observed=True,
                description_value="cabinets look worn",
                condition_value="damaged condition",
                flagged=True,
            ),
        }
    )
    followups = generate_followups(room_sections)
    assert _values(followups, "section") == ["other", "door"]


def test_questions_are_capped():
    # 8 flagged sections but only 6 questions are kept
    overrides = {}
    for section in SECTIONS:
        overrides[section] = _slot(observed=True, condition_value="damaged", flagged=True)
    followups = generate_followups(_room_sections(overrides))
    assert len(followups) == MAX_QUESTIONS
    for followup in followups:
        assert followup["reason"] == REASON_LOW_CONFIDENCE


def test_question_shows_photo_evidence():
    question = _flagged_question([8, 3, 1, 2])
    assert question == (
        "WINDOW:\n"
        "Photos unclear or disagree with each other.\n"
        "Of 14 photos, 8 look new, 3 look good, 1 looks worn and 2 look damaged."
    )


def test_answer_clears_flag():
    room_sections, questions = _questions_and_sections()
    confirmed = apply_answers(room_sections, questions, {"floor_condition": "it's actually fine, just dusty"})
    slot = confirmed["sections"]["floor"]
    assert slot["condition"] == {
        "value": "it's actually fine, just dusty",
        "source": "answer",
        "confidence": None,
        "flagged_for_review": False,
    }
    assert slot["observed"] is True


def test_empty_answer_ignored():
    room_sections, questions = _questions_and_sections()
    for answer in [None, "none", "n/a", ""]:
        confirmed = apply_answers(room_sections, questions, {"floor_condition": answer})
        assert confirmed["sections"]["floor"]["condition"] == room_sections["sections"]["floor"]["condition"]


def test_apply_answers_keeps_original():
    room_sections, questions = _questions_and_sections()
    before = dict(room_sections["sections"]["floor"]["condition"])
    apply_answers(room_sections, questions, {"floor_condition": "fine actually"}, confirm=True)
    assert room_sections["sections"]["floor"]["condition"] == before
    assert room_sections["confirmed"] is False


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
