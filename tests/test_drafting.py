import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.drafting import (  # noqa: E402
    generate_stage_draft,
)
from src.pipeline.schema import SECTIONS  # noqa: E402

ROOM = "livingroom"
PROPERTY_ID = "8705"

# --------------------------------------------------- FAKE LANGUAGE MODELS
def returns_correct_door_sentence(prompt):
    return "The door is a varnished wooden door in good condition."

def returns_hallucinated_door_sentence(prompt):
    return "The door is a varnished wooden door with three marble panels."

def returns_correct_window_sentence(prompt):
    return "The window is a sliding window with a worn frame."

def returns_hallucination(prompt):
    return "The window has golden shutters and a balcony."

def returns_nonsense(prompt):
    return "The cat ate the lasagna."


class RecordingFlan:
    """A fake model that gives one fixed answer and remembers the prompt it was asked"""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.answer


# ---------------------------------------------------- HELPERS

def _slot(observed=False, description=None, condition=None, flagged=False):
    """One section"""
    return {
        "observed": observed,
        "description": {"value": description, "source": "transcript" if description else "none"},
        "condition": {
            "value": condition,
            "source": "vision" if condition else "none",
            "confidence": 0.5 if condition else None,
            "flagged_for_review": flagged,
        },
    }


def _room_sections(overrides, confirmed=True, room=ROOM, property_id=PROPERTY_ID):
    """All the sections of a room"""
    sections = {section: _slot() for section in SECTIONS}
    sections.update(overrides)
    return {
        "schema_version": 1,
        "property_id": property_id,
        "room": room,
        "confirmed": confirmed,
        "sections": sections,
    }


# ---------------------------------------------------- TESTS

def test_unconfirmed_raises():
    room_sections = _room_sections({}, confirmed=False)
    try:
        generate_stage_draft(room_sections)
    except ValueError as error:
        assert "confirmed" in str(error).lower()
        return
    raise AssertionError("expected a ValueError")


def test_grounded_sentence_kept():
    room_sections = _room_sections({"door": _slot(observed=True, description="varnished wooden door", condition="good condition")})
    
    draft = generate_stage_draft(room_sections, generate_fn=returns_correct_door_sentence)
    assert draft["sections"]["door"]["written_by"] == "model"
    assert draft["sections"]["door"]["text"] == "The door is a varnished wooden door in good condition."


def test_ungrounded_sentence_falls_back():
    room_sections = _room_sections({"door": _slot(observed=True, description="varnished wooden door", condition="good condition")})
    # "three" and "marble" should reject the answer
    draft = generate_stage_draft(room_sections, generate_fn=returns_hallucinated_door_sentence)
    assert draft["sections"]["door"]["written_by"] == "template"
    assert draft["sections"]["door"]["text"] == "Varnished wooden door. Good condition."


def test_flagged_gets_suffix():
    room_sections = _room_sections(
        {"window": _slot(observed=True, description="sliding window", condition="worn frame", flagged=True)}
    )
    grounded = generate_stage_draft(room_sections, generate_fn=returns_correct_window_sentence)
    ungrounded = generate_stage_draft(room_sections, generate_fn=returns_hallucination)
    assert grounded["sections"]["window"]["text"].endswith("[to be confirmed by the inspector]")
    assert ungrounded["sections"]["window"]["text"].endswith("[to be confirmed by the inspector]")


def test_prompt_is_not_form_like():
    room_sections = _room_sections({"door": _slot(observed=True, description="a door", condition="new")})
    fake = RecordingFlan("none")
    generate_stage_draft(room_sections, generate_fn=fake.ask)
    assert "Section:" not in fake.prompts[0]
    assert "Condition:" not in fake.prompts[0]
    assert "Details:" not in fake.prompts[0]


LONG_LABEL = "damaged ceiling with holes, cracks, broken panels, peeling paint or water stains"
def _photo_slot(level, description=None, flagged=False):
    """A section whose condition came from the photos"""
    entry = _slot(observed=True, description=description, condition=LONG_LABEL, flagged=flagged)
    entry["condition"]["level"] = level
    return entry


def test_long_label_not_in_report():
    room_sections = _room_sections({"ceiling": _photo_slot(3, description="a painted ceiling")})
    draft = generate_stage_draft(room_sections, generate_fn=returns_nonsense)
    assert draft["sections"]["ceiling"]["text"] == "A painted ceiling. Damaged."


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
