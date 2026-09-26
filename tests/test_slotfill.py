import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.slotfill import (  # noqa: E402
    assign_sentences_to_sections,
    fill_sections,
    is_grounded,
    worst_condition_per_section,
)
from src.pipeline.schema import SECTION_NAMES, SECTIONS, room_display_name  # noqa: E402
from src.pipeline.vision import CONDITION_LABELS  # noqa: E402

ROOM = "livingroom"
PROPERTY_ID = "8705"
TRANSCRIPT = (
    "The oak-toned laminate flooring shows scuffs at the entrance and minor chips "
    "in front of the bedroom and window. Both the walls and the ceiling have a "
    "fresh coat of matte snow-white acrylic paint."
)


# ---------------------------------------------------- FAKE LANGUAGE MODEL


class FakeFlan:
    """Gives the answer set for a section and "none" for the others"""

    def __init__(self, answers, room=ROOM):
        self.answers = answers
        self.display_room = room_display_name(room)
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        for section in SECTION_NAMES:
            name = SECTION_NAMES[section]
            if f"about the {name} in the {self.display_room}" in prompt:
                return self.answers.get(section, "none")
        raise AssertionError("prompt matched no known section")


# ---------------------------------------------------- HELPERS


def label(component, level):
    """The label of a component at a level"""
    return CONDITION_LABELS[component][level]


def _component(name, condition, confidence, flagged=False):
    return {
        "component": name,
        "condition": condition,
        "condition_confidence": confidence,
        "condition_distribution": {},
        "flagged_for_review": flagged,
        "n_observations": 1,
        "source_images": ["a.jpg"],
    }


def _fill(transcript, damage_analysis, answers, room=ROOM, property_id=PROPERTY_ID):
    fake = FakeFlan(answers, room)
    return fill_sections(transcript, damage_analysis, room, property_id, generate_fn=fake.ask)


# ---------------------------------------------------- TESTS


def test_grounded_passes():
    assert is_grounded("oak-toned laminate flooring shows scuffs at the entrance", TRANSCRIPT)


def test_invented_fails():
    assert not is_grounded("marble countertop with granite sink", TRANSCRIPT)


def test_worst_condition_wins():
    analysis = {
        "components": [
            _component("fixture", label("fixture", 0), 0.9),
            _component("plumbing fitting", label("plumbing fitting", 1), 0.35),
            _component("cabinetry", label("cabinetry", 3), 0.5, flagged=True),
        ]
    }
    other = worst_condition_per_section(analysis)["other"]
    assert other["value"] == label("cabinetry", 3)
    assert other["confidence"] == 0.5
    assert other["flagged_for_review"] is True


def test_old_label_gives_clear_error():
    analysis = {"components": [_component("floor", "an old label that no longer exists", 0.8)]}
    try:
        worst_condition_per_section(analysis)
    except ValueError as error:
        message = str(error)
        assert "an old label that no longer exists" in message
        assert "floor" in message
        return
    raise AssertionError("expected a ValueError")


def test_sentence_goes_to_both_sections():
    text = "Both the walls and the ceiling have a fresh coat of matte snow-white acrylic paint."
    sections = assign_sentences_to_sections(text)
    assert sections["wall"] == text
    assert sections["ceiling"] == text


def test_cabinet_door_is_not_the_door():
    # the door belongs to the vanity not the room
    text = "The cabinetry includes a white laminate vanity with a single door and an MDF shelf."
    sections = assign_sentences_to_sections(text)
    assert "door" not in sections
    assert sections["other"] == text


def test_fill_sections_shape():
    room_sections = _fill(TRANSCRIPT, None, {})
    assert room_sections["schema_version"] == 1
    assert room_sections["property_id"] == PROPERTY_ID
    assert room_sections["room"] == ROOM
    assert room_sections["confirmed"] is False
    assert set(room_sections["sections"]) == set(SECTIONS)
    for slot in room_sections["sections"].values():
        assert set(slot) == {"observed", "description", "condition"}
        assert set(slot["description"]) == {"value", "source"}
        assert set(slot["condition"]) == {"value", "source", "confidence", "flagged_for_review"}


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
