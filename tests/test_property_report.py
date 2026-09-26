import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx.enum.text import WD_COLOR_INDEX  # noqa: E402

from src.pipeline.property_report import generate_property_report  # noqa: E402


# ---------------------------------------------------- HELPERS


def _entry(text, flagged=False):
    """One section of a draft"""
    return {"text": text, "written_by": "model", "flagged_for_review": flagged}


def _draft(room, sections, omitted=None):
    """A draft for one room"""
    if omitted is None:
        omitted = []
    return {"property_id": "8705", "room": room, "sections": sections, "omitted_sections": omitted}


def _find_paragraph(document, wanted):
    """The first paragraph containing wanted"""
    for paragraph in document.paragraphs:
        if wanted in paragraph.text:
            return paragraph
    raise AssertionError(f"no paragraph contains {wanted!r}")


# ---------------------------------------------------- TESTS


def test_room_heading():
    draft = _draft("livingroom", {"door": _entry("A wooden door.")})
    document = generate_property_report("8705", [draft])
    heading = _find_paragraph(document, "1.")
    assert heading.text == "1. LIVING ROOM:"
    run = heading.runs[0]
    assert run.bold is True
    assert run.underline is True


def test_bold_label():
    draft = _draft("bathroom", {"door": _entry("A metal door.")})
    document = generate_property_report("8705", [draft])
    field_paragraph = _find_paragraph(document, "A metal door.")
    label_run, content_run = field_paragraph.runs
    assert label_run.text == "Door: "
    assert label_run.bold is True
    assert content_run.text == "A metal door."
    assert not content_run.bold


def test_flagged_line_is_yellow():
    draft = _draft(
        "livingroom",
        {
            "door": _entry("A door."),
            "floor": _entry("A floor. [to be confirmed by the inspector]", flagged=True),
        },
    )
    document = generate_property_report("8705", [draft])
    door = _find_paragraph(document, "Door:")
    floor = _find_paragraph(document, "Floor:")

    for run in door.runs:
        assert run.font.highlight_color is None

    # the label and the text are both highlighted
    assert len(floor.runs) == 2
    for run in floor.runs:
        assert run.font.highlight_color == WD_COLOR_INDEX.YELLOW


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
