"""Write the draft of the report for one room"""

from src.pipeline.language import generate
from src.pipeline.schema import SECTION_NAMES, SECTIONS, condition_wording
from src.pipeline.slotfill import is_grounded

# Every important word of a written sentence must come from the section notes
FAITHFULNESS_MIN_OVERLAP = 1.0

DRAFT_MAX_NEW_TOKENS = 128  # the longest sentence the model may write

PROMPT_TEMPLATE = (
    "Write two formal sentences in the present tense for a property inspection report "
    "about the {section_name}. Known facts: {facts}. Use only these facts.\n"
    "Sentence:"
)

# Added to the end of a sentence whose condition is still uncertain.
FLAGGED_SUFFIX = " [to be confirmed by the inspector]"


def _ask_flan(prompt: str) -> str:
    return generate(prompt, max_new_tokens=DRAFT_MAX_NEW_TOKENS)


def _build_prompt(section_name: str, description: str, condition: str) -> str:
    """Make the prompt for one section"""
    known = []
    if description:
        known.append(description)
    if condition:
        known.append(condition)
    return PROMPT_TEMPLATE.format(section_name=section_name, facts="; ".join(known))


def _as_sentence(text: str) -> str:
    """Make a piece of text into a sentence"""
    text = text.strip().rstrip(".;, ") # clear from full stops, commas, semicolons, and spaces at the end
    if text == "":
        return ""
    return text[0].upper() + text[1:] + "."


def _plain_text(description: str, condition: str) -> str:
    """Backup sentence from the description and condition"""
    sentences = []
    if description:
        sentences.append(_as_sentence(description))
    if condition:
        sentences.append(_as_sentence(condition))
    return " ".join(sentences)


def generate_stage_draft(room_sections: dict, generate_fn=_ask_flan) -> dict:
    """Write the draft for one room with sections confirmed by the inspector"""
    if room_sections["confirmed"] is not True:
        raise ValueError("Sections not confirmed by the inspector")

    sections = {}
    omitted_sections = []

    for section in SECTIONS:
        entry = room_sections["sections"][section]
        if not entry["observed"]:
            omitted_sections.append(section)
            continue

        section_name = SECTION_NAMES[section]

        description = entry["description"]["value"] or ""
        condition = condition_wording(entry["condition"])

        # Nothing to write about this section, omit from the report
        if description == "" and condition == "":
            omitted_sections.append(section)
            continue

        prompt = _build_prompt(section_name, description, condition)
        model_text = generate_fn(prompt).strip()

        # The words the sentence may use
        allowed_text = f"{section_name} {description} {condition}"

        if model_text != "" and is_grounded(model_text, allowed_text, min_overlap=FAITHFULNESS_MIN_OVERLAP):
            text = model_text
            written_by = "model"
        else:
            text = _plain_text(description, condition)
            written_by = "template"

        flagged = entry["condition"]["flagged_for_review"]
        if flagged:
            text = text + FLAGGED_SUFFIX

        sections[section] = {"text": text, "written_by": written_by, "flagged_for_review": flagged}

    return {
        "property_id": room_sections["property_id"],
        "room": room_sections["room"],
        "sections": sections,
        "omitted_sections": omitted_sections,
    }
