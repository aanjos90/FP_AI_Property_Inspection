"""Fill in the report sections for one room"""

import re

from src.pipeline.language import generate
from src.pipeline.schema import (
    AMBIGUOUS_SECTIONS,
    COMPONENT_TO_SECTION,
    CONDITION_LABELS,
    SECTION_KEYWORDS,
    SECTION_NAMES,
    SECTIONS,
    room_display_name,
)

# How many words the model may use for one answer
SECTION_MAX_NEW_TOKENS = 225

# The question that goes into the language model
PROMPT_TEMPLATE = (
    'Read the text and answer the question. If the text does not say, answer "none".\n'
    "Question: What does the speaker say about the {section_name} in the {room}?\n"
    "Text: {text}\n"
    "Answer:"
)

# An answer is only trusted if at least this much comes from the text it came from
# This catches hallucinations
GROUNDING_MIN_OVERLAP = 0.6

NONE_ANSWERS = ["", "none", "n/a", "na", "no"]

STOP_WORDS = [
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "has", "have",
    "had", "also", "some", "both", "while", "which", "then", "so",
]

# Split the text into sentences
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def normalise_answer(answer: str):
    text = answer.strip()
    without_punctuation = text.strip(".,;:!?\"' ").lower()
    if without_punctuation in NONE_ANSWERS:
        return None
    return text

def content_words(text: str) -> list:
    every_word = re.findall(r"[a-z0-9]+", text.lower())
    important = []
    for word in every_word:
        if word not in STOP_WORDS:
            important.append(word)
    return important

def is_grounded(answer: str, text: str, min_overlap: float = GROUNDING_MIN_OVERLAP) -> bool:
    """Counts how many words in the answer also appear in the text"""
    answer_words = content_words(answer)
    if len(answer_words) == 0:
        return False

    text_words = content_words(text)
    found = 0
    for word in answer_words:
        if word in text_words:
            found = found + 1

    return found / len(answer_words) >= min_overlap

def _severity(component: dict) -> int:
    """Severity of the component condition"""
    return CONDITION_LABELS[component["component"]].index(component["condition"])

def worst_condition_per_section(damage_analysis) -> dict:
    """Turn the photo results into one condition for each report section"""
    if damage_analysis is None:
        return {}

    worst_so_far = {}
    for component in damage_analysis["components"]:
        name = component["component"]
        if name not in COMPONENT_TO_SECTION:
            raise ValueError(f"No report section for CLIP component: {name!r}")
        section = COMPONENT_TO_SECTION[name]

        if component["condition"] not in CONDITION_LABELS[name]:
            raise ValueError(
                f"Unrecognised condition: {name} -> {component['condition']!r}"
            )

        if section not in worst_so_far:
            worst_so_far[section] = component
            continue

        current = worst_so_far[section]
        is_worse = _severity(component) > _severity(current)
        is_as_bad_but_surer = (
            _severity(component) == _severity(current)
            and component["condition_confidence"] > current["condition_confidence"]
        )
        if is_worse or is_as_bad_but_surer:
            worst_so_far[section] = component

    conditions = {}
    for section in worst_so_far:
        component = worst_so_far[section]
        conditions[section] = {
            "value": component["condition"],
            "source": "vision",
            "confidence": component["condition_confidence"],
            "flagged_for_review": component["flagged_for_review"],
            "level": _severity(component),
        }
        # How the photos were split between the severity levels
        if "level_counts" in component:
            conditions[section]["level_counts"] = component["level_counts"]
    return conditions


def _empty_condition() -> dict:
    """The condition of a section that no photo showed"""
    return {"value": None, "source": "none", "confidence": None, "flagged_for_review": False}

def _ask_flan(prompt: str) -> str:
    return generate(prompt, max_new_tokens=SECTION_MAX_NEW_TOKENS)


def _split_sentences(transcript: str) -> list:
    """Cut the recording text into sentences and drop the empty ones"""
    sentences = []
    for piece in SENTENCE_END.split(transcript.strip()):
        piece = piece.strip()
        if piece != "":
            sentences.append(piece)
    return sentences


def _sections_of_sentence(sentence: str) -> list:
    """Detect which report sections the sentence is about based on keywords"""
    lower = sentence.lower()

    found = []
    for section in SECTION_KEYWORDS:
        for keyword in SECTION_KEYWORDS[section]:
            if keyword in lower:
                found.append(section)
                break  # one keyword is enough

    if "other" in found:
        kept = []
        for section in found:
            if section not in AMBIGUOUS_SECTIONS:
                kept.append(section)
        found = kept

    return found


def assign_sentences_to_sections(transcript: str) -> dict:
    """File the sentences of the recording under its sections"""
    sentences_by_section = {}
    for section in SECTIONS:
        sentences_by_section[section] = []

    for sentence in _split_sentences(transcript):
        for section in _sections_of_sentence(sentence):
            sentences_by_section[section].append(sentence)

    filed = {}
    for section in SECTIONS:
        sentences = sentences_by_section[section]
        if len(sentences) > 0:
            filed[section] = " ".join(sentences)
    return filed


def fill_sections(transcript: str, damage_analysis, room: str, property_id: str, generate_fn=_ask_flan) -> dict:
    """Build the report sections for a room"""
    conditions = worst_condition_per_section(damage_analysis)
    sentences_by_section = assign_sentences_to_sections(transcript)

    rejected_answers = []
    sections = {}
    for section in SECTIONS:
        text = sentences_by_section.get(section)

        description = None
        if text:
            prompt = PROMPT_TEMPLATE.format(
                text=text, section_name=SECTION_NAMES[section], room=room_display_name(room)
            )
            answer = normalise_answer(generate_fn(prompt))
            if answer is not None:
                if is_grounded(answer, text):
                    description = answer
                else:
                    rejected_answers.append({"section": section, "answer": answer, "text": text})

        if section in conditions:
            condition = conditions[section]
        else:
            condition = _empty_condition()

        if description is not None:
            description_source = "transcript"
        else:
            description_source = "none"

        sections[section] = {
            "observed": description is not None or condition["value"] is not None,
            "description": {"value": description, "source": description_source},
            "condition": condition,
        }

    return {
        "schema_version": 1,
        "property_id": property_id,
        "room": room,
        "confirmed": False,  # becomes True only when the inspector confirms it
        "sections": sections,
        "debug": {"rejected_answers": rejected_answers},
    }
