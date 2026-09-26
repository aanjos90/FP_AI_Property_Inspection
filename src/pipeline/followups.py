"""Ask the inspector about anything the pipeline is not sure of"""

import copy
import json
from pathlib import Path

from src.pipeline.schema import SECTION_NAMES, SECTIONS, condition_wording
from src.pipeline.slotfill import normalise_answer

# Reasons for asking a question
REASON_LOW_CONFIDENCE = "photos_disagree_or_low_confidence"  # the photos are unclear
REASON_NOT_DESCRIBED = "seen_but_not_described"  # something was seen but not described in the transcription

FIELD_BY_REASON = {REASON_LOW_CONFIDENCE: "condition", REASON_NOT_DESCRIBED: "description"}

# Threshold for number of questions
MAX_QUESTIONS = 6 

# The four levels of a condition
LEVEL_WORDS = ["new", "good", "worn", "damaged"]


def _followup(section: str, reason: str, question: str) -> dict:
    field = FIELD_BY_REASON[reason]
    return {
        "id": f"{section}_{field}",
        "section": section,
        "field": field,
        "reason": reason,
        "question": question,
    }

def _photo_evidence(level_counts) -> str:
    """Verbose of the photos' damages to be shown in a followup question"""
    if not level_counts:
        return ""

    total = 0
    parts = []
    for level in range(len(level_counts)):
        count = level_counts[level]
        total = total + count
        if count == 0:
            continue
        if count == 1:
            verb = "looks"
        else:
            verb = "look"
        parts.append(f"{count} {verb} {LEVEL_WORDS[level]}")

    if total == 0:
        return ""

    # "a", "a and b", "a, b and c"
    if len(parts) == 1:
        list_text = parts[0]
    else:
        list_text = ", ".join(parts[:-1]) + " and " + parts[-1]

    if total == 1:
        photos = "photo"
    else:
        photos = "photos"
    return f"Of {total} {photos}, {list_text}."


def generate_followups(room_sections: dict, max_questions: int = MAX_QUESTIONS) -> list:
    """Work out which questions to ask
          1. A condition that is flagged for review (the photos were unclear).
          2. A section that was seen but not described.
        """
    sections = room_sections["sections"]
    questions = []

    for section in SECTIONS:
        condition = sections[section]["condition"]
        if condition["flagged_for_review"]:
            name = SECTION_NAMES[section]
            evidence = _photo_evidence(condition.get("level_counts"))
            text = f"{name.upper()}:\nPhotos unclear or disagree with each other."
            if evidence:
                text = text + "\n" + evidence
            questions.append(_followup(section, REASON_LOW_CONFIDENCE, text))

    for section in SECTIONS:
        entry = sections[section]
        if entry["observed"] and entry["description"]["value"] is None:
            text = f"What can you tell me about the {SECTION_NAMES[section]} in the {room_sections['room']}?"
            questions.append(_followup(section, REASON_NOT_DESCRIBED, text))

    return questions[:max_questions]  # keep the threshold at max_questions


def apply_answers(room_sections: dict, questions: list, answers: dict, confirm: bool = False) -> dict:
    """Write the inspector's answers into the report"""
    room_sections = copy.deepcopy(room_sections)  # a full copy, so the caller's sections are not changed

    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            continue
        answer = normalise_answer(answer)
        if answer is None:
            continue

        entry = room_sections["sections"][question["section"]]
        if question["field"] == "condition":
            # The inspector's word replaces the photo reading
            entry["condition"] = {"value": answer, "source": "answer", "confidence": None, "flagged_for_review": False}
        else:
            entry["description"] = {"value": answer, "source": "answer"}
        entry["observed"] = True

    if confirm:
        room_sections["confirmed"] = True
    return room_sections


def _ask_yes_no(prompt: str, input_fn, print_fn) -> bool:
    """Ask until the reply is y/yes or n/no"""
    while True:
        reply = input_fn(f"{prompt} [y/n]: ").strip().lower()
        if reply == "y" or reply == "yes":
            return True
        if reply == "n" or reply == "no":
            return False
        print_fn("  Please type y or n.")


def _ask_text(prompt: str, input_fn, print_fn) -> str:
    """Ask until something is typed"""
    while True:  # keep asking until we get a usable reply
        reply = input_fn(f"{prompt}: ").strip()
        if reply != "":
            return reply
        print_fn("  Please type something.")


def ask_answers(room_sections: dict, questions: list, input_fn=input, print_fn=print) -> dict:
    """Ask the inspector each question and return their answers in a dictionary keyed by question ID"""
    answers = {}
    total = len(questions)

    for number, question in enumerate(questions, start=1):
        print_fn(f"\n[{number}/{total}] {question['question']}")

        if question["field"] == "condition":
            reading = condition_wording(room_sections["sections"][question["section"]]["condition"])
            if _ask_yes_no(f"  Is '{reading}' correct?", input_fn, print_fn):
                answers[question["id"]] = reading
            else:
                answers[question["id"]] = _ask_text("  What is the correct condition?", input_fn, print_fn)
        else:
            reply = _ask_text("  Describe it (or type 'skip')", input_fn, print_fn)
            if reply.lower() != "skip":
                answers[question["id"]] = reply

    return answers


def load_answers(path: Path):
    """Read answers that were saved in a file"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
