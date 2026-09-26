"""Runs all five stages sequentially for one room

  1. transcription     audio -> text                       (Whisper)
  2. vision            photos -> conditions                (CLIP)
  3. fill_sections         text + conditions -> the sections   (Flan-T5)
  4. followups         questions to the inspector -> answers written into the sections
  5. draft             the sections -> the written report (Flan-T5)
"""

import json
import tempfile
import zipfile
from pathlib import Path

from src.pipeline.drafting import generate_stage_draft
from src.pipeline.followups import apply_answers, ask_answers, generate_followups, load_answers
from src.pipeline.slotfill import fill_sections
from src.pipeline.transcription import transcribe_audio
from src.pipeline.vision import classify_photo_condition, reshape_to_damage_analysis

PROPERTIES_DIR = Path("properties")
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png"]


# -------------------------------------------------------------- FIND ROOMS AND THEIR PHOTOS

def _room_name_from_stem(stem: str) -> str:
    """Get the room name from a photo zip or folder name "4-Bathroom" to "bathroom" """
    name = stem
    if "-" in stem:
        name = stem.split("-", 1)[1]
    return name.lower()


def _find_photo_source(photos_dir: Path, room: str):
    """Find the zip file or folder of photos for a room"""
    if not photos_dir.exists():
        return None

    for zip_path in sorted(photos_dir.glob("*.zip")):
        if _room_name_from_stem(zip_path.stem) == room:
            return zip_path

    for item in sorted(photos_dir.iterdir()):
        if item.is_dir() and _room_name_from_stem(item.name) == room:
            return item

    return None


def discover_rooms(property_id=None) -> list:
    """Find every room that has something to work on"""
    found = []
    if not PROPERTIES_DIR.exists():
        return found

    for property_dir in sorted(PROPERTIES_DIR.iterdir()):
        if not property_dir.is_dir():
            continue
        if property_id is not None and property_dir.name != property_id:
            continue

        names = []  # the room names in this property

        audios_dir = property_dir / "audios"
        for pattern in ["*.m4a", "*.txt"]:
            for audio_file in audios_dir.glob(pattern):
                names.append(audio_file.stem)

        photos_dir = property_dir / "photos"
        if photos_dir.exists():
            for zip_file in photos_dir.glob("*.zip"):
                names.append(_room_name_from_stem(zip_file.stem))
            for item in photos_dir.iterdir():
                if item.is_dir():
                    names.append(_room_name_from_stem(item.name))
            ending = "_damage_analysis.json"
            for analysis_file in photos_dir.glob("*" + ending):
                names.append(analysis_file.name[: -len(ending)])  # drop the ending

        for name in names:
            pair = (property_dir.name, name)
            if pair not in found:
                found.append(pair)

    return sorted(found)


def _classify_zip(zip_path: Path) -> dict:
    """Run the photo model on every photo in a zip file"""
    readings = {}
    with tempfile.TemporaryDirectory() as temp_folder:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    extracted_path = archive.extract(name, path=temp_folder)
                    readings[Path(name).name] = classify_photo_condition(extracted_path)
    return readings


def _classify_folder(folder: Path) -> dict:
    """Run the photo model on every photo in a folder"""
    readings = {}
    for photo in sorted(folder.rglob("*")):
        if photo.suffix.lower() in IMAGE_SUFFIXES:
            readings[photo.name] = classify_photo_condition(str(photo))
    return readings

# -------------------------------------------------------------- FIVE STAGES

def _run_transcription(property_id: str, room: str, force: bool):
    """1 Transcription"""
    txt_file = PROPERTIES_DIR / property_id / "audios" / f"{room}.txt"
    if txt_file.exists() and not force:
        return txt_file.read_text(encoding="utf-8"), True

    audio_file = txt_file.with_suffix(".m4a")
    if not audio_file.exists():
        return "", True  # nothing was recorded for this room: not an error

    transcript = transcribe_audio(str(audio_file))
    txt_file.write_text(transcript, encoding="utf-8")
    return transcript, False


def _run_vision(property_id: str, room: str, force: bool):
    """2 Vision"""
    analysis_file = PROPERTIES_DIR / property_id / "photos" / f"{room}_damage_analysis.json"
    if analysis_file.exists() and not force:
        return json.loads(analysis_file.read_text(encoding="utf-8")), True

    source = _find_photo_source(PROPERTIES_DIR / property_id / "photos", room)
    if source is None:
        return None, True  # no photos for this room: not an error

    if source.suffix.lower() == ".zip":
        readings = _classify_zip(source)
    else:
        readings = _classify_folder(source)

    # Save the reading of every single photo, and then the summary of them.
    conditions_file = analysis_file.with_name(f"{room}_conditions.json")
    conditions_file.write_text(json.dumps(readings, indent=2), encoding="utf-8")

    damage_analysis = reshape_to_damage_analysis(readings, room, property_id)
    analysis_file.write_text(json.dumps(damage_analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return damage_analysis, False


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_fill_sections(out_dir: Path, transcript: str, damage_analysis, property_id: str, room: str, force: bool):
    """3 Fill Sections"""
    sections_file = out_dir / "sections.json"
    if sections_file.exists() and not force:
        return json.loads(sections_file.read_text(encoding="utf-8")), True

    room_sections = fill_sections(transcript, damage_analysis, room=room, property_id=property_id)
    _save_json(sections_file, room_sections)
    return room_sections, False


def _get_answers(room_sections: dict, questions: list, out_dir: Path, answers_path, ask_followup: bool) -> dict:
    """4.1 Get the answers to the questions"""
    if len(questions) == 0:
        return {}

    if ask_followup:
        return ask_answers(room_sections, questions)

    if answers_path:
        path = Path(answers_path)
    else:
        path = out_dir / "answers.json"

    answers = load_answers(path)
    if answers is None:
        return {}
    return answers


def _run_followups(room_sections: dict, out_dir: Path, answers_path, ask_followup: bool, force: bool):
    """4 Follow-ups"""
    confirmed_file = out_dir / "sections_confirmed.json"
    if confirmed_file.exists() and not force:
        return json.loads(confirmed_file.read_text(encoding="utf-8")), True

    questions = generate_followups(room_sections)
    _save_json(out_dir / "followups.json", questions)

    answers = _get_answers(room_sections, questions, out_dir, answers_path, ask_followup)
    confirmed_sections = apply_answers(room_sections, questions, answers, confirm=True)
    _save_json(confirmed_file, confirmed_sections)
    return confirmed_sections, False


def _run_draft(confirmed_sections: dict, out_dir: Path, force: bool):
    """5 Draft"""
    draft_file = out_dir / "draft.json"
    if draft_file.exists() and not force:
        return json.loads(draft_file.read_text(encoding="utf-8")), True

    draft = generate_stage_draft(confirmed_sections)
    _save_json(draft_file, draft)
    return draft, False


# -------------------------------------------------------------- RUN ONE ROOM

def _record(run_log: dict, name: str, was_skipped: bool) -> None:
    """Log if stage was skipped"""
    run_log["skipped"][name] = was_skipped


def run_room(property_id: str, room: str, answers_path=None, ask_followup: bool = False, force: bool = False) -> dict:
    """Run all five stages for one room"""
    out_dir = PROPERTIES_DIR / property_id / "outputs" / room
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log = {"property_id": property_id, "room": room, "skipped": {}}

    transcript, was_skipped = _run_transcription(property_id, room, force)
    _record(run_log, "transcription", was_skipped)

    damage_analysis, was_skipped = _run_vision(property_id, room, force)
    _record(run_log, "vision", was_skipped)

    room_sections, was_skipped = _run_fill_sections(out_dir, transcript, damage_analysis, property_id, room, force)
    _record(run_log, "fill_sections", was_skipped)

    confirmed_sections, was_skipped = _run_followups(room_sections, out_dir, answers_path, ask_followup, force)
    _record(run_log, "followups_and_answers", was_skipped)

    draft, was_skipped = _run_draft(confirmed_sections, out_dir, force)
    _record(run_log, "draft", was_skipped)

    _save_json(out_dir / "run_log.json", run_log)
    return run_log
