import json
import tempfile
import zipfile
from pathlib import Path

from src.pipeline.vision import classify_photo_condition

PHOTOS_DIR = Path("properties/8705/photos")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def room_name_from_stem(stem: str) -> str:
    # "4-Bathroom" -> "bathroom", "2-LivingRoom" -> "livingroom"
    name = stem.split("-", 1)[1] if "-" in stem else stem
    return name.lower()


def classify_zip(zip_path: Path) -> dict:
    readings = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path) as archive:
            image_names = [n for n in archive.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES]
            for name in image_names:
                extracted_path = archive.extract(name, path=tmp_dir)
                photo_name = Path(name).name
                reading = classify_photo_condition(extracted_path)
                readings[photo_name] = reading
                print(f"    {photo_name} -> {reading['component']} / {reading['condition']}")
    return readings


def classify_dir(dir_path: Path) -> dict:
    readings = {}
    for photo_file in sorted(dir_path.rglob("*")):
        if photo_file.suffix.lower() in IMAGE_SUFFIXES:
            reading = classify_photo_condition(str(photo_file))
            readings[photo_file.name] = reading
            print(f"    {photo_file.name} -> {reading['component']} / {reading['condition']}")
    return readings


def run_room(room_name: str, source_name: str, readings: dict):
    out_path = PHOTOS_DIR / f"{room_name}_conditions.json"
    out_path.write_text(json.dumps(readings, indent=2), encoding="utf-8")
    print(f"    Done -> {out_path.name}\n")


def main():
    if not PHOTOS_DIR.exists():
        print(f"No photos directory found at {PHOTOS_DIR}")
        return

    zip_files = sorted(PHOTOS_DIR.glob("*.zip"))
    room_dirs = sorted(p for p in PHOTOS_DIR.iterdir() if p.is_dir())

    if not zip_files and not room_dirs:
        print(f"No photos found in {PHOTOS_DIR}")
        return

    for zip_path in zip_files:
        room_name = room_name_from_stem(zip_path.stem)
        print(f"Classifying {room_name} ({zip_path.name})...")
        readings = classify_zip(zip_path)
        run_room(room_name, zip_path.name, readings)

    for room_dir in room_dirs:
        room_name = room_name_from_stem(room_dir.name)
        print(f"Classifying {room_name} ({room_dir.name})...")
        readings = classify_dir(room_dir)
        run_room(room_name, room_dir.name, readings)


if __name__ == "__main__":
    main()
