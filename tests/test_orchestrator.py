import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import orchestrator  # noqa: E402
from src.pipeline.vision import CONDITION_LABELS  # noqa: E402

FLOOR = CONDITION_LABELS["floor"]


# ---------------------------------------------------- FAKE PHOTO MODEL


class FakeClip:
    """Says every photo is a floor in the same condition and counts the photos"""

    def __init__(self):
        self.photos_seen = 0

    def classify(self, photo_path):
        self.photos_seen = self.photos_seen + 1
        scores = {}
        for label in FLOOR:
            scores[label] = 0.1
        scores[FLOOR[1]] = 0.7
        return {
            "component": "floor",
            "component_confidence": 0.9,
            "condition": FLOOR[1],
            "condition_confidence": 0.7,
            "component_scores": {"floor": 0.9},
            "condition_scores": scores,
        }


# ---------------------------------------------------- HELPERS


def _fresh_property(root: Path, property_id: str = "9001") -> Path:
    """A property with raw inputs only"""
    (root / property_id / "audios").mkdir(parents=True)
    (root / property_id / "photos").mkdir(parents=True)
    (root / property_id / "audios" / "livingroom.m4a").write_bytes(b"")
    with zipfile.ZipFile(root / property_id / "photos" / "2-LivingRoom.zip", "w"):
        pass
    with zipfile.ZipFile(root / property_id / "photos" / "3-Kitchen.zip", "w"):
        pass
    return root


def _discover_in(root):
    """Run discover_rooms with root as the properties folder"""
    original = orchestrator.PROPERTIES_DIR
    orchestrator.PROPERTIES_DIR = root
    try:
        return orchestrator.discover_rooms()
    finally:
        orchestrator.PROPERTIES_DIR = original


def _saved_analysis(condition):
    return {
        "room": "kitchen",
        "property_id": "9001",
        "components": [
            {
                "component": "floor",
                "condition": condition,
                "condition_confidence": 0.9,
                "condition_distribution": {},
                "flagged_for_review": False,
                "n_observations": 1,
                "source_images": ["a.jpg"],
            }
        ],
    }


def _kitchen_with_saved_results(root, condition, with_photos=True):
    """A kitchen with saved photo results"""
    photos = root / "9001" / "photos"
    photos.mkdir(parents=True)
    if with_photos:
        with zipfile.ZipFile(photos / "3-Kitchen.zip", "w") as archive:
            archive.writestr("a.jpg", b"pretend photo")
    analysis_file = photos / "kitchen_damage_analysis.json"
    analysis_file.write_text(json.dumps(_saved_analysis(condition)), encoding="utf-8")
    return analysis_file


def _run_vision_with_fake_clip(root, fake):
    """Run the vision stage for the kitchen with the fakes in place"""
    original_folder = orchestrator.PROPERTIES_DIR
    original_classify = orchestrator.classify_photo_condition
    orchestrator.PROPERTIES_DIR = root
    orchestrator.classify_photo_condition = fake.classify
    try:
        return orchestrator._run_vision("9001", "kitchen", force=False)
    finally:
        orchestrator.PROPERTIES_DIR = original_folder
        orchestrator.classify_photo_condition = original_classify


# ---------------------------------------------------- TESTS


def test_discovers_rooms_from_raw_inputs():
    # discovery used to look only at stage outputs, so raw inputs found no rooms
    with tempfile.TemporaryDirectory() as tmp:
        root = _fresh_property(Path(tmp))
        original = orchestrator.PROPERTIES_DIR
        orchestrator.PROPERTIES_DIR = root
        try:
            assert orchestrator.discover_rooms() == [("9001", "kitchen"), ("9001", "livingroom")]
        finally:
            orchestrator.PROPERTIES_DIR = original


def test_discovers_audio_only_room():
    # a brand-new room: only a recording
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "9001" / "audios").mkdir(parents=True)
        (Path(tmp) / "9001" / "audios" / "bedroom.m4a").write_bytes(b"")
        assert _discover_in(Path(tmp)) == [("9001", "bedroom")]


def test_saved_photo_results_reused():
    with tempfile.TemporaryDirectory() as tmp:
        _kitchen_with_saved_results(Path(tmp), FLOOR[2])
        fake = FakeClip()
        analysis, was_skipped = _run_vision_with_fake_clip(Path(tmp), fake)
        assert was_skipped is True
        assert fake.photos_seen == 0
        assert analysis is not None
        assert analysis["components"][0]["condition"] == FLOOR[2]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
