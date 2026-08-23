# damage_analysis()

import torch
from transformers import CLIPModel, CLIPProcessor

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to("cuda")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _processor

# Condition labels for property components
# These are used to generate prompts for CLIP model
CONDITION_LABELS = {
    "door": [
        "new or like-new condition",
        "good condition with minor wear",
        "worn condition, visible marks or scuffs",
        "damaged condition, holes cracks or broken hardware",
    ],
    "floor": [
        "new flooring",
        "good condition with minor wear",
        "worn or scratched flooring",
        "damaged flooring, chips stains or cracks",
    ],
    "wall": [
        "freshly painted, like-new condition",
        "good condition with minor marks",
        "worn or faded paint",
        "damaged paintwork, peeling cracked or holed",
    ],
    "window": [
        "new or like-new frame and glass",
        "good condition with minor wear",
        "worn frame or scratched glass",
        "damaged frame or glass, cracks rust or broken seals",
    ],
    "ceiling": [
        "freshly painted, like-new condition",
        "good condition",
        "worn or discoloured",
        "damaged, cracks stains or peeling",
    ],
    "electrical fitting": [
        "new or like-new fitting",
        "good condition fitting",
        "worn or aged fitting",
        "damaged or missing fitting",
    ],
    "light fixture": [
        "new fixture",
        "good condition fixture",
        "worn or aged fixture",
        "damaged or non-functional fixture",
    ],
    "fixture": [
        "new or like-new condition",
        "good condition",
        "worn condition",
        "damaged condition",
    ],
}

CONDITION_PROMPT_TEMPLATE = "a photo of a {component} in {label}"