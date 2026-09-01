# damage_analysis()

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        _model = _model.to("cuda")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
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
    "plumbing fitting": [
        "new or like-new fitting",
        "good condition fitting, minor wear",
        "worn or oxidised fitting",
        "damaged or non-functional fitting",
    ],
    "sanitary ware": [
        "new or like-new condition",
        "good condition with minor marks",
        "worn or stained condition",
        "damaged condition, chips or cracks",
    ],
    "shower enclosure": [
        "new or like-new condition",
        "good condition with minor wear",
        "worn condition, marks or stiff runners",
        "damaged condition, cracked glass or broken frame",
    ],
    "cabinetry": [
        "new or like-new condition",
        "good condition with minor wear",
        "worn condition, misaligned doors or worn surfaces",
        "damaged condition, broken doors, drawers or countertop",
    ],
    "railing": [
        "new or like-new condition",
        "good condition with minor wear",
        "worn condition, faded paint or stiff hardware",
        "damaged condition, rust, cracks or broken sections",
    ],
}

CONDITION_PROMPT_TEMPLATE = "a photo of a {component} in {label}"

COMPONENT_LABELS = list(CONDITION_LABELS.keys())
COMPONENT_PROMPT_TEMPLATE = "a photo of a {component}"


def _score_prompts(image: Image.Image, prompts: list[str]) -> list[float]:
    """Scores an image against a set of text prompts with CLIP and returns softmax distribution"""
    model, processor = _load_model()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.softmax(dim=1)[0].tolist()


def classify_photo_condition(photo_path: str) -> dict:
    """Two steps zero-shot classification of a property photo
    Step 1: Classify the component in the photo (door, floor, wall, etc.)
    Step 2: Classify the condition of that component (new, good, worn, damaged)
    """
    image = Image.open(photo_path).convert("RGB")

    component_prompts = [
        COMPONENT_PROMPT_TEMPLATE.format(component=component) for component in COMPONENT_LABELS
    ]
    component_scores = _score_prompts(image, component_prompts)
    component_idx = max(range(len(COMPONENT_LABELS)), key=component_scores.__getitem__)
    component = COMPONENT_LABELS[component_idx]

    condition_labels = CONDITION_LABELS[component]
    condition_prompts = [
        CONDITION_PROMPT_TEMPLATE.format(component=component, label=label) for label in condition_labels
    ]
    condition_scores = _score_prompts(image, condition_prompts)
    condition_idx = max(range(len(condition_labels)), key=condition_scores.__getitem__)
    condition = condition_labels[condition_idx]

    return {
        "component": component,
        "component_confidence": component_scores[component_idx],
        "condition": condition,
        "condition_confidence": condition_scores[condition_idx],
        "component_scores": dict(zip(COMPONENT_LABELS, component_scores)),
        "condition_scores": dict(zip(condition_labels, condition_scores)),
    }