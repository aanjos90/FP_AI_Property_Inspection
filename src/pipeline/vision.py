"""The photo model (CLIP). It looks at a photo and says what it shows and what condition it is in"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.pipeline.schema import CONDITION_LABELS

# Loaded once and kept, so the model is not reloaded for every photo
_model = None
_processor = None


def _load_model():
    """Loads CLIP the first time it is needed"""
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        _model = _model.to("cuda")  # move it onto the graphics card
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    return _model, _processor


# CLIP matches a photo against sentences built from these templates
CONDITION_PROMPT_TEMPLATE = "a photo of a {component} in {label}"
COMPONENT_PROMPT_TEMPLATE = "a photo of a {component}"

# Threshold for the human in the loop trigger
CONFIDENCE_FLAG_THRESHOLD = 0.45

COMPONENT_LABELS = list(CONDITION_LABELS.keys())


def _score_prompts(image, prompts: list) -> list:
    """Compare a photo with a list of sentences"""
    model, processor = _load_model()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.softmax(dim=1)[0].tolist()


def _index_of_highest(scores: list) -> int:
    """The position of the biggest number in the list"""
    best = 0
    for position in range(len(scores)):
        if scores[position] > scores[best]:
            best = position
    return best


def _scores_by_name(names: list, scores: list) -> dict:
    """Pair each name with its score"""
    result = {}
    for position in range(len(names)):
        result[names[position]] = scores[position]
    return result


def classify_photo_condition(photo_path: str) -> dict:
    """Two steps zero-shot classification of a property photo
    Step 1: Classify the component in the photo (door, floor, wall, etc.)
    Step 2: Classify the condition of that component (new, good, worn, damaged)
    """
    image = Image.open(photo_path).convert("RGB")

    # Step 1
    component_prompts = []
    for component in COMPONENT_LABELS:
        component_prompts.append(COMPONENT_PROMPT_TEMPLATE.format(component=component))
    component_scores = _score_prompts(image, component_prompts)
    component_position = _index_of_highest(component_scores)
    component = COMPONENT_LABELS[component_position]

    # Step 2
    condition_labels = CONDITION_LABELS[component]
    condition_prompts = []
    for label in condition_labels:
        condition_prompts.append(CONDITION_PROMPT_TEMPLATE.format(component=component, label=label))
    condition_scores = _score_prompts(image, condition_prompts)
    condition_position = _index_of_highest(condition_scores)

    return {
        "component": component,
        "component_confidence": component_scores[component_position],
        "condition": condition_labels[condition_position],
        "condition_confidence": condition_scores[condition_position],
        "component_scores": _scores_by_name(COMPONENT_LABELS, component_scores),
        "condition_scores": _scores_by_name(condition_labels, condition_scores),
    }


def _severity(component: str, condition: str) -> int:
    """Condition severity from 0 (best) to 3 (worst)"""
    return CONDITION_LABELS[component].index(condition)


def reshape_to_damage_analysis(
    raw_conditions: dict, room: str, property_id: str, confidence_threshold: float = CONFIDENCE_FLAG_THRESHOLD
) -> dict:
    """Combine the readings of many photos into one summary per kind of thing"""
    # Step 1 - put the photos into groups, one group per kind of thing
    groups = {}
    for filename, reading in raw_conditions.items():
        component = reading["component"]
        if component not in groups:
            groups[component] = []
        groups[component].append((filename, reading))

    # Step 2 - make one summary for each group
    components = []
    for component in sorted(groups):
        photos = groups[component]

        severities = []
        for filename, reading in photos:
            severities.append(_severity(component, reading["condition"]))
        worst = max(severities)

        # From the photos with the worst condition, keep the one with the highest confidence
        selected = None
        for filename, reading in photos:
            is_worst = _severity(component, reading["condition"]) == worst
            if is_worst and (selected is None or reading["condition_confidence"] > selected["condition_confidence"]):
                selected = reading

        photos_disagree = len(photos) > 1 and (worst - min(severities)) > 1
        not_sure_enough = selected["condition_confidence"] < confidence_threshold

        filenames = []
        for filename, reading in photos:
            filenames.append(filename)

        # How many photos fell at each level [best, good, worn, worst] for followup
        level_counts = []
        for _label in CONDITION_LABELS[component]:
            level_counts.append(0)
        for severity in severities:
            level_counts[severity] = level_counts[severity] + 1

        components.append(
            {
                "component": component,
                "condition": selected["condition"],
                "condition_confidence": selected["condition_confidence"],
                "condition_distribution": selected["condition_scores"],
                "flagged_for_review": photos_disagree or not_sure_enough,
                "n_observations": len(photos),
                "level_counts": level_counts,
                "source_images": sorted(filenames),
            }
        )

    return {"room": room, "property_id": property_id, "components": components}
