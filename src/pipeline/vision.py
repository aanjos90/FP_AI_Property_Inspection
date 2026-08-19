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