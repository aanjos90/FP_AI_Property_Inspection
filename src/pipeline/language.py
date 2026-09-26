"""The language model (Flan-T5). It answers one short prompt at a time."""

import logging

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.utils import logging as transformers_logging

logger = logging.getLogger(__name__)

FLAN_MODEL = "google/flan-t5-large"
MAX_INPUT_TOKENS = 512  # a longer prompt is cut to this length
DEFAULT_MAX_NEW_TOKENS = 64  # the longest answer we allow
DEFAULT_NUM_BEAMS = 4  # how many candidate answers the model weighs up

# The model is big, so we load it once and keep it here. "None" means "not loaded yet".
_tokenizer = None
_model = None


def get_flan():
    """Load the model the first time it is needed, then reuse it. Returns (tokenizer, model)."""
    global _tokenizer, _model  # we are changing the two variables above, not making new ones

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL, local_files_only=True)

        # Loading prints a harmless warning about "tied weights" (the checkpoint keeps two sets of
        # weights separate on purpose). We hide the library's warnings only while loading, then
        # put the old level back. We do NOT change the model's settings: that breaks its answers.
        old_level = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            # bfloat16 on purpose: this model gives NaN (not-a-number) results in float16.
            _model = T5ForConditionalGeneration.from_pretrained(
                FLAN_MODEL, local_files_only=True, torch_dtype=torch.bfloat16
            )
        finally:
            transformers_logging.set_verbosity(old_level)
        _model = _model.to("cuda")  # move it onto the graphics card
        _model.eval()  # switch from training mode to answering mode

    return _tokenizer, _model


def generate(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, num_beams: int = DEFAULT_NUM_BEAMS) -> str:
    """Answer one prompt. The same prompt always gives the same answer."""
    tokenizer, model = get_flan()

    # Tell the user if the prompt is too long, because the end of it will be cut off.
    full_length = len(tokenizer(prompt).input_ids)
    if full_length > MAX_INPUT_TOKENS:
        logger.warning("Prompt has %d tokens. Cutting it to %d.", full_length, MAX_INPUT_TOKENS)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    inputs = inputs.to(model.device)

    with torch.no_grad():  # we are not training, so skip the extra bookkeeping
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
