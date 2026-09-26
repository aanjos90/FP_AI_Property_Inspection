"""Turns an audio recording into text using Whisper"""

import whisper

# Loaded once and kept, so the model is not reloaded on every call
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = whisper.load_model("medium", device="cuda")
    return _model


def transcribe_audio(audio: str) -> str:
    model = _load_model()
    result = model.transcribe(audio, language="en")
    return str(result["text"]).strip()
