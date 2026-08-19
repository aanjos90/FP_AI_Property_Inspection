# transcribe_audio()

import whisper

_model = None

# cache model so it's not loaded every time the function is called
def _load_model():
    global _model
    if _model is None:
        _model = whisper.load_model("medium", device="cuda")
    return _model

def transcribe_audio(audio: str) -> str:
    """Transcribe an english audio recording to plain text using Whisper"""
    model = _load_model()
    result = model.transcribe(audio, language="en")
    return str(result["text"]).strip()