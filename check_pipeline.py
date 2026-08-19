import argparse

from pathlib import Path
from src.pipeline.transcription import transcribe_audio
from src.pipeline.vision import _load_model as load_clip

AUDIO_DIR = Path("properties/8705/audios")


def check_transcription():
    # get all audio files in the dir
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    for audio_file in audio_files:
        room_name = audio_file.stem
        print(f"Transcribing {room_name}...")

        transcript = transcribe_audio(str(audio_file))

        # save transcript to a text file
        transcript_file = audio_file.with_suffix(".txt")
        transcript_file.write_text(transcript, encoding="utf-8")

        preview = transcript[:120] + ("..." if len(transcript) > 120 else "")
        print(f"    Done -> {transcript_file.name}")
        print(f"    Preview: {preview}\n")

def check_clip():
    model, processor = load_clip()

    print(f"CLIP loaded on:", next(model.parameters()).device)

def main():
    parser = argparse.ArgumentParser(description="Check pipeline stages")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["transcription", "clip"],
        help="Pipeline stage to check",
    )
    args = parser.parse_args()

    if args.stage == "transcription":
        check_transcription()
    elif args.stage == "clip":
        check_clip()
    

if __name__ == "__main__":
    main()