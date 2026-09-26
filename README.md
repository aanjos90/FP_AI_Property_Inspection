# FINAL PROJECT

## About

Turns the audio and photos of a property inspection into a report.  

**Template**: 4.1 Orchestrating AI Models to achieve a goal  
**Models Used**: Whisper, CLIP, Flan-T5  

| # | Model     | Domain        | Pipeline Stage                        |
| - | --------- | ------------- | ------------------------------------- |
| 1 | Whisper   | audio-to-text | Audio Transcription                   |
| 2 | CLIP      | image-to-text | Damage Detection                      |
| 3 | Flan-T5   | text-to-text  | Slot Filling -> Report generation   |

### Pipeline Stages

Each room goes through these five stages. A stage is skipped if its result is already saved.

| # | Stage              | What it does                                                         | Model   |
| - | ------------------ | -------------------------------------------------------------------- | ------- |
| 1 | Transcription      | Turns the room's recording into text                                 | Whisper |
| 2 | Photos analysis    | Reads each photo and detects the condition of each part of the room  | CLIP    |
| 3 | Fill sections      | Sorts the text and photo results into door, floor, walls, etc.       | Flan-T5 |
| 4 | Follow-up questions| Asks user to confirm or fix anything unclear                         | User    |
| 5 | Draft              | Writes about each section                                            | Flan-T5 |

The drafts of all rooms are put together in the docx report.

## How to Run

```
pip install -r requirements.txt
python run_pipeline.py --property 8705 --room livingroom   # one room
python run_pipeline.py --property 8705 --all               # all rooms of a property
python run_pipeline.py --all                               # all rooms of all properties
```

The follow-up questions are asked in the terminal. Options:
- `--no-followup` skips the questions
- `--force` redoes every stage, ignoring saved results

The report is saved at `properties/{property id}/outputs/{property id}_report.docx`.

## Directory Structure

In order to run the pipeline, the property folder must be in this format:

```
properties/
└── 8705/                    one folder per property
    ├── audios/
    │   └── livingroom.m4a   one recording per room, named after the room
    ├── photos/
    │   └── 2-LivingRoom.zip one zip or folder of photos per room: "{number}-{room}"
    └── outputs/             created by the pipeline
```
