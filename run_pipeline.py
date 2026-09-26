"""Run the full pipeline: audio and photos input, a Word report output

  python run_pipeline.py --property 8705 --room livingroom     one room
  python run_pipeline.py --property 8705 --all                 all rooms of a property
  python run_pipeline.py --all                                 all rooms of all properties

The questions are asked on the terminal.
Add --no-followup to skip asking.
Add --force to redo all stages
"""

import argparse
import json
from pathlib import Path

from src.pipeline.orchestrator import discover_rooms, run_room
from src.pipeline.property_report import save_property_report

PROPERTIES_DIR = Path("properties")


def _run_all_rooms(rooms: list, followup: bool, force: bool) -> list:
    """Run each room in turn. Returns one summary row per room
    """
    summary = []
    for property_id, room in rooms:
        print(f"\n=== {property_id}/{room} ===")
        try:
            run_log = run_room(property_id, room, ask_followup=followup, force=force)
        except Exception as error:
            print(f"ERROR: {error}")
            summary.append(
                {
                    "property_id": property_id,
                    "room": room,
                    "status": "error",
                    "error": str(error),
                }
            )
            continue

        summary.append(
            {
                "property_id": property_id,
                "room": room,
                "status": "ok",
                "error": None,
            }
        )
    return summary


def _print_summary(summary: list) -> None:
    """Show the summary as a table"""
    print(f"\n{'property':<10} {'room':<12} {'status':<8}  error")
    for row in summary:
        error = row["error"] or ""
        print(f"{row['property_id']:<10} {row['room']:<12} {row['status']:<8}  {error}")


def _make_reports(summary: list) -> None:
    """Make one report for each property that had at least one room finish"""
    property_ids = []
    for row in summary:
        if row["status"] == "ok" and row["property_id"] not in property_ids:
            property_ids.append(row["property_id"])

    for property_id in sorted(property_ids):
        try:
            report_path = save_property_report(property_id)
            print(f"Report saved at {report_path}")
        except ValueError as error:  # a failed report must not stop the other reports
            print(f"Report ({property_id}) FAILED: {error}")


def run_all(followup: bool, force: bool, property_id=None) -> None:
    """Run all rooms found then make the reports"""
    rooms = discover_rooms(property_id)
    if len(rooms) == 0:
        if property_id:
            where = f"property {property_id}"
        else:
            where = "any property"
        print(f"No rooms found for {where} under {PROPERTIES_DIR.resolve()}")
        print("Looked for audio (.m4a), photo zips/folders, transcripts and damage analyses.")
        return

    summary = _run_all_rooms(rooms, followup, force)

    summary_file = PROPERTIES_DIR / "run_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(summary)
    print(f"\nSaved -> {summary_file}")

    _make_reports(summary)


def main():
    parser = argparse.ArgumentParser(description="Run the full 5 stages pipeline for one room or all rooms found")
    parser.add_argument("--property", dest="property_id", help="property id, e.g. 8705")
    parser.add_argument("--room", help="run just this room, e.g. livingroom (needs --property)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all rooms found of --property if given, otherwise of every property",
    )
    parser.add_argument("--answers", dest="answers_path", help="answers.json path (single room only)")
    parser.add_argument(
        "--no-followup",
        action="store_true",
        help="don't ask the follow-up questions",
    )
    parser.add_argument("--followup", action="store_true", help=argparse.SUPPRESS) # default
    parser.add_argument("--force", action="store_true", help="redo all stages ignoring existing outputs")
    args = parser.parse_args()

    # The questions are asked unless --no-followup is given
    followup = not args.no_followup

    if args.room:
        if not args.property_id:
            parser.error("--room needs --property")
        run_log = run_room(args.property_id, args.room, args.answers_path, followup, args.force)
        print(json.dumps(run_log, indent=2))
        print(f"\nReport saved at {save_property_report(args.property_id)}")
        return

    if not (args.all or args.property_id):
        parser.error("give --property (and --room for one room), or --all")

    run_all(followup, args.force, args.property_id)


if __name__ == "__main__":
    main()
