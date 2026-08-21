"""
CLI entry point for Pod B.

Usage:
    python main.py "I got promoted today."
    python main.py            # interactive prompt loop
"""
import json
import sys

from analyzer import analyze_event


def run_once(event: str) -> None:
    try:
        result = analyze_event(event)
        print(json.dumps(result, indent=2))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_interactive() -> None:
    print("Pod B - Event Understanding Engine. Ctrl+C to exit.")
    while True:
        try:
            event = input("\nEvent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not event:
            continue
        run_once(event)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        run_interactive()
