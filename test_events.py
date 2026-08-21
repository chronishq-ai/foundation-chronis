import json
import time

from analyzer import analyze_event
from dataset import get_event_texts

TEST_EVENTS = get_event_texts()


def main() -> None:
    results = []
    for i, event in enumerate(TEST_EVENTS, start=1):
        print(f"[{i:2d}/{len(TEST_EVENTS)}] {event}")
        try:
            output = analyze_event(event)
            print(json.dumps(output, indent=2))
        except RuntimeError as exc:
            output = {"error": str(exc)}
            print(f"  FAILED: {exc}")
        results.append({"event": event, "result": output})
        print("-" * 60)
        time.sleep(5)  # gpt-oss-120b: 8K TPM -> ~12 calls/min safe headroom

    with open("logs/test_events_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Summary written to logs/test_events_summary.json")


if __name__ == "__main__":
    main()