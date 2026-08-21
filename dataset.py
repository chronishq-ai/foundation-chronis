"""
Pod B event dataset - v0.2.

Single source of truth for test events, so test_events.py, ground_truth.py,
and evaluate.py all reference the exact same set by stable id. Splitting
this out of test_events.py (where it lived in v0.1) is the only structural
change here - the 15 original sentences are unchanged.

Each entry:
    id       - stable identifier, referenced by ground_truth.py ratings
    text     - the event sentence
    category - one of: positive, negative, neutral, ambiguous, mixed
    author   - "v0.1" for the original 15 (unchanged from the first sprint),
               "rohit-draft-pending-review" for the 10 new ones added for
               v0.2. Per spec, these need review/replacement by someone
               other than the author (Hrithika) before being treated as
               final - do not treat "code exists" as "requirement met".
"""
from __future__ import annotations

from typing import TypedDict


class DatasetEntry(TypedDict):
    id: str
    text: str
    category: str
    author: str


DATASET: list[DatasetEntry] = [
    # --- v0.1 original 15 (unchanged) ---
    {"id": "e01", "text": "I got promoted today.", "category": "positive", "author": "v0.1"},
    {"id": "e02", "text": "I got fired from my job.", "category": "negative", "author": "v0.1"},
    {"id": "e03", "text": "My best friend lied to me about something important.", "category": "negative", "author": "v0.1"},
    {"id": "e04", "text": "I went for a walk this morning.", "category": "neutral", "author": "v0.1"},
    {"id": "e05", "text": "I finished a big project ahead of deadline.", "category": "positive", "author": "v0.1"},
    {"id": "e06", "text": "I had a huge argument with my roommate.", "category": "negative", "author": "v0.1"},
    {"id": "e07", "text": "I bought milk and eggs at the store.", "category": "neutral", "author": "v0.1"},
    {"id": "e08", "text": "I watered the plants on the balcony.", "category": "neutral", "author": "v0.1"},
    {"id": "e09", "text": "I had a job interview today.", "category": "ambiguous", "author": "v0.1"},
    {"id": "e10", "text": "I submitted my thesis for review.", "category": "ambiguous", "author": "v0.1"},
    {"id": "e11", "text": "Great, I missed the bus again.", "category": "negative", "author": "v0.1"},
    {"id": "e12", "text": "I did NOT enjoy that meeting at all.", "category": "negative", "author": "v0.1"},
    {"id": "e13", "text": "I hung out with friends all evening and it was great.", "category": "positive", "author": "v0.1"},
    {"id": "e14", "text": "I canceled plans with everyone this week to be alone.", "category": "negative", "author": "v0.1"},
    {"id": "e15", "text": "I stayed up all night studying and I'm exhausted but I nailed the exam.", "category": "mixed", "author": "v0.1"},

    # --- v0.2 new 10 (draft - needs Hrithika's review before final) ---
    {"id": "e16", "text": "I finally paid off my student loan.", "category": "positive", "author": "rohit-draft-pending-review"},
    {"id": "e17", "text": "My landlord raised the rent without any notice.", "category": "negative", "author": "rohit-draft-pending-review"},
    {"id": "e18", "text": "I folded the laundry and put it away.", "category": "neutral", "author": "rohit-draft-pending-review"},
    {"id": "e19", "text": "I found out I didn't get the scholarship, but my professor offered to write me a strong recommendation anyway.", "category": "mixed", "author": "rohit-draft-pending-review"},
    {"id": "e20", "text": "Someone from my old school messaged me out of nowhere.", "category": "ambiguous", "author": "rohit-draft-pending-review"},
    {"id": "e21", "text": "I skipped the gym again this week.", "category": "ambiguous", "author": "rohit-draft-pending-review"},
    {"id": "e22", "text": "My parents are proud of how I handled everything this semester.", "category": "positive", "author": "rohit-draft-pending-review"},
    {"id": "e23", "text": "I overheard my coworkers talking about me during lunch.", "category": "negative", "author": "rohit-draft-pending-review"},
    {"id": "e24", "text": "I tried a new recipe and it turned out fine, nothing special.", "category": "neutral", "author": "rohit-draft-pending-review"},
    {"id": "e25", "text": "I finally confronted my sibling about something that's bothered me for years, and it went better than I expected but it's still unresolved.", "category": "mixed", "author": "rohit-draft-pending-review"},
]


def get_event_texts() -> list[str]:
    """Plain list of sentences, for callers that don't need id/category/author."""
    return [entry["text"] for entry in DATASET]


def get_by_id(event_id: str) -> DatasetEntry:
    for entry in DATASET:
        if entry["id"] == event_id:
            return entry
    raise KeyError(f"No dataset entry with id={event_id!r}")