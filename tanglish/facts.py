"""The hotel both agents play, read from facts/resort.yaml.

One file feeds the local agent's prompt and Vaani's per-call override, so both
agents are tested against the same facts.
"""

from pathlib import Path

import yaml

FACTS_PATH = Path(__file__).resolve().parents[1] / "facts" / "resort.yaml"


def load_facts(path: str | Path = FACTS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def hotel_prompt(facts: dict | None = None) -> str:
    f = facts or load_facts()
    early = f["early_checkin"]
    return (
        f"You are the phone booking assistant for a hotel in {f['place']}. "
        "Callers often mix Tamil and English (Tanglish). Reply in English, in one or two "
        f"short spoken sentences. Facts: {f['availability']}; "
        f"the tariff is Rs {f['tariff_per_room_per_night']:,} per room per night; "
        f"early check-in before {early['before']} costs Rs {early['extra']:,} extra."
    )


def welcome_message(facts: dict | None = None) -> str:
    return (facts or load_facts())["welcome_message"]
