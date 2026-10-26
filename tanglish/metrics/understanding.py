"""Did the agent understand and answer correctly? (Not just: were the words right.)

  * key details: did the agent hear the name, date, guests, place correctly?
    A wrong name is 1 word to WER but loses the booking, so this is the headline.
  * invented numbers: numbers in the reply that are in neither the agent's given
    facts nor anything the caller said (e.g. a made-up price).
  * reply language: what language the agent answered in.

All three are plain string checks, so every verdict can be traced by hand.
"""

import re

from tanglish.metrics.wer import ONES, TENS, normalize

# Common Tamil words in romanised Tanglish, used to spot a Tanglish reply.
TAMIL_WORDS = {"illa", "illai", "aama", "irukku", "irukkum", "venum", "vendum", "sollunga",
               "panren", "pannunga", "kidaikum", "kedaikum", "seri", "ungalukku", "naan",
               "neenga", "enna", "evlo", "evalo", "romba", "nandri", "vanakkam"}
NUMBER_WORDS = {w: i for i, w in enumerate(ONES)} | {w: i * 10 for i, w in enumerate(TENS) if w}


def details_heard(text: str, key_details: dict[str, list[str]]) -> dict[str, bool]:
    """For each key detail: does any accepted spelling appear in the text?"""
    padded = f" {normalize(text)} "
    return {name: any(f" {normalize(s)} " in padded for s in spellings)
            for name, spellings in key_details.items()}


def numbers_in(text: str) -> set[int]:
    """All numbers in a text, written as digits ("3,500", "8:30") or words ("four")."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 3,500 -> 3500
    found = {int(n) for n in re.findall(r"\d+", text)}
    words = re.findall(r"[a-z]+", text)
    return found | {NUMBER_WORDS[w] for w in words if w in NUMBER_WORDS}


def invented_numbers(reply: str, allowed_sources: list[str]) -> list[int]:
    """Numbers the agent said that it was never given and the caller never said."""
    allowed = set().union(*(numbers_in(s) for s in allowed_sources)) if allowed_sources else set()
    return sorted(numbers_in(reply) - allowed)


def reply_language(text: str) -> str:
    if not text.strip():
        return "none"
    if re.search(r"[஀-௿]", text):  # Tamil script
        return "tamil"
    return "tanglish" if set(normalize(text).split()) & TAMIL_WORDS else "english"
