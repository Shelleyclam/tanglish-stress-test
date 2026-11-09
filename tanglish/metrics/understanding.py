"""Did the agent understand and answer correctly? (Not just: were the words right.)

  * key details: did the agent hear the name, date, guests, place correctly?
    A wrong name is 1 word to WER but loses the booking, so this is the headline.
  * invented numbers: numbers in the reply that are in neither the agent's given
    facts nor anything the caller said (e.g. a made-up price).
    A date the agent worked out ("this Friday, November 13th") is not invented when
    it names a weekday and that date really is that weekday, soon after the call.
  * reply language: what language the agent answered in.

All three are plain string checks, so every verdict can be traced by hand.
"""

import re
from datetime import date, timedelta

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
    text = re.sub(r"(?<=\d):00\b", "", text)            # 12:00 pm -> 12 pm, not 12 and 0
    found = {int(n) for n in re.findall(r"\d+", text)}
    words = re.findall(r"[a-z]+", text)
    return found | {NUMBER_WORDS[w] for w in words if w in NUMBER_WORDS}


def invented_numbers(reply: str, allowed_sources: list[str],
                     also_allowed: set[int] = frozenset()) -> list[int]:
    """Numbers the agent said that it was never given and the caller never said."""
    allowed = set().union(*(numbers_in(s) for s in allowed_sources)) if allowed_sources else set()
    return sorted(numbers_in(reply) - allowed - set(also_allowed))


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH = "(" + "|".join(MONTHS) + ")"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(\d{4}))?"
_DATE_PATTERNS = [(re.compile(rf"\b{_MONTH}\s+{_DAY}\b{_YEAR}"), (1, 2, 3)),          # November 13th
                  (re.compile(rf"\b{_DAY}\s+(?:of\s+)?{_MONTH}\b{_YEAR}"), (2, 1, 3))]  # 13th of November
DATE_WINDOW_DAYS = 366  # a worked-out date must fall within this many days after the call


def checked_dates(reply: str, call_date: date | None) -> list[dict]:
    """Dates in the reply that name a weekday which is correct for that date.

    Each is {"text", "date", "numbers"}; its numbers (day, and year if said) are then not
    counted as invented. A date with no weekday, or the wrong one, is not accepted:
    nothing in the reply lets us check it.
    """
    if call_date is None:
        return []
    text, found = reply.lower(), []
    for pattern, (m_i, d_i, y_i) in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            month, day = MONTHS.index(m.group(m_i)) + 1, int(m.group(d_i))
            year = int(m.group(y_i)) if m.group(y_i) else None
            nearby = text[max(0, m.start() - 30):m.end() + 20]
            weekdays = {w for w in WEEKDAYS if re.search(rf"\b{w}\b", nearby)}
            when = _resolve(day, month, year, call_date)
            if len(weekdays) == 1 and when and WEEKDAYS[when.weekday()] in weekdays:
                found.append({"text": m.group(0), "date": when.isoformat(),
                              "numbers": sorted({day} | ({year} if year else set()))})
    return found


def _resolve(day: int, month: int, year: int | None, call_date: date) -> date | None:
    """The date meant, if it is a real date within the window after the call."""
    for y in ([year] if year else [call_date.year, call_date.year + 1]):
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if call_date <= d <= call_date + timedelta(days=DATE_WINDOW_DAYS):
            return d
    return None


def reply_language(text: str) -> str:
    if not text.strip():
        return "none"
    if re.search(r"[஀-௿]", text):  # Tamil script
        return "tamil"
    return "tanglish" if set(normalize(text).split()) & TAMIL_WORDS else "english"
