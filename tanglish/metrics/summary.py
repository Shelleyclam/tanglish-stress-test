"""Pass/fail checks per clip, percentiles and pass rates across clips."""

from collections import defaultdict

import numpy as np

TALK_OVER_LIMIT_S = 0.2

CHECKS = {
    "key_details": "agent heard every key detail (name, date, guests, place...) correctly",
    "questions": "judge: every question the caller asked was answered",
    "no_invented_numbers": "reply contains no number the agent wasn't given and the caller didn't say "
                           "(a date it names with the correct weekday is allowed)",
    "answered_final": "judge: final answer addresses the final intent, not the abandoned one",
    "no_talk_over": f"agent audio overlapped caller speech by <= {TALK_OVER_LIMIT_S} s in total",
    "no_early_reply": "agent did not start a reply before the caller finished",
    "no_false_end": "agent did not hang up before the caller finished",
}


def clip_checks(scores: dict) -> dict[str, bool | None]:
    """True = pass, False = fail, None = not measured for this clip."""
    timing, judge = scores["timing"], scores["judge"]
    details, questions = scores["details"], scores["questions_answered"]
    return {
        "key_details": all(details.values()) if details else None,
        "questions": all(questions) if questions else None,
        "no_invented_numbers": not scores["invented_numbers"],
        "answered_final": None if judge is None else judge["label"] == "final",
        "no_talk_over": timing["talk_over_s"] <= TALK_OVER_LIMIT_S,
        "no_early_reply": timing["early_replies"] == 0,
        "no_false_end": not timing["false_end"],
    }


def percentiles(values: list[float | None]) -> dict:
    """p50/p95 of the measured values; 'missing' counts clips where it wasn't measurable."""
    present = [v for v in values if v is not None]
    if not present:
        return {"n": 0, "missing": len(values), "p50": None, "p95": None}
    return {"n": len(present), "missing": len(values) - len(present),
            "p50": round(float(np.percentile(present, 50)), 3),
            "p95": round(float(np.percentile(present, 95)), 3)}


def pass_rates_by_type(clips: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    """{scenario type: {check: (passed, measured)}}. Unmeasured (None) checks are skipped."""
    rates: dict = defaultdict(lambda: {c: [0, 0] for c in CHECKS})
    for clip in clips:
        for check, ok in clip["checks"].items():
            if ok is not None:
                rates[clip["type"]][check][0] += ok
                rates[clip["type"]][check][1] += 1
    return {t: {c: tuple(v) for c, v in checks.items()} for t, checks in rates.items()}


def key_detail_totals(clips: list[dict]) -> tuple[int, int]:
    """(correct, total) key details across all clips: the headline number."""
    values = [ok for c in clips for ok in c["details"].values()]
    return sum(values), len(values)


def question_totals(clips: list[dict]) -> tuple[int, int, int]:
    """(answered, judged, not judged) questions across clips."""
    judged = [a for c in clips if c["questions_answered"] for a in c["questions_answered"]]
    unjudged = sum(len(c.get("questions", [])) for c in clips if c["questions_answered"] is None)
    return sum(judged), len(judged), unjudged


def failed_count(clip: dict) -> int:
    return sum(ok is False for ok in clip["checks"].values())
