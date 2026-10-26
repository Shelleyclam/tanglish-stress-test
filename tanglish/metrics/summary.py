"""Pass/fail checks per clip, percentiles and pass rates across clips."""

from collections import defaultdict

import numpy as np

TALK_OVER_LIMIT_S = 0.2

CHECKS = {
    "no_false_end": "agent did not hang up before the caller finished",
    "no_talk_over": f"agent audio overlapped caller speech by <= {TALK_OVER_LIMIT_S} s in total",
    "no_early_reply": "agent did not start a reply before the caller finished",
    "answered_final": "judge: final answer addresses the final intent, not the abandoned one",
}


def clip_checks(timing: dict, judge_label: str | None) -> dict[str, bool | None]:
    return {
        "no_false_end": not timing["false_end"],
        "no_talk_over": timing["talk_over_s"] <= TALK_OVER_LIMIT_S,
        "no_early_reply": timing["early_replies"] == 0,
        "answered_final": None if judge_label is None else judge_label == "final",
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


def failed_count(clip: dict) -> int:
    return sum(ok is False for ok in clip["checks"].values())
