"""Score one recorded call. Kept separate from running calls, so changed metrics or
an edited scenarios.yaml can be re-scored from saved results without re-running
the agent (python run.py --rescore <results.json>).
"""

from datetime import date

from tanglish.agents.base import AgentEvent
from tanglish.audio import load_clip
from tanglish.metrics.judge import judge_answer, judge_question
from tanglish.metrics.summary import clip_checks
from tanglish.metrics.timing import timing_metrics
from tanglish.metrics.understanding import (checked_dates, details_heard, invented_numbers,
                                            reply_language)
from tanglish.metrics.wer import has_tamil_script, word_error_rate
from tanglish.scenarios import Scenario
from tanglish.vad import Segment, VadSettings, find_speech


def caller_speech(scenario: Scenario, vad: VadSettings) -> list[list[float]]:
    return [[s.start, s.end] for s in find_speech(load_clip(scenario.clip_path), vad)]


def score_clip(scenario: Scenario, raw: dict, agent_facts: str, use_judge: bool,
               call_date: date | None = None) -> dict:
    """raw = what the call produced (events, transcripts, speech). Returns all metrics.
    call_date lets a weekday date the agent worked out count as correct, not invented."""
    events = [AgentEvent(**e) for e in raw["events"]]
    speech = [Segment(*s) for s in raw["caller_speech"]]
    timing = timing_metrics(speech, scenario.pauses, events).to_dict()
    after_end = [e for e in events if e.kind == "reply" and e.t >= timing["speech_end"]]
    final_answer = " ".join(e.text for e in after_end)
    # A reply can span several audio segments, and only some have a transcript line.
    heard = next((e.heard for e in after_end if e.heard), raw["stt_transcript"] or "")
    details = details_heard(heard, scenario.key_details)
    dates = checked_dates(final_answer, call_date)
    wer, wer_skipped = _wer(scenario.transcript, raw["stt_transcript"])
    scores = {
        "timing": timing,
        "final_answer": final_answer,
        "final_heard": heard,
        "wer": wer,
        "wer_skipped": wer_skipped,
        "details": details,
        "invented_numbers": invented_numbers(final_answer, [agent_facts, scenario.transcript,
                                                            heard, raw["stt_transcript"] or ""],
                                             {n for d in dates for n in d["numbers"]}),
        "checked_dates": dates,
        "reply_language": reply_language(final_answer),
        "fillers_per_min": round(sum(e.kind == "filler" for e in events)
                                 / (raw["clip_seconds"] / 60), 1),
        "judge": None,
        "questions_answered": None,
    }
    if use_judge:
        scores["judge"] = judge_answer(scenario.transcript, scenario.final_intent,
                                       scenario.abandoned_intent, final_answer)
        verdicts = [judge_question(q, final_answer) for q in scenario.questions]
        scores["questions_answered"] = [v["answered"] for v in verdicts]
        scores["question_reasons"] = [v["reason"] for v in verdicts]
    scores["checks"] = clip_checks(scores)
    return scores


def _wer(reference: str, hypothesis: str | None) -> tuple[float | None, str | None]:
    """(WER, None), or (None, reason) when the two texts can't be fairly compared."""
    if hypothesis is None:
        return None, None
    if has_tamil_script(hypothesis) != has_tamil_script(reference):
        return None, "script mismatch"  # Tamil script vs romanised: every Tamil word counts as wrong
    return word_error_rate(reference, hypothesis), None
