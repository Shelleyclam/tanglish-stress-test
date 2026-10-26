import pytest

from tanglish.agents.base import AgentEvent
from tanglish.metrics.judge import judge_answer, to_label
from tanglish.metrics.summary import clip_checks, pass_rates_by_type, percentiles
from tanglish.metrics.timing import timing_metrics
from tanglish.metrics.wer import normalize, word_error_rate
from tanglish.vad import Segment

# Caller speaks 1-3 s, pauses (labelled) 3-4 s, speaks 4-6 s.
SPEECH = [Segment(1, 3), Segment(4, 6)]
PAUSES = [(3.0, 4.0)]


def test_latency_counts_from_the_last_word():
    events = [AgentEvent(6.1, "filler", duration=0.3), AgentEvent(6.8, "reply", duration=2)]

    m = timing_metrics(SPEECH, PAUSES, events)

    assert m.first_audio_latency == pytest.approx(0.1)
    assert m.reply_latency == pytest.approx(0.8)
    assert m.talk_over_s == 0
    assert m.early_replies == 0


def test_filler_in_a_pause_that_runs_into_speech_is_talk_over():
    events = [AgentEvent(3.1, "filler", duration=1.2)]  # plays until 4.3, caller resumed at 4.0

    m = timing_metrics(SPEECH, PAUSES, events)

    assert m.fillers_in_pauses == 1
    assert m.fillers_talked_over == 1
    assert m.talk_over_s == pytest.approx(0.3)


def test_reply_in_mid_turn_pause_is_early_and_hangup_before_end_is_false_end():
    events = [AgentEvent(3.3, "reply", duration=0.5), AgentEvent(5.0, "call_end")]

    m = timing_metrics(SPEECH, PAUSES, events)

    assert m.early_replies == 1
    assert m.false_end is True
    assert m.reply_latency is None  # no reply after the caller finished


def test_wer_ignores_case_punctuation_and_digit_spelling():
    assert normalize("Rooms for 2 People!") == "rooms for two people"
    assert word_error_rate("rooms for 2 people", "Rooms for two people.") == 0
    assert word_error_rate("saturday room kedaikuma please", "saturday room get please") == 0.25


def test_judge_labels():
    assert to_label(True, False) == "final"
    assert to_label(False, True) == "abandoned"
    assert to_label(True, True) == "both"
    assert to_label(False, False) == "neither"


def test_empty_answer_is_no_answer_without_calling_the_llm():
    assert judge_answer("hi", "x", None, "  ")["label"] == "no_answer"


def test_checks_and_pass_rates():
    timing = {"false_end": False, "talk_over_s": 0.5, "early_replies": 0}
    checks = clip_checks(timing, None)
    assert checks == {"no_false_end": True, "no_talk_over": False,
                      "no_early_reply": True, "answered_final": None}

    rates = pass_rates_by_type([{"type": "code_switch", "checks": checks}])
    assert rates["code_switch"]["no_talk_over"] == (0, 1)
    assert rates["code_switch"]["answered_final"] == (0, 0)  # not measured


def test_percentiles_skip_missing_values():
    p = percentiles([1.0, None, 3.0])
    assert p["n"] == 2 and p["missing"] == 1 and p["p50"] == 2.0
