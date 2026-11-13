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
    scores = {"timing": {"false_end": False, "talk_over_s": 0.5, "early_replies": 0},
              "judge": None, "details": {"name": False, "date": True},
              "questions_answered": None, "invented_numbers": []}
    checks = clip_checks(scores)
    assert checks == {"key_details_kept": None, "key_details": False, "questions": None, "no_invented_numbers": True,
                      "answered_final": None, "no_talk_over": False,
                      "no_early_reply": True, "no_false_end": True}

    rates = pass_rates_by_type([{"type": "code_switch", "checks": checks}])
    assert rates["code_switch"]["no_talk_over"] == (0, 1)
    assert rates["code_switch"]["answered_final"] == (0, 0)  # not measured


def test_percentiles_skip_missing_values():
    p = percentiles([1.0, None, 3.0])
    assert p["n"] == 2 and p["missing"] == 1 and p["p50"] == 2.0


def test_detail_is_kept_only_if_heard_and_said_back():
    from tanglish.agents.base import AgentEvent
    from tanglish.scenarios import Scenario
    from tanglish.scoring import score_clip
    s = Scenario(id="x", type="code_switch", transcript="my name is Raghu, 4 people",
                 final_intent="book", clip_path="x.wav",
                 key_details={"name": ["raghu"], "guests": ["4"]})
    reply = AgentEvent(5.0, "reply", "Hello Rahul! Rooms for 4 people.", 2.0,
                       heard="my name is Raghu, 4 people")
    raw = {"clip_seconds": 10.0, "caller_speech": [(0.0, 3.0)], "stream": {},
           "events": [reply.to_dict()], "stt_transcript": None, "notes": [], "agent_stats": {}}
    scores = score_clip(s, raw, "", use_judge=False)
    assert scores["details"] == {"name": True, "guests": True}         # heard right
    assert scores["details_kept"] == {"name": False, "guests": True}   # but said "Rahul"
    assert scores["checks"]["key_details_kept"] is False


def test_rescore_without_judge_keeps_saved_verdicts():
    from run import _keep_saved_judge
    from tanglish.scenarios import Scenario
    s = Scenario(id="x", type="code_switch", transcript="t", final_intent="f",
                 clip_path="x.wav", questions=["q1", "q2"])
    old = {"final_answer": "same", "questions": ["q1", "q2"], "judge": {"label": "final"},
           "questions_answered": [True, False], "question_reasons": ["a", "b"]}
    new = {"final_answer": "same", "judge": None, "questions_answered": None,
           "details": {}, "details_kept": {}, "invented_numbers": [],
           "timing": {"talk_over_s": 0, "early_replies": 0, "false_end": False}}
    _keep_saved_judge(new, old, s)
    assert new["questions_answered"] == [True, False] and new["checks"]["questions"] is False
    changed = dict(new, final_answer="different", judge=None, questions_answered=None)
    _keep_saved_judge(changed, old, s)
    assert changed["questions_answered"] is None  # reply changed: old verdicts don't apply
