"""VaaniAgent's offline parts: cutting received audio into events, reading transcripts."""

import json

import numpy as np

from tanglish.agents.base import AgentEvent
from tanglish.agents.vaani import (VaaniAgent, VaaniConfig, _attach_text, _caller_text,
                                   _persona_override, _timed_turns)
from tanglish.audio import SAMPLE_RATE
from tanglish.scenarios import Scenario
from tanglish.scoring import score_clip


def loud(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), 0.2, dtype=np.float32)


def test_segments_split_on_gaps_and_short_ones_count_as_fillers():
    agent = VaaniAgent()
    agent._frames = [(1.0, loud(0.3)), (3.0, loud(2.0)), (5.5, loud(1.0))]
    events = agent._segments(8.0)
    assert [e.kind for e in events] == ["filler", "reply"]  # 0.5 s pause stays in one turn
    assert abs(events[0].t - 1.0) < 0.03 and abs(events[1].duration - 3.5) < 0.03


def test_scores_from_vaani_events_are_json_serialisable():
    agent = VaaniAgent()
    agent._frames = [(1.0, loud(0.3)), (3.0, loud(2.0))]
    scenario = Scenario(id="x", type="commitment", transcript="hello", final_intent="book",
                        clip_path="x.wav")
    raw = {"clip_seconds": 6.0, "caller_speech": [(0.0, 2.5)], "stream": {},
           "events": [e.to_dict() for e in agent._segments(6.0)], "stt_transcript": None,
           "notes": [], "agent_stats": {}}
    json.dumps(score_clip(scenario, raw, "", use_judge=False))


def test_per_call_override_carries_the_hotel_prompt_and_greeting():
    config = VaaniConfig(system_prompt="Hotel facts", welcome_message="Vanakkam")
    assert _persona_override(config) == {"identity": {
        "system_prompt": "Hotel facts",
        "greeting_message": {"agent_message": "Vanakkam", "interruptible": True},
    }}


def test_key_details_use_the_reply_segment_that_has_a_transcript():
    scenario = Scenario(id="x", type="commitment", transcript="for 2 people", final_intent="book",
                        clip_path="x.wav", key_details={"guests": ["2"]})
    reply = {"kind": "reply", "duration": 1.0}
    raw = {"clip_seconds": 6.0, "caller_speech": [(0.0, 2.5)], "stream": {},
           "events": [{**reply, "t": 3.0, "heard": ""},  # audio with no transcript line
                      {**reply, "t": 4.5, "heard": "for 2 people", "text": "Yes"}],
           "stt_transcript": None, "notes": [], "agent_stats": {}}
    assert score_clip(scenario, raw, "", use_judge=False)["details"] == {"guests": True}


def test_wer_is_left_out_when_scripts_differ():
    scenario = Scenario(id="x", type="commitment", transcript="saturday rooms kedaikuma",
                        final_intent="book", clip_path="x.wav")
    raw = {"clip_seconds": 6.0, "caller_speech": [(0.0, 2.5)], "stream": {}, "events": [],
           "stt_transcript": "சாட்டர்டே ரூம்ஸ் கிடைக்குமா", "notes": [], "agent_stats": {}}
    scores = score_clip(scenario, raw, "", use_judge=False)
    assert scores["wer"] is None and scores["wer_skipped"] == "script mismatch"


TRANSCRIPT = ("[11:47:46] AGENT: Vanakkam, booking line\n"
              "[11:47:50] AGENT: Hello?\n"
              "[11:48:02] USER: Saturday rooms kidaikkuma? For 2 people.\n"
              "[11:48:19] AGENT: Friday appointment book pannava?")
T0 = 11 * 3600 + 47 * 60 + 45.6  # caller clock started at 11:47:45.6 UTC


def test_caller_text_keeps_only_caller_lines():
    assert _caller_text(TRANSCRIPT) == "Saturday rooms kidaikkuma? For 2 people."


def test_transcript_lines_attach_to_nearest_segment_with_what_was_heard():
    turns = _timed_turns(TRANSCRIPT, T0)
    assert [round(t, 1) for t, *_ in turns] == [0.4, 4.4, 16.4, 33.4]
    # the reply's line is logged 6.6 s into its audio, as on the real run
    events = [AgentEvent(4.0, "filler", duration=0.4), AgentEvent(26.8, "reply", duration=9.5),
              AgentEvent(40.0, "filler", duration=0.3)]
    unmatched = _attach_text(events, turns, within_s=2.0)
    assert unmatched == 1  # the 0.4 s greeting line was before any recorded segment
    assert events[0].kind == "prompt" and events[0].text == "Hello?"  # before the caller spoke
    assert events[1].text.startswith("Friday") and events[1].heard.startswith("Saturday")
    assert events[2].kind == "filler" and events[2].text == ""  # unlogged and short


def test_hotel_prompt_from_facts_is_the_baseline_prompt_unchanged():
    from tanglish.agents.local_pipeline import SYSTEM_PROMPT
    from tanglish.facts import hotel_prompt
    frozen = ("You are the phone booking assistant for a hotel in Yelagiri, Tamil Nadu. "
              "Callers often mix Tamil and English (Tanglish). Reply in English, in one or two "
              "short spoken sentences. Facts: rooms are available on all dates; the tariff is "
              "Rs 3,500 per room per night; early check-in before 12 pm costs Rs 1,000 extra.")
    assert hotel_prompt() == SYSTEM_PROMPT == frozen


def test_vaani_sends_the_hotel_prompt_and_reports_it_as_facts():
    from tanglish.facts import hotel_prompt
    assert VaaniAgent().describe()["system_prompt"] == hotel_prompt()
