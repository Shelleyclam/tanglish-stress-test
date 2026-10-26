import asyncio

import numpy as np
import pytest

from tanglish.agents.base import AgentAdapter, CallRecording
from tanglish.agents.local_pipeline import LocalPipelineAgent, LocalPipelineConfig
from tanglish.clock import FakeClock
from tanglish.streamer import stream_call
from tanglish.synth import concat, silence, speech_like


# --- streamer, with an agent that just counts chunks ---

class CountingAgent(AgentAdapter):
    async def start_call(self, clock):
        self.times = []

    async def send_audio(self, chunk, t):
        self.times.append(t)

    async def finish_call(self):
        return CallRecording([], None)

    def is_busy(self):
        return False

    def call_ended(self):
        return False


def test_streamer_sends_chunks_on_schedule_then_stops_when_agent_is_idle():
    agent = CountingAgent()

    _, stats = asyncio.run(stream_call(agent, silence(1.0), FakeClock(), idle_tail_s=0.5))

    assert agent.times[:2] == [pytest.approx(0.02), pytest.approx(0.04)]  # chunk end times
    assert stats.stopped_by == "agent_done"
    assert len(agent.times) == 50 + 25  # 1 s of clip + 0.5 s idle tail
    assert stats.max_lateness_ms == 0  # fake clock is never late


# --- local agent with fake models, run instantly on a fake clock ---

class FakeSTT:
    def transcribe(self, audio):
        return "saturday room kedaikuma"


class FakeLLM:
    def first_sentence(self, history, text):
        return "Yes, rooms are available."

    def continue_reply(self, history, text, first):
        return "Shall I book one?"


class FakeTTS:
    def __init__(self, seconds=0.5):
        self.seconds = seconds

    def synthesize(self, text):
        return np.zeros(int(self.seconds * 16000), dtype=np.float32)


async def run_inline(fn, *args):
    return fn(*args)


def run_agent(audio, tts_seconds=0.5, **config):
    agent = LocalPipelineAgent(LocalPipelineConfig(**config), FakeSTT(), FakeLLM(),
                               FakeTTS(tts_seconds), run_blocking=run_inline)
    recording, stats = asyncio.run(stream_call(agent, audio, FakeClock()))
    return recording, stats


def test_filler_comes_quickly_then_reply_follows():
    recording, _ = run_agent(concat(speech_like(1.0), silence(3.0)))

    kinds = [e.kind for e in recording.events]
    filler = recording.events[0]
    assert kinds == ["filler", "reply", "reply"]
    assert 1.0 < filler.t < 1.3  # ~75 ms after speech ends (+ the agent VAD's own lag)
    assert recording.events[1].heard == "saturday room kedaikuma"


def test_caller_resuming_cuts_the_agent_off():
    audio = concat(speech_like(1.0), silence(1.0), speech_like(1.0), silence(4.0))

    recording, _ = run_agent(audio, tts_seconds=3.0)

    cut = [e for e in recording.events if e.interrupted]
    assert cut, "something should have been cut off when the caller resumed"
    assert all(e.end <= 2.2 for e in cut)
    assert recording.events[-1].kind == "reply" and recording.events[-1].t > 3.0


def test_silence_timer_hangs_up():
    recording, stats = run_agent(concat(speech_like(1.0), silence(4.0)), hangup_after_s=1.0)

    assert stats.stopped_by == "call_end"
    assert recording.events[-1].kind == "call_end"


def test_silence_timer_waits_while_agent_is_speaking():
    # Reply audio lasts 3 s; a 1 s timer must not fire until after it finishes.
    recording, _ = run_agent(concat(speech_like(1.0), silence(6.0)), tts_seconds=3.0,
                             hangup_after_s=1.0)

    last_audio = max(e.end for e in recording.events if e.kind != "call_end")
    hang_up = recording.events[-1]
    assert hang_up.kind == "call_end"
    assert hang_up.t >= last_audio + 1.0
    assert recording.stats["longest_timer_run_s"] >= 1.0
