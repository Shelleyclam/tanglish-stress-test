"""Timing metrics: latency, talking over the caller, early replies, false hang-ups.

Caller speech comes from our referee VAD on the original clip; mid-turn pauses
come from the hand labels in scenarios.yaml (only you know you weren't finished).
"""

from dataclasses import asdict, dataclass

from tanglish.agents.base import AgentEvent
from tanglish.vad import Segment


@dataclass
class TimingMetrics:
    speech_end: float                  # when the caller's last word ended
    first_audio_latency: float | None  # first agent sound (filler or reply) after that
    reply_latency: float | None        # first real reply after that
    talk_over_s: float                 # seconds of agent audio on top of caller speech
    fillers_in_pauses: int             # fillers that started in a labelled mid-turn pause
    fillers_talked_over: int           # fillers still playing when the caller spoke
    early_replies: int                 # replies started before the caller had finished
    false_end: bool                    # agent hung up before the caller had finished

    def to_dict(self) -> dict:
        return asdict(self)


def timing_metrics(speech: list[Segment], pauses: list[tuple[float, float]],
                   events: list[AgentEvent]) -> TimingMetrics:
    end = speech[-1].end
    audio = [e for e in events if e.kind in ("filler", "reply")]
    replies = [e for e in audio if e.kind == "reply"]
    fillers = [e for e in audio if e.kind == "filler"]
    return TimingMetrics(
        speech_end=end,
        first_audio_latency=_first_start_after(audio, end),
        reply_latency=_first_start_after(replies, end),
        talk_over_s=round(sum(overlap(e, speech) for e in audio), 3),
        fillers_in_pauses=sum(_starts_in(e, pauses) for e in fillers),
        fillers_talked_over=sum(overlap(e, speech) > 0 for e in fillers),
        early_replies=sum(e.t < end for e in replies),
        false_end=any(e.kind == "call_end" and e.t < end for e in events),
    )


def overlap(event: AgentEvent, speech: list[Segment]) -> float:
    return sum(max(0.0, min(event.end, s.end) - max(event.t, s.start)) for s in speech)


def _first_start_after(events: list[AgentEvent], t: float) -> float | None:
    starts = [e.t for e in events if e.t >= t]
    return round(min(starts) - t, 3) if starts else None


def _starts_in(event: AgentEvent, pauses: list[tuple[float, float]]) -> bool:
    return any(start <= event.t < end for start, end in pauses)
