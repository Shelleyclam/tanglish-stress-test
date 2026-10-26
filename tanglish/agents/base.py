"""The adapter interface: how the harness talks to any voice agent.

The harness acts like a phone line. It sends 20 ms of caller audio every 20 ms
and records whatever audio the agent sends back, with timestamps on one clock.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np

from tanglish.audio import SAMPLE_RATE


@dataclass
class AgentEvent:
    t: float                  # seconds since call start, when the audio started playing
    kind: str                 # "filler" | "reply" | "call_end"
    text: str = ""
    duration: float = 0.0     # seconds actually played (shortened if the caller barged in)
    interrupted: bool = False
    heard: str = ""           # for replies: what the agent's speech-to-text heard
    audio: np.ndarray | None = field(default=None, repr=False)

    @property
    def end(self) -> float:
        return self.t + self.duration

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        return d


@dataclass
class CallRecording:
    events: list[AgentEvent]
    stt_transcript: str | None  # what the agent heard over the whole call; None if unknown
    notes: list[str] = field(default_factory=list)

    def agent_audio(self, total_seconds: float) -> np.ndarray:
        """All agent audio placed on the caller's timeline, for listening back."""
        end = max([total_seconds] + [e.end for e in self.events])
        track = np.zeros(int(end * SAMPLE_RATE) + 1, dtype=np.float32)
        for e in self.events:
            if e.audio is None:
                continue
            start = int(e.t * SAMPLE_RATE)
            clip = e.audio[: int(e.duration * SAMPLE_RATE)]
            track[start:start + len(clip)] += clip
        return track


class AgentAdapter(ABC):
    name: str = "agent"

    async def warm_up(self) -> None:
        """Load models etc. before the first call, so it isn't counted as latency."""

    def describe(self) -> dict:
        """Settings to print in the report."""
        return {}

    @abstractmethod
    async def start_call(self, clock) -> None: ...

    @abstractmethod
    async def send_audio(self, chunk: np.ndarray, t: float) -> None:
        """Receive one 20 ms chunk of caller audio. t = when that chunk ends."""

    @abstractmethod
    async def finish_call(self) -> CallRecording: ...

    @abstractmethod
    def is_busy(self) -> bool:
        """True while the agent is thinking or speaking."""

    @abstractmethod
    def call_ended(self) -> bool:
        """True once the agent has hung up."""
