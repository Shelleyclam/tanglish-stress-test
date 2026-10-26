"""Vaani Research Labs agent: placeholder until their API is public (launch Oct 29).

To fill in: open a full-duplex audio session, forward each 20 ms chunk in
send_audio, record every audio packet Vaani sends back with clock.now(), and
map their transcript (if exposed) to CallRecording.stt_transcript.
"""

import numpy as np

from tanglish.agents.base import AgentAdapter, CallRecording


class VaaniAgent(AgentAdapter):
    name = "vaani"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "VaaniAgent is a stub: Vaani's API is not public yet. "
            "See tanglish/agents/vaani.py for what to implement.")

    async def start_call(self, clock) -> None: ...
    async def send_audio(self, chunk: np.ndarray, t: float) -> None: ...
    async def finish_call(self) -> CallRecording: ...
    def is_busy(self) -> bool: return False
    def call_ended(self) -> bool: return False
