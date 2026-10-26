"""Play a caller clip into an agent at real-time pace, like a phone line.

Each 20 ms chunk is scheduled against a fixed start time (chunk i is sent when
its audio has fully "arrived", at (i+1) x 20 ms), so sleep errors never add up.
After the clip ends we keep sending silence until the agent has finished replying.
"""

from dataclasses import dataclass

import numpy as np

from tanglish.agents.base import AgentAdapter, CallRecording
from tanglish.audio import CHUNK_MS, split_into_chunks

CHUNK_S = CHUNK_MS / 1000


@dataclass
class StreamStats:
    chunks_sent: int
    p95_lateness_ms: float  # how late chunks were sent vs schedule
    max_lateness_ms: float
    stopped_by: str         # "agent_done" | "call_end" | "tail_timeout"


async def stream_call(agent: AgentAdapter, audio: np.ndarray, clock,
                      idle_tail_s: float = 1.5,
                      max_tail_s: float = 45.0) -> tuple[CallRecording, StreamStats]:
    clip_chunks = split_into_chunks(audio)
    quiet = np.zeros_like(clip_chunks[0])
    max_chunks = len(clip_chunks) + int(max_tail_s / CHUNK_S)
    lateness, idle_since, stopped_by = [], None, "tail_timeout"

    clock.reset()
    await agent.start_call(clock)
    for i in range(max_chunks):
        due = (i + 1) * CHUNK_S
        await clock.sleep_until(due)
        lateness.append(clock.now() - due)
        await agent.send_audio(clip_chunks[i] if i < len(clip_chunks) else quiet, due)

        if agent.call_ended():
            stopped_by = "call_end"
            break
        if i + 1 >= len(clip_chunks):  # clip is over; wait for the agent to go quiet
            idle_since = None if agent.is_busy() else (idle_since or due)
            if idle_since is not None and due - idle_since >= idle_tail_s:
                stopped_by = "agent_done"
                break

    recording = await agent.finish_call()
    return recording, _stats(lateness, stopped_by)


def _stats(lateness: list[float], stopped_by: str) -> StreamStats:
    ms = np.array(lateness) * 1000
    return StreamStats(len(ms), round(float(np.percentile(ms, 95)), 2),
                       round(float(ms.max()), 2), stopped_by)
