"""Baseline agent built from local models, mimicking Vaani's published design:

  * full-duplex: keeps listening while speaking; caller speech cuts the agent off
  * a backchannel filler ("Mm-hmm") after filler_after_ms of caller silence
  * commits to a reply after only reply_after_ms of silence, and a small fast
    model says the first sentence while a bigger one continues
  * a plain timer ends the call after hangup_after_s of silence

All timings are settings, so we can test which values break on Tanglish callers.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import webrtcvad

from tanglish.agents.base import AgentAdapter, AgentEvent, CallRecording
from tanglish.agents.components import OllamaLLM, PiperTTS, WhisperSTT
from tanglish.audio import CHUNK_MS, SAMPLE_RATE

SYSTEM_PROMPT = (
    "You are the phone booking assistant for a hotel in Yelagiri, Tamil Nadu. "
    "Callers often mix Tamil and English (Tanglish). Reply in English, in one or two "
    "short spoken sentences. Facts: rooms are available on all dates; the tariff is "
    "Rs 3,500 per room per night; early check-in before 12 pm costs Rs 1,000 extra."
)


@dataclass
class LocalPipelineConfig:
    filler_after_ms: int = 75      # Vaani's published backchannel timing
    reply_after_ms: int = 200      # assumption: needed to reach ~300 ms perceived latency
    hangup_after_s: float = 5.0    # assumption: Vaani's timer value is not published
    speech_onset_ms: int = 60      # caller must speak this long to count (and to barge in)
    vad_aggressiveness: int = 3
    filler_text: str = "Mm-hmm."
    whisper_model: str = "small"
    whisper_language: str | None = "en"  # auto-detect heard Tanglish as Telugu
    fast_model: str = "qwen2.5:1.5b"
    main_model: str = "qwen2.5:7b"
    voice_path: str = "models/en_US-lessac-medium.onnx"
    system_prompt: str = SYSTEM_PROMPT


class OneWorker:
    """Runs model calls one at a time in a background thread, like a single GPU."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def __call__(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)


class LocalPipelineAgent(AgentAdapter):
    name = "local"

    def __init__(self, config: LocalPipelineConfig, stt, llm, tts, run_blocking=None):
        self.config, self.stt, self.llm, self.tts = config, stt, llm, tts
        self._run = run_blocking or OneWorker()  # tests pass a version that runs inline
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._filler_audio = None

    @classmethod
    def from_config(cls, config: LocalPipelineConfig | None = None) -> "LocalPipelineAgent":
        config = config or LocalPipelineConfig()
        return cls(config,
                   WhisperSTT(config.whisper_model, config.whisper_language),
                   OllamaLLM(config.fast_model, config.main_model, config.system_prompt),
                   PiperTTS(config.voice_path))

    def describe(self) -> dict:
        return asdict(self.config)

    async def warm_up(self) -> None:
        self._filler_audio = await self._run(self.tts.synthesize, self.config.filler_text)
        await self._run(self.stt.transcribe, np.zeros(SAMPLE_RATE, dtype=np.float32))
        await self._run(self.llm.first_sentence, [], "Hello")
        await self._run(self.llm.continue_reply, [], "Hello", "Hello.")

    # --- the call ---

    async def start_call(self, clock) -> None:
        self.clock = clock
        if self._filler_audio is None:
            self._filler_audio = self.tts.synthesize(self.config.filler_text)
        self._events: list[AgentEvent] = []
        self._notes: list[str] = []
        self._all_audio: list[np.ndarray] = []
        self._turn_audio: list[np.ndarray] = []
        self._history: list[tuple[str, str]] = []
        self._speech_run = 0
        self._last_speech_t: float | None = None
        self._filler_due = self._reply_due = False
        self._play_until = 0.0
        self._generation = 0      # bumped whenever an in-progress reply becomes stale
        self._task: asyncio.Task | None = None
        self._pending: tuple | None = None  # (generation, heard, reply) until fully played
        self._ended = False

    async def send_audio(self, chunk: np.ndarray, t: float) -> None:
        self._all_audio.append(chunk)
        self._turn_audio.append(chunk)
        if self._is_speech(chunk):
            self._speech_run += 1
            if self._speech_run * CHUNK_MS >= self.config.speech_onset_ms:
                self._on_caller_speech(t)
        else:
            self._speech_run = 0
        self._finish_completed_reply()
        self._check_timers(t)

    async def finish_call(self) -> CallRecording:
        self._generation += 1  # ignore anything still being generated
        transcript = await self._run(self.stt.transcribe, np.concatenate(self._all_audio))
        events = sorted(self._events, key=lambda e: e.t)
        return CallRecording(events, transcript, self._notes)

    def is_busy(self) -> bool:
        thinking = self._task is not None and not self._task.done()
        return thinking or self._playing()

    def call_ended(self) -> bool:
        return self._ended

    # --- listening ---

    def _is_speech(self, chunk: np.ndarray) -> bool:
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        return self._vad.is_speech(pcm, SAMPLE_RATE)

    def _on_caller_speech(self, t: float) -> None:
        self._last_speech_t = t
        self._filler_due = self._reply_due = True
        if self.is_busy():
            self._barge_in()

    def _check_timers(self, t: float) -> None:
        if self._last_speech_t is None:
            return
        silence = t - self._last_speech_t
        if self._filler_due and silence * 1000 >= self.config.filler_after_ms:
            self._filler_due = False
            if not self._playing():
                self._play("filler", self._filler_audio, self.config.filler_text)
        if self._reply_due and silence * 1000 >= self.config.reply_after_ms:
            self._reply_due = False
            self._start_reply()
        quiet_since = max(self._last_speech_t, self._play_until)
        if not self.is_busy() and t - quiet_since >= self.config.hangup_after_s:
            self._hang_up()

    # --- speaking ---

    def _start_reply(self) -> None:
        audio = np.concatenate(self._turn_audio)
        self._task = asyncio.create_task(self._reply(self._generation, audio))
        self._task.add_done_callback(self._note_task_error)

    async def _reply(self, gen: int, audio: np.ndarray) -> None:
        heard = await self._step(gen, self.stt.transcribe, audio)
        if not heard:
            return
        first = await self._step(gen, self.llm.first_sentence, list(self._history), heard)
        if not first:
            return
        await self._speak(gen, first, heard)
        rest = await self._step(gen, self.llm.continue_reply, list(self._history), heard, first)
        if rest is None:
            return
        if rest:
            await self._speak(gen, rest, heard)
        self._pending = (gen, heard, f"{first} {rest}".strip())

    async def _speak(self, gen: int, text: str, heard: str) -> None:
        audio = await self._step(gen, self.tts.synthesize, text)
        if audio is not None:
            self._play("reply", audio, text, heard)

    async def _step(self, gen: int, fn, *args):
        """Run one model call, unless the reply went stale (then return None).
        Checked when the job actually starts, so queued stale jobs cost nothing."""
        def job():
            return fn(*args) if gen == self._generation else None
        result = await self._run(job)
        return result if gen == self._generation else None

    def _play(self, kind: str, audio: np.ndarray, text: str, heard: str = "") -> None:
        start = max(self.clock.now(), self._play_until)
        duration = len(audio) / SAMPLE_RATE
        self._events.append(AgentEvent(round(start, 3), kind, text, round(duration, 3),
                                       heard=heard, audio=audio))
        self._play_until = start + duration

    def _playing(self) -> bool:
        return self.clock.now() < self._play_until

    def _barge_in(self) -> None:
        """Caller is talking: stop speaking now and drop any reply in progress."""
        now = self.clock.now()
        self._generation += 1
        self._task, self._pending = None, None
        for e in self._events:
            if e.t < now < e.end:
                e.duration, e.interrupted = round(now - e.t, 3), True
        self._events = [e for e in self._events if e.t < now or e.kind == "call_end"]
        self._play_until = min(self._play_until, now)

    def _finish_completed_reply(self) -> None:
        """Once a reply has fully played, remember it and start a fresh turn."""
        if self._pending is None or self._playing():
            return
        gen, heard, reply = self._pending
        if gen == self._generation:
            self._history.append((heard, reply))
            self._turn_audio = []
        self._pending = None

    def _hang_up(self) -> None:
        self._events.append(AgentEvent(round(self.clock.now(), 3), "call_end"))
        self._ended = True
        self._generation += 1

    def _note_task_error(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            self._notes.append(f"reply failed: {task.exception()!r}")
