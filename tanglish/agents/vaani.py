"""Vaani Research Labs agent, called over WebRTC.

  1. vaani-sdk starts a WebRTC call and returns a LiveKit room + token.
  2. We join the room as the caller, publish a microphone track, and wait for
     the agent's greeting to finish. Only then does the caller's clock start,
     so the greeting is not on the timeline being scored.
  3. Each 20 ms caller chunk is published as one audio frame. Every frame the
     agent sends back is kept with the time it arrived on that same clock.
  4. After the call, the transcript, call details and Vaani's own recording are
     fetched. WebRTC calls return no call_id, so it is looked up in the call
     history by room name.

What we can and cannot see: Vaani sends audio, not labelled events, so agent
speech is split into segments by loudness. Short segments are *counted* as
fillers (filler_max_s); that is a guess from duration, not something Vaani
reports. A segment that has a line in Vaani's transcript is a reply, or a
"prompt" if it came before the caller's first line (e.g. "Hello?" to a silent
line); prompts are not scored as replies or fillers. Times are when decoded audio reached us, which includes LiveKit's
jitter buffer: roughly what a caller would hear.
"""

import asyncio
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import webrtcvad

from tanglish.agents.base import AgentAdapter, AgentEvent, CallRecording
from tanglish.audio import CHUNK_MS, SAMPLE_RATE
from tanglish.facts import hotel_prompt, welcome_message

CALLER_ROLES = {"user", "customer", "caller", "human", "client"}


@dataclass
class VaaniConfig:
    primary_language: str = "ta"
    # Same hotel as the local agent, sent as a per-call override, so the agent saved in
    # Vaani's dashboard is not changed. "system_prompt" is also what scoring treats as facts.
    system_prompt: str = field(default_factory=hotel_prompt)
    welcome_message: str = field(default_factory=welcome_message)
    greeting_timeout_s: float = 20.0  # give up waiting for a greeting after this long
    greeting_quiet_s: float = 1.0     # greeting is over after this much agent silence
    voiced_rms: float = 0.01          # agent audio louder than this (~-40 dBFS) is speech
    gap_s: float = 1.0                # agent silence that splits two turns (Vaani's TTS
                                      # pauses 0.4-0.6 s between sentences of one reply)
    filler_max_s: float = 0.8         # short segments with no transcript line count as fillers
    reply_wait_s: float = 12.0        # how long after the caller's last word to wait for a reply
                                      # (first run: Vaani took 5.6 s, by its own numbers)
    speech_onset_ms: int = 60         # caller must speak this long to count (same as local)
    fetch_timeout_s: float = 600.0    # post-processing took 2-4 min on the first runs
    text_match_s: float = 2.0         # transcript times are whole seconds; match within this
    recordings_dir: str = "reports/vaani_server_audio"


def _persona_override(config: VaaniConfig) -> dict:
    """Per-call persona. The top-level welcome_message alone left the saved greeting playing
    (run of 2026-11-09), so the greeting is also set here, using the SDK's field names."""
    return {"identity": {
        "system_prompt": config.system_prompt,
        "greeting_message": {"agent_message": config.welcome_message, "interruptible": True},
    }}


class VaaniAgent(AgentAdapter):
    name = "vaani"

    def __init__(self, config: VaaniConfig | None = None) -> None:
        self.config = config or VaaniConfig()
        self._vad = webrtcvad.Vad(3)
        self._api_key = self._agent_id = None

    def describe(self) -> dict:
        return {"agent_id": self._agent_id, "medium": "webrtc", **asdict(self.config)}

    async def warm_up(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        self._api_key = os.getenv("VAANI_API_KEY")
        self._agent_id = os.getenv("VAANI_AGENT_ID")
        if not self._api_key or not self._agent_id:
            raise SystemExit("Set VAANI_API_KEY and VAANI_AGENT_ID in .env")

    # --- the call ---

    async def start_call(self, clock) -> None:
        from livekit import rtc
        from vaani_sdk import AsyncVaaniClient
        from vaani_sdk.models import ModifyAgentConfig

        if self._api_key is None:
            await self.warm_up()
        self.clock = clock
        self._live = False                 # False while the greeting plays
        self._frames: list[tuple[float, np.ndarray]] = []  # (start t, samples), caller timeline
        self._greeting: list[np.ndarray] = []
        self._last_voice_perf: float | None = None   # wall clock, for the greeting wait
        self._last_agent_voice_t = float("-inf")    # caller timeline
        self._last_caller_speech_t: float | None = None
        self._speech_run = 0
        self._ended_t: float | None = None
        self._notes: list[str] = []
        self._rx_task: asyncio.Task | None = None
        self._started_wall = time.time()

        self._client = AsyncVaaniClient(api_key=self._api_key)
        self._resp = await self._client.trigger_call(
            self._agent_id, medium="webrtc", primary_language=self.config.primary_language,
            welcome_message=self.config.welcome_message,
            modify_agent=ModifyAgentConfig(persona=_persona_override(self.config)))
        if not self._resp.token or not self._resp.connection_url:
            raise RuntimeError(f"trigger_call returned no LiveKit credentials: "
                               f"{sorted(self._resp.raw)}")

        self._room = room = rtc.Room()

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO and self._rx_task is None:
                self._rx_task = asyncio.create_task(self._receive(rtc, track))

        @room.on("participant_disconnected")
        def _on_left(participant):
            self._agent_hung_up()

        @room.on("disconnected")
        def _on_disconnected(*_):
            self._agent_hung_up()

        await room.connect(self._resp.connection_url, self._resp.token)
        self._source = rtc.AudioSource(SAMPLE_RATE, 1)
        mic = rtc.LocalAudioTrack.create_audio_track("caller", self._source)
        await room.local_participant.publish_track(
            mic, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self._rtc = rtc
        await self._wait_for_greeting()
        # The streamer starts the caller clock as soon as this returns (no await in
        # between), so frames from here on are on the caller timeline.
        self._t0_utc = time.time()  # for placing Vaani's transcript times on that timeline
        self._live = True

    async def _wait_for_greeting(self) -> None:
        cfg, start = self.config, time.perf_counter()
        while time.perf_counter() - start < cfg.greeting_timeout_s:
            await asyncio.sleep(0.05)
            last = self._last_voice_perf
            if last is not None and time.perf_counter() - last >= cfg.greeting_quiet_s:
                self._greeting_s = round(sum(len(g) for g in self._greeting) / SAMPLE_RATE, 2)
                return
        self._greeting_s = None
        self._notes.append(f"no greeting finished within {cfg.greeting_timeout_s} s; "
                           "started the clip anyway")

    async def _receive(self, rtc, track) -> None:
        async for event in rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1):
            samples = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768
            voiced = float(np.sqrt(np.mean(samples ** 2))) >= self.config.voiced_rms
            if voiced:
                self._last_voice_perf = time.perf_counter()
            if not self._live:
                self._greeting.append(samples)
                continue
            end = self.clock.now()
            self._frames.append((end - len(samples) / SAMPLE_RATE, samples))
            if voiced:
                self._last_agent_voice_t = end

    async def send_audio(self, chunk: np.ndarray, t: float) -> None:
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
        if self._vad.is_speech(pcm.tobytes(), SAMPLE_RATE):
            self._speech_run += 1
            if self._speech_run * CHUNK_MS >= self.config.speech_onset_ms:
                self._last_caller_speech_t = t
        else:
            self._speech_run = 0
        if self._ended_t is not None:
            return
        frame = self._rtc.AudioFrame(pcm.tobytes(), SAMPLE_RATE, 1, len(pcm))
        await self._source.capture_frame(frame)

    def is_busy(self) -> bool:
        now = self.clock.now()
        if now - self._last_agent_voice_t < self.config.gap_s:
            return True  # speaking
        caller = self._last_caller_speech_t
        waiting = caller is not None and caller > self._last_agent_voice_t
        return waiting and now - caller < self.config.reply_wait_s  # presumably thinking

    def call_ended(self) -> bool:
        return self._ended_t is not None

    def _agent_hung_up(self) -> None:
        if self._live and self._ended_t is None:
            self._ended_t = round(self.clock.now(), 3)

    async def finish_call(self) -> CallRecording:
        live_until = self.clock.now()
        self._live = False
        if self._rx_task is not None:
            self._rx_task.cancel()
        await self._room.disconnect()
        events = self._segments(live_until)
        if self._ended_t is not None:
            events.append(AgentEvent(self._ended_t, "call_end"))
        if self._rx_task is None:
            self._notes.append("never received an audio track from the agent")

        stats = {"room_name": self._resp.room_name, "greeting_s": self._greeting_s,
                 "clock_start_utc": self._t0_utc,
                 "frames_received": len(self._frames)}
        try:
            stt = await self._fetch_after_call(stats, events)
        finally:
            await self._client.aclose()
        return CallRecording(events, stt, self._notes, stats)

    # --- turning received audio into events ---

    def _segments(self, live_until: float) -> list[AgentEvent]:
        """Lay received frames on the caller timeline and cut them into speech segments."""
        cfg = self.config
        track = np.zeros(int(max(live_until, 0) * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
        for start, samples in self._frames:
            i = max(int(start * SAMPLE_RATE), 0)
            piece = samples[: max(len(track) - i, 0)]
            track[i:i + len(piece)] = piece
        hop = SAMPLE_RATE * CHUNK_MS // 1000
        n = len(track) // hop
        rms = np.sqrt(np.mean(track[: n * hop].reshape(n, hop) ** 2, axis=1))
        voiced = np.flatnonzero(rms >= cfg.voiced_rms)
        if len(voiced) == 0:
            return []
        gap = int(cfg.gap_s * 1000 / CHUNK_MS)
        runs, first, prev = [], voiced[0], voiced[0]
        for v in voiced[1:]:
            if v - prev > gap:
                runs.append((first, prev))
                first = v
            prev = v
        runs.append((first, prev))

        events = []
        for a, b in runs:
            # plain floats: numpy scalars leak into the scores and break JSON output
            t, end = float(a * hop / SAMPLE_RATE), float((b + 1) * hop / SAMPLE_RATE)
            kind = "filler" if end - t < cfg.filler_max_s else "reply"
            events.append(AgentEvent(round(t, 3), kind, duration=round(end - t, 3),
                                     audio=track[a * hop:(b + 1) * hop].copy()))
        return events

    # --- after the call: Vaani's own records ---

    async def _fetch_after_call(self, stats: dict, events: list[AgentEvent]) -> str | None:
        call_id = self._resp.call_id or await self._find_call_id()
        stats["call_id"] = call_id
        if call_id is None:
            return None
        details = await self._poll(lambda: self._client.get_call_details(call_id),
                                   lambda d: d.raw.get("post_processing_status") != "in_processing"
                                   and "transcription" in d.raw, "call post-processing")
        if details is not None:
            stats["details"] = details.raw
            stats["vaani_turn_latency"] = [
                {"turn": t.get("turn"), **(t.get("e2e") or {})}
                for t in details.raw.get("turns_with_percentile_rankings") or []]
        stats["recording_path"] = await self._save_recording(call_id)
        transcript = None
        try:
            transcript = (await self._client.get_transcript(call_id)).transcript
        except Exception as exc:
            self._notes.append(f"transcript failed: {exc!r}")
        if not isinstance(transcript, str) or not TRANSCRIPT_LINE.search(transcript):
            self._notes.append(f"transcript not usable: {str(transcript)[:120]!r}; WER skipped")
            return None
        stats["transcript"] = transcript
        turns = _timed_turns(transcript, self._t0_utc)
        unmatched = _attach_text(events, turns, self.config.text_match_s)
        if unmatched:
            self._notes.append(f"{unmatched} agent transcript line(s) matched no audio segment")
        return _caller_text(transcript)

    async def _find_call_id(self) -> str | None:
        room = self._resp.room_name

        async def lookup():
            history = await self._client.get_call_history(page=1, page_size=50)
            for record in history.data:
                if room and any(room in str(v) for v in record.raw.values()):
                    return record.call_id
            self._history_keys = sorted(history.data[0].raw) if history.data else []
            return None

        call_id = await self._poll(lookup, lambda r: r is not None, "call_id in call history")
        if call_id is None:
            self._notes.append(f"no history record mentions room {room}; record fields: "
                               f"{getattr(self, '_history_keys', [])}")
        return call_id

    async def _poll(self, fetch, ready, what: str):
        """Vaani writes records after the call ends, so retry with backoff."""
        deadline, delay, last_error = time.perf_counter() + self.config.fetch_timeout_s, 2.0, None
        while True:
            try:
                result = await fetch()
                if ready(result):
                    return result
            except Exception as exc:  # e.g. 404 until the record exists
                last_error = exc
            if time.perf_counter() + delay > deadline:
                self._notes.append(f"gave up waiting for {what} after "
                                   f"{self.config.fetch_timeout_s:.0f} s (last error: {last_error!r})")
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 15.0)

    async def _save_recording(self, call_id: str) -> str | None:
        from pathlib import Path
        audio = await self._poll(lambda: self._client.stream_audio(call_id),
                                 lambda a: bool(a.content), "recording")
        if audio is None:
            return None
        ext = ".mp3" if "mpeg" in audio.content_type else ".ogg"
        path = Path(self.config.recordings_dir) / f"{call_id}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio.content)
        return str(path)


TRANSCRIPT_LINE = re.compile(r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s*([A-Za-z_ ]+?):\s*(.*)$", re.M)


def _timed_turns(transcript: str, t0_utc: float) -> list[tuple[float, str, str]]:
    """(t on the caller timeline, role, text) for each '[HH:MM:SS] ROLE: text' line.
    Vaani's times are UTC time of day, whole seconds."""
    t0_day = t0_utc % 86400
    turns = []
    for h, m, sec, role, text in TRANSCRIPT_LINE.findall(transcript):
        offset = (int(h) * 3600 + int(m) * 60 + int(sec)) - t0_day
        offset = (offset + 43200) % 86400 - 43200  # calls that cross midnight
        turns.append((float(offset), role.strip().lower(), text.strip()))
    return turns


def _attach_text(events: list[AgentEvent], turns: list[tuple[float, str, str]],
                 within_s: float) -> int:
    """Give each agent transcript line to the speech segment it was logged during (Vaani
    logs a line while it is being spoken, not when audio starts), and record what the
    caller had said just before it as what the agent heard. Segments with a line are
    replies whatever their length; lines before the caller's first line (e.g. "Hello?"
    to a silent line) are prompts, not replies to anything. Returns lines unmatched."""
    speech = [e for e in events if e.kind in ("filler", "reply")]
    claimed, unmatched, heard, caller_spoke = set(), 0, "", False

    def distance(e: AgentEvent, t: float) -> float:
        return 0.0 if e.t <= t <= e.end else min(abs(e.t - t), abs(e.end - t))

    for t, role, text in turns:
        if role in CALLER_ROLES:
            heard, caller_spoke = f"{heard} {text}".strip(), True
            continue
        free = [e for e in speech if id(e) not in claimed and distance(e, t) <= within_s]
        if not free:
            unmatched += 1
            continue
        e = min(free, key=lambda e: distance(e, t))
        claimed.add(id(e))
        e.kind, e.text, e.heard = ("reply" if caller_spoke else "prompt"), text, heard
        heard = ""
    return unmatched


def _caller_text(transcript: str) -> str | None:
    """Everything the caller said, as Vaani's speech-to-text heard it."""
    words = [text.strip() for *_, role, text in TRANSCRIPT_LINE.findall(transcript)
             if role.strip().lower() in CALLER_ROLES]
    return " ".join(words) if words else None
