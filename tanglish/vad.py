"""Voice activity detection: find when someone is speaking and when they pause.

Steps:
  1. WebRTC VAD labels every 20 ms chunk as speech or not.
  2. Neighbouring speech chunks are joined into segments.
  3. Each segment's quiet edges are trimmed. WebRTC VAD keeps saying "speech"
     for ~80-120 ms after speech stops (its "hangover"); left in, that would make
     every latency we measure look ~100 ms better than it really is.
  4. Segments much quieter than the caller's loudest speech are dropped
     (background noise that sounds voice-like, e.g. in long silences).
  5. Gaps shorter than min_pause_ms are filled in (gaps between words).
  6. Segments shorter than min_speech_ms are dropped (breaths, clicks, and the
     80-100 ms false alarm WebRTC VAD gives at the start of any new sound).

Defaults were chosen by comparing against hand-labelled pauses on real clips
(scripts/vad_check.py): aggressiveness 3 matched the labels better than 2.
"""

from dataclasses import dataclass

import numpy as np
import webrtcvad

from tanglish.audio import CHUNK_MS, SAMPLE_RATE, split_into_chunks

CHUNK_S = CHUNK_MS / 1000


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VadSettings:
    aggressiveness: int = 3   # 0 = lenient ... 3 = strict
    min_pause_ms: int = 300   # shorter gaps are just gaps between words
    min_speech_ms: int = 120  # shorter bursts are noise, not speech
    edge_trim_db: float = 35  # trim edge chunks this much quieter than the segment's peak
    faint_db: float = 15      # drop segments this much quieter than the loudest speech


def find_speech(audio: np.ndarray, settings: VadSettings = VadSettings()) -> list[Segment]:
    """Return the cleaned-up speech segments in the audio."""
    flags = chunk_is_speech(audio, settings.aggressiveness)
    loudness = chunk_loudness_db(audio)
    segments = flags_to_segments(flags, CHUNK_S)
    segments = [trim_quiet_edges(s, loudness, settings.edge_trim_db) for s in segments]
    segments = drop_faint_segments(segments, loudness, settings.faint_db)
    segments = merge_short_gaps(segments, settings.min_pause_ms / 1000)
    return drop_short_segments(segments, settings.min_speech_ms / 1000)


def chunk_is_speech(audio: np.ndarray, aggressiveness: int) -> list[bool]:
    """Ask WebRTC VAD about each 20 ms chunk. It needs 16-bit integer samples."""
    vad = webrtcvad.Vad(aggressiveness)
    return [vad.is_speech(_to_pcm16(chunk), SAMPLE_RATE)
            for chunk in split_into_chunks(audio)]


def _to_pcm16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()


def chunk_loudness_db(audio: np.ndarray) -> np.ndarray:
    """Loudness of each 20 ms chunk in decibels (0 dB = full scale, silence ~ -100)."""
    rms = np.array([np.sqrt(np.mean(c ** 2)) for c in split_into_chunks(audio)])
    return 20 * np.log10(rms + 1e-9)


def flags_to_segments(flags: list[bool], chunk_s: float) -> list[Segment]:
    """Turn [F, T, T, F, T] into segments covering each run of True."""
    segments, run_start = [], None
    for i, is_speech in enumerate(flags + [False]):  # extra False closes a final run
        if is_speech and run_start is None:
            run_start = i
        elif not is_speech and run_start is not None:
            segments.append(Segment(round(run_start * chunk_s, 3), round(i * chunk_s, 3)))
            run_start = None
    return segments


def trim_quiet_edges(segment: Segment, loudness_db: np.ndarray, trim_db: float) -> Segment:
    """Shrink a segment so it starts and ends on a chunk near its loudest level."""
    first, last = round(segment.start / CHUNK_S), round(segment.end / CHUNK_S)
    threshold = _peak_db(segment, loudness_db) - trim_db
    while first < last - 1 and loudness_db[first] < threshold:
        first += 1
    while last - 1 > first and loudness_db[last - 1] < threshold:
        last -= 1
    return Segment(round(first * CHUNK_S, 3), round(last * CHUNK_S, 3))


def drop_faint_segments(segments: list[Segment], loudness_db: np.ndarray,
                        faint_db: float) -> list[Segment]:
    """Keep only segments whose peak is within faint_db of the loudest segment's peak."""
    if not segments:
        return []
    peaks = [_peak_db(s, loudness_db) for s in segments]
    threshold = max(peaks) - faint_db
    return [s for s, peak in zip(segments, peaks) if peak >= threshold]


def _peak_db(segment: Segment, loudness_db: np.ndarray) -> float:
    return loudness_db[round(segment.start / CHUNK_S):round(segment.end / CHUNK_S)].max()


def merge_short_gaps(segments: list[Segment], min_gap_s: float) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if merged and seg.start - merged[-1].end < min_gap_s:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)
    return merged


def drop_short_segments(segments: list[Segment], min_length_s: float) -> list[Segment]:
    return [seg for seg in segments if seg.duration >= min_length_s]


def find_pauses(speech: list[Segment]) -> list[Segment]:
    """The silent gaps *between* speech segments (not before the first or after the last)."""
    return [Segment(a.end, b.start) for a, b in zip(speech, speech[1:])]


def speech_end(speech: list[Segment]) -> float | None:
    """When the caller finished speaking, or None if they never spoke."""
    return speech[-1].end if speech else None
