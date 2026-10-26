import numpy as np
import pytest

from tanglish.synth import concat, silence, speech_like, tone
from tanglish.vad import (Segment, VadSettings, drop_short_segments, find_pauses,
                          find_speech, flags_to_segments, merge_short_gaps, speech_end,
                          trim_quiet_edges)


# --- the clean-up steps, on hand-made inputs ---

def test_flags_become_segments():
    flags = [False, True, True, False, True]

    assert flags_to_segments(flags, 0.02) == [Segment(0.02, 0.06), Segment(0.08, 0.10)]


def test_short_gap_is_merged_long_gap_is_kept():
    segments = [Segment(0, 1), Segment(1.2, 2), Segment(3, 4)]

    merged = merge_short_gaps(segments, min_gap_s=0.3)

    assert merged == [Segment(0, 2), Segment(3, 4)]


def test_short_burst_is_dropped():
    segments = [Segment(0, 1), Segment(2, 2.05), Segment(3, 4)]

    assert drop_short_segments(segments, 0.1) == [Segment(0, 1), Segment(3, 4)]


def test_pauses_are_gaps_between_speech():
    speech = [Segment(1, 2), Segment(3.5, 5)]

    assert find_pauses(speech) == [Segment(2, 3.5)]
    assert speech_end(speech) == 5


def test_no_speech_means_no_end():
    assert speech_end([]) is None


# --- the whole pipeline, on synthetic audio ---

def test_finds_a_thinking_pause():
    # 1 s quiet, 2 s talking, 1.5 s thinking, 1 s talking, 1 s quiet
    audio = concat(silence(1), speech_like(2), silence(1.5), speech_like(1), silence(1))

    speech = find_speech(audio)

    assert len(speech) == 2
    [pause] = find_pauses(speech)
    assert pause.start == pytest.approx(3.0, abs=0.1)
    assert pause.end == pytest.approx(4.5, abs=0.1)
    assert speech_end(speech) == pytest.approx(5.5, abs=0.03)  # within ~1 chunk


def test_hangover_is_trimmed_from_speech_end():
    # Raw WebRTC VAD reports this ending ~100 ms late; the trimmed end must be exact.
    audio = concat(silence(1), speech_like(2), silence(1))

    assert speech_end(find_speech(audio)) == pytest.approx(3.0, abs=0.03)


def test_quiet_edge_chunks_are_trimmed():
    loudness = np.array([-100, -30, -25, -30, -90, -100], dtype=float)
    segment = Segment(0.0, 0.12)  # all 6 chunks

    assert trim_quiet_edges(segment, loudness, trim_db=35) == Segment(0.02, 0.08)


def test_pause_shorter_than_threshold_is_ignored():
    audio = concat(speech_like(1), silence(0.2), speech_like(1))

    assert len(find_speech(audio, VadSettings(min_pause_ms=300))) == 1


def test_faint_background_noise_inside_a_pause_is_ignored():
    # Like silence_01: a long pause with a faint voice-like sound in the middle.
    faint = speech_like(0.3, volume=0.02)  # ~23 dB quieter than the caller
    audio = concat(speech_like(2), silence(2), faint, silence(2), speech_like(1))

    [pause] = find_pauses(find_speech(audio))

    assert pause.duration == pytest.approx(4.3, abs=0.05)


def test_pure_tone_and_silence_are_not_speech():
    assert find_speech(concat(silence(1), tone(1), silence(1))) == []
