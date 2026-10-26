"""Generate fake audio (tones and silence) so tests run without real recordings."""

import numpy as np

from tanglish.audio import SAMPLE_RATE


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def tone(seconds: float, freq: float = 220.0, volume: float = 0.3,
         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A plain sine wave. Stands in for 'someone is talking'."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech_like(seconds: float, pitch: float = 150.0, volume: float = 0.3,
                sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A buzzy voice-like sound that VAD accepts as speech (a pure tone is rejected).

    A voice is a base pitch plus many harmonics, getting louder and softer about
    4 times a second (roughly one syllable each).
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    harmonics = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 15))
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    wave = harmonics * syllables
    return (volume * wave / np.abs(wave).max()).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)
