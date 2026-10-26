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


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)
