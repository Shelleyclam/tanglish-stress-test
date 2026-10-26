"""Load caller recordings into the one format every other module expects.

That format is: 16 kHz, mono, float32 samples between -1.0 and 1.0.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000  # samples per second; what speech-to-text models expect
CHUNK_MS = 20  # one "packet" of audio, like a phone call sends


def load_clip(path: str | Path) -> np.ndarray:
    """Read any audio file (wav, m4a, mp3...) and return 16 kHz mono samples."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _run_ffmpeg_to_raw(path)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _run_ffmpeg_to_raw(path: Path) -> bytes:
    """Ask ffmpeg to decode, downmix to mono, resample, and print raw floats."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
        "-i", str(path),
        "-ac", "1",                 # 1 audio channel = mono
        "-ar", str(SAMPLE_RATE),    # resample to 16 kHz
        "-f", "f32le", "-",         # raw 32-bit floats to stdout
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg could not read {path}: {result.stderr.decode()}")
    return result.stdout


def save_wav(path: str | Path, audio: np.ndarray) -> None:
    """Write samples as a normal 16-bit wav file you can play."""
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def samples_per_chunk(chunk_ms: int = CHUNK_MS) -> int:
    """20 ms at 16 kHz = 320 samples."""
    return SAMPLE_RATE * chunk_ms // 1000


def split_into_chunks(audio: np.ndarray, chunk_ms: int = CHUNK_MS) -> list[np.ndarray]:
    """Cut audio into equal chunks. The last one is padded with silence."""
    size = samples_per_chunk(chunk_ms)
    padding = (-len(audio)) % size
    padded = np.pad(audio, (0, padding))
    return list(padded.reshape(-1, size))


def duration_seconds(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE
