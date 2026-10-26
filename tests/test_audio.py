import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

from tanglish.audio import (SAMPLE_RATE, duration_seconds, load_clip, save_wav,
                            samples_per_chunk, split_into_chunks)
from tanglish.synth import concat, silence, tone


def test_stereo_44k_wav_becomes_16k_mono(tmp_path):
    # A typical phone export: 44.1 kHz stereo, 2 seconds long.
    left = tone(2.0, sample_rate=44_100)
    stereo = np.stack([left, left], axis=1)
    path = tmp_path / "phone.wav"
    sf.write(path, stereo, 44_100)

    audio = load_clip(path)

    assert audio.ndim == 1  # mono
    assert duration_seconds(audio) == pytest.approx(2.0, abs=0.01)


def test_m4a_can_be_loaded(tmp_path):
    wav = tmp_path / "in.wav"
    m4a = tmp_path / "in.m4a"
    save_wav(wav, tone(1.0))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-v", "error", "-i", str(wav), str(m4a)], check=True)

    audio = load_clip(m4a)

    assert duration_seconds(audio) == pytest.approx(1.0, abs=0.05)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_clip("does/not/exist.wav")


def test_one_second_is_fifty_chunks_of_20ms():
    chunks = split_into_chunks(silence(1.0))

    assert samples_per_chunk() == 320
    assert len(chunks) == 50
    assert all(len(c) == 320 for c in chunks)


def test_last_chunk_is_padded_with_silence():
    audio = concat(silence(1.0), tone(0.005))  # 5 ms extra

    chunks = split_into_chunks(audio)

    assert len(chunks) == 51
    assert np.all(chunks[-1][80:] == 0)  # 5 ms = 80 samples, rest is padding


def test_save_then_load_roundtrip(tmp_path):
    original = concat(tone(0.5), silence(0.5))
    path = tmp_path / "roundtrip.wav"

    save_wav(path, original)
    loaded = load_clip(path)

    assert len(loaded) == len(original)
    assert np.max(np.abs(loaded - original)) < 0.001


def test_three_seconds_is_150_chunks():
    audio = concat(tone(1.0), silence(1.0), tone(1.0))

    chunks = split_into_chunks(audio)

    assert len(chunks) == 150
