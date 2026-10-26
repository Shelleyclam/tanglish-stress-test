import numpy as np

from tanglish.agents.components import _first_sentence, trim_silence
from tanglish.synth import concat, silence, tone


def test_first_sentence_stops_at_sentence_end():
    assert _first_sentence("Yes, rooms are free. Shall I book?") == "Yes, rooms are free."


def test_initial_is_not_a_sentence_end():
    assert _first_sentence("You can visit A. Lagvi on Friday. Thanks.") == \
        "You can visit A. Lagvi on Friday."


def test_tts_silence_padding_is_trimmed():
    audio = concat(silence(0.3), tone(0.5), silence(0.2))

    trimmed = trim_silence(audio)

    assert len(trimmed) / 16000 < 0.51
    assert np.abs(trimmed[0]) <= 0.3
