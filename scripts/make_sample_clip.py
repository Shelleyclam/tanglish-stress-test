"""Write clips/sample_synthetic.wav: a voice-like buzz with a mid-turn pause.

It has no words, so understanding metrics are meaningless on it. It is there so
you can run the whole pipeline (streaming, VAD, fillers, timing, report) before
recording your own calls. Described in scenarios.sample.yaml.

  python scripts/make_sample_clip.py
"""

from pathlib import Path

from tanglish.audio import save_wav
from tanglish.synth import concat, silence, speech_like

OUT = Path("clips/sample_synthetic.wav")


def main() -> None:
    audio = concat(silence(0.5),
                   speech_like(2.0),            # "caller" starts talking
                   silence(1.2),                # thinking pause mid-turn, 2.5-3.7 s
                   speech_like(2.5, pitch=140),  # carries on
                   silence(0.5))
    OUT.parent.mkdir(exist_ok=True)
    save_wav(OUT, audio)
    print(f"Wrote {OUT} ({len(audio) / 16000:.1f} s)")


if __name__ == "__main__":
    main()
