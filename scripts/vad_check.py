"""Compare VAD-detected pauses with the hand-labelled pauses in scenarios.yaml.

Usage:  .venv\\Scripts\\python scripts\\vad_check.py
"""

from tanglish.audio import load_clip
from tanglish.scenarios import load_scenarios
from tanglish.vad import VadSettings, find_pauses, find_speech, speech_end

SETTINGS = VadSettings()  # try e.g. VadSettings(aggressiveness=3)


def show(scenario) -> None:
    speech = find_speech(load_clip(scenario.clip_path), SETTINGS)
    pauses = find_pauses(speech)
    print(f"{scenario.id}: speech {speech[0].start:.2f}-{speech_end(speech):.2f}s")
    print("   hand:", ", ".join(f"{a:.1f}-{b:.1f}" for a, b in scenario.pauses) or "-")
    print("   VAD :", ", ".join(f"{p.start:.1f}-{p.end:.1f}" for p in pauses) or "-")


if __name__ == "__main__":
    print(SETTINGS)
    for s in load_scenarios("scenarios.yaml"):
        show(s)
