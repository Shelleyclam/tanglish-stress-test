"""The three model pieces of the local pipeline: hear, think, speak.

Each is a small class with one method, so tests can swap in fakes.
"""

import re

import numpy as np

from tanglish import ollama
from tanglish.audio import resample


class WhisperSTT:
    def __init__(self, model_size: str = "small", language: str | None = "en",
                 download_root: str = "models/whisper") -> None:
        from faster_whisper import WhisperModel  # heavy import, only when used
        self.language = language
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                  download_root=download_root)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()


class PiperTTS:
    def __init__(self, voice_path: str) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(voice_path)

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.voice.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        return resample(audio, chunks[0].sample_rate)


class OllamaLLM:
    """A fast small model says the first sentence; a bigger model continues it.
    This mimics Vaani's 'small model for first tokens' design."""

    def __init__(self, fast_model: str, main_model: str, system_prompt: str) -> None:
        self.fast_model, self.main_model = fast_model, main_model
        self.system_prompt = system_prompt

    def first_sentence(self, history: list[tuple[str, str]], user_text: str) -> str:
        prompt = ollama.qwen_prompt(self.system_prompt, history, user_text)
        text = ollama.generate(self.fast_model, prompt, raw=True, options={
            "temperature": 0, "num_predict": 30, "stop": ["<|im_end|>", "\n"]})
        return _first_sentence(text)

    def continue_reply(self, history: list[tuple[str, str]], user_text: str,
                       first: str) -> str:
        prompt = ollama.qwen_prompt(self.system_prompt, history, user_text, first + " ")
        text = ollama.generate(self.main_model, prompt, raw=True, options={
            "temperature": 0, "num_predict": 60, "stop": ["<|im_end|>", "\n"]})
        return _first_sentence(text)  # keep phone replies short: one more sentence


def _first_sentence(text: str) -> str:
    match = re.match(r"\s*(.+?[.!?])(\s|$)", text)
    return (match.group(1) if match else text).strip()
