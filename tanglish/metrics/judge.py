"""LLM-as-judge: did the agent's answer address the caller's final or abandoned intent?

An LLM judge is fast but can be wrong, so every verdict is shown next to the
manual label column you fill in, and the report shows how often the two agree.
"""

import json

from tanglish import ollama

JUDGE_MODEL = "qwen2.5:7b"

PROMPT = """You are grading a phone agent. The caller said (exact transcript):
"{transcript}"

The caller's FINAL request: {final}
{abandoned_line}
The agent's answer, after the caller finished speaking:
"{answer}"

Return JSON with keys:
  "addresses_final": true if the answer responds to the FINAL request,
  "acts_on_abandoned": true if the answer responds to the ABANDONED request as if it still stood,
  "reason": one short sentence."""


def judge_answer(transcript: str, final_intent: str, abandoned_intent: str | None,
                 answer: str, model: str = JUDGE_MODEL) -> dict:
    if not answer.strip():
        return {"label": "no_answer", "reason": "the agent gave no reply after the caller finished"}
    abandoned_line = (f"The caller's ABANDONED request (they changed their mind): {abandoned_intent}\n"
                      if abandoned_intent else "There is no abandoned request.\n")
    prompt = PROMPT.format(transcript=transcript, final=final_intent,
                           abandoned_line=abandoned_line, answer=answer)
    raw = ollama.generate(model, prompt, json_format=True, options={"temperature": 0})
    verdict = json.loads(raw)
    return {"label": to_label(bool(verdict.get("addresses_final")),
                              bool(verdict.get("acts_on_abandoned")) and bool(abandoned_intent)),
            "reason": verdict.get("reason", ""), "model": model}


def to_label(addresses_final: bool, acts_on_abandoned: bool) -> str:
    if addresses_final and acts_on_abandoned:
        return "both"
    if addresses_final:
        return "final"
    if acts_on_abandoned:
        return "abandoned"
    return "neither"
