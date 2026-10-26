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
    raw = ollama.generate(model, prompt, json_format=True,
                          options={"temperature": 0, "num_ctx": 2048})  # same as agent: no reload
    verdict = json.loads(raw)
    return {"label": to_label(bool(verdict.get("addresses_final")),
                              bool(verdict.get("acts_on_abandoned")) and bool(abandoned_intent)),
            "reason": verdict.get("reason", ""), "model": model}


# One question per call: asking about all questions at once made the 7b judge
# miscount and mix questions up (see git history for the comparison).
QUESTION_PROMPT = """A caller asked a phone agent this question:
"{question}"

The agent replied:
"{answer}"

Does the reply answer THIS question? Judge only this question; ignore anything else the
caller asked. A reply that gives the key fact asked for counts as answered, even if it is
short. Acknowledging without giving the fact ("I can help with that") does not count.
Return JSON: {{"reason": "<why, in one sentence>", "answered": true or false}}"""


def judge_question(question: str, answer: str, model: str = JUDGE_MODEL) -> dict:
    """{"answered": bool, "reason": str}. Unverified, like every judge verdict."""
    if not answer.strip():
        return {"answered": False, "reason": "no reply after the caller finished"}
    raw = ollama.generate(model, QUESTION_PROMPT.format(question=question, answer=answer),
                          json_format=True, options={"temperature": 0, "num_ctx": 2048})
    verdict = json.loads(raw)
    return {"answered": bool(verdict.get("answered")), "reason": verdict.get("reason", "")}


def to_label(addresses_final: bool, acts_on_abandoned: bool) -> str:
    if addresses_final and acts_on_abandoned:
        return "both"
    if addresses_final:
        return "final"
    if acts_on_abandoned:
        return "abandoned"
    return "neither"
