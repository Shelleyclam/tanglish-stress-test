"""Tiny client for a local Ollama server (no extra dependency, just urllib)."""

import json
import urllib.request

HOST = "http://localhost:11434"


def generate(model: str, prompt: str, raw: bool = False, options: dict | None = None,
             json_format: bool = False, host: str = HOST) -> str:
    body = {"model": model, "prompt": prompt, "raw": raw, "stream": False,
            "options": options or {}, "keep_alive": "30m"}
    if json_format:
        body["format"] = "json"
    request = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["response"]


def qwen_prompt(system: str, history: list[tuple[str, str]], user_text: str,
                assistant_prefix: str = "") -> str:
    """Build Qwen's chat format by hand, so we can make the model *continue*
    a reply that another (faster) model started."""
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for user, assistant in history:
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts) + assistant_prefix
