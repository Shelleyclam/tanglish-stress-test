"""Read scenarios.yaml: one entry per caller clip, describing what was said and meant.

The loader checks every entry and reports *all* problems at once, so you can fix
the file in one pass instead of one error at a time.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCENARIO_TYPES = {
    "change_of_mind", "code_switch", "thinking_pause", "interruption",
    "long_silence", "callback_request", "messy_details",
}
REQUIRED_FIELDS = ("id", "type", "transcript", "final_intent")
AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")


class ScenarioError(ValueError):
    """scenarios.yaml has one or more problems. The message lists all of them."""


@dataclass
class Scenario:
    id: str
    type: str
    transcript: str
    final_intent: str
    abandoned_intent: str | None = None
    pauses: list[tuple[float, float]] = field(default_factory=list)  # (start_s, end_s)
    clip_path: Path | None = None


def load_scenarios(yaml_path: str | Path, clips_dir: str | Path = "clips",
                   require_clips: bool = True) -> list[Scenario]:
    """Read, check, and return all scenarios. Raises ScenarioError on any problem."""
    entries = _read_entries(Path(yaml_path))
    problems = _find_duplicate_ids(entries)
    scenarios = []
    for i, entry in enumerate(entries):
        label = f"entry {i + 1} ({entry.get('id', 'no id')})"
        entry_problems = _check_entry(entry)
        if not entry_problems:
            scenario = _to_scenario(entry, Path(clips_dir))
            if require_clips and scenario.clip_path is None:
                entry_problems.append(f"no audio file named '{scenario.id}.*' in {clips_dir}")
            scenarios.append(scenario)
        problems += [f"{label}: {p}" for p in entry_problems]
    if problems:
        raise ScenarioError("scenarios.yaml has problems:\n  - " + "\n  - ".join(problems))
    return scenarios


def _read_entries(yaml_path: Path) -> list[dict]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return data.get("scenarios") or []


def _find_duplicate_ids(entries: list[dict]) -> list[str]:
    seen, duplicates = set(), set()
    for entry in entries:
        if entry.get("id") in seen:
            duplicates.add(entry["id"])
        seen.add(entry.get("id"))
    return [f"id '{d}' is used more than once" for d in sorted(duplicates)]


def _check_entry(entry: dict) -> list[str]:
    problems = [f"missing '{f}'" for f in REQUIRED_FIELDS if not entry.get(f)]
    if entry.get("type") and entry["type"] not in SCENARIO_TYPES:
        problems.append(f"unknown type '{entry['type']}'")
    if entry.get("type") == "change_of_mind" and not entry.get("abandoned_intent"):
        problems.append("change_of_mind needs an 'abandoned_intent'")
    problems += _check_pauses(entry.get("pauses") or [])
    return problems


def _check_pauses(pauses: list) -> list[str]:
    problems = []
    for pause in pauses:
        if not (isinstance(pause, list) and len(pause) == 2):
            problems.append(f"pause {pause} should be [start, end]")
        elif not 0 <= pause[0] < pause[1]:
            problems.append(f"pause {pause} must have 0 <= start < end")
    return problems


def _to_scenario(entry: dict, clips_dir: Path) -> Scenario:
    return Scenario(
        id=str(entry["id"]),
        type=entry["type"],
        transcript=entry["transcript"].strip(),
        final_intent=entry["final_intent"].strip(),
        abandoned_intent=(entry.get("abandoned_intent") or None),
        pauses=[(float(s), float(e)) for s, e in entry.get("pauses") or []],
        clip_path=find_clip(clips_dir, str(entry["id"])),
    )


def find_clip(clips_dir: Path, scenario_id: str) -> Path | None:
    """Find clips/<id>.wav or .m4a etc. Returns None if there is none."""
    for ext in AUDIO_EXTENSIONS:
        candidate = clips_dir / f"{scenario_id}{ext}"
        if candidate.exists():
            return candidate
    return None
