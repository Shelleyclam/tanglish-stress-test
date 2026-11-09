"""Run the stress test.

  python run.py --agent local                 # all scenarios
  python run.py --agent local --clips 2       # first 2 scenarios
  python run.py --agent local --only com_01   # one scenario
  python run.py --rescore reports/local_2026-10-27.json       # new metrics / edited scenarios.yaml,
                                                              # without re-running the agent
  python run.py --report-only reports/local_2026-10-27.json   # rebuild report after labelling
"""

import argparse
import asyncio
import json
import platform
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from tanglish.audio import SAMPLE_RATE, duration_seconds, load_clip
from tanglish.clock import RealClock
from tanglish.report import read_manual_labels, render_report, write_labels_csv
from tanglish.scenarios import Scenario, load_scenarios
from tanglish.scoring import caller_speech, score_clip
from tanglish.streamer import stream_call
from tanglish.vad import VadSettings

MIN_FREE_RAM_GB = 1.5  # the local agent needs ~6.6 GB; below this headroom, Whisper can fail

CAVEATS = {
    "local": [
        "Baseline runs on a CPU-only laptop. Its latencies reflect this hardware, not Vaani's; "
        "they show how the *design* behaves, not how fast a production system is. H1 and H3 "
        "need an agent fast enough to reply inside a pause, so they are left for the Vaani run.",
        "reply_after_ms and hangup_after_s are assumptions: Vaani has not published them.",
        "The agent's system prompt gives it facts (all dates available, Rs 3,500 per night, "
        "Rs 1,000 early check-in). 'Invented numbers' means numbers outside those facts and "
        "outside what the caller said; only numbers are checked, not other claims.",
        "Reply language is English because the system prompt says so: a setting, not a finding. "
        "Whether a Tanglish caller minds an English reply needs a listener, not this harness.",
        "Key details are checked in what the agent's speech-to-text heard before its final "
        "reply, using the accepted spellings in scenarios.yaml.",
        "WER uses the agent's own speech-to-text over the whole call, compared with your "
        "romanised transcript. Whisper is forced to English (auto-detect labelled Tanglish as "
        "Telugu) and often *translates* Tamil phrases ('wait pannunga' -> 'wait for me'), "
        "which WER counts as errors even when the meaning survives.",
        "Fillers that start in a pause are counted, but whether that feels like talking over "
        "you is your call when listening: stereo recordings are saved next to this report "
        "(left = you, right = agent).",
    ],
}


AGENT_NOTES = {
    "local": "Baseline: local models mimicking Vaani's published design.",
    "vaani": "Vaani's hosted agent over WebRTC; this machine only streams audio.",
}


def build_agent(name: str):
    if name == "local":
        from tanglish.agents.local_pipeline import LocalPipelineAgent
        return LocalPipelineAgent.from_config()
    if name == "vaani":
        from tanglish.agents.vaani import VaaniAgent
        return VaaniAgent()
    raise SystemExit(f"unknown agent '{name}'")


def free_ram_gb() -> float:
    return round(psutil.virtual_memory().available / 1e9, 1)


async def run_call(agent, scenario: Scenario, vad: VadSettings, audio_dir: Path) -> dict:
    """Stream one clip into the agent and keep everything needed for scoring."""
    audio = load_clip(scenario.clip_path)
    ram_before = free_ram_gb()
    recording, stats = await stream_call(agent, audio, RealClock())
    _save_stereo(audio_dir / f"{scenario.id}.wav", audio, recording.agent_audio(duration_seconds(audio)))
    return {
        "clip_seconds": round(duration_seconds(audio), 2),
        "caller_speech": caller_speech(scenario, vad),
        "stream": asdict(stats), "free_ram_gb_at_start": ram_before,
        "events": [e.to_dict() for e in recording.events],
        "stt_transcript": recording.stt_transcript,
        "notes": recording.notes, "agent_stats": recording.stats,
    }


def clip_record(scenario: Scenario, raw: dict, agent_facts: str, use_judge: bool) -> dict:
    """Scenario description + raw call + scores, as stored in the results JSON."""
    described = {"id": scenario.id, "type": scenario.type, "language": scenario.language,
                 "transcript": scenario.transcript, "final_intent": scenario.final_intent,
                 "abandoned_intent": scenario.abandoned_intent, "pauses": scenario.pauses,
                 "key_details": scenario.key_details, "questions": scenario.questions}
    return described | raw | score_clip(scenario, raw, agent_facts, use_judge) | {"manual_label": None}


def _save_stereo(path: Path, caller: np.ndarray, agent: np.ndarray) -> None:
    n = max(len(caller), len(agent))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[:len(caller), 0], stereo[:len(agent), 1] = caller, agent
    sf.write(str(path), stereo, SAMPLE_RATE, subtype="PCM_16")


async def run(args) -> None:
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only.split(",")]
    scenarios = scenarios[: args.clips] if args.clips else scenarios
    agent = build_agent(args.agent)
    print(f"Warming up {agent.name} agent...")
    await agent.warm_up()
    _warn_if_low_memory()

    stem = Path(args.out) / f"{agent.name}_{date.today().isoformat()}"
    audio_dir = Path(f"{stem}_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    vad = VadSettings()
    config = agent.describe()
    clips = []
    for s in scenarios:
        print(f"  {s.id}: streaming {s.clip_path} in real time...")
        raw = await run_call(agent, s, vad, audio_dir)
        clips.append(clip_record(s, raw, config.get("system_prompt", ""), not args.no_judge))
    results = {"agent": agent.name, "date": date.today().isoformat(),
               "machine": f"{platform.processor()} ({platform.system()}), CPU only",
               "agent_note": AGENT_NOTES.get(agent.name, ""),
               "agent_config": config, "vad_settings": asdict(vad),
               "caveats": CAVEATS.get(agent.name, []), "clips": clips}
    write_outputs(stem, results)


def _warn_if_low_memory() -> None:
    free = free_ram_gb()
    if free < MIN_FREE_RAM_GB:
        print(f"WARNING: only {free} GB RAM free with models loaded (want >= {MIN_FREE_RAM_GB}). "
              "Close other apps, or expect speech-to-text failures (logged in the report).")


def rescore(json_path: Path, scenarios_path: str, use_judge: bool) -> None:
    """Recompute every metric from saved calls, using the current scenarios.yaml."""
    results = json.loads(json_path.read_text(encoding="utf-8"))
    scenarios = {s.id: s for s in load_scenarios(scenarios_path)}
    facts = results["agent_config"].get("system_prompt", "")
    results["vad_settings"] = asdict(VadSettings())
    results["caveats"] = CAVEATS.get(results["agent"], results["caveats"])
    rescored = []
    for clip in results["clips"]:
        s = scenarios[clip["id"]]
        raw = {k: clip[k] for k in ("clip_seconds", "stream", "events", "stt_transcript",
                                     "notes", "agent_stats") if k in clip}
        raw["caller_speech"] = clip.get("caller_speech") or caller_speech(s, VadSettings())
        raw["free_ram_gb_at_start"] = clip.get("free_ram_gb_at_start")
        print(f"  rescoring {s.id}...")
        rescored.append(clip_record(s, raw, facts, use_judge))
    results["clips"] = rescored
    write_outputs(json_path.with_suffix(""), results)


def write_outputs(stem: Path, results: dict) -> None:
    labels_path = Path(f"{stem}_labels.csv")
    mine = read_manual_labels(labels_path)
    for c in results["clips"]:
        labels = mine.get(c["id"], {})
        c["manual_label"], c["manual_questions"] = labels.get("intent"), labels.get("questions")
    Path(f"{stem}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_labels_csv(labels_path, results["clips"])
    Path(f"{stem}.md").write_text(render_report(results), encoding="utf-8")
    print(f"Wrote {stem}.md, {stem}.json, {labels_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="local", choices=["local", "vaani"])
    parser.add_argument("--scenarios", default="scenarios.yaml")
    parser.add_argument("--clips", type=int, help="only run the first N scenarios")
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--rescore", help="re-score an existing results JSON (no agent run)")
    parser.add_argument("--report-only", help="rebuild the report from an existing results JSON")
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.rescore), args.scenarios, not args.no_judge)
    elif args.report_only:
        json_path = Path(args.report_only)
        write_outputs(json_path.with_suffix(""), json.loads(json_path.read_text(encoding="utf-8")))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
