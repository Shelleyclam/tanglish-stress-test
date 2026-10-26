"""Run the stress test.

  python run.py --agent local                 # all scenarios
  python run.py --agent local --clips 2       # first 2 scenarios
  python run.py --agent local --only com_01   # one scenario
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
import soundfile as sf

from tanglish.audio import SAMPLE_RATE, duration_seconds, load_clip
from tanglish.clock import RealClock
from tanglish.metrics.judge import judge_answer
from tanglish.metrics.summary import clip_checks
from tanglish.metrics.timing import timing_metrics
from tanglish.metrics.wer import word_error_rate
from tanglish.report import read_manual_labels, render_report, write_labels_csv
from tanglish.scenarios import Scenario, load_scenarios
from tanglish.streamer import stream_call
from tanglish.vad import VadSettings, find_speech

CAVEATS = {
    "local": [
        "Baseline runs on a CPU-only laptop. Its latencies reflect this hardware, not Vaani's; "
        "they show how the *design* behaves, not how fast a production system is.",
        "reply_after_ms and hangup_after_s are assumptions: Vaani has not published them.",
        "WER uses the agent's own speech-to-text over the whole call, compared with your "
        "romanised transcript. Romanised Tamil has no fixed spelling, so some 'errors' are "
        "spelling variants.",
        "Whisper is forced to English; with auto-detect it labelled Tanglish as Telugu and "
        "returned non-Latin text. Forced to English, it often *translates* Tamil phrases "
        "(e.g. 'wait pannunga' -> 'wait for me'), which WER counts as errors even when the "
        "meaning survives. WER here measures 'words kept', not 'meaning kept'.",
        "Fillers that start in a pause are counted, but whether that feels like talking over "
        "you is your call when listening: stereo recordings are saved next to this report "
        "(left = you, right = agent).",
    ],
}


def build_agent(name: str):
    if name == "local":
        from tanglish.agents.local_pipeline import LocalPipelineAgent
        return LocalPipelineAgent.from_config()
    if name == "vaani":
        from tanglish.agents.vaani import VaaniAgent
        return VaaniAgent()
    raise SystemExit(f"unknown agent '{name}'")


async def run_clip(agent, scenario: Scenario, vad: VadSettings, audio_dir: Path,
                   use_judge: bool) -> dict:
    audio = load_clip(scenario.clip_path)
    speech = find_speech(audio, vad)
    recording, stats = await stream_call(agent, audio, RealClock())
    timing = timing_metrics(speech, scenario.pauses, recording.events)
    final_answer = " ".join(e.text for e in recording.events
                            if e.kind == "reply" and e.t >= timing.speech_end)
    judge = (judge_answer(scenario.transcript, scenario.final_intent,
                          scenario.abandoned_intent, final_answer) if use_judge else None)
    _save_stereo(audio_dir / f"{scenario.id}.wav", audio, recording.agent_audio(duration_seconds(audio)))
    return {
        "id": scenario.id, "type": scenario.type, "transcript": scenario.transcript,
        "final_intent": scenario.final_intent, "abandoned_intent": scenario.abandoned_intent,
        "pauses": scenario.pauses, "clip_seconds": round(duration_seconds(audio), 2),
        "stream": asdict(stats), "events": [e.to_dict() for e in recording.events],
        "stt_transcript": recording.stt_transcript,
        "wer": (word_error_rate(scenario.transcript, recording.stt_transcript)
                if recording.stt_transcript is not None else None),
        "timing": timing.to_dict(), "final_answer": final_answer, "judge": judge,
        "manual_label": None, "notes": recording.notes, "agent_stats": recording.stats,
        "checks": clip_checks(timing.to_dict(), judge["label"] if judge else None),
    }


def _save_stereo(path: Path, caller: np.ndarray, agent: np.ndarray) -> None:
    n = max(len(caller), len(agent))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[:len(caller), 0], stereo[:len(agent), 1] = caller, agent
    sf.write(str(path), stereo, SAMPLE_RATE, subtype="PCM_16")


async def run(args) -> Path:
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only.split(",")]
    scenarios = scenarios[: args.clips] if args.clips else scenarios
    agent = build_agent(args.agent)
    print(f"Warming up {agent.name} agent...")
    await agent.warm_up()

    stem = Path(args.out) / f"{agent.name}_{date.today().isoformat()}"
    audio_dir = Path(f"{stem}_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    vad = VadSettings()
    clips = []
    for s in scenarios:
        print(f"  {s.id}: streaming {s.clip_path} in real time...")
        clips.append(await run_clip(agent, s, vad, audio_dir, not args.no_judge))
    results = {"agent": agent.name, "date": date.today().isoformat(),
               "machine": f"{platform.processor()} ({platform.system()}), CPU only",
               "agent_note": "Baseline: local models mimicking Vaani's published design.",
               "agent_config": agent.describe(), "vad_settings": asdict(vad),
               "caveats": CAVEATS.get(agent.name, []), "clips": clips}
    write_outputs(stem, results)
    return stem


def write_outputs(stem: Path, results: dict) -> None:
    labels_path = Path(f"{stem}_labels.csv")
    mine = read_manual_labels(labels_path)
    for c in results["clips"]:
        c["manual_label"] = mine.get(c["id"])
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
    parser.add_argument("--report-only", help="rebuild the report from an existing results JSON")
    args = parser.parse_args()
    if args.report_only:
        json_path = Path(args.report_only)
        write_outputs(json_path.with_suffix(""), json.loads(json_path.read_text(encoding="utf-8")))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
