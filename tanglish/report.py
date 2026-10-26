"""Turn the raw results (JSON) into the Markdown report, and handle manual labels.

Every number in the report comes from the results file. Anything that could not
be measured is shown as "not measured" with the reason, never estimated.
"""

import csv
from pathlib import Path

from tanglish.metrics.summary import (CHECKS, failed_count, pass_rates_by_type,
                                      percentiles)

LABELS = {"final", "abandoned", "both", "neither", "no_answer"}
LABEL_COLUMNS = ["clip_id", "type", "final_intent", "abandoned_intent",
                 "agent_final_answer", "judge_label", "my_label"]


# --- manual labels ---

def write_labels_csv(path: Path, clips: list[dict]) -> None:
    """Create/refresh the labels file, keeping any my_label values already filled in."""
    mine = read_manual_labels(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for c in clips:
            writer.writerow({"clip_id": c["id"], "type": c["type"],
                             "final_intent": c["final_intent"],
                             "abandoned_intent": c["abandoned_intent"] or "",
                             "agent_final_answer": c["final_answer"],
                             "judge_label": c["judge"]["label"] if c["judge"] else "",
                             "my_label": mine.get(c["id"], "")})


def read_manual_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return {r["clip_id"]: r["my_label"].strip().lower() for r in rows
                if r.get("my_label", "").strip().lower() in LABELS}


def judge_agreement(clips: list[dict]) -> dict:
    pairs = [(c["id"], c["judge"]["label"], c["manual_label"]) for c in clips
             if c.get("manual_label") and c["judge"]]
    agreed = [p for p in pairs if p[1] == p[2]]
    return {"labelled": len(pairs), "agreed": len(agreed),
            "disagreements": [p for p in pairs if p[1] != p[2]]}


# --- the report ---

def render_report(results: dict) -> str:
    clips = results["clips"]
    sections = [
        _header(results),
        _summary_table(clips),
        _across_clips(clips),
        _pass_rates(clips),
        _judge_section(clips),
        _what_it_means(results),
        _worst_clips(clips),
        _not_measured(results),
        _setup(results),
    ]
    return "\n\n".join(sections) + "\n"


def _fmt(value, unit: str = "", digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _header(results: dict) -> str:
    return (f"# Tanglish stress test: `{results['agent']}` agent, {results['date']}\n\n"
            f"{len(results['clips'])} caller clips · machine: {results['machine']}\n\n"
            f"> {results['agent_note']}")


def _summary_table(clips: list[dict]) -> str:
    rows = ["## Summary per clip", "",
            "| clip | type | first audio after you stopped | reply after you stopped "
            "| talk-over | early replies | fillers in your pauses | hung up early | WER | judge | checks passed |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        t = c["timing"]
        passed = sum(ok is True for ok in c["checks"].values())
        measured = sum(ok is not None for ok in c["checks"].values())
        rows.append(
            f"| {c['id']} | {c['type']} | {_fmt(t['first_audio_latency'], ' s')} "
            f"| {_fmt(t['reply_latency'], ' s')} | {_fmt(t['talk_over_s'], ' s')} "
            f"| {t['early_replies']} | {t['fillers_in_pauses']} | {_fmt(t['false_end'])} "
            f"| {_fmt(c['wer'])} | {c['judge']['label'] if c['judge'] else '–'} "
            f"| {passed}/{measured} |")
    rows.append("\n– = not measurable for that clip (e.g. no reply after you finished).")
    return "\n".join(rows)


def _across_clips(clips: list[dict]) -> str:
    metrics = {
        "First agent audio after caller stopped (s)": [c["timing"]["first_audio_latency"] for c in clips],
        "First reply after caller stopped (s)": [c["timing"]["reply_latency"] for c in clips],
        "Agent talking over caller (s)": [c["timing"]["talk_over_s"] for c in clips],
        "Speech-to-text WER": [c["wer"] for c in clips],
    }
    rows = ["## Across clips", "", "| metric | p50 | p95 | clips measured |", "|---|---|---|---|"]
    for name, values in metrics.items():
        p = percentiles(values)
        rows.append(f"| {name} | {_fmt(p['p50'])} | {_fmt(p['p95'])} | {p['n']} of {len(values)} |")
    if len(clips) < 20:
        rows.append(f"\nWith only {len(clips)} clips, p95 is essentially the worst clip. "
                    "Treat these as early signals, not statistics.")
    return "\n".join(rows)


def _pass_rates(clips: list[dict]) -> str:
    rates = pass_rates_by_type(clips)
    names = list(CHECKS)
    rows = ["## Pass rate per scenario type", "",
            "| type | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for scenario_type, checks in sorted(rates.items()):
        cells = [f"{p}/{n}" if n else "–" for p, n in (checks[c] for c in names)]
        rows.append(f"| {scenario_type} | " + " | ".join(cells) + " |")
    rows += ["", "Checks:"] + [f"- **{name}**: {desc}" for name, desc in CHECKS.items()]
    return "\n".join(rows)


def _judge_section(clips: list[dict]) -> str:
    agreement = judge_agreement(clips)
    lines = ["## LLM judge vs your labels", ""]
    if agreement["labelled"] == 0:
        lines.append("**Not measured yet:** no manual labels. Fill in `my_label` in the labels "
                     "CSV next to this report (final / abandoned / both / neither / no_answer), "
                     "then run `python run.py --report-only <results.json>`. Until then, "
                     "judge verdicts are unverified and should not be trusted on their own.")
        return "\n".join(lines)
    lines.append(f"Agreement: **{agreement['agreed']}/{agreement['labelled']}** labelled clips.")
    for clip_id, judge, mine in agreement["disagreements"]:
        lines.append(f"- {clip_id}: judge said **{judge}**, you said **{mine}**")
    return "\n".join(lines)


def _what_it_means(results: dict) -> str:
    clips = results["clips"]
    n = len(clips)
    early = [c["id"] for c in clips if c["timing"]["early_replies"] > 0]
    filler_over = [c["id"] for c in clips if c["timing"]["fillers_talked_over"] > 0]
    false_end = [c["id"] for c in clips if c["timing"]["false_end"]]
    no_reply = [c["id"] for c in clips if c["timing"]["reply_latency"] is None]
    lines = ["## What this means for the product", "",
             "_Generated from the numbers above; each line states only what was measured._", "",
             f"- **H1 – answering before the caller changes their mind:** the agent started a reply "
             f"before the caller had finished in {len(early)} of {n} clips"
             f"{_ids(early)}. Judge verdicts on the final answer: {_label_counts(clips)}."
             + _h1_caveat(clips, early),
             f"- **H2 – filler talking over mid-thought pauses:** a filler was still playing when the "
             f"caller resumed in {len(filler_over)} of {n} clips{_ids(filler_over)}.",
             f"- **H3 – silence timer cutting calls:** the agent hung up before the caller finished in "
             f"{len(false_end)} of {n} clips{_ids(false_end)}"
             f" (timer = {results['agent_config'].get('hangup_after_s', '?')} s)."
             + _h3_caveat(clips, results["agent_config"].get("hangup_after_s")),
             f"- **H4 – speech-to-text on code-switched speech:** median WER "
             f"{_fmt(percentiles([c['wer'] for c in clips])['p50'])}; see the transcripts below for "
             f"which words were lost. Without English-only control clips this cannot yet show "
             f"that WER *jumps* on code-switching, only how high it is.",
             ]
    if no_reply:
        lines.append(f"- No reply at all after the caller finished in {len(no_reply)} clip(s){_ids(no_reply)}.")
    return "\n".join(lines)


def _h1_caveat(clips: list[dict], early: list[str]) -> str:
    """Zero early replies only means something if the agent was fast enough to make one."""
    abandoned = sum(c.get("agent_stats", {}).get("reply_attempts_abandoned", 0) for c in clips)
    if early or not abandoned:
        return ""
    latency = percentiles([c["timing"]["reply_latency"] for c in clips])["p50"]
    return (f" However, the agent began preparing a reply {abandoned} times during the caller's "
            f"turn and abandoned every one when the caller resumed, before making a sound: its "
            f"replies take ~{_fmt(latency)} s here, longer than any pause. **So this run can "
            f"neither confirm nor rule out H1**; it needs an agent fast enough to speak inside "
            f"a pause (Vaani, or this baseline on a GPU).")


def _h3_caveat(clips: list[dict], timer_s: float | None) -> str:
    """Zero hang-ups only means something if the timer actually got to run."""
    runs = {c["id"]: c.get("agent_stats", {}).get("longest_timer_run_s") for c in clips}
    runs = {k: v for k, v in runs.items() if v is not None}
    if not runs or timer_s is None:
        return ""
    detail = ", ".join(f"{k} {v:.1f} s" for k, v in runs.items())
    text = f" Longest the timer actually ran per clip: {detail}."
    if max(runs.values()) < timer_s * 0.8:
        text += (" The timer pauses while the agent is thinking or speaking, and this slow "
                 "baseline was thinking through most pauses, so **H3 is untested here**, not passed.")
    return text


def _ids(ids: list[str]) -> str:
    return f" ({', '.join(ids)})" if ids else ""


def _label_counts(clips: list[dict]) -> str:
    counts: dict[str, int] = {}
    for c in clips:
        label = c["judge"]["label"] if c["judge"] else "not judged"
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def _worst_clips(clips: list[dict], limit: int = 5) -> str:
    ranked = sorted(clips, key=lambda c: (-failed_count(c), -(c["timing"]["reply_latency"] or 99)))
    lines = [f"## Worst {min(limit, len(clips))} clips"]
    for c in ranked[:limit]:
        failed = [k for k, ok in c["checks"].items() if ok is False]
        lines += ["", f"### {c['id']} ({c['type']}) – failed: {', '.join(failed) or 'none'}", "",
                  f"- **You said:** {c['transcript']}",
                  f"- **Agent's speech-to-text heard (whole call):** {c['stt_transcript'] or '–'}",
                  f"- **You finished speaking at:** {c['timing']['speech_end']:.2f} s",
                  f"- **Judge:** {c['judge']['label']} – {c['judge'].get('reason', '')}" if c["judge"] else "- **Judge:** –",
                  "", "| time | agent | duration | cut off by you | text | heard before replying |",
                  "|---|---|---|---|---|---|"]
        for e in c["events"]:
            lines.append(f"| {e['t']:.2f} s | {e['kind']} | {_fmt(e['duration'], ' s')} "
                         f"| {_fmt(e['interrupted'])} | {e['text']} | {e['heard']} |")
    return "\n".join(lines)


def _not_measured(results: dict) -> str:
    lines = ["## Not measured / caveats", ""] + [f"- {note}" for note in results["caveats"]]
    for c in results["clips"]:
        for note in c.get("notes", []):
            lines.append(f"- {c['id']}: {note}")
    return "\n".join(lines)


def _setup(results: dict) -> str:
    cfg = {k: v for k, v in results["agent_config"].items() if k != "system_prompt"}
    lines = ["## Setup", "", "Agent settings:", ""]
    lines += [f"- `{k}` = {v}" for k, v in cfg.items()]
    lines += ["", f"VAD (referee) settings: {results['vad_settings']}", "",
              "Streamer timing (how late each 20 ms chunk was sent vs schedule):", ""]
    for c in results["clips"]:
        s = c["stream"]
        lines.append(f"- {c['id']}: p95 {s['p95_lateness_ms']} ms, max {s['max_lateness_ms']} ms, "
                     f"stopped by {s['stopped_by']}")
    return "\n".join(lines)
