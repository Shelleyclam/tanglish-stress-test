"""Turn the raw results (JSON) into the Markdown report, and handle manual labels.

Every number in the report comes from the results file. Anything that could not
be measured is shown as "not measured" with the reason, never estimated.
"""

import csv
from datetime import date
from pathlib import Path

from tanglish.metrics.summary import (CHECKS, failed_count, key_detail_totals,
                                      pass_rates_by_type, percentiles, question_totals)

LABELS = {"final", "abandoned", "both", "neither", "no_answer"}
LABEL_COLUMNS = ["clip_id", "type", "final_intent", "abandoned_intent", "agent_final_answer",
                 "judge_label", "my_label", "questions", "judge_questions", "my_questions"]
CONTROL_LANGUAGES = ("english", "tamil")


# --- manual labels ---
# my_label: final / abandoned / both / neither / no_answer
# my_questions: one y or n per question, in order, e.g. "y,y,n"

def write_labels_csv(path: Path, clips: list[dict]) -> None:
    """Create/refresh the labels file, keeping anything you already filled in."""
    previous = _read_rows(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for c in clips:
            kept = previous.get(c["id"], {})
            writer.writerow({"clip_id": c["id"], "type": c["type"],
                             "final_intent": c["final_intent"],
                             "abandoned_intent": c["abandoned_intent"] or "",
                             "agent_final_answer": c["final_answer"],
                             "judge_label": c["judge"]["label"] if c["judge"] else "",
                             "my_label": kept.get("my_label", ""),
                             "questions": " | ".join(c["questions"]),
                             "judge_questions": _yn(c["questions_answered"]),
                             "my_questions": kept.get("my_questions", "")})


def read_manual_labels(path: Path) -> dict[str, dict]:
    """{clip_id: {"intent": label or None, "questions": [bool, ...] or None}}"""
    labels = {}
    for clip_id, row in _read_rows(path).items():
        intent = row.get("my_label", "").strip().lower()
        labels[clip_id] = {"intent": intent if intent in LABELS else None,
                           "questions": _parse_yn(row.get("my_questions", ""))}
    return labels


def _read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def _yn(values: list[bool] | None) -> str:
    return ",".join("y" if v else "n" for v in values) if values else ""


def _parse_yn(text: str) -> list[bool] | None:
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts or any(p not in ("y", "n") for p in parts):
        return None
    return [p == "y" for p in parts]


def judge_agreement(clips: list[dict]) -> dict:
    pairs = [(c["id"], c["judge"]["label"], c["manual_label"]) for c in clips
             if c.get("manual_label") and c["judge"]]
    agreed = [p for p in pairs if p[1] == p[2]]
    q_pairs = [(c["id"], q, j, m) for c in clips
               if c.get("manual_questions") and c["questions_answered"]
               and len(c["manual_questions"]) == len(c["questions_answered"])
               for q, j, m in zip(c["questions"], c["questions_answered"], c["manual_questions"])]
    return {"labelled": len(pairs), "agreed": len(agreed),
            "disagreements": [p for p in pairs if p[1] != p[2]],
            "questions_labelled": len(q_pairs),
            "questions_agreed": sum(j == m for _, _, j, m in q_pairs),
            "question_disagreements": [p for p in q_pairs if p[2] != p[3]]}


# --- the report ---

def render_report(results: dict) -> str:
    clips = results["clips"]
    sections = [
        _header(results),
        _headline(clips),
        _understanding_table(clips),
        _timing_table(clips),
        _across_clips(clips),
        _by_language(clips),
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


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "not measured"


def _header(results: dict) -> str:
    return (f"# Tanglish stress test: `{results['agent']}` agent, {results['date']}\n\n"
            f"{len(results['clips'])} caller clips · machine: {results['machine']}\n\n"
            f"> {results['agent_note']}")


def _headline(clips: list[dict]) -> str:
    correct, total = key_detail_totals(clips)
    answered, judged, unjudged = question_totals(clips)
    invented = sum(len(c["invented_numbers"]) for c in clips)
    fillers = percentiles([c["fillers_per_min"] for c in clips])["p50"]
    missed = [f"{c['id']}: {', '.join(k for k, ok in c['details'].items() if not ok)}"
              for c in clips if not all(c["details"].values())]
    lines = ["## Headline", "",
             f"**Key details heard correctly: {_pct(correct, total)}**"
             + (f" (missed: {'; '.join(missed)})" if missed else ""), "",
             f"- Questions answered (LLM judge, unverified): {_pct(answered, judged)}"
             + (f"; {unjudged} not judged" if unjudged else ""),
             f"- Invented numbers in replies: {invented}",
             f"- Fillers per minute (median): {_fmt(fillers, digits=1)}"]
    skipped = [c["id"] for c in clips if c.get("wer_skipped")]
    if skipped:
        lines.append(f"- Speech-to-text WER: **left out** for {len(skipped)} of {len(clips)} clips. "
                     + WER_SCRIPT_NOTE)
    return "\n".join(lines)


WER_SCRIPT_NOTE = ("The agent's speech-to-text wrote Tamil words in Tamil script and your "
                   "transcript is romanised, so every Tamil word would count as an error even "
                   "when heard correctly. That is not a fair comparison; key details are checked "
                   "with Tamil-script spellings instead.")


def _understanding_table(clips: list[dict]) -> str:
    rows = ["## Understanding per clip", "",
            "| clip | type | caller language | key details heard | questions answered "
            "| invented numbers | judge (intent) | WER | reply language |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        details = c["details"]
        missed = [k for k, ok in details.items() if not ok]
        detail_cell = (f"{len(details) - len(missed)}/{len(details)}"
                       + (f" (missed {', '.join(missed)})" if missed else "")) if details else "–"
        q = c["questions_answered"]
        q_cell = f"{sum(q)}/{len(q)}" if q else "–"
        rows.append(f"| {c['id']} | {c['type']} | {c['language']} | {detail_cell} | {q_cell} "
                    f"| {', '.join(map(str, c['invented_numbers'])) or 'none'} "
                    f"| {c['judge']['label'] if c['judge'] else '–'} "
                    f"| {'n/c' if c.get('wer_skipped') else _fmt(c['wer'])} "
                    f"| {c['reply_language']} |")
    rows.append("\n– = not measured for that clip (no key details / questions listed, or judge off).")
    if any(c.get("wer_skipped") for c in clips):
        rows.append("n/c = WER not comparable: Tamil script vs romanised transcript.")
    return "\n".join(rows)


def _timing_table(clips: list[dict]) -> str:
    rows = ["## Timing per clip", "",
            "| clip | first audio after you stopped | reply after you stopped | talk-over "
            "| fillers/min | fillers in your pauses | early replies | hung up early |",
            "|---|---|---|---|---|---|---|---|"]
    for c in clips:
        t = c["timing"]
        rows.append(f"| {c['id']} | {_fmt(t['first_audio_latency'], ' s')} "
                    f"| {_fmt(t['reply_latency'], ' s')} | {_fmt(t['talk_over_s'], ' s')} "
                    f"| {_fmt(c['fillers_per_min'], digits=1)} | {t['fillers_in_pauses']} "
                    f"| {t['early_replies']} | {_fmt(t['false_end'])} |")
    return "\n".join(rows)


def _across_clips(clips: list[dict]) -> str:
    metrics = {
        "First agent audio after caller stopped (s)": [c["timing"]["first_audio_latency"] for c in clips],
        "First reply after caller stopped (s)": [c["timing"]["reply_latency"] for c in clips],
        "Agent talking over caller (s)": [c["timing"]["talk_over_s"] for c in clips],
        "Fillers per minute": [c["fillers_per_min"] for c in clips],
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


def _by_language(clips: list[dict]) -> str:
    rows = ["## By caller language (H4: does code-switching hurt speech-to-text?)", "",
            "| caller language | clips | median WER | key details heard |", "|---|---|---|---|"]
    for language in ("tanglish",) + CONTROL_LANGUAGES:
        group = [c for c in clips if c["language"] == language]
        if group:
            correct, total = key_detail_totals(group)
            rows.append(f"| {language} | {len(group)} | "
                        f"{_fmt(percentiles([c['wer'] for c in group])['p50'])} | {_pct(correct, total)} |")
    if any(c.get("wer_skipped") for c in clips):
        rows.append("\nWER is left out where the scripts differ (Tamil script vs romanised).")
    missing = [lang for lang in CONTROL_LANGUAGES if not any(c["language"] == lang for c in clips)]
    if missing:
        rows.append(f"\n**Not measured:** no {' or '.join(missing)} control clips yet, so this run "
                    "cannot show whether errors *increase* on code-switched speech.")
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
    a = judge_agreement(clips)
    lines = ["## LLM judge vs your labels", ""]
    if a["labelled"] == 0 and a["questions_labelled"] == 0:
        lines.append("**Not measured yet:** no manual labels. In the labels CSV next to this "
                     "report, fill in `my_label` (final / abandoned / both / neither / no_answer) "
                     "and `my_questions` (one y/n per question, e.g. `y,y,n`), then run "
                     "`python run.py --report-only <results.json>`. Until then, judge verdicts "
                     "(intent and questions answered) are unverified.")
        return "\n".join(lines)
    lines.append(f"- Intent: agreed on **{_pct(a['agreed'], a['labelled'])}** labelled clips.")
    for clip_id, judge, mine in a["disagreements"]:
        lines.append(f"  - {clip_id}: judge said **{judge}**, you said **{mine}**")
    lines.append(f"- Questions answered: agreed on **{_pct(a['questions_agreed'], a['questions_labelled'])}** "
                 "labelled questions.")
    for clip_id, question, judge, mine in a["question_disagreements"]:
        lines.append(f"  - {clip_id} \"{question}\": judge said **{_yn([judge])}**, "
                     f"you said **{_yn([mine])}**")
    return "\n".join(lines)


def _what_it_means(results: dict) -> str:
    clips = results["clips"]
    n = len(clips)
    correct, total = key_detail_totals(clips)
    filler_over = [c["id"] for c in clips if c["timing"]["fillers_talked_over"] > 0]
    early = [c["id"] for c in clips if c["timing"]["early_replies"] > 0]
    false_end = [c["id"] for c in clips if c["timing"]["false_end"]]
    lines = ["## What this means for the product", "",
             "_Generated from the numbers above; each line states only what was measured._", "",
             f"- **Understanding:** the agent heard {_pct(correct, total)} of key details correctly"
             f"{_detail_misses(clips)}. A wrong name or place costs one word in WER but can lose the booking.",
             f"- **H2 – filler talking over mid-thought pauses:** a filler was still playing when the "
             f"caller resumed in {len(filler_over)} of {n} clips{_ids(filler_over)}; median "
             f"{_fmt(percentiles([c['fillers_per_min'] for c in clips])['p50'], digits=1)} fillers per minute.",
             _h4_line(clips)
             + ("" if any(c["language"] in CONTROL_LANGUAGES for c in clips)
                else " Needs English-only and Tamil-only control clips to show an *increase*."),
             f"- **H1 – answering before the caller changes their mind:** early replies in "
             f"{len(early)} of {n} clips{_ids(early)}." + _h1_caveat(clips, early),
             f"- **H3 – silence timer cutting calls:** hung up before the caller finished in "
             f"{len(false_end)} of {n} clips{_ids(false_end)} "
             f"(timer = {results['agent_config'].get('hangup_after_s', '?')} s)."
             + _h3_caveat(clips, results["agent_config"].get("hangup_after_s")),
             ]
    return "\n".join(lines)


def _h4_line(clips: list[dict]) -> str:
    head = "- **H4 – speech-to-text on code-switched speech:** "
    p = percentiles([c["wer"] for c in clips])
    if p["n"] == 0 and any(c.get("wer_skipped") for c in clips):
        return head + "not measured by WER: " + WER_SCRIPT_NOTE
    return head + f"median WER {_fmt(p['p50'])} on Tanglish ({p['n']} of {len(clips)} clips)."


def _detail_misses(clips: list[dict]) -> str:
    misses = [f"{name} in {c['id']}" for c in clips for name, ok in c["details"].items() if not ok]
    return f" (wrong: {', '.join(misses)})" if misses else ""


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


def _worst_clips(clips: list[dict], limit: int = 5) -> str:
    ranked = sorted(clips, key=lambda c: (-failed_count(c), -(c["timing"]["reply_latency"] or 99)))
    lines = [f"## Worst {min(limit, len(clips))} clips"]
    for c in ranked[:limit]:
        failed = [k for k, ok in c["checks"].items() if ok is False]
        lines += ["", f"### {c['id']} ({c['type']}) – failed: {', '.join(failed) or 'none'}", "",
                  f"- **You said:** {c['transcript']}",
                  f"- **Agent heard before its final reply:** {c['final_heard'] or '–'}",
                  f"- **Key details:** " + (", ".join(f"{k} {'✓' if ok else '✗'}"
                                                     for k, ok in c["details"].items()) or "–"),
                  f"- **Agent's final answer:** {c['final_answer'] or '–'}",
                  f"- **Questions answered (judge):** {_questions(c)}",
                  f"- **Invented numbers:** {', '.join(map(str, c['invented_numbers'])) or 'none'}"
                  + "".join(f"; \"{d['text']}\" accepted (it is a {_weekday(d['date'])})"
                            for d in c.get("checked_dates", [])),
                  f"- **Judge (intent):** {c['judge']['label']} – {c['judge'].get('reason', '')}"
                  if c["judge"] else "- **Judge (intent):** –",
                  f"- **You finished speaking at:** {c['timing']['speech_end']:.2f} s",
                  "", "| time | agent | duration | cut off by you | text |", "|---|---|---|---|---|"]
        for e in c["events"]:
            lines.append(f"| {e['t']:.2f} s | {e['kind']} | {_fmt(e['duration'], ' s')} "
                         f"| {_fmt(e['interrupted'])} | {e['text']} |")
    return "\n".join(lines)


def _weekday(iso: str) -> str:
    return f"{date.fromisoformat(iso):%A}, {iso}"


def _questions(clip: dict) -> str:
    answered = clip["questions_answered"]
    if not clip["questions"]:
        return "none listed"
    if answered is None:
        return "not judged"
    reasons = clip.get("question_reasons") or [""] * len(answered)
    return "".join(f"\n  - {'✓' if ok else '✗'} {q} _(judge: {why})_"
                   for q, ok, why in zip(clip["questions"], answered, reasons))


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
    if "system_prompt" in results["agent_config"]:
        lines += ["", f"System prompt (the agent's given facts): \"{results['agent_config']['system_prompt']}\""]
    lines += ["", f"VAD (referee) settings: {results['vad_settings']}", "",
              "Per clip: streamer timing (how late each 20 ms chunk was sent) and free RAM at start:", ""]
    for c in results["clips"]:
        s = c["stream"]
        lines.append(f"- {c['id']}: p95 {s['p95_lateness_ms']} ms, max {s['max_lateness_ms']} ms, "
                     f"stopped by {s['stopped_by']}; free RAM {_fmt(c.get('free_ram_gb_at_start'), ' GB', 1)}")
    return "\n".join(lines)
