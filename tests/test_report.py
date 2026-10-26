from tanglish.report import (judge_agreement, read_manual_labels, render_report,
                             write_labels_csv)


def fake_clip(clip_id="com_01", judge_label="final", manual=None):
    return {
        "id": clip_id, "type": "change_of_mind", "transcript": "saturday... no, friday",
        "final_intent": "friday", "abandoned_intent": "saturday", "pauses": [[1, 2]],
        "clip_seconds": 5.0, "events": [{"t": 3.1, "kind": "reply", "text": "Friday is free.",
                                          "duration": 1.0, "interrupted": False, "heard": "friday"}],
        "stream": {"chunks_sent": 300, "p95_lateness_ms": 1.0, "max_lateness_ms": 2.0,
                   "stopped_by": "agent_done"},
        "stt_transcript": "saturday no friday", "wer": 0.0, "final_answer": "Friday is free.",
        "timing": {"speech_end": 3.0, "first_audio_latency": 0.1, "reply_latency": 0.1,
                   "talk_over_s": 0.0, "fillers_in_pauses": 0, "fillers_talked_over": 0,
                   "early_replies": 0, "false_end": False},
        "judge": {"label": judge_label, "reason": "ok"}, "manual_label": manual, "notes": [],
        "language": "tanglish", "final_heard": "friday", "fillers_per_min": 6.0,
        "key_details": {"date": ["friday"]}, "details": {"date": True},
        "questions": ["Rooms on Friday?"], "questions_answered": [True],
        "invented_numbers": [], "reply_language": "english",
        "checks": {"key_details": True, "questions": True, "no_invented_numbers": True,
                   "answered_final": judge_label == "final", "no_talk_over": True,
                   "no_early_reply": True, "no_false_end": True},
    }


def fake_results(clips):
    return {"agent": "local", "date": "2026-10-27", "machine": "test", "agent_note": "test run",
            "agent_config": {"hangup_after_s": 5.0}, "vad_settings": {}, "caveats": ["a caveat"],
            "clips": clips}


def test_report_has_every_section_and_says_labels_are_missing():
    report = render_report(fake_results([fake_clip()]))

    for heading in ["## Headline", "## Understanding per clip", "## Timing per clip",
                    "## Across clips", "## By caller language", "## Pass rate per scenario type",
                    "## LLM judge vs your labels", "## What this means for the product",
                    "## Worst 1 clips", "## Not measured / caveats"]:
        assert heading in report
    assert "Not measured yet" in report
    assert "a caveat" in report
    assert "Key details heard correctly: 1/1 (100%)" in report
    assert "no english or tamil control clips yet" in report


def test_zero_early_replies_from_a_slow_agent_is_flagged_as_inconclusive():
    clip = fake_clip()
    clip["agent_stats"] = {"reply_attempts_abandoned": 3}

    report = render_report(fake_results([clip]))

    assert "neither confirm nor rule out H1" in report


def test_labels_csv_keeps_what_you_filled_in(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels_csv(path, [fake_clip()])
    path.write_text(path.read_text(encoding="utf-8").replace("final,\n", "final,abandoned\n"),
                    encoding="utf-8")

    write_labels_csv(path, [fake_clip()])  # e.g. after a re-run

    assert read_manual_labels(path) == {"com_01": "abandoned"}


def test_agreement_counts_only_labelled_clips():
    clips = [fake_clip("a", "final", "final"), fake_clip("b", "final", "abandoned"),
             fake_clip("c", "final", None)]

    agreement = judge_agreement(clips)

    assert agreement["labelled"] == 2 and agreement["agreed"] == 1
    assert agreement["disagreements"] == [("b", "final", "abandoned")]
