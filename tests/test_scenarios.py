import pytest

from tanglish.audio import save_wav
from tanglish.scenarios import ScenarioError, load_scenarios
from tanglish.synth import tone

GOOD_YAML = """
scenarios:
  - id: com_01
    type: change_of_mind
    transcript: "morning 10 venum... illa, evening 5"
    final_intent: "book 5 pm"
    abandoned_intent: "book 10 am"
    pauses:
      - [1.0, 2.5]
"""


def write(tmp_path, text):
    path = tmp_path / "scenarios.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_good_entry_loads_with_its_clip(tmp_path):
    save_wav(tmp_path / "com_01.wav", tone(1.0))

    [scenario] = load_scenarios(write(tmp_path, GOOD_YAML), clips_dir=tmp_path)

    assert scenario.type == "change_of_mind"
    assert scenario.abandoned_intent == "book 10 am"
    assert scenario.pauses == [(1.0, 2.5)]
    assert scenario.clip_path.name == "com_01.wav"


def test_uppercase_extension_is_found(tmp_path):
    save_wav(tmp_path / "com_01.WAV", tone(1.0))

    [scenario] = load_scenarios(write(tmp_path, GOOD_YAML), clips_dir=tmp_path)

    assert scenario.clip_path.name == "com_01.WAV"


def test_missing_clip_is_reported(tmp_path):
    with pytest.raises(ScenarioError, match="no audio file named 'com_01"):
        load_scenarios(write(tmp_path, GOOD_YAML), clips_dir=tmp_path)


def test_missing_clip_allowed_when_not_required(tmp_path):
    [scenario] = load_scenarios(write(tmp_path, GOOD_YAML), clips_dir=tmp_path,
                                require_clips=False)

    assert scenario.clip_path is None


def test_all_problems_reported_together(tmp_path):
    bad = """
scenarios:
  - id: a
    type: not_a_real_type
    transcript: "hi"
    final_intent: "x"
  - id: b
    type: change_of_mind
    transcript: "hi"
    final_intent: "x"
    pauses:
      - [3.0, 2.0]
"""
    with pytest.raises(ScenarioError) as error:
        load_scenarios(write(tmp_path, bad), clips_dir=tmp_path, require_clips=False)

    message = str(error.value)
    assert "unknown type 'not_a_real_type'" in message
    assert "needs an 'abandoned_intent'" in message
    assert "must have 0 <= start < end" in message


def test_duplicate_ids_are_reported(tmp_path):
    entry = """
  - id: same
    type: code_switch
    transcript: "hi"
    final_intent: "x"
"""
    with pytest.raises(ScenarioError, match="used more than once"):
        load_scenarios(write(tmp_path, "scenarios:" + entry + entry), clips_dir=tmp_path,
                       require_clips=False)


def test_blank_abandoned_intent_counts_as_none(tmp_path):
    text = """
scenarios:
  - id: p
    type: thinking_pause
    transcript: "hi"
    final_intent: "x"
    abandoned_intent: " "
"""
    [scenario] = load_scenarios(write(tmp_path, text), clips_dir=tmp_path,
                                require_clips=False)

    assert scenario.abandoned_intent is None


def test_empty_file_gives_no_scenarios(tmp_path):
    assert load_scenarios(write(tmp_path, "scenarios: []"), clips_dir=tmp_path) == []


def test_shipped_scenarios_yaml_is_valid():
    load_scenarios("scenarios.yaml", require_clips=False)
